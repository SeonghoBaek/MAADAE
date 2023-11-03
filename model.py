# MAADAE[meidei]: Memory Augmented Adversarial Dual AUto Encoder
# Author: Seongho Baek
# e-mail: seonghobaek@gmail.com

USE_TF_2 = False

if USE_TF_2 is True:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
else:
    import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.utils import shuffle
import layers
import argparse
import time
import util

# scope
Qeury_Encoder_Scope = 'generator_encoder'
Image_Reconstructor_Scope = 'generator_decoder'
LATENT_Memory_scope = 'generator_mem'
SEGMENT_Encoder_scope = 'unet_generator_encoder'
SEGMENT_Decoder_scope = 'unet_generator_decoder'
DISC_scope = 'discriminator'


def load_images(file_name_list, base_dir=None, noise_mask=None, cutout=False, cutout_mask=None,
                add_eps=False, rotate=False, flip=False, gray_scale=False, shift=False, contrast=False):
    try:
        images = []
        gt_images = []
        seg_images = []

        for file_name in file_name_list:
            fullname = file_name
            if base_dir is not None:
                fullname = os.path.join(base_dir, file_name).replace("\\", "/")
            img = cv2.imread(fullname)

            if img is None:
                print('Load failed: ' + fullname)
                return None

            if contrast is True:
                if np.random.randint(0, 10) < 5:
                    img = cv2.resize(img, dsize=(input_width//2, input_height//2), interpolation=cv2.INTER_AREA)

            h, w, c = img.shape
            img = cv2.resize(img, dsize=(input_width, input_height), interpolation=cv2.INTER_AREA)

            if gray_scale is True:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if img is not None:
                img = np.array(img) * 1.0

                if rotate is True:
                    rot = np.random.randint(-10, 10)
                    img = util.rotate_image(img, rot)

                if flip is True:
                    if np.random.randint(low=0, high=10) > 5:
                        img = cv2.flip(img, 1)

                if shift is True:
                    x_shift = np.random.randint(-7, 7)
                    y_shift = np.random.randint(-7, 7)
                    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
                    img = cv2.warpAffine(img, M, (input_width, input_height))

                gt_img = img.copy()
                gt_img = np.array(gt_img)

                if add_eps is True:
                    img = img + np.random.uniform(low=-2.55, high=2.55, size=img.shape)

                if gray_scale is True:
                    img = np.expand_dims(img, axis=-1)
                    gt_img = np.expand_dims(gt_img, axis=-1)

                cut_mask = np.zeros_like(img)
                seg_img = cut_mask

                if cutout is True:
                    # square cut out
                    co_w = np.random.randint(low=1, high=input_width // 10)
                    co_h = np.random.randint(low=1, high=input_height // 10)
                    num_cutout = np.random.randint(low=5, high=20)

                    if np.random.randint(low=0, high=10) < 5:
                        if co_w <= co_h:
                            co_w = np.random.randint(low=1, high=5)
                        else:
                            co_h = np.random.randint(low=1, high=5)
                        noise_mask = None

                    for _ in range(num_cutout):
                        r_x = np.random.randint(low=60, high=input_width - co_w - 60)
                        r_y = np.random.randint(low=60, high=input_height - co_h - 60)
                        # img[r_x:r_x + co_w, r_y:r_y + co_h] = 0.0
                        cut_mask[r_x:r_x + co_w, r_y:r_y + co_h, :] = 1.0

                    if noise_mask is not None:
                        # cut_mask = mask_noise + cut_mask
                        cut_mask = noise_mask * cut_mask
                        cut_mask = np.where(cut_mask > 0.5, 1.0, 0)

                    rot = np.random.randint(-90, 90)
                    cut_mask = util.rotate_image(cut_mask, rot)
                    if cutout_mask is not None:
                        cut_mask = cut_mask * cutout_mask

                    # Segmentation Mode
                    seg_img = cut_mask
                    bg_img = (1.0 - seg_img) * img
                    fg_img = seg_img * img
                    alpha = np.random.uniform(low=0.5, high=0.7)

                    random_pixels = 2 * (0.1 + np.random.rand())
                    structural_noise = util.rotate_image(img, np.random.randint(0, 360))
                    cut_mask = np.abs(cut_mask * (structural_noise * random_pixels))

                    cut_mask = np.where(cut_mask > 255, 255, cut_mask)
                    img = bg_img + (alpha * fg_img + (1 - alpha) * cut_mask)
                else:
                    if noise_mask is not None:
                        cut_mask = noise_mask

                        # Segmentation Mode
                        seg_img = cut_mask
                        bg_img = (1.0 - seg_img) * img
                        fg_img = seg_img * img
                        alpha = np.random.uniform(low=0.5, high=0.7)

                        random_pixels = 2 * (0.1 + np.random.rand())
                        structural_noise = util.rotate_image(img, np.random.randint(0, 360))
                        cut_mask = np.abs(cut_mask * (structural_noise * random_pixels))
                        cut_mask = np.abs(np.where(cut_mask > 255, 255, cut_mask))
                        img = bg_img + (alpha * fg_img + (1 - alpha) * cut_mask)
                        # img = (1.0 - cut_mask) * img

                seg_img = np.average(seg_img, axis=-1)
                seg_img = np.expand_dims(seg_img, axis=-1)
                seg_images.append(seg_img)

                n_img = (img * 1.0) / 255.0
                n_gt_img = (gt_img * 1.0) / 255.0

                images.append(n_img)
                gt_images.append(n_gt_img)

    except cv2.error as e:
        print(e)
        return None

    return np.array(images), np.array(gt_images), np.array(seg_images)


def get_gradient_loss(img1, img2):
    # Laplacian second derivation
    image_a = img1  # tf.expand_dims(img1, axis=0)
    image_b = img2  # tf.expand_dims(img2, axis=0)

    dx_a, dy_a = tf.image.image_gradients(image_a)
    dx_b, dy_b = tf.image.image_gradients(image_b)

    # d2x_ax, d2y_ax = tf.image.image_gradients(dx_a)
    # d2x_bx, d2y_bx = tf.image.image_gradients(dx_b)
    # d2x_ay, d2y_ay = tf.image.image_gradients(dy_a)
    # d2x_by, d2y_by = tf.image.image_gradients(dy_b)

    # loss1 = tf.reduce_mean(tf.abs(tf.subtract(d2x_ax, d2x_bx))) + tf.reduce_mean(tf.abs(tf.subtract(d2y_ax, d2y_bx)))
    # loss2 = tf.reduce_mean(tf.abs(tf.subtract(d2x_ay, d2x_by))) + tf.reduce_mean(tf.abs(tf.subtract(d2y_ay, d2y_by)))

    loss1 = tf.reduce_mean(tf.abs(tf.subtract(dx_a, dx_b)))
    loss2 = tf.reduce_mean(tf.abs(tf.subtract(dy_a, dy_b)))

    return loss1 + loss2


def get_residual_loss(value, target, type='l1', alpha=1.0):
    if type == 'mse':
        loss = alpha * tf.reduce_mean(tf.square(tf.subtract(target, value)))
    elif type == 'ce':
        eps = 1e-10
        loss = alpha * tf.reduce_mean(-1 * target * tf.log(value + eps) - 1 * (1 - target) * tf.log(1 - value + eps))
        # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=value, labels=target))
    elif type == 'l1':
        loss = alpha * tf.reduce_mean(tf.abs(tf.subtract(target, value)))
    elif type == 'l2':
        loss = alpha * tf.reduce_mean(tf.square(tf.subtract(target, value)))
    elif type == 'entropy':
        eps = 1e-10
        loss = alpha * tf.reduce_mean(-1 * value * tf.log(value + eps))
    elif type == 'focal':
        eps = 1e-10
        q = tf.math.maximum(1 - value, eps)
        p = tf.math.maximum(value, eps)
        pos_loss = -(q ** 2) * tf.math.log(p)
        neg_loss = -(p ** 2) * tf.math.log(q)
        loss = alpha * tf.reduce_mean(target * pos_loss + (1 - target) * neg_loss)
    elif type == 'l1_focal':
        eps = 1e-10
        q = tf.math.maximum(1 - value, eps)
        p = tf.math.maximum(value, eps)
        pos_loss = -(q ** 4) * tf.math.log(p)
        neg_loss = -(p ** 4) * tf.math.log(q)
        f_loss = tf.reduce_mean(target * pos_loss + (1 - target) * neg_loss)
        l1_loss = tf.reduce_mean(tf.abs(tf.subtract(target, value)))
        loss = f_loss + l1_loss
    elif type == 'ssim_focal':
        eps = 1e-10
        q = tf.math.maximum(1 - value, eps)
        p = tf.math.maximum(value, eps)
        pos_loss = -(q ** 2) * tf.math.log(p)
        neg_loss = -(p ** 2) * tf.math.log(q)
        f_loss = tf.reduce_mean(target * pos_loss + (1 - target) * neg_loss)
        loss = tf.reduce_mean(1 - tf.image.ssim_multiscale(value, target, max_val=1.0)) + \
               f_loss
    elif type == 'ssim_l1':
        m = tf.reduce_mean(tf.abs(tf.subtract(value, target)))
        # num_patches = 16
        # img1 = tf.reshape(value, shape=[-1, num_patches, input_height // 4, input_width // 4, num_channel])
        # img2 = tf.reshape(target, shape=[-1, num_patches, input_height // 4, input_width // 4, num_channel])
        loss = alpha * tf.reduce_mean(1 - tf.image.ssim_multiscale(value, target, max_val=1.0)) + (1 - alpha) * m
    elif type == 'dice':
        y = tf.layers.flatten(target)
        y_pred = tf.layers.flatten(value)
        nominator = 2 * tf.reduce_sum(tf.multiply(y_pred, y)) + 1e-6
        denominator = tf.reduce_sum(y_pred) + tf.reduce_sum(y) + 1e-6
        loss = 1 - tf.divide(nominator, denominator)
    elif type == 'ft':
        # Focal Tversky
        y = tf.layers.flatten(target)
        y_pred = tf.layers.flatten(value)

        tp = tf.reduce_sum(y * y_pred) + 1e-6
        fn = tf.reduce_sum(y * (1 - y_pred)) + 1e-6
        fp = tf.reduce_sum((1 - y) * y_pred) + 1e-6
        tv = (tp + 1) / (tp + alpha * fn + (1 - alpha) * fp + 1)

        loss = (1 - tv) ** 2
    else:
        loss = 0.0

    return loss


def get_discriminator_loss(real, fake, type='wgan', gamma=1.0):
    if type == 'wgan':
        # wgan loss
        d_loss_real = tf.reduce_mean(real)
        d_loss_fake = tf.reduce_mean(fake)

        # W Distant: f(real) - f(fake). Maximizing W Distant.
        return gamma * (d_loss_fake - d_loss_real), d_loss_real, d_loss_fake
    elif type == 'ce':
        # cross entropy
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=real, labels=tf.ones_like(real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.zeros_like(fake)))
        return gamma * (d_loss_fake + d_loss_real), d_loss_real, d_loss_fake
    elif type == 'hinge':
        d_loss_real = tf.reduce_mean(tf.nn.relu(1.0 - real))
        d_loss_fake = tf.reduce_mean(tf.nn.relu(1.0 + fake))
        return gamma * (d_loss_fake + d_loss_real), d_loss_real, d_loss_fake
    elif type == 'ls':
        return tf.reduce_mean((real - fake) ** 2)


def query_encoder(in_tensor, activation=tf.nn.relu, norm='instance', scope='encoder', b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        print('Query encoder input: ' + str(in_tensor.get_shape().as_list()))
        num_bottleneck = bottleneck_num
        block_depth = unit_block_depth

        l = layers.conv(in_tensor, scope='init', filter_dims=[5, 5, block_depth], stride_dims=[1, 1],
                        non_linear_fn=activation)

        print(' Downsample: ' + str(l.get_shape().as_list()))

        for i in range(downsample_num + 1):
            block_depth = block_depth * 2
            l = layers.conv(l, scope='downsapmple_' + str(i), filter_dims=[4, 4, block_depth], stride_dims=[2, 2],
                            non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='encoder_norm_' + str(i))
            l = activation(l)

            print(' Downsample: ' + str(l.get_shape().as_list()))

        for i in range(num_bottleneck):
            l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], norm=norm, b_train=b_train,
                                          act_func=activation, scope=scope + '_bt_block_' + str(i))

            print(' Bottleneck Block : ' + str(l.get_shape().as_list()))

        if num_bottleneck > 0:
            l = activation(l)

        l = layers.conv(l, scope='latent', filter_dims=[3, 3, query_dimension], stride_dims=[1, 1],
                        non_linear_fn=activation)

        query = l

    return query


def image_reconstructor(content, activation=tf.nn.relu, norm='instance', scope='decoder', b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        print('Image reconstructor input: ' + str(content.get_shape().as_list()))
        block_depth = unit_block_depth * (2 ** upsample_num)

        l = content
        #l = layers.conv(l, scope='init', filter_dims=[3, 3, block_depth], stride_dims=[1, 1], non_linear_fn=activation)

        for i in range(decoder_bottleneck_num):
            l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], norm=norm, b_train=b_train,
                                          act_func=activation, scope=scope + '_bt_block_' + str(i))

            print(' Bottleneck Block : ' + str(l.get_shape().as_list()))

        if decoder_bottleneck_num > 0:
            l = layers.conv(l, scope='upsample_init', filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
                            non_linear_fn=activation)
        latents = []
        for i in range(upsample_num + 1):
            block_depth = block_depth // 2
            # l = layers.conv(l, scope='espcn_1' + str(i), filter_dims=[3, 3, block_depth * 2 * 2], stride_dims=[1, 1],
            #                non_linear_fn=None)
            # l = tf.nn.depth_to_space(l, 2)
            latents.append(l)
            _, h, w, _ = l.get_shape().as_list()
            l = tf.image.resize_images(l, size=[2 * h, 2 * w])
            l = layers.conv(l, scope='upsample_conv_' + str(i), filter_dims=[3, 3, block_depth],
                            stride_dims=[1, 1], non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='upsample_norm_' + str(i))
            l = activation(l)
            print(' Upsampling: ' + str(l.get_shape().as_list()))

        l = activation(l)
        img = layers.conv(l, scope='last', filter_dims=[5, 5, num_channel], stride_dims=[1, 1], non_linear_fn=None)
        print(' Output: ' + str(img.get_shape().as_list()))

    return img, latents


def segment_encoder(in_tensor, mem_latents, activation=tf.nn.relu, norm='instance', scope='unet_encoder', b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        print('Segment encoder input: ' + str(in_tensor.get_shape().as_list()))
        block_depth = segment_unit_block_depth
        lateral_layers = []
        feature_layers = []

        l = layers.conv(in_tensor, scope='init', filter_dims=[3, 3, 2*segment_unit_block_depth], stride_dims=[2, 2],
                        non_linear_fn=None)
        l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='init_norm')
        l = activation(l)

        print(' init1: ' + str(l.get_shape().as_list()))
        _, h, w, _ = l.get_shape().as_list()

        ml = tf.image.resize_images(mem_latents, size=[h, w])
        l = tf.concat([l, ml], axis=-1)
        l = layers.conv(l, scope='init2', filter_dims=[3, 3, segment_unit_block_depth], stride_dims=[1, 1], non_linear_fn=None)
        print(' init2: ' + str(l.get_shape().as_list()))
        l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='init2_norm')
        l = activation(l)
        l = layers.add_residual_block(l, filter_dims=[3, 3, segment_unit_block_depth], norm=norm, act_func=activation,
                                      b_train=b_train, scope='init_resblock')

        print(' Add Lateral: ' + str(l.get_shape().as_list()))
        lateral_layers.append(l)
        feature_layers.append(l)

        for i in range(segmentation_downsample_num):
            block_depth = block_depth * 2
            l = layers.conv(l, scope='downsapmple1_' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
                            non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='encoder_norm1_' + str(i))
            l = activation(l)
            _, h, w, _ = l.get_shape().as_list()
            ml = tf.image.resize_images(mem_latents, size=[h, w])
            l = tf.concat([l, ml], axis=-1)
            l = layers.conv(l, scope='mem_fusion_' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[1, 1], non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='mem_fusion_norm' + str(i))
            l = activation(l)
            l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], norm=norm, act_func=activation,
                                          b_train=b_train, scope='dnsample_block_' + str(i))
            print(' Add Lateral: ' + str(l.get_shape().as_list()))
            lateral_layers.append(l)
            feature_layers.append(l)

        for n_loop in range(num_shuffle_car):
            for i in range(len(lateral_layers)):
                _, h, w, c = lateral_layers[i].get_shape().as_list()
                for num_rblock in range(num_car):
                    print(' Conv Layer: ' + str(lateral_layers[i].get_shape().as_list()))
                    lateral_layers[i] = layers.add_residual_block(lateral_layers[i], filter_dims=[3, 3, c], norm=norm,
                                                                  b_train=b_train, act_func=activation,
                                                                  scope='loop_sqblock_' + str(n_loop) + str(i) + str(num_rblock))

            print('Shuffling...')
            fused_layers = []

            for i in range(len(lateral_layers)):
                _, l_h, l_w, l_c = lateral_layers[i].get_shape().as_list()
                mixed_layer = lateral_layers[i]
                print(' Shuffle to: ' + str(lateral_layers[i].get_shape().as_list()))

                for j in range(len(lateral_layers)):
                    if i != j:
                        l_lat = lateral_layers[j]
                        _, h, w, _ = l_lat.get_shape().as_list()

                        if l_h > h:
                            # Resize
                            l_lat = layers.conv(l_lat, scope='shuffling_' + str(n_loop) + str(i) + str(j),
                                                filter_dims=[3, 3, l_c],
                                                stride_dims=[1, 1], non_linear_fn=None)
                            l_lat = layers.conv_normalize(l_lat, norm=norm, b_train=b_train,
                                                          scope='shuffling_norm' + str(n_loop) + str(i) + str(j))
                            l_lat = activation(l_lat)
                            l_lat = tf.image.resize_images(l_lat, size=[l_h, l_w])
                        elif l_h < h:
                            ratio = h // l_h
                            num_downsample = int(np.log2(ratio))
                            for k in range(num_downsample):
                                l_lat = layers.conv(l_lat,
                                                    scope='shuffling_dn_' + str(n_loop) + str(i) + str(j) + str(k),
                                                    filter_dims=[3, 3, l_c], stride_dims=[2, 2], non_linear_fn=None)
                                l_lat = layers.conv_normalize(l_lat, norm=norm, b_train=b_train,
                                                              scope='shuffling_dn_norm' + str(n_loop) + str(i) + str(j) + str(k))
                            l_lat = activation(l_lat)
                        mixed_layer = tf.add(mixed_layer, l_lat)
                fused_layers.append(mixed_layer)
                if n_loop == (num_shuffle_car - 1):
                    break

            lateral_layers = fused_layers

    return lateral_layers, feature_layers


def lateral_add_merge(lateral_layers, activation=tf.nn.relu, norm='instance', b_train=False, scope='unet3_add_merge'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        _, target_h, target_w, target_depth = lateral_layers[-1].get_shape().as_list()

        l = lateral_layers[-1]
        for i in range(len(lateral_layers) - 1):
            lowl = lateral_layers[i]
            _, h, w, _ = lowl.get_shape().as_list()
            lowl = layers.conv(lowl, scope='ll_concat' + str(i), filter_dims=[3, 3, target_depth], stride_dims=[1, 1],
                               non_linear_fn=None)
            lowl = layers.conv_normalize(lowl, norm=norm, b_train=b_train, scope='ll_concat_norm' + str(i))
            lowl = activation(lowl)
            lowl = tf.image.resize_images(lowl, size=[target_h, target_w])

            l = tf.add(l, lowl)

        return l


def lateral_merge(lateral_layers, activation=tf.nn.relu, norm='instance',
                b_train=False, scope='unet3_merge'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        _, target_h, target_w, target_depth = lateral_layers[-1].get_shape().as_list()

        concat_layers = [lateral_layers[-1]]

        for i in range(len(lateral_layers) - 1):
            lowl = lateral_layers[i]
            _, h, w, _ = lowl.get_shape().as_list()

            lowl = layers.conv_norm_activation(lowl, filter_dims=[1, 1, target_depth], stride_dims=[1, 1],
                                               norm=norm, b_train=b_train, activation=activation,
                                               scope='ll_concat' + str(i))
            lowl = tf.image.resize_images(lowl, size=[target_h, target_w])
            '''
            ratio = 2 ** (len(lateral_layers) - 1 - i)
            lowl = layers.conv(lowl, scope=str(i) + '_ll_concat',
                               filter_dims=[3, 3, target_depth * ratio * ratio],
                               stride_dims=[1, 1], non_linear_fn=None)
            lowl = layers.conv_normalize(lowl, norm=norm, b_train=b_train, scope='norm_low_' + str(i))
            lowl = activation(lowl)
            lowl = tf.nn.depth_to_space(lowl, ratio)
            '''
            concat_layers.append(lowl)

        l = tf.concat(concat_layers, axis=-1)

        return l


def lateral_concat(lateral_layers, start_step=0, unit_block=8, activation=tf.nn.relu, norm='instance',
                 b_train=False, scope='unet3_concat'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        _, target_h, target_w, _ = lateral_layers[start_step].get_shape().as_list()

        concat_layers = []
        for i in range(len(lateral_layers) - start_step):
            lat = lateral_layers[i + start_step]

            _, h, w, c = lat.get_shape().as_list()

            if h > target_h:
                # lat = layers.conv(lat, scope='lat_' + str(i), filter_dims=[1, 1, unit_block], stride_dims=[1, 1],
                #                  non_linear_fn=None)
                # lat = layers.conv_normalize(lat, norm=norm, b_train=b_train, scope='norm_' + str(i))
                # lat = activation(lat)

                ratio = h // target_h
                num_downsample = int(np.log2(ratio))
                for j in range(num_downsample):
                    lat = layers.conv(lat, scope='dn_' + str(i) + str(j), filter_dims=[3, 3, c],
                                      stride_dims=[2, 2], non_linear_fn=None)

                    lat = layers.conv_normalize(lat, norm=norm, b_train=b_train,
                                                scope='dn_espcn_norm_' + str(i) + str(j))
                    lat = activation(lat)

            concat_layers.append(lat)

        l = tf.concat(concat_layers, axis=-1)

        return l


def segment_decoder(lateral_layers, feature_layers, activation=tf.nn.relu, norm='instance', scope='unet3_decoder', b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        print('Segment Decoder')

        lateral_layers.reverse()
        feature_layers.reverse()
        segment_layer_depth = segment_unit_block_depth // 2

        '''
        for i in range(len(lateral_layers)):
            _, h, w, c = lateral_layers[i].get_shape().as_list()
            lateral_layers[i] = tf.concat([lateral_layers[i], feature_layers[i]], axis=-1)
            lateral_layers[i] = layers.conv(lateral_layers[i], scope='feature_concat_' + str(i), filter_dims=[1, 1, c],
                                            stride_dims=[1, 1], non_linear_fn=None)
            lateral_layers[i] = layers.conv_normalize(lateral_layers[i], norm=norm, b_train=b_train,
                                                      scope='feature_concat_norm_' + str(i))
            lateral_layers[i] = activation(lateral_layers[i])

            for num_rblock in range(2):
                print('Decoder Residual: ' + str(lateral_layers[i].get_shape().as_list()))
                lateral_layers[i] = layers.add_residual_block(lateral_layers[i], filter_dims=[3, 3, c], norm=norm, b_train=b_train,
                                                              act_func=activation, scope='decoder_residual_' + str(i) + str(num_rblock))
            r = input_height // h
            lateral_layers[i] = layers.conv(lateral_layers[i], scope='upsacle_conv' + str(i),
                                            filter_dims=[3, 3, r * r * segment_layer_depth],
                                            stride_dims=[1, 1], non_linear_fn=None)
            lateral_layers[i] = layers.conv_normalize(lateral_layers[i], norm=norm, b_train=b_train,
                                                      scope='upscale_norm' + str(i))
            lateral_layers[i] = activation(lateral_layers[i])
            lateral_layers[i] = tf.nn.depth_to_space(lateral_layers[i], r)

        average_map = tf.concat(lateral_layers, axis=-1)
        average_map = layers.conv(average_map, scope='average_map_resize', filter_dims=[1, 1, segment_layer_depth],
                                    stride_dims=[1, 1], non_linear_fn=None)
        average_map = layers.conv_normalize(average_map, norm=norm, b_train=b_train,
                                              scope='average_map_resize_norm')
        average_map = activation(average_map)
        '''
        average_map = lateral_layers[-1]
        feature_map = feature_layers[-1]
        _, h, w, c = average_map.get_shape().as_list()
        average_map = tf.concat([average_map, feature_map], axis=-1)
        average_map = layers.conv(average_map, scope='feature_concat', filter_dims=[1, 1, c],
                                  stride_dims=[1, 1], non_linear_fn=None)
        average_map = layers.conv_normalize(average_map, norm=norm, b_train=b_train,
                                            scope='feature_concat_norm')
        average_map = activation(average_map)

        for num_rblock in range(2):
            print('Decoder Residual: ' + str(average_map.get_shape().as_list()))
            average_map = layers.add_residual_block(average_map, filter_dims=[3, 3, c], norm=norm, b_train=b_train,
                                                    act_func=activation, scope='decoder_residual_' + str(num_rblock))
        r = input_height // h
        average_map = layers.conv(average_map, scope='upsacle_conv',
                                  filter_dims=[3, 3, r * r * segment_layer_depth],
                                  stride_dims=[1, 1], non_linear_fn=None)
        average_map = layers.conv_normalize(average_map, norm=norm, b_train=b_train,
                                            scope='upscale_norm')
        average_map = activation(average_map)
        average_map = tf.nn.depth_to_space(average_map, r)

        # Multi Resolution Effect
        average_map_narrow = layers.conv(average_map, scope='average_map_narrow',
                                         filter_dims=[3, 3, segment_layer_depth],
                                         stride_dims=[1, 1], non_linear_fn=None)
        average_map_narrow = layers.conv_normalize(average_map_narrow, norm=norm, b_train=b_train,
                                                   scope='average_map_narrow_norm')
        average_map_narrow = activation(average_map_narrow)
        average_map_wide = layers.conv(average_map, scope='average_map_wide',
                                       filter_dims=[3, 3, segment_layer_depth],
                                       stride_dims=[1, 1], non_linear_fn=None, dilation=[1, 2, 2, 1])
        average_map_wide = layers.conv_normalize(average_map_wide, norm=norm, b_train=b_train,
                                                 scope='average_map_wide_norm')
        average_map_wide = activation(average_map_wide)
        average_map = tf.add(average_map_narrow, average_map_wide)

        segment_layer = layers.conv(average_map, scope='segment_resize', filter_dims=[3, 3, segment_layer_depth],
                                    stride_dims=[1, 1], non_linear_fn=None)
        segment_layer = layers.conv_normalize(segment_layer, norm=norm, b_train=b_train,
                                              scope='segment_resize_norm')
        segment_layer = activation(segment_layer)
        residual_layer = layers.add_residual_block(average_map, filter_dims=[3, 3, segment_layer_depth],
                                                  act_func=activation, norm=norm,  b_train=b_train, scope='refinement1')
        segment_layer = tf.add(segment_layer, residual_layer)
        segment_layer = layers.conv(segment_layer, scope='segment_merge', filter_dims=[3, 3, segment_layer_depth],
                                    stride_dims=[1, 1], non_linear_fn=None)
        segment_layer = layers.conv_normalize(segment_layer, norm=norm, b_train=b_train,
                                              scope='segment_merge_norm')
        segment_layer = activation(segment_layer)
        segment_layer = layers.conv(segment_layer, scope='segment_output', filter_dims=[3, 3, 1],
                                    stride_dims=[1, 1], non_linear_fn=tf.nn.sigmoid)

    return segment_layer


def spatial_memory(query, size, dims, scope='spatial_aug_mem'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        aug_mem = tf.get_variable('mem_vars', [size, dims], dtype=tf.float32,
                                  initializer=tf.random_normal_initializer)
        print(scope + ' mem size: ' + str(aug_mem.get_shape().as_list()))
        # query = [B, H, W, R], aug_mem = [Q, R]
        norm_q = tf.nn.l2_normalize(query, axis=[0, 1, 2])
        print(scope + ' norm_q: ' + str(norm_q.get_shape().as_list()))
        norm_mem = tf.nn.l2_normalize(aug_mem, axis=0)
        print(scope + ' norm mem: ' + str(norm_mem.get_shape().as_list()))

        _, h, w, dim = norm_q.get_shape().as_list()
        norm_q = tf.reshape(norm_q, [-1, dim])

        distance = tf.matmul(norm_q, norm_mem, transpose_b=True)  # [B x H x W, Q]

        sm = tf.nn.softmax(distance, axis=-1)  # [B x H x W, Q]
        print(scope + ' softmax: ' + str(sm.get_shape().as_list()))
        threashold = tf.constant(1 / aug_mem_size, dtype=tf.float32)
        attention = tf.multiply(tf.nn.relu(sm - threashold), sm)
        attention = attention / (1e-7 + tf.abs(sm - threashold))
        print(scope + ' attention: ' + str(attention.get_shape().as_list()))
        l1 = tf.expand_dims(tf.reduce_sum(attention, axis=-1), axis=-1)
        # print(scope + ' l1: ' + str(l1.get_shape().as_list()))
        attention = attention / (l1 + 1e-7)

        latent = tf.matmul(attention, aug_mem)  # [B x H x W, Q] x [Q, R] = [B x H x W, R]
        latent = tf.reshape(latent, shape=[-1, h, w, dim])
        # norm_q = tf.reshape(norm_q, shape=[-1, h, w, dim])
        # latent = tf.add(latent, norm_q)
        print(scope + ' latent: ' + str(latent.get_shape().as_list()))
        return attention, latent


def memory(query, size, dims, scope='aug_mem'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        aug_mem = tf.get_variable('mem_vars', [size, dims], dtype=tf.float32,
                                  initializer=tf.random_normal_initializer)
        print(scope + ' mem size: ' + str(aug_mem.get_shape().as_list()))
        # query = [B, R], aug_mem = [Q, R]
        norm_q = tf.nn.l2_normalize(query, axis=0)
        print(scope + ' norm_q: ' + str(norm_q.get_shape().as_list()))
        norm_mem = tf.nn.l2_normalize(aug_mem, axis=0)
        print(scope + ' norm mem: ' + str(norm_mem.get_shape().as_list()))

        distance = tf.matmul(norm_q, norm_mem, transpose_b=True)  # [B, Q]

        # Simple Adaptive Scaling
        dist_scale = layers.fc(query, aug_mem_size // 8, non_linear_fn=tf.nn.relu, scope='mem_dist_scale')
        scale = layers.fc(dist_scale, aug_mem_size, non_linear_fn=None, scope='mem_linear_scale')
        shift = layers.fc(dist_scale, aug_mem_size, non_linear_fn=None, scope='mem_linear_shift')

        distance = scale * distance + shift  # [B, Q]

        sm = tf.nn.softmax(distance, axis=-1)  # [B, Q]
        print(scope + ' softmax: ' + str(sm.get_shape().as_list()))
        threashold = tf.constant(1 / aug_mem_size, dtype=tf.float32)
        attention = tf.multiply(tf.nn.relu(sm - threashold), sm)
        attention = attention / (1e-4 + tf.abs(sm - threashold))
        print(scope + ' attention: ' + str(attention.get_shape().as_list()))
        l1 = tf.expand_dims(tf.reduce_sum(attention, axis=-1), axis=-1)
        print(scope + ' l1: ' + str(l1.get_shape().as_list()))
        attention = attention / (l1 + 1e-7)

        latent = tf.matmul(attention, aug_mem)  # [B, Q] x [Q, R] = [B, R]
        print(scope + ' attention: ' + str(attention.get_shape().as_list()))

        return attention, latent


def categorical_sample(logits):
    u = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(u)), axis=-1)


def discriminator(x1, x2=None, activation='swish', scope='discriminator_network', norm='layer', b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if activation == 'swish':
            act_func = layers.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        elif activation == 'lrelu':
            act_func = tf.nn.leaky_relu
        else:
            act_func = tf.nn.sigmoid

        block_depth = disc_unit_block_depth

        if x2 is not None:
            x = tf.concat([x1, x2], axis=-1)
        else:
            x = x1

        print(scope + ' Input: ' + str(x.get_shape().as_list()))

        l = layers.conv(x, scope='init', filter_dims=[3, 3, block_depth], stride_dims=[1, 1], non_linear_fn=act_func)

        downsample_num_itr = downsample_num

        features = []

        for i in range(downsample_num_itr + 1):
            block_depth = block_depth * 2
            l = layers.conv(l, scope='disc_dn_' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
                            non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='disc_dn_norm_' + str(i))
            l = act_func(l)
            features.append(l)

        last_layer = l

        print(scope + ' Discriminator GP Dims: ' + str(last_layer.get_shape().as_list()))

        logit = layers.conv(last_layer, scope='conv_pred', filter_dims=[1, 1, 1], stride_dims=[1, 1],
                            non_linear_fn=tf.nn.sigmoid)
        print(scope + ' Discriminator Logit Dims: ' + str(logit.get_shape().as_list()))

    return features, logit


def create_roi_mask(width, height, offset=60):
    empty_img = np.ones((width, height, num_channel), dtype=np.uint8) * 1.0
    center_x = width // 2
    center_y = height // 2
    center = np.array((center_x, center_y))
    radius = 1.0 * (width // 2) - offset

    for i in range(width):
        for j in range(height):
            p = np.array((i, j))
            d = np.linalg.norm(p - center)
            if d > radius:
                empty_img[i, j] = 0

    mask_img = empty_img

    return mask_img


def train(model_path='None', mode='train'):
    print('Please wait. Preparing to start training...')
    train_start_time = time.time()

    cutout_mask = create_roi_mask(input_width, input_height, offset=8)
    outlier_files = []
    if use_outlier_samples is True:
        # Classes
        raw_aug_files = os.listdir(aug_data)
        print('Load augmentation samples, Total Num of Samples: ' + str(len(raw_aug_files)))

        for a_file in raw_aug_files:
            a_file_path = os.path.join(aug_data, a_file).replace("\\", "/")
            outlier_files.append(a_file_path)
        outlier_files = shuffle(outlier_files)

    learning_rate = 2e-4
    sparsity = 5e-4

    tf.reset_default_graph()

    X_IN = tf.placeholder(tf.float32, [None, input_height, input_width, num_channel])
    Y_IN = tf.placeholder(tf.float32, [None, input_height, input_width, num_channel])
    S_IN = tf.placeholder(tf.float32, [None, input_height, input_width, 1])
    B_TRAIN = True
    LR = tf.placeholder(tf.float32, None)

    query = query_encoder(X_IN, norm='instance', activation=layers.swish, scope=Qeury_Encoder_Scope, b_train=B_TRAIN)
    attention_g, latent_g = spatial_memory(query, size=aug_mem_size, dims=representation_dimension, scope=LATENT_Memory_scope)
    laterals, features = segment_encoder(X_IN, mem_latents=latent_g, norm='instance', scope=SEGMENT_Encoder_scope,
                                         activation=layers.swish, b_train=B_TRAIN)
    U_G_X = segment_decoder(laterals, features, norm='instance', activation=layers.swish,
                            scope=SEGMENT_Decoder_scope, b_train=B_TRAIN)

    print('Segment decoder images: ' + str(U_G_X.get_shape().as_list()))
    segment_residual_loss = 0.0
    segment_residual_loss += get_residual_loss(U_G_X, S_IN, type='l1_focal')
    segment_residual_loss += get_residual_loss(U_G_X, S_IN, type='ft', alpha=0.3)

    if freeze_reconstructor is False:
        G_X, _ = image_reconstructor(latent_g, activation=tf.nn.relu, norm='instance', scope=Image_Reconstructor_Scope, b_train=True)
        reconstructor_residual_loss = get_residual_loss(G_X, Y_IN, type='ssim_l1', alpha=0.8)

        if use_categorical_constraints is True:
            cat_samples = categorical_sample(attention_g)
            cat_onehot = tf.one_hot(cat_samples, aug_mem_size)

            use_strict_categorical_dist = True

            if use_strict_categorical_dist is False:
                smoothing_p = 0.3
                smoothing_factor = smoothing_p / aug_mem_size
                cat_onehot = tf.nn.relu(cat_onehot - smoothing_p - smoothing_factor) + smoothing_factor

            sparsity_reg_loss = get_residual_loss(attention_g, cat_onehot, type='focal', alpha=sparsity)  # More discrete
        else:
            sparsity_reg_loss = get_residual_loss(attention_g, None, type='entropy', alpha=sparsity)  # Less discrete

        reconstructor_residual_loss += sparsity_reg_loss

    if use_discriminator is True:
        fake_features, fake_logit = discriminator(x1=U_G_X, x2=X_IN, norm='layer', scope=DISC_scope, b_train=B_TRAIN)
        real_features, real_logit = discriminator(x1=S_IN, x2=X_IN, norm='layer', scope=DISC_scope, b_train=B_TRAIN)
        d_real_loss = get_discriminator_loss(tf.ones_like(real_logit), real_logit, type='ls') + \
                      get_discriminator_loss(tf.zeros_like(fake_logit), fake_logit, type='ls')
        d_fake_loss = get_discriminator_loss(tf.ones_like(fake_logit), fake_logit, type='ls')
        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=DISC_scope)
        segment_residual_loss = segment_residual_loss + 1e-2 * d_fake_loss

    segment_encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=SEGMENT_Encoder_scope)
    segment_decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=SEGMENT_Decoder_scope)
    segment_generator_vars = segment_encoder_vars + segment_decoder_vars

    query_encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=Qeury_Encoder_Scope)
    image_decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=Image_Reconstructor_Scope)
    reconstructor_vars = query_encoder_vars + image_decoder_vars
    memory_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=LATENT_Memory_scope)
    image_reconstuctor_vars = memory_vars + reconstructor_vars
    total_joint_variable = segment_generator_vars + image_reconstuctor_vars
    # Optimizer
    segment_loss = segment_residual_loss
    segmentation_optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(segment_loss, var_list=segment_generator_vars)

    if freeze_reconstructor is False:
        reconstruct_loss = reconstructor_residual_loss
        reconstruction_optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(reconstruct_loss, var_list=image_reconstuctor_vars)
        if meta_training is False:
            joint_optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(segment_loss + reconstruct_loss, var_list=total_joint_variable)

    if use_discriminator is True:
        # d_opt = tf.train.AdamOptimizer(learning_rate=LR)
        # d_gradients = d_opt.compute_gradients(-disc_loss, disc_vars)
        # clipped_d_gradients = [(tf.clip_by_value(grad, -0.01, 0.01), var) for grad, var in d_gradients]
        # disc_optimizer = d_opt.apply_gradients(clipped_d_gradients)
        disc_optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(d_real_loss, var_list=disc_vars)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    reconstructor_model_path = os.path.join(model_path, 'recon/m.chpt').replace("\\", "/")
    segment_model_path = os.path.join(model_path, 'seg/m.chpt').replace("\\", "/")
    if use_discriminator is True:
        disc_model_path = os.path.join(model_path, 'disc/m.chpt').replace("\\", "/")
        discriminator_saver = tf.train.Saver(disc_vars)
    image_reconstructor_saver = tf.train.Saver(image_reconstuctor_vars)
    query_encoder_savers = tf.train.Saver(query_encoder_vars)
    if freeze_reconstructor is False:
        image_decoder_savers = tf.train.Saver(image_decoder_vars)
    latent_mem_savers = tf.train.Saver(memory_vars)
    segment_generator_saver = tf.train.Saver(segment_generator_vars)
    segment_encoder_saver = tf.train.Saver(segment_encoder_vars)
    segment_decoder_saver = tf.train.Saver(segment_decoder_vars)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        try:
            query_encoder_savers.restore(sess, reconstructor_model_path)
            print('Load query encoder.')
            latent_mem_savers.restore(sess, reconstructor_model_path)
            print('Load latent memory.')
            if freeze_reconstructor is False:
                image_decoder_savers.restore(sess, reconstructor_model_path)
                print('Load image decoder.')
        except:
            print('Reconstructor Load Failed')
        try:
            if use_discriminator is True:
                discriminator_saver.restore(sess, disc_model_path)
                print('Load discriminator.')
        except:
            print('Discriminator Load Failed')
        try:
            # unet_generator_saver.restore(sess, uae_model_path)
            segment_encoder_saver.restore(sess, segment_model_path)
            print('Load segmentation encoder.')
        except:
            print('Segment Encoder Load Failed')
        try:
            segment_decoder_saver.restore(sess, segment_model_path)
            print('Load segmentation decoder.')
        except:
            print('Segment Decoder Load Failed')

        tr_dir = train_data
        up_dir = update_data

        # Classes
        classes = os.listdir(tr_dir)
        print(' Train classes: ' + str(len(classes)))

        if mode == 'update':
            update_classes = os.listdir(up_dir)
            print(' Update classes: ' + str(len(update_classes)))

        # Supervised Settings
        labeled_list_X = os.listdir('data/supervised/X')
        labeled_list_Y = os.listdir('data/supervised/Y')

        labeled_X = []
        labeled_Y = []
        for file_x in labeled_list_X:
            labeled_file_X = 'data/supervised/X/' + file_x
            labeled_file_Y = 'data/supervised/Y/' + file_x.split('.')[0] + '.png'
            labeled_X.append(labeled_file_X)
            labeled_Y.append(labeled_file_Y)

        labeled_X = np.array(labeled_X)
        labeled_Y = np.array(labeled_Y)

        use_semisupervised = True

        for e in range(num_epoch):
            tr_files = []
            for cls in classes:
                class_path = os.path.join(tr_dir, cls).replace("\\", "/")
                samples = os.listdir(class_path)
                if mode == 'update':
                    samples = np.random.choice(samples, size=1)
                else:
                    samples = np.random.choice(samples, size=num_samples_per_class)  # (1000//len(classes)))
                for s in samples:
                    sample_path = os.path.join(class_path, s).replace("\\", "/")
                    tr_files.append(sample_path)
            if mode == 'update':
                for cls in update_classes:
                    class_path = os.path.join(up_dir, cls).replace("\\", "/")
                    samples = os.listdir(class_path)
                    samples = np.random.choice(samples, size=num_samples_per_class)  # (1000//len(classes)))
                    for s in samples:
                        sample_path = os.path.join(class_path, s).replace("\\", "/")
                        tr_files.append(sample_path)
            print(' Num samples per epoch: ' + str(len(tr_files)))
            total_input_size = len(tr_files)
            tr_files = shuffle(tr_files)
            training_batch = zip(range(0, total_input_size, batch_size),
                                 range(batch_size, total_input_size + 1, batch_size))
            itr = 0

            # Learning rate schedule
            #lr = 0.5 * learning_rate * (1.0 + np.cos(np.pi * (e / num_epoch)))
            lr = learning_rate
            train_with_normal_sample = True

            for start, end in training_batch:
                itr = itr + 1

                b_use_cutdout = True
                b_use_outlier_samples = use_outlier_samples
                b_use_bg_samples = use_bg_samples

                if np.random.randint(1, 10) < 3:
                    b_use_outlier_samples = False

                if np.random.randint(1, 10) < 3:
                    b_use_bg_samples = False

                if b_use_outlier_samples is True:
                    sample_outlier_files = np.random.choice(outlier_files,
                                                            size=np.random.random_integers(low=1, high=7))
                    sample_outlier_imgs, _, _ = load_images(sample_outlier_files, rotate=True)
                    sample_outlier_imgs = np.sum(sample_outlier_imgs, axis=0)
                    # sample_outlier_imgs = aug_noise + sample_outlier_imgs
                    aug_noise = sample_outlier_imgs
                    aug_noise = np.where(aug_noise > 0.9, 1.0, 0.0)
                    b_use_cutdout = False
                else:
                    # Perlin Noise
                    perlin_res = int(np.random.choice([16, 32, 64], size=1))  # 1024 x 1024
                    # perlin_res = int(np.random.choice([8, 16, 32], size=1)) # 512 x 512
                    # perlin_res = 2, perlin_octave = 4 : for large smooth object augmentation.
                    perlin_octave = 5
                    noise = util.generate_fractal_noise_2d((input_width, input_height), (perlin_res, perlin_res),
                                                           perlin_octave)
                    # noise = util.generate_perlin_noise_2d((input_width, input_height), (perlin_res, perlin_res))
                    perlin_noise = np.where(noise > np.average(noise), 1.0, 0.0)
                    perlin_noise = np.expand_dims(perlin_noise, axis=-1)
                    aug_noise = perlin_noise

                if train_with_normal_sample is True:
                    batch_imgs, gt_imgs, seg_imgs = load_images(tr_files[start + 1:end], rotate=True, shift=True, flip=True, contrast=True,
                                                                noise_mask=aug_noise, cutout=b_use_cutdout, cutout_mask=cutout_mask, add_eps=True)
                    b_use_bg_samples = False
                else:
                    batch_imgs, gt_imgs, seg_imgs = load_images(tr_files[start:end], rotate=True, shift=True,
                                                                flip=True, contrast=True,
                                                                noise_mask=aug_noise, cutout=b_use_cutdout,
                                                                cutout_mask=cutout_mask)
                seg_imgs = np.where(seg_imgs > 0, 1.0, 0.0)

                if b_use_bg_samples is True:
                    # noise_samples = np.random.choice(tr_files, size=batch_size)
                    random_index = np.random.choice(len(labeled_X), size=batch_size, replace=False)
                    noise_sample_files_X = labeled_X[random_index]
                    noise_sample_files_Y = labeled_Y[random_index]
                    noise_sample_images, _, _ = load_images(noise_sample_files_X)
                    noise_sample_segments, _, _ = load_images(noise_sample_files_Y, gray_scale=True)
                    flip_axis = np.random.random_integers(low=1, high=2)
                    noise_sample_images = np.flip(noise_sample_images, axis=flip_axis)
                    noise_sample_segments = np.flip(noise_sample_segments, axis=flip_axis)
                    # noise_sample_imgs, _, _ = load_images(noise_samples, rotate=True)
                    blending_a = np.random.uniform(low=0.1, high=0.5)
                    noise_sample_images = (1 - blending_a) * noise_sample_images + blending_a * batch_imgs
                    # fg = seg_imgs * noise_sample_imgs
                    fg = noise_sample_segments * noise_sample_images
                    # bg = (1 - seg_imgs) * batch_imgs
                    bg = (1 - noise_sample_segments) * batch_imgs
                    batch_imgs = fg + bg
                    seg_imgs = seg_imgs + noise_sample_segments
                    seg_imgs = np.where(seg_imgs > 0, 1.0, 0.0)

                if train_with_normal_sample is True:
                    b_img, gt, seg = load_images([tr_files[start]], rotate=True, shift=True, flip=True, contrast=True,
                                                 noise_mask=None, cutout=False, cutout_mask=cutout_mask)
                    batch_imgs = np.append(batch_imgs, b_img, axis=0)
                    gt_imgs = np.append(gt_imgs, gt, axis=0)
                    seg_imgs = np.append(seg_imgs, seg, axis=0)

                batch_imgs, gt_imgs, seg_imgs = shuffle(batch_imgs, gt_imgs, seg_imgs)

                if meta_training is True:  # Meta Training
                    if freeze_reconstructor is False:
                        if e < (num_epoch // 2):
                            _, pseudo_g_loss = sess.run([reconstruction_optimizer, reconstruct_loss],
                                                        feed_dict={X_IN: batch_imgs, Y_IN: gt_imgs, S_IN: seg_imgs, LR: lr})

                    _, segment_g_loss, u_g_x_imgs = sess.run([segmentation_optimizer, segment_loss, U_G_X],
                                                             feed_dict={X_IN: batch_imgs, S_IN: seg_imgs, LR: lr})
                else:
                    _, segment_g_loss, pseudo_g_loss, u_g_x_imgs = sess.run(
                        [joint_optimizer, segment_loss, reconstruct_loss, U_G_X],
                        feed_dict={X_IN: batch_imgs, Y_IN: gt_imgs, S_IN: seg_imgs, LR: lr})
                if freeze_reconstructor is True:
                    print('epoch: ' + str(e) + ', segment loss: ' + str(segment_g_loss))
                else:
                    print('epoch: ' + str(e) + ', segment loss: ' + str(segment_g_loss) + ', reconstruct loss: ' +
                          str(pseudo_g_loss))

                if segment_g_loss > 0.1:
                    for i in range(batch_size):
                        cv2.imwrite('hard/Y/' + str(itr) + str(i) + '.jpg', 255 * seg_imgs[i])
                        cv2.imwrite('hard/X/' + str(itr) + str(i) + '.jpg', 255 * batch_imgs[i])

                if itr % 10 == 0:
                    for i in range(batch_size):
                        cv2.imwrite(out_dir + '/' + str(itr) + str(i) + '_pred.jpg', 255 * u_g_x_imgs[i])
                        cv2.imwrite(out_dir + '/' + str(itr) + str(i) + '_gt.jpg', 255 * seg_imgs[i])
                        cv2.imwrite(out_dir + '/' + str(itr) + str(i) + '.jpg', 255 * batch_imgs[i])
                    # cv2.imwrite(out_dir + '/' + str(itr) + '_recon.jpg', 255 * g_x_imgs[0])
                    print('Elapsed Time at  ' + str(e) + '/' + str(num_epoch) + ' epochs, ' +
                          str(time.time() - train_start_time) + ' sec')
                if use_discriminator is True:
                    _ = sess.run([disc_optimizer],
                                 feed_dict={X_IN: batch_imgs, S_IN: seg_imgs, LR: lr})
                if itr % 30 == 0:
                    try:
                        print('Saving model...')
                        # total_saver = tf.train.Saver()
                        # total_saver.save(sess, model_path, write_meta_graph=False)
                        if freeze_reconstructor is False:
                            image_reconstructor_saver.save(sess, reconstructor_model_path, write_meta_graph=False)
                        segment_generator_saver.save(sess, segment_model_path, write_meta_graph=False)
                        if use_discriminator is True:
                            discriminator_saver.save(sess, disc_model_path, write_meta_graph=False)
                        print('Saved.')
                    except:
                        print('Save failed')

                if itr % 100 == 0:
                    te_dir = test_data
                    te_files = os.listdir(te_dir)
                    te_batch = zip(range(0, len(te_files), batch_size),
                                   range(batch_size, len(te_files) + 1, batch_size))

                    for t_s, t_e in te_batch:
                        test_imgs, _, _ = load_images(te_files[t_s:t_e], base_dir=te_dir)
                        u_gx_imgs = sess.run(U_G_X, feed_dict={X_IN: test_imgs})

                        for i in range(batch_size):
                            cv2.imwrite('out/' + te_files[t_s + i], 255 * u_gx_imgs[i])
                            s = np.sum(u_gx_imgs[i])
                            print('anomaly score of ' + te_files[t_s + i] + ': ' + str(s))

            if e % 2 == 0:
                hard_X = []
                hard_Y = []
                hard_list_X = os.listdir('hard/X')
                for file_x in hard_list_X:
                    labeled_file_X = 'hard/X/' + file_x
                    labeled_file_Y = 'hard/Y/' + file_x
                    hard_X.append(labeled_file_X)
                    hard_Y.append(labeled_file_Y)

                hard_X = np.array(hard_X)
                hard_Y = np.array(hard_Y)

                if len(hard_X) > batch_size*10:
                    random_index = np.random.choice(len(hard_X), size=10*batch_size, replace=False)
                    hard_X = hard_X[random_index]
                    hard_Y = hard_Y[random_index]
                training_batch = zip(range(0, len(hard_X), batch_size),
                                     range(batch_size, len(hard_X) + 1, batch_size))

                for start, end in training_batch:
                    hard_file_X = hard_X[start:end]
                    hard_file_Y = hard_Y[start:end]

                    hard_img_X, _, _ = load_images(hard_file_X)
                    hard_img_Y, _, _ = load_images(hard_file_Y, gray_scale=True)

                    sess.run([segmentation_optimizer], feed_dict={X_IN: hard_img_X, S_IN: hard_img_Y, LR: lr})

            if use_semisupervised is True and e % 4 == 0:
                itr = 0
                #random_index = np.random.choice(len(labeled_Y), size=batch_size*10, replace=False)
                #sup_X = labeled_X[random_index]
                #sup_Y = labeled_Y[random_index]
                sup_X = labeled_X
                sup_Y = labeled_Y
                training_batch = zip(range(0, len(sup_X), batch_size),
                                     range(batch_size, len(sup_X) + 1, batch_size))

                for start, end in training_batch:
                    labeled_file_X = sup_X[start:end]
                    labeled_file_Y = sup_Y[start:end]

                    labeled_img_X, _, _ = load_images(labeled_file_X, contrast=True)
                    labeled_img_Y, _, _ = load_images(labeled_file_Y, gray_scale=True)

                    _, u_loss, u_g_x_imgs = sess.run([segmentation_optimizer, segment_loss, U_G_X],
                                                     feed_dict={X_IN: labeled_img_X, S_IN: labeled_img_Y, LR: lr})
                    itr += 1
                    print('epoch: ' + str(e) + ', segment loss: ' + str(u_loss))
                    if itr % 10 == 0:
                        cv2.imwrite(out_dir + '/' + str(itr) + '_pred.jpg', 255 * u_g_x_imgs[0])
                        cv2.imwrite(out_dir + '/' + str(itr) + '_gt.jpg', 255 * labeled_img_Y[0])
                        cv2.imwrite(out_dir + '/' + str(itr) + '.jpg', 255 * labeled_img_X[0])

            try:
                print('Saving model...')
                if freeze_reconstructor is False:
                    image_reconstructor_saver.save(sess, reconstructor_model_path, write_meta_graph=False)
                segment_generator_saver.save(sess, segment_model_path, write_meta_graph=False)
                if use_discriminator is True:
                    discriminator_saver.save(sess, disc_model_path, write_meta_graph=False)
                print('Saved.')
            except:
                print('Save failed')


def calculate_anomaly_score(img, width, height, filter_size=16):
    step = filter_size // 2
    score_list = [0.0]
    threshold = 0
    alpha = 0.8

    for p_y in range(0, height - step, step):
        for p_x in range(0, width - step, step):
            roi = img[p_x:p_x + filter_size, p_y:p_y + filter_size]
            score = np.mean(roi)

            if score > threshold:
                score_list.append(score)

    max_score = np.max(score_list)
    mean_score = np.mean(score_list)
    anomaly_score = (1 - alpha) * mean_score + alpha * max_score

    return anomaly_score


def test(model_path):
    print('Please wait. Preparing to test...')

    tf.reset_default_graph()

    X_IN = tf.placeholder(tf.float32, [None, input_height, input_width, num_channel])
    B_TRAIN = False

    # Generator
    query = query_encoder(X_IN, norm='instance', activation=layers.swish, scope=Qeury_Encoder_Scope, b_train=B_TRAIN)
    _, latent_g = spatial_memory(query, size=aug_mem_size, dims=representation_dimension, scope=LATENT_Memory_scope)
    laterals, features = segment_encoder(X_IN, mem_latents=latent_g, norm='instance', activation=layers.swish,
                                      scope=SEGMENT_Encoder_scope, b_train=B_TRAIN)
    U_G_X = segment_decoder(laterals, features, norm='instance', activation=layers.swish,
                            scope=SEGMENT_Decoder_scope, b_train=B_TRAIN)

    segment_encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=SEGMENT_Encoder_scope)
    segment_decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=SEGMENT_Decoder_scope)
    segment_generator_vars = segment_encoder_vars + segment_decoder_vars
    query_encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=Qeury_Encoder_Scope)

    memory_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=LATENT_Memory_scope)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    reconstructor_model_path = os.path.join(model_path, 'recon/m.chpt').replace("\\", "/")
    segment_model_path = os.path.join(model_path, 'seg/m.chpt').replace("\\", "/")

    latent_saver = tf.train.Saver(query_encoder_vars + memory_vars)
    segment_generator_saver = tf.train.Saver(segment_generator_vars)

    with tf.Session(config=config) as sess:
        try:
            latent_saver.restore(sess, reconstructor_model_path)
            print('Load latent memory.')
            segment_generator_saver.restore(sess, segment_model_path)
            print('Load segment generator.')
        except:
            print('Fail to load ...')
            return
        test_start_time = time.time()
        te_dir = test_data
        te_files = os.listdir(te_dir)
        te_batch = zip(range(0, len(te_files), batch_size),
                       range(batch_size, len(te_files) + 1, batch_size))

        for t_s, t_e in te_batch:
            test_imgs, _, _ = load_images(te_files[t_s:t_e], base_dir=te_dir)
            u_gx_imgs = sess.run(U_G_X, feed_dict={X_IN: test_imgs})

            for i in range(batch_size):
                cv2.imwrite('out/' + te_files[t_s + i], 255 * u_gx_imgs[i])
                s = np.sum(u_gx_imgs[i])
                print('anomaly score of ' + te_files[t_s + i] + ': ' + str(s))

        print('Total ' + str(len(te_files)) + ' Samples. Elapsed Time:  ' + str(time.time() - test_start_time) + ' sec')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, help='train/test/update', default='train')
    parser.add_argument('--model_path', type=str, help='model check point file path', default='model/m.ckpt')
    parser.add_argument('--train_data', type=str, help='training data directory', default='data/train')
    parser.add_argument('--test_data', type=str, help='test data directory', default='data/test')
    parser.add_argument('--update_data', type=str, help='update data directory', default='data/update')
    parser.add_argument('--aug_data', type=str, help='augmentation samples', default='data/augmentation')
    parser.add_argument('--noise_data', type=str, help='specific noise data samples', default='data/noise')
    parser.add_argument('--out_dir', type=str, help='output directory', default='imgs')
    parser.add_argument('--bgmask_data', type=str, help='background mask sample director', default='bgmask')
    parser.add_argument('--img_size', type=int, help='training image size', default=512)
    parser.add_argument('--epoch', type=int, help='num epoch', default=1000)
    parser.add_argument('--batch_size', type=int, help='Training batch size', default=16)
    parser.add_argument('--alpha', type=int, help='AE loss weight', default=1)

    args = parser.parse_args()

    input_width = args.img_size
    input_height = args.img_size
    batch_size = args.batch_size
    mode = args.mode
    model_path = args.model_path
    train_data = args.train_data
    test_data = args.test_data
    update_data = args.update_data
    out_dir = args.out_dir
    num_epoch = args.epoch
    alpha = args.alpha
    aug_data = args.aug_data
    bg_mask_data = args.bgmask_data
    noise_data = args.noise_data
    unit_block_depth = 16
    segment_unit_block_depth = 48
    disc_unit_block_depth = 8
    bottleneck_num = 2
    decoder_bottleneck_num = 0
    query_dimension = 128
    representation_dimension = query_dimension
    segmentation_upsample_ratio = 4
    segmentation_downsample_num = int(np.log2(segmentation_upsample_ratio))
    segmentation_upsample_num = segmentation_downsample_num
    upsample_ratio = 8
    downsample_num = int(np.log2(upsample_ratio))
    upsample_num = downsample_num
    aug_mem_size = 2048
    num_channel = 3
    num_shuffle_car = 2
    num_car = 2
    use_categorical_constraints = True
    use_outlier_samples = True
    use_bg_samples = True
    num_samples_per_class = 8
    meta_training = True
    freeze_reconstructor = True

    if meta_training is False:
        freeze_reconstructor = False
    use_discriminator = False

    if mode == 'test':
        test(model_path)
    else:
        train(model_path, mode)
