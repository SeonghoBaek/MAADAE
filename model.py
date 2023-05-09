# MAADAE[meidei]: Memory Augmented Adversarial Dual AUto Encoder
# Author: Seongho Baek
# e-mail: seonghobaek@gmail.com

import math

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
G_Encoder_scope = 'generator_encoder'
G_Decoder_scope = 'generator_decoder'
Style_Encoder_scope = 'style_encoder'
Style_M_scope = 'style_mem'
G_M_scope = 'generator_mem'
G_UNet_Encoder_scope = 'unet_generator_encoder'
G_UNet_Decoder_scope = 'unet_generator_decoder'
D_scope = 'discriminator'


def load_images(file_name_list, base_dir=None, mask=None, mask_noise=None, cutout=False, cutout_mask=None,
                add_eps=False, rotate=False, flip=False, gray_scale=False, shift=False, scaleabs=False):
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

            h, w, c = img.shape
            img = cv2.resize(img, dsize=(input_width, input_height), interpolation=cv2.INTER_AREA)

            if scaleabs is True:
                contrast = 1.5  # np.random.uniform(low=0.5, high=1.5)
                brightness = 0  # np.random.randint(low=-10, high=10)
                img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)

            if gray_scale is True:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if img is not None:
                img = np.array(img) * 1.0

                if rotate is True:
                    rot = np.random.randint(0, 360)
                    img = util.rotate_image(img, rot)

                if flip is True:
                    img = cv2.flip(img, np.random.randint(low=0, high=2))

                if shift is True:
                    x_shift = np.random.randint(-7, 7)
                    y_shift = np.random.randint(-7, 7)
                    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
                    img = cv2.warpAffine(img, M, (input_width, input_height))

                gt_img = img.copy()
                gt_img = np.array(gt_img)

                if add_eps is True:
                    img = img + np.random.normal(size=img.shape)

                if gray_scale is True:
                    img = np.expand_dims(img, axis=-1)
                    gt_img = np.expand_dims(gt_img, axis=-1)

                cut_mask = np.zeros_like(img)
                seg_img = cut_mask

                if cutout is True:
                    # square cut out
                    co_w = np.random.randint(low=2, high=input_width // 4)
                    co_h = np.random.randint(low=2, high=input_height // 4)
                    num_cutout = np.random.randint(low=1, high=5)

                    if np.random.randint(low=0, high=10) < 5:
                        if co_w <= co_h:
                            co_w = np.random.randint(low=1, high=5)
                        else:
                            co_h = np.random.randint(low=1, high=5)
                        mask_noise = None

                    for _ in range(num_cutout):
                        r_x = np.random.randint(low=60, high=input_width - co_w - 60)
                        r_y = np.random.randint(low=60, high=input_height - co_h - 60)
                        # img[r_x:r_x + co_w, r_y:r_y + co_h] = 0.0
                        cut_mask[r_x:r_x + co_w, r_y:r_y + co_h, :] = 1.0

                    if mask_noise is not None:
                        # cut_mask = mask_noise + cut_mask
                        cut_mask = mask_noise * cut_mask
                        cut_mask = np.where(cut_mask > 0.5, 1.0, 0)

                    rot = np.random.randint(-90, 90)
                    cut_mask = util.rotate_image(cut_mask, rot)
                    if cutout_mask is not None:
                        cut_mask = cut_mask * cutout_mask

                    # Segmentation Mode
                    seg_img = cut_mask
                    bg_img = (1.0 - seg_img) * img
                    fg_img = seg_img * img
                    alpha = np.random.uniform(low=0.0, high=0.5)
                    cut_mask = np.random.randint(1, 255, 3) * cut_mask
                    cut_mask = np.where(cut_mask > 255, 255, cut_mask)
                    img = bg_img + (alpha * fg_img + (1 - alpha) * cut_mask)
                else:
                    if mask_noise is not None:
                        cut_mask = mask_noise

                        if cutout_mask is not None:
                            cut_mask = cut_mask * cutout_mask

                        # Segmentation Mode
                        seg_img = cut_mask
                        bg_img = (1.0 - cut_mask) * img
                        fg_img = cut_mask * img
                        alpha = np.random.uniform(low=0.5, high=0.9)
                        cut_mask = np.random.randint(1, 255, 3) * cut_mask
                        cut_mask = np.where(cut_mask > 255, 255, cut_mask)
                        img = bg_img + (alpha * fg_img + (1 - alpha) * cut_mask)
                        # img = (1.0 - cut_mask) * img

                seg_img = np.average(seg_img, axis=-1)
                seg_img = np.expand_dims(seg_img, axis=-1)
                seg_images.append(seg_img)

                if mask is not None:
                    img = img * mask
                    gt_img = gt_img * mask

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
    elif type == 'mix_focal':
        eps = 1e-10
        q = tf.math.maximum(1 - value, eps)
        p = tf.math.maximum(value, eps)
        pos_loss = -(q ** 2) * tf.math.log(p)
        neg_loss = -(p ** 2) * tf.math.log(q)
        f_loss = tf.reduce_mean(target * pos_loss + (1 - target) * neg_loss)
        loss = tf.reduce_mean(1 - tf.image.ssim_multiscale(value, target, max_val=1.0)) + \
               f_loss
    elif type == 'mix_l1':
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


def encoder(in_tensor, activation=tf.nn.relu, norm='instance', scope='encoder', b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        print(scope + ' encoder input: ' + str(in_tensor.get_shape().as_list()))
        num_bottleneck = bottleneck_num
        block_depth = unit_block_depth

        l = layers.conv(in_tensor, scope='init', filter_dims=[5, 5, block_depth], stride_dims=[1, 1],
                        non_linear_fn=activation)

        print(scope + ' Downsample: ' + str(l.get_shape().as_list()))

        for i in range(downsample_num + 1):
            block_depth = block_depth * 2
            l = layers.conv(l, scope='downsapmple_' + str(i), filter_dims=[4, 4, block_depth], stride_dims=[2, 2],
                            non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='encoder_norm_' + str(i))
            l = activation(l)

            print(scope + ' Downsample: ' + str(l.get_shape().as_list()))

        for i in range(num_bottleneck):
            l = layers.add_se_residual_block(l, filter_dims=[3, 3, block_depth],
                                             act_func=activation, scope=scope + '_bt_block_' + str(i))

            print(scope + ' Bottleneck Block : ' + str(l.get_shape().as_list()))

        if num_bottleneck > 0:
            l = activation(l)

        l = layers.conv(l, scope='latent', filter_dims=[3, 3, query_dimension], stride_dims=[1, 1],
                        non_linear_fn=None)

        query = l

    return query


def decoder(content, style=None, activation=tf.nn.relu, norm='instance', scope='decoder', b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        print(scope + ' decoder input: ' + str(content.get_shape().as_list()))
        block_depth = unit_block_depth * (2 ** (1 + upsample_num))

        l = content
        l = layers.conv(l, scope='init', filter_dims=[3, 3, block_depth], stride_dims=[1, 1], non_linear_fn=activation)

        if style is not None:
            target_depth = block_depth
            style_alpha = layers.fc(style, target_depth, non_linear_fn=None, scope='style_alpha')
            style_beta = layers.fc(style, target_depth, non_linear_fn=None, scope='style_beta')
            style_alpha = tf.reshape(style_alpha, shape=[-1, 1, 1, target_depth])
            style_beta = tf.reshape(style_beta, shape=[-1, 1, 1, target_depth])

            for i in range(decoder_bottleneck_num):
                l = layers.add_se_adain_residual_block(l, style_alpha, style_beta, filter_dims=[3, 3, block_depth],
                                                       act_func=activation, scope=scope + '_bt_block_' + str(i))

                print(scope + ' Bottleneck Block : ' + str(l.get_shape().as_list()))
        else:
            for i in range(decoder_bottleneck_num):
                l = layers.add_se_residual_block(l, filter_dims=[3, 3, block_depth],
                                                 act_func=activation, scope=scope + '_bt_block_' + str(i))

                print(scope + ' Bottleneck Block : ' + str(l.get_shape().as_list()))

        if decoder_bottleneck_num > 0:
            l = layers.conv(l, scope='upsample_init', filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
                            non_linear_fn=activation)

        for i in range(upsample_num + 1):
            block_depth = block_depth // 2
            # l = layers.conv(l, scope='espcn_1' + str(i), filter_dims=[3, 3, block_depth * 2 * 2], stride_dims=[1, 1],
            #                non_linear_fn=None)
            # l = tf.nn.depth_to_space(l, 2)
            _, h, w, _ = l.get_shape().as_list()
            l = tf.image.resize_images(l, size=[2 * h, 2 * w])
            l = layers.conv(l, scope='upsample_conv_' + str(i), filter_dims=[3, 3, block_depth],
                            stride_dims=[1, 1], non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='upsample_norm_' + str(i))
            l = activation(l)
            print(scope + ' Upsampling: ' + str(l.get_shape().as_list()))

        l = activation(l)
        img = layers.conv(l, scope='last', filter_dims=[5, 5, num_channel], stride_dims=[1, 1], non_linear_fn=None)
        print(scope + ' Output: ' + str(img.get_shape().as_list()))

    return img


def style_encoder(in_tensor, activation=util.swish, norm='layer', scope='style_encoder', b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        print(scope + ' encoder input: ' + str(in_tensor.get_shape().as_list()))
        block_depth = unit_block_depth
        style_tensor = layers.conv(in_tensor, scope='style_init1', filter_dims=[5, 5, block_depth], stride_dims=[1, 1],
                                   non_linear_fn=activation)
        style_tensor = layers.conv(style_tensor, scope='style_init2', filter_dims=[3, 3, block_depth],
                                   stride_dims=[2, 2],
                                   non_linear_fn=activation)
        # Downsample stage.
        l = style_tensor

        for i in range(downsample_num):
            l = layers.conv(l, scope='style_downsapmple1_' + str(i), filter_dims=[3, 3, block_depth],
                            stride_dims=[1, 1],
                            non_linear_fn=activation)
            block_depth = block_depth + unit_block_depth
            l = layers.conv(l, scope='style_downsapmple2_' + str(i), filter_dims=[3, 3, block_depth],
                            stride_dims=[2, 2], non_linear_fn=activation)

            print(scope + ' Style Downsample: ' + str(l.get_shape().as_list()))

        l = layers.conv(l, scope='latent', filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
                        non_linear_fn=None)
        _, h, w, c = l.get_shape().as_list()
        l = tf.reshape(l, shape=[-1, h // 16, w // 16, block_depth * 16 * 16])
        style = layers.global_avg_pool(l, style_dimension, scope='gap_style')

    return style


def unet_encoder(in_tensor, activation=tf.nn.relu, norm='instance', scope='unet_encoder', b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        print(scope + ' encoder input: ' + str(in_tensor.get_shape().as_list()))
        block_depth = unet_unit_block_depth
        lateral_layers = []

        l = layers.conv(in_tensor, scope='init', filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
                        non_linear_fn=activation)
        l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='init_norm')
        l = activation(l)
        print(scope + ' Add Lateral: ' + str(l.get_shape().as_list()))
        lateral_layers.append(l)

        for i in range(segmentation_downsample_num):
            for j in range(len(lateral_layers)):
                _, h, w, c = lateral_layers[j].get_shape().as_list()
                for num_rblock in range(2):
                    print('Residual Layer: ' + str(lateral_layers[j].get_shape().as_list()))
                    lateral_layers[j] = layers.add_residual_block(lateral_layers[j], filter_dims=[3, 3, c],
                                                                     scope='sqblock_' + str(j) + str(i) + str(num_rblock))

            if len(lateral_layers) > 1:
                print('Shuffling..')
                fused_layers = []
                for j in range(len(lateral_layers)):
                    _, l_h, l_w, l_c = lateral_layers[j].get_shape().as_list()
                    mixed_layer = lateral_layers[j]
                    for k in range(len(lateral_layers)):
                        if j != k:
                            l_lat = lateral_layers[k]
                            _, h, w, _ = l_lat.get_shape().as_list()

                            if l_h > h:
                                ratio = l_h // h
                                l_lat = layers.conv(l_lat, scope='espcn_' + str(i) + str(j) + str(k),
                                                    filter_dims=[3, 3, l_c * ratio * ratio], stride_dims=[1, 1],
                                                    non_linear_fn=None)
                                l_lat = layers.conv_normalize(l_lat, norm=norm, b_train=b_train,
                                                              scope='espcn_norm_' + str(i) + str(j) + str(k))
                                l_lat = activation(l_lat)
                                l_lat = tf.nn.depth_to_space(l_lat, ratio)
                            elif l_h < h:
                                ratio = h // l_h
                                num_downsample = int(np.log2(ratio))
                                for num_down in range(num_downsample):
                                    l_lat = layers.conv(l_lat, scope='dn_' + str(i) + str(j) + str(k) + str(num_down),
                                                        filter_dims=[3, 3, l_c], stride_dims=[2, 2], non_linear_fn=None)
                                    l_lat = layers.conv_normalize(l_lat, norm=norm, b_train=b_train,
                                                                  scope='norm_' + str(i) + str(j) + str(k) + str(num_down))
                                    l_lat = activation(l_lat)

                            mixed_layer = tf.add(mixed_layer, l_lat)
                    fused_layers.append(mixed_layer)
                lateral_layers = fused_layers

            l = lateral_layers[-1]
            block_depth = block_depth * 2
            l = layers.conv(l, scope='downsapmple1_' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
                            non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='encoder_norm1_' + str(i))
            l = activation(l)

            #l = layers.conv(l, scope='downsapmple2_' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
            #                non_linear_fn=None)
            #l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='encoder_norm2_' + str(i))
            #l = activation(l)
            print(scope + ' Add Lateral: ' + str(l.get_shape().as_list()))
            lateral_layers.append(l)

        # l = layers.context_attention(l)
        latent = layers.conv(l, scope='latent', filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
                             non_linear_fn=None)
        latent = layers.conv_normalize(latent, norm=norm, b_train=b_train, scope='latent_norm')
        latent = activation(latent)
        #print(scope + ' Encoder Out: ' + str(l.get_shape().as_list()))

        for n_loop in range(8):
            for i in range(len(lateral_layers)):
                _, h, w, c = lateral_layers[i].get_shape().as_list()
                for num_rblock in range(2):
                    print('Residual Layer: ' + str(lateral_layers[i].get_shape().as_list()))
                    lateral_layers[i] = layers.add_residual_block(lateral_layers[i], filter_dims=[3, 3, c],
                                                                     scope='loop_sqblock_' + str(n_loop) + str(i) + str(
                                                                         num_rblock))
            print('Shuffling...')
            fused_layers = []
            for i in range(len(lateral_layers)):
                _, l_h, l_w, l_c = lateral_layers[i].get_shape().as_list()
                mixed_layer = lateral_layers[i]
                for j in range(len(lateral_layers)):
                    if i != j:
                        l_lat = lateral_layers[j]
                        _, h, w, _ = l_lat.get_shape().as_list()

                        if l_h > h:
                            ratio = l_h // h
                            l_lat = layers.conv(l_lat, scope='loop_espcn_' + str(n_loop) + str(i) + str(j),
                                                filter_dims=[3, 3, l_c * ratio * ratio], stride_dims=[1, 1],
                                                non_linear_fn=None)
                            l_lat = layers.conv_normalize(l_lat, norm=norm, b_train=b_train,
                                                          scope='loop_espcn_norm_' + str(n_loop) + str(i) + str(j))
                            l_lat = activation(l_lat)
                            l_lat = tf.nn.depth_to_space(l_lat, ratio)
                        elif l_h < h:
                            ratio = h // l_h
                            num_downsample = int(np.log2(ratio))
                            for k in range(num_downsample):
                                l_lat = layers.conv(l_lat, scope='loop_dn_' + str(n_loop) + str(i) + str(j) + str(k),
                                                    filter_dims=[3, 3, l_c], stride_dims=[2, 2], non_linear_fn=None)
                                l_lat = layers.conv_normalize(l_lat, norm=norm, b_train=b_train,
                                                              scope='loop_norm_' + str(n_loop) + str(i) + str(j) + str(k))
                                l_lat = activation(l_lat)

                        mixed_layer = tf.add(mixed_layer, l_lat)
                fused_layers.append(mixed_layer)
            lateral_layers = fused_layers
            l0 = layers.conv(in_tensor, scope='init_injection_' + str(n_loop), filter_dims=[3, 3, unet_unit_block_depth], stride_dims=[2, 2],
                            non_linear_fn=activation)
            l0 = layers.conv_normalize(l0, norm=norm, b_train=b_train, scope='init_injection_norm_' + str(n_loop))
            l0 = activation(l0)
            lateral_layers[0] = tf.add(lateral_layers[0], l0)

    return latent, lateral_layers


def unet3_concat(lateral_layers, start_step=0, unit_block=8, activation=tf.nn.relu, norm='instance',
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


def unet3_decoder(latent, lateral_layers, activation=tf.nn.relu, norm='instance', scope='unet3_decoder', b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        print(scope + ' Decoder Input: ' + str(latent.get_shape().as_list()))
        block_depth = unet3_unit_block_depth
        lateral_layers.append(latent)
        lateral_layers.reverse()

        # Full concat version
        low_layers = []

        for i in range(len(lateral_layers)):
            concat_laterals = unet3_concat(lateral_layers, start_step=i, unit_block=block_depth, norm=norm,
                                           activation=activation, b_train=b_train, scope='unet3_concat_' + str(i))
            print(scope + ' Concat Laterals ' + str(i) + ': ' + str(concat_laterals.get_shape().as_list()))
            l = concat_laterals
            _, _, _, target_depth = lateral_layers[i].get_shape().as_list()

            for j in range(len(low_layers)):
                ratio = 2 ** (len(low_layers) - j)
                _, h, w, _ = low_layers[j].get_shape().as_list()
                lowl = low_layers[j]

                lowl = layers.conv(lowl, scope=str(i) + '_ll_concat_' + str(j),
                                   filter_dims=[3, 3, target_depth * ratio * ratio],
                                   stride_dims=[1, 1], non_linear_fn=None)
                lowl = layers.conv_normalize(lowl, norm=norm, b_train=b_train, scope='norm_low_' + str(i) + str(j))
                lowl = activation(lowl)
                lowl = tf.nn.depth_to_space(lowl, ratio)
                # lowl = tf.image.resize_images(lowl, size=[ratio * h, ratio * w])
                # print(scope + ' low step resize: ' + str(lowl.get_shape().as_list()))
                l = tf.concat([l, lowl], axis=-1)
                # print(scope + ' step: ' + str(j))

            _, _, _, concat_c = l.get_shape().as_list()

            l = layers.conv(l, scope='decoder_layer_' + str(i), filter_dims=[3, 3, concat_c],
                            stride_dims=[1, 1], non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='decoder_layer_norm_' + str(i))
            l = activation(l)
            print(scope + ' Features Layer: ' + str(l.get_shape().as_list()))

            low_layers.append(l)
        l = low_layers[-1]

        '''
        l = lateral_layers[-1]
        _, h, w, num_ch = l.get_shape().as_list()

        for layer_num in range(len(lateral_layers) - 1):
            lowl = lateral_layers[layer_num]
            _, l_h, l_w, l_ch = lowl.get_shape().as_list()

            #lowl = tf.image.resize_images(lowl, size=[h, w])

            ratio = h // l_h
            lowl = layers.conv(lowl, scope='ll_concat_' + str(layer_num), filter_dims=[1, 1, num_ch],
                               stride_dims=[1, 1], non_linear_fn=None)
            lowl = layers.conv_normalize(lowl, norm=norm, b_train=b_train, scope='norm_low_' + str(layer_num))
            lowl = activation(lowl)

            lowl = layers.conv(lowl, scope='espcn_' + str(layer_num), filter_dims=[3, 3, block_depth * ratio * ratio],
                               stride_dims=[1, 1], non_linear_fn=None)
            lowl = layers.conv_normalize(lowl, norm=norm, b_train=b_train, scope='espcn_norm_' + str(layer_num))
            lowl = activation(lowl)
            lowl = tf.nn.depth_to_space(lowl, ratio)

            l = tf.concat([l, lowl], axis=-1)
        _, h, w, num_ch = l.get_shape().as_list()
        l = layers.conv(l, scope='out_sum', filter_dims=[1, 1, num_ch], stride_dims=[1, 1], non_linear_fn=None)
        l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='out_sum_norm')
        l = activation(l)
        '''
        img_list = []
        # segment_layer = layers.conv(l, scope='segment', filter_dims=[1, 1, block_depth], stride_dims=[1, 1],
        #                            non_linear_fn=None)
        # segment_layer = layers.conv_normalize(segment_layer, norm=norm, b_train=b_train, scope='out_sum_norm')
        # segment_layer = activation(segment_layer)
        segment_layer = layers.conv(l, scope='espcn_hr', filter_dims=[1, 1, 1 * 2 * 2],
                                    stride_dims=[1, 1], non_linear_fn=tf.nn.sigmoid)
        # segment_layer = layers.conv_normalize(segment_layer, norm=norm, b_train=b_train, scope='espcn_norm_hr')
        # segment_layer = activation(segment_layer)
        segment_layer = tf.nn.depth_to_space(segment_layer, 2)
        # segment_layer = tf.image.resize_images(segment_layer, size=[input_height, input_width])
        # segment_layer = layers.conv(segment_layer, scope='segment_hr', filter_dims=[1, 1, 1], stride_dims=[1, 1],
        #                            non_linear_fn=tf.nn.sigmoid)
        img_list.append(segment_layer)

    return img_list


def unet_decoder(latent, lateral_layers, activation=tf.nn.relu, norm='instance', scope='unet_decoder', b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        print(scope + ' decoder input: ' + str(latent.get_shape().as_list()))
        block_depth = unet_unit_block_depth * (2 ** (1 + segmentation_upsample_num))

        l = latent
        lateral_layers.reverse()

        for i in range(segmentation_upsample_num + 1):
            block_depth = block_depth // 2

            # ESPCN
            l = layers.conv(l, scope='espcn_1_' + str(i), filter_dims=[3, 3, block_depth * 2 * 2], stride_dims=[1, 1],
                            non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='upsample_norm_' + str(i))
            l = activation(l)
            l = tf.nn.depth_to_space(l, 2)

            # Resize & Conv
            # _, h, w, _ = l.get_shape().as_list()
            # l = tf.image.resize_images(l, size=[2 * h, 2 * w])
            # l = layers.conv(l, scope='upsample_conv_' + str(i), filter_dims=[3, 3, block_depth],
            #                stride_dims=[1, 1], non_linear_fn=None)
            # l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='upsample_norm_' + str(i))
            # l = activation(l)
            print(scope + ' Upsampling: ' + str(l.get_shape().as_list()))

            l = tf.concat([l, lateral_layers[i]], axis=-1)
            l = layers.conv(l, scope='upsample_conv2_' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
                            non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='upsample_norm2_' + str(i))
            l = activation(l)

        # l = layers.conv(l, scope='summation', filter_dims=[3, 3, block_depth], stride_dims=[1, 1], non_linear_fn=activation)
        img = layers.conv(l, scope='last', filter_dims=[1, 1, 1], stride_dims=[1, 1], non_linear_fn=tf.nn.sigmoid)
        img = tf.image.resize_images(img, size=[input_height, input_width])
        print(scope + ' Output: ' + str(img.get_shape().as_list()))

    return img


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
            act_func = util.swish
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


def train(model_path='None'):
    print('Please wait. Preparing to start training...')
    train_start_time = time.time()

    # Mask Creation
    if use_roi_mask is True:
        print('Create RoI Mask Image.')
        roi_mask_img = create_roi_mask(input_width, input_height)
    else:
        roi_mask_img = None

    outlier_files = []
    if use_outlier_samples is True:

        # Classes
        raw_aug_files = os.listdir(aug_data)
        print('Load augmentation samples, Total Num of Samples: ' + str(len(raw_aug_files)))

        for a_file in raw_aug_files:
            a_file_path = os.path.join(aug_data, a_file).replace("\\", "/")
            outlier_files.append(a_file_path)
        outlier_files = shuffle(outlier_files)

    learning_rate = 1e-3
    sparsity = 2e-5

    tf.reset_default_graph()

    X_IN = tf.placeholder(tf.float32, [None, input_height, input_width, num_channel])
    Y_IN = tf.placeholder(tf.float32, [None, input_height, input_width, num_channel])
    S_IN = tf.placeholder(tf.float32, [None, input_height, input_width, 1])
    B_TRAIN = tf.placeholder(tf.bool)
    LR = tf.placeholder(tf.float32, None)

    # Generator
    query = encoder(X_IN, norm='layer', activation=util.swish, scope=G_Encoder_scope, b_train=B_TRAIN)
    if use_style is True:
        style = style_encoder(X_IN, norm='layer', scope=Style_Encoder_scope, b_train=B_TRAIN)
    else:
        style = None
    attention_g, latent_g = spatial_memory(query, size=aug_mem_size, dims=representation_dimension, scope=G_M_scope)
    # attention_g, latent_g = memory(query, size=aug_mem_size, dims=representation_dimension, scope=G_M_scope)
    G_X = decoder(latent_g, style, norm='layer', scope=G_Decoder_scope, b_train=B_TRAIN)
    unet_input = tf.concat([X_IN, G_X], axis=-1)
    z_gen, laterals = unet_encoder(unet_input, norm='layer', activation=tf.nn.leaky_relu, scope=G_UNet_Encoder_scope,
                                   b_train=B_TRAIN)

    unet_residual_loss = 0.0
    if use_unet3 is True:
        IMGS = unet3_decoder(z_gen, laterals, norm='layer', scope=G_UNet_Decoder_scope, b_train=B_TRAIN)
        for i in range(len(IMGS)):
            print('unet3 decoder images: ' + str(IMGS[i].get_shape().as_list()))
            unet_residual_loss += get_residual_loss(IMGS[i], S_IN, type='mix_focal')
            unet_residual_loss += get_residual_loss(IMGS[i], S_IN, type='ft', alpha=0.6)
        U_G_X = IMGS[-1]
    else:
        U_G_X = unet_decoder(z_gen, laterals, norm='layer', scope=G_UNet_Decoder_scope, b_train=B_TRAIN)
        unet_residual_loss = get_residual_loss(U_G_X, S_IN, type='ft', alpha=0.8)

    pseudo_residual_loss = get_residual_loss(G_X, Y_IN, type='mix_l1', alpha=0.8)

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

    pseudo_residual_loss += sparsity_reg_loss

    if use_gan is True:
        fake_features, fake_logit = discriminator(x1=U_G_X, x2=X_IN, norm='layer', scope=D_scope, b_train=B_TRAIN)
        real_features, real_logit = discriminator(x1=S_IN, x2=X_IN, norm='layer', scope=D_scope, b_train=B_TRAIN)
        d_real_loss = get_discriminator_loss(tf.ones_like(real_logit), real_logit, type='ls') + \
                      get_discriminator_loss(tf.zeros_like(fake_logit), fake_logit, type='ls')
        d_fake_loss = get_discriminator_loss(tf.ones_like(fake_logit), fake_logit, type='ls')

        # d_feature_loss = 0.0
        # for num_feature in range(len(fake_features)):
        #    d_feature_loss = d_feature_loss + get_residual_loss(real_features[num_feature], fake_features[num_feature], type='l1')
        # d_feature_loss = d_feature_loss / len(fake_features)

        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=D_scope)
        unet_residual_loss = unet_residual_loss + 1e-2 * d_fake_loss

    unet_encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=G_UNet_Encoder_scope)
    unet_decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=G_UNet_Decoder_scope)
    unet_generator_vars = unet_encoder_vars + unet_decoder_vars

    ae_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=G_Encoder_scope) + \
              tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=G_Decoder_scope)
    style_encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=Style_Encoder_scope)
    memory_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=G_M_scope)
    pseudo_generator_vars = memory_vars + ae_vars + style_encoder_vars
    total_joint_variable = unet_generator_vars + pseudo_generator_vars
    # Optimizer
    unet_loss = unet_residual_loss
    mnet_loss = pseudo_residual_loss

    unet_generator_optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(unet_loss,
                                                                                 var_list=unet_generator_vars)
    pseudo_generator_optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(mnet_loss,
                                                                                   var_list=pseudo_generator_vars)
    joint_optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(unet_loss + mnet_loss,
                                                                        var_list=total_joint_variable)
    if use_gan is True:
        # d_opt = tf.train.AdamOptimizer(learning_rate=LR)
        # d_gradients = d_opt.compute_gradients(-disc_loss, disc_vars)
        # clipped_d_gradients = [(tf.clip_by_value(grad, -0.01, 0.01), var) for grad, var in d_gradients]
        # disc_optimizer = d_opt.apply_gradients(clipped_d_gradients)
        disc_optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(d_real_loss, var_list=disc_vars)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    num_pretrain = encoder_pretrain

    ae_model_path = os.path.join(model_path, 'ae/m.chpt').replace("\\", "/")
    uae_model_path = os.path.join(model_path, 'uae/m.chpt').replace("\\", "/")
    if use_gan is True:
        disc_model_path = os.path.join(model_path, 'disc/m.chpt').replace("\\", "/")
        discriminator_saver = tf.train.Saver(disc_vars)
    pseudo_generator_saver = tf.train.Saver(pseudo_generator_vars)
    unet_generator_saver = tf.train.Saver(unet_generator_vars)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        try:
            pseudo_generator_saver.restore(sess, ae_model_path)
            print('Load pseudo-generator.')
            num_pretrain = 0
            if use_gan is True:
                discriminator_saver.restore(sess, disc_model_path)
                print('Load discriminator.')
            unet_generator_saver.restore(sess, uae_model_path)
            print('Load unet-generator.')
        except:
            if num_pretrain == 0:
                print('Start Training UNet Generator Only. Wait ...')
            else:
                print('Start New Training. Wait ...')

        te_dir = test_data
        tr_dir = train_data
        cutout_mask = create_roi_mask(input_width, input_height, offset=45)
        test_mask = create_roi_mask(input_width, input_height, offset=65)
        tr_files = []
        # Classes
        classes = os.listdir(tr_dir)

        print(' Num classes: ' + str(len(classes)))
        for cls in classes:
            class_path = os.path.join(tr_dir, cls).replace("\\", "/")
            samples = os.listdir(class_path)
            samples = np.random.choice(samples, size=num_samples_per_class)  # (1000//len(classes)))
            for s in samples:
                sample_path = os.path.join(class_path, s).replace("\\", "/")
                tr_files.append(sample_path)
        print(' Num samples per epoch: ' + str(len(tr_files)))
        total_input_size = len(tr_files)

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
            tr_files = shuffle(tr_files)
            training_batch = zip(range(0, total_input_size, batch_size),
                                 range(batch_size, total_input_size + 1, batch_size))
            itr = 0

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

            # Learning rate schedule
            if e <= num_pretrain:
                lr = learning_rate
            else:
                lr = 0.5 * learning_rate * (1.0 + np.cos(np.pi * ((e - encoder_pretrain) / num_epoch)))

            for start, end in training_batch:
                itr = itr + 1

                b_use_cutdout = True
                b_use_outlier_samples = use_outlier_samples
                b_use_bg_samples = use_bg_samples

                if np.random.randint(1, 10) < 5:
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
                    aug_noise = perlin_noise

                batch_imgs, gt_imgs, seg_imgs = load_images(tr_files[start:end], rotate=False, shift=False, flip=True,
                                                            mask_noise=aug_noise, mask=roi_mask_img,
                                                            cutout=b_use_cutdout, cutout_mask=cutout_mask,
                                                            scaleabs=False)
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
                    blending_a = np.random.uniform(low=0.2, high=0.9)
                    noise_sample_images = (1 - blending_a) * noise_sample_images + blending_a * batch_imgs
                    # fg = seg_imgs * noise_sample_imgs
                    fg = noise_sample_segments * noise_sample_images
                    # bg = (1 - seg_imgs) * batch_imgs
                    bg = (1 - noise_sample_segments) * batch_imgs
                    batch_imgs = fg + bg
                    seg_imgs = seg_imgs + noise_sample_segments
                    seg_imgs = np.where(seg_imgs > 0, 1.0, 0.0)

                if e >= num_pretrain:
                    if meta_training is True:  # Meta Training
                        if freeze_ae is False:
                            _, pseudo_g_loss = sess.run([pseudo_generator_optimizer, mnet_loss],
                                                        feed_dict={X_IN: batch_imgs, Y_IN: gt_imgs, LR: lr,
                                                                   B_TRAIN: True})
                        _, unet_g_loss, u_g_x_imgs = sess.run(
                            [unet_generator_optimizer, unet_loss, U_G_X],
                            feed_dict={X_IN: batch_imgs, S_IN: seg_imgs, LR: lr, B_TRAIN: True})
                    else:
                        _, unet_g_loss, pseudo_g_loss, u_g_x_imgs = sess.run(
                            [joint_optimizer, unet_loss, mnet_loss, U_G_X],
                            feed_dict={X_IN: batch_imgs, Y_IN: gt_imgs, S_IN: seg_imgs, LR: lr, B_TRAIN: True})
                    if freeze_ae is True:
                        print('epoch: ' + str(e) + ', unet_g_loss: ' + str(unet_g_loss))
                    else:
                        print('epoch: ' + str(e) + ', unet_g_loss: ' + str(unet_g_loss) + ', pseudo_g_loss: ' +
                              str(pseudo_g_loss))

                    if itr % 10 == 0:
                        cv2.imwrite(out_dir + '/' + str(itr) + '.jpg', 255 * u_g_x_imgs[0])
                        cv2.imwrite(out_dir + '/' + str(itr) + '_gt.jpg', 255 * seg_imgs[0])
                        cv2.imwrite(out_dir + '/' + str(itr) + '_anomal.jpg', 255 * batch_imgs[0])
                        # cv2.imwrite(out_dir + '/' + str(itr) + '_recon.jpg', 255 * g_x_imgs[0])
                        print('Elapsed Time at  ' + str(e) + '/' + str(num_epoch) + ' epochs, ' +
                              str(time.time() - train_start_time) + ' sec')
                    if use_gan is True:
                        _ = sess.run([disc_optimizer],
                                     feed_dict={X_IN: batch_imgs, S_IN: seg_imgs, LR: lr, B_TRAIN: True})
                else:
                    _, g_x_imgs, pseudo_g_loss = sess.run([pseudo_generator_optimizer, G_X, mnet_loss],
                                                          feed_dict={X_IN: batch_imgs, Y_IN: gt_imgs, LR: lr,
                                                                     B_TRAIN: True})
                    print('epoch: ' + str(e) + ', pseudo_g_loss: ' + str(pseudo_g_loss))
                    if itr % 10 == 0:
                        cv2.imwrite(out_dir + '/' + str(itr) + '.jpg', 255 * g_x_imgs[0])

                        print('Elapsed Time at  ' + str(e) + '/' + str(num_epoch) + ' epochs, ' + str(
                            time.time() - train_start_time) + ' sec')

            if use_semisupervised is True:
                # Semi Supervised.
                if e % 2 == 0:
                    itr = 0
                    labeled_X, labeled_Y = shuffle(labeled_X, labeled_Y)
                    training_batch = zip(range(0, len(labeled_X), batch_size),
                                         range(batch_size, len(labeled_X) + 1, batch_size))

                    for start, end in training_batch:
                        labeled_file_X = labeled_X[start:end]
                        labeled_file_Y = labeled_Y[start:end]

                        labeled_img_X, _, _ = load_images(labeled_file_X)
                        labeled_img_Y, _, _ = load_images(labeled_file_Y, gray_scale=True)

                        _, u_loss, u_g_x_imgs = sess.run([unet_generator_optimizer, unet_loss, U_G_X],
                                                         feed_dict={X_IN: labeled_img_X, S_IN: labeled_img_Y, LR: lr,
                                                                    B_TRAIN: True})
                        itr += 1
                        print('epoch: ' + str(e) + ', unet_g_loss: ' + str(u_loss))
                        if itr % 10 == 0:
                            cv2.imwrite(out_dir + '/' + str(itr) + '.jpg', 255 * u_g_x_imgs[0])
                            cv2.imwrite(out_dir + '/' + str(itr) + '_gt.jpg', 255 * labeled_img_Y[0])
                            cv2.imwrite(out_dir + '/' + str(itr) + '_anomal.jpg', 255 * labeled_img_X[0])
            try:
                print('Saving model...')
                # total_saver = tf.train.Saver()
                # total_saver.save(sess, model_path, write_meta_graph=False)
                if freeze_ae is False:
                    pseudo_generator_saver.save(sess, ae_model_path, write_meta_graph=False)
                unet_generator_saver.save(sess, uae_model_path, write_meta_graph=False)
                if use_gan is True:
                    discriminator_saver.save(sess, disc_model_path, write_meta_graph=False)
                print('Saved.')
            except:
                print('Save failed')

            if (e + 1) % 30 == 0:
                te_files = os.listdir(te_dir)
                te_batch = zip(range(0, len(te_files), batch_size),
                               range(batch_size, len(te_files) + 1, batch_size))
                for t_s, t_e in te_batch:
                    test_imgs, _, _ = load_images(te_files[t_s:t_e], base_dir=te_dir)
                    u_gx_imgs = sess.run(U_G_X, feed_dict={X_IN: test_imgs, B_TRAIN: False})
                    # u_gx_imgs = np.where(u_gx_imgs > 0.9, 1.0, 0.0)
                    delta_imgs = np.abs(u_gx_imgs - test_imgs) * test_mask

                    for i in range(batch_size):
                        cv2.imwrite('out/' + te_files[t_s + i], 255 * u_gx_imgs[i])
                        cv2.imwrite('out/delta_img_' + te_files[t_s + i], 255 * delta_imgs[i])
                        # s = calculate_anomaly_score(255 * delta_imgs[i], input_width, input_height, filter_size=32)
                        print('Test: ' + te_files[t_s + i])


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

    X_IN = tf.placeholder(tf.float32, [None, input_height, input_width, num_channel])
    B_TRAIN = tf.placeholder(tf.bool)

    # Generator
    query = encoder(X_IN, norm='layer', scope=G_Encoder_scope, b_train=B_TRAIN)
    if use_style is True:
        style = style_encoder(X_IN, norm='layer', scope=Style_Encoder_scope, b_train=B_TRAIN)
    else:
        style = None
    attention_g, latent_g = spatial_memory(query, size=aug_mem_size, dims=representation_dimension, scope=G_M_scope)
    # attention_g, latent_g = memory(query, size=aug_mem_size, dims=representation_dimension, scope=G_M_scope)
    G_X = decoder(latent_g, style, norm='layer', scope=G_Decoder_scope, b_train=B_TRAIN)

    unet_input = tf.concat([G_X, X_IN], axis=-1)
    z_gen, laterals = unet_encoder(unet_input, norm='layer', scope=G_UNet_Encoder_scope, b_train=B_TRAIN)

    if use_unet3 is True:
        U_G_X = unet3_decoder(z_gen, laterals, norm='layer', scope=G_UNet_Decoder_scope, b_train=B_TRAIN)
    else:
        U_G_X = unet_decoder(z_gen, laterals, norm='layer', scope=G_UNet_Decoder_scope, b_train=B_TRAIN)

    unet_encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=G_UNet_Encoder_scope)
    unet_decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=G_UNet_Decoder_scope)
    unet_generator_vars = unet_encoder_vars + unet_decoder_vars

    ae_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=G_Encoder_scope) + \
              tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=G_Decoder_scope)
    style_encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=Style_Encoder_scope)
    memory_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=G_M_scope)
    pseudo_generator_vars = memory_vars + ae_vars + style_encoder_vars
    total_joint_variable = unet_generator_vars + pseudo_generator_vars

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    test_mask = create_roi_mask(input_width, input_height, offset=60)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        try:
            saver = tf.train.Saver(var_list=total_joint_variable)
            saver.restore(sess, model_path)
            print('Model Restored')
        except:
            print('Start New Training. Wait ...')

        te_dir = test_data
        te_files = os.listdir(te_dir)
        batch_size = 1
        te_batch = zip(range(0, len(te_files), batch_size),
                       range(batch_size, len(te_files) + 1, batch_size))

        for t_s, t_e in te_batch:
            test_imgs, _ = load_images(te_files[t_s:t_e], base_dir=te_dir)
            u_gx_imgs = sess.run(U_G_X, feed_dict={X_IN: test_imgs, B_TRAIN: False})

            for i in range(batch_size):
                cv2.imwrite('out/' + te_files[t_s + i], 255 * u_gx_imgs[i])
                delta_imgs = np.abs(u_gx_imgs - test_imgs) * test_mask
                cv2.imwrite('out/delta_img_' + te_files[t_s + i], 255 * delta_imgs[i])
                s = calculate_anomaly_score(delta_imgs[i], input_width, input_height, filter_size=32)
                print('anomaly score of ' + te_files[t_s + i] + ': ' + str(s))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, help='train/test', default='train')
    parser.add_argument('--model_path', type=str, help='model check point file path', default='model/m.ckpt')
    parser.add_argument('--train_data', type=str, help='training data directory', default='data/train')
    parser.add_argument('--test_data', type=str, help='test data directory', default='data/test')
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
    out_dir = args.out_dir
    num_epoch = args.epoch
    alpha = args.alpha
    aug_data = args.aug_data
    bg_mask_data = args.bgmask_data
    noise_data = args.noise_data

    use_unet3 = True
    unet3_unit_block_depth = 16
    unit_block_depth = 16
    unet_unit_block_depth = 16

    disc_unit_block_depth = 8
    bottleneck_num = 0
    decoder_bottleneck_num = 0
    query_dimension = 128
    style_dimension = 128

    representation_dimension = query_dimension

    segmentation_upsample_ratio = 4
    segmentation_downsample_num = int(np.log2(segmentation_upsample_ratio))
    segmentation_upsample_num = segmentation_downsample_num
    upsample_ratio = 8
    downsample_num = int(np.log2(upsample_ratio))
    upsample_num = downsample_num
    aug_mem_size = 2048
    num_channel = 3
    use_style = False
    use_categorical_constraints = True
    use_roi_mask = False
    use_outlier_samples = True
    use_bg_samples = True
    num_samples_per_class = 10
    meta_training = True
    encoder_pretrain = 20
    use_deep_supervision = False
    freeze_ae = False
    use_gan = False

    if mode == 'train':
        train(model_path)
    elif mode == 'test':
        test(model_path)
    else:
        print('Train or Test?')
