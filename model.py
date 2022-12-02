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


def load_images(file_name_list, base_dir=None, mask=None, cutout=False, add_eps=False, rotate=False, gray_scale=False,
                shift=False, brightness=False, mask_noise=None, masking_img=None):
    try:
        images = []
        gt_images = []
        m_img = None

        if masking_img is not None:
            fullname = masking_img[0]
            #print('masking_img: ' + str(fullname))
            if base_dir is not None:
                fullname = os.path.join(base_dir, masking_img).replace("\\", "/")

            m_img = cv2.imread(fullname)

            if m_img is None:
                print('Load failed: ' + fullname)
                return None

            if gray_scale is True:
                m_img = cv2.cvtColor(m_img, cv2.COLOR_BGR2GRAY)

            h, w, c = m_img.shape
            m_img = m_img[90:h - 90, 90:w - 90]
            m_img = cv2.resize(m_img, dsize=(input_width, input_height), interpolation=cv2.INTER_AREA)
        #else:
        #    # Random Reverse Augmentation
        #    m_img = 1 - np.ones((input_height, input_width, 3), dtype=np.uint8) * 1.0

        for file_name in file_name_list:
            fullname = file_name
            if base_dir is not None:
                fullname = os.path.join(base_dir, file_name).replace("\\", "/")
            img = cv2.imread(fullname)

            if img is None:
                print('Load failed: ' + fullname)
                return None

            if gray_scale is True:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            h, w, c = img.shape
            img = img[90:h - 90, 90:w - 90]
            img = cv2.resize(img, dsize=(input_width, input_height), interpolation=cv2.INTER_AREA)

            if img is not None:
                img = np.array(img)

                if brightness is True:
                    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    value = np.uint8((np.random.randint(0, 10)))
                    h, s, v = cv2.split(hsv_img)
                    v += value
                    v[v > 255] = 255
                    hsv_img = cv2.merge((h, s, v))
                    img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

                if rotate is True:
                    rot = np.random.randint(-45, 45)
                    img = util.rotate_image(img, rot)
                    if mask_noise is not None:
                        rot = np.random.randint(-90, 90)
                        mask_noise = util.rotate_image(mask_noise, rot)

                if shift is True:
                    x_shift = np.random.randint(-7, 7)
                    y_shift = np.random.randint(-7, 7)
                    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
                    img = cv2.warpAffine(img, M, (input_width, input_height))
                    if mask_noise is not None:
                        x_shift = np.random.randint(-15, 15)
                        y_shift = np.random.randint(-15, 15)
                        M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
                        mask_noise = cv2.warpAffine(mask_noise, M, (input_width, input_height))

                gt_img = img.copy()
                gt_img = np.array(gt_img)

                if add_eps is True:
                    img = img + np.random.normal(size=img.shape)
                    if m_img is not None:
                        m_img = m_img + 10 * np.random.normal(size=m_img.shape)

                if m_img is None:
                    #m_img = 1 - np.ones((input_height, input_width, 3), dtype=np.uint8) * 1.0
                    m_img = 255 * np.random.normal(size=img.shape)

                if cutout is True:
                    if mask_noise is not None:
                        # Random Reverse Augmentation
                        if np.random.randint(low=0, high=10) < 5:
                            mask_noise = 1 - mask_noise
                        if m_img is not None:
                            #img = img * mask_noise
                            r_mask_noise = 1 - mask_noise
                            m_img = m_img * r_mask_noise
                            bg_img = img * mask_noise
                            m_img = m_img + bg_img

                    # square cut out
                    co_w = np.random.randint(low=2, high=input_width // 10)
                    co_h = co_w
                    num_cutout = np.random.randint(low=0, high=input_width // co_w)

                    b_cutmix = True

                    for _ in range(num_cutout):
                        r_x = np.random.randint(low=0, high=input_width - co_w)
                        r_y = np.random.randint(low=0, high=input_height - co_h)
                        if mask_noise is not None:
                            if m_img is not None:
                                if b_cutmix is True:
                                    img[r_x:r_x + co_w, r_y:r_y + co_h, :] = m_img[r_x:r_x + co_w, r_y:r_y + co_h, :]
                                else:
                                    img[r_x:r_x + co_w, r_y:r_y + co_h, :] *= mask_noise[r_x:r_x + co_w, r_y:r_y + co_h, :]
                            else:
                                img[r_x:r_x + co_w, r_y:r_y + co_h, :] *= mask_noise[r_x:r_x + co_w, r_y:r_y + co_h, :]
                        else:
                            img[r_x:r_x + co_w, r_y:r_y + co_h, :] = np.random.randint(low=0, high=255)

                if gray_scale is True:
                    img = np.expand_dims(img, axis=-1)
                    gt_img = np.expand_dims(gt_img, axis=-1)

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

    return np.array(images), np.array(gt_images)


def get_gradient_loss(img1, img2, effective_pixel=0):
    # Laplacian second derivation
    image_a = img1  # tf.expand_dims(img1, axis=0)
    image_b = img2  # tf.expand_dims(img2, axis=0)

    dx_a, dy_a = tf.image.image_gradients(image_a)
    dx_b, dy_b = tf.image.image_gradients(image_b)

    #d2x_ax, d2y_ax = tf.image.image_gradients(dx_a)
    #d2x_bx, d2y_bx = tf.image.image_gradients(dx_b)
    #d2x_ay, d2y_ay = tf.image.image_gradients(dy_a)
    #d2x_by, d2y_by = tf.image.image_gradients(dy_b)

    # loss1 = tf.reduce_mean(tf.abs(tf.subtract(d2x_ax, d2x_bx))) + tf.reduce_mean(tf.abs(tf.subtract(d2y_ax, d2y_bx)))
    # loss2 = tf.reduce_mean(tf.abs(tf.subtract(d2x_ay, d2x_by))) + tf.reduce_mean(tf.abs(tf.subtract(d2y_ay, d2y_by)))

    if effective_pixel == 0:
        loss1 = tf.reduce_mean(tf.abs(tf.subtract(dx_a, dx_b)))
        loss2 = tf.reduce_mean(tf.abs(tf.subtract(dy_a, dy_b)))
    else:
        loss1 = tf.reduce_sum(tf.abs(tf.subtract(dx_a, dx_b))) / effective_pixel
        loss2 = tf.reduce_sum(tf.abs(tf.subtract(dy_a, dy_b))) / effective_pixel

    return loss1+loss2


def get_residual_loss(value, target, type='l1', alpha=1.0, effective_pixel=0):
    if type == 'rmse':
        loss = alpha * tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(target, value))))
    elif type == 'ce':
        eps = 1e-10
        loss = alpha * tf.reduce_mean(-1 * target * tf.log(value + eps) - 1 * (1 - target) * tf.log(1 - value + eps))
        # loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=value, labels=target))
    elif type == 'l1':
        if effective_pixel > 0:
            loss = alpha * tf.reduce_sum(tf.abs(tf.subtract(target, value))) / effective_pixel
        else:
            loss = alpha * tf.reduce_mean(tf.abs(tf.subtract(target, value)))
    elif type == 'l2':
        loss = alpha * tf.reduce_mean(tf.square(tf.subtract(target, value)))
    elif type == 'entropy':
        eps = 1e-10
        loss = alpha * tf.reduce_mean(-1 * value * tf.log(value + eps))
    elif type == 'focal':
        eps = 1e-10
        fc_t = tf.math.pow(1 - value, 2)
        fc_f = tf.math.pow(value, 2)
        loss = alpha * tf.reduce_mean(
            -1 * target * fc_t * tf.log(value + eps) - 1 * (1 - target) * fc_f * tf.log(1 - value + eps))
    elif type == 'mix_l1':
        if effective_pixel > 0:
            loss = alpha * tf.reduce_sum(1 - tf.image.ssim_multiscale(value, target, max_val=1.0)) + \
                   (1 - alpha) * tf.reduce_sum(tf.abs(tf.subtract(target, value))) + get_gradient_loss(value, target, effective_pixel=effective_pixel)
            loss = loss / effective_pixel
        else:
            loss = alpha * tf.reduce_mean(1 - tf.image.ssim_multiscale(value, target, max_val=1.0)) + \
                (1 - alpha) * tf.reduce_mean(tf.abs(tf.subtract(target, value))) + get_gradient_loss(value, target, effective_pixel=effective_pixel)

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


def encoder(in_tensor, activation=util.swish, norm='layer', scope='encoder', b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        print(scope + ' encoder input: ' + str(in_tensor.get_shape().as_list()))
        num_bottleneck = bottleneck_num
        block_depth = unit_block_depth

        init_tensor = layers.conv(in_tensor, scope='init1', filter_dims=[7, 7, block_depth], stride_dims=[1, 1],
                                  non_linear_fn=activation)
        init_tensor = layers.conv(init_tensor, scope='init2', filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
                                  non_linear_fn=activation)
        print(scope + ' Downsample: ' + str(init_tensor.get_shape().as_list()))

        l = init_tensor

        for i in range(downsample_num):
            l = layers.conv(l, scope='downsapmple1_' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
                            non_linear_fn=activation)
            block_depth = block_depth * 2
            l = layers.conv(l, scope='downsapmple2_' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
                            non_linear_fn=activation)

            print(scope + ' Downsample: ' + str(l.get_shape().as_list()))

        for i in range(num_bottleneck):
            l = layers.add_se_residual_block(l, filter_dims=[3, 3, block_depth],
                                             act_func=activation, scope=scope + '_bt_block_' + str(i),
                                             num_groups=dec_conv_num_groups)

            print(scope + ' Bottleneck Block : ' + str(l.get_shape().as_list()))

        l = activation(l)
        l = layers.conv(l, scope='latent', filter_dims=[3, 3, query_dimension], stride_dims=[1, 1], non_linear_fn=None)
        query = l

    return query


def decoder(content, style=None, activation=util.swish, norm='instance', scope='decoder', b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        print(scope + ' decoder input: ' + str(content.get_shape().as_list()))
        block_depth = unit_block_depth * upsample_ratio
        decoder_bottleneck_num = bottleneck_num

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
                                                       act_func=activation, scope=scope + '_bt_block_' + str(i),
                                                       num_groups=dec_conv_num_groups)

                print(scope + ' Bottleneck Block : ' + str(l.get_shape().as_list()))
        '''
        else:
            for i in range(decoder_bottleneck_num):
                l = layers.add_se_residual_block(l, filter_dims=[3, 3, block_depth],
                                                 act_func=activation, scope=scope + '_bt_block_' + str(i),
                                                 num_groups=dec_conv_num_groups)

                print(scope + ' Bottleneck Block : ' + str(l.get_shape().as_list()))

        l = layers.conv(l, scope='upsample_init', filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
                        non_linear_fn=activation)
        '''
        for i in range(upsample_num + 1):
            block_depth = block_depth // 2
            l = layers.conv(l, scope='espcn_1' + str(i), filter_dims=[3, 3, block_depth * 2 * 2], stride_dims=[1, 1],
                            non_linear_fn=None)
            l = tf.nn.depth_to_space(l, 2)
            l = layers.conv_normalize(l, norm='layer', b_train=b_train, scope='espcn_norm2_' + str(i))
            l = activation(l)

            print(scope + ' Upsampling: ' + str(l.get_shape().as_list()))

        img = layers.conv(l, scope='last', filter_dims=[3, 3, num_channel], stride_dims=[1, 1], non_linear_fn=None)
        print(scope + ' Output: ' + str(img.get_shape().as_list()))

    return img


def style_encoder(in_tensor, activation=util.swish, norm='layer', scope='style_encoder', b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        print(scope + ' encoder input: ' + str(in_tensor.get_shape().as_list()))
        block_depth = unit_block_depth
        style_tensor = layers.conv(in_tensor, scope='style_init1', filter_dims=[5, 5, block_depth], stride_dims=[1, 1],
                                   non_linear_fn=activation)
        style_tensor = layers.conv(style_tensor, scope='style_init2', filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
                                   non_linear_fn=activation)
        # Downsample stage.
        l = style_tensor

        for i in range(downsample_num):
            l = layers.conv(l, scope='style_downsapmple1_' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
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


def unet_encoder(in_tensor, activation=util.swish, norm='layer', scope='unet_encoder', b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        print(scope + ' encoder input: ' + str(in_tensor.get_shape().as_list()))
        num_bottleneck = bottleneck_num
        block_depth = unit_block_depth

        init_tensor = layers.conv(in_tensor, scope='init1', filter_dims=[5, 5, block_depth], stride_dims=[1, 1],
                                  non_linear_fn=activation)
        init_tensor = layers.conv(init_tensor, scope='init2', filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
                                  non_linear_fn=None)
        print(scope + ' Downsample: ' + str(init_tensor.get_shape().as_list()))

        l = init_tensor
        lateral_layers = [l]

        for i in range(downsample_num - 1):
            l = activation(l)
            l = layers.conv(l, scope='downsapmple1_' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
                            non_linear_fn=activation)
            block_depth = block_depth * 2
            l = layers.conv(l, scope='downsapmple2_' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
                            non_linear_fn=None)
            # l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='dw_norm_' + str(i))
            lateral_layers.append(l)
            print(scope + ' Downsample: ' + str(l.get_shape().as_list()))

        l = activation(l)
        l = layers.conv(l, scope='latent_downsapmple_1' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
                        non_linear_fn=activation)
        block_depth = block_depth * 2
        l = layers.conv(l, scope='latent_downsapmple_2', filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
                        non_linear_fn=activation)

        '''
        # Bottleneck stage
        for i in range(num_bottleneck):
            print(scope + ' Bottleneck Block : ' + str(l.get_shape().as_list()))
            l = layers.add_se_residual_block(l, filter_dims=[3, 3, block_depth], act_func=activation,
                                             num_groups=enc_conv_num_groups, norm=norm, b_train=b_train, scope='bt_block_' + str(i))
        '''

        l = layers.conv(l, scope='latent', filter_dims=[3, 3, block_depth], stride_dims=[1, 1], non_linear_fn=None)
        latent = l

    return latent, lateral_layers


def unet_decoder(latent, lateral_layers, activation=util.swish, norm='layer', scope='unet_decoder', b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        print(scope + ' decoder input: ' + str(latent.get_shape().as_list()))
        block_depth = unit_block_depth * upsample_ratio

        l = latent

        for i in range(upsample_num):
            block_depth = block_depth // 2
            l = layers.conv(l, scope='espcn_1_' + str(i), filter_dims=[3, 3, block_depth * 2 * 2], stride_dims=[1, 1],
                            non_linear_fn=None)
            l = tf.nn.depth_to_space(l, 2)
            l = layers.conv_normalize(l, norm='layer', b_train=b_train, scope='espcn_norm1_' + str(i))
            # l = layers.conv(l, scope='espcn_2_' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
            #                non_linear_fn=activation)
            l = activation(l)
            l = tf.concat([l, lateral_layers[upsample_num - 1 - i]], axis=-1)
            print(scope + ' Upsampling: ' + str(l.get_shape().as_list()))

        block_depth = block_depth // 2
        l = layers.conv(l, scope='espcn_2_last', filter_dims=[3, 3, block_depth * 2 * 2],
                        stride_dims=[1, 1], non_linear_fn=None)
        l = tf.nn.depth_to_space(l, 2)
        l = layers.conv_normalize(l, norm='layer', b_train=b_train, scope='espcn_norm2')
        l = activation(l)

        print(scope + ' Upsampling: ' + str(l.get_shape().as_list()))

        img = layers.conv(l, scope='last', filter_dims=[3, 3, num_channel], stride_dims=[1, 1], non_linear_fn=None)
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
        attention = attention / (1e-4 + tf.abs(sm - threashold))
        print(scope + ' attention: ' + str(attention.get_shape().as_list()))
        l1 = tf.expand_dims(tf.reduce_sum(attention, axis=-1), axis=-1)
        # print(scope + ' l1: ' + str(l1.get_shape().as_list()))
        attention = attention / (l1 + 1e-7)

        latent = tf.matmul(attention, aug_mem)  # [B x H x W, Q] x [Q, R] = [B x H x W, R]
        latent = tf.reshape(latent, shape=[-1, h, w, dim])
        #norm_q = tf.reshape(norm_q, shape=[-1, h, w, dim])
        #latent = tf.add(latent, norm_q)
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


def discriminator(x1, x2, activation='swish', scope='discriminator_network', norm='layer', b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        elif activation == 'lrelu':
            act_func = tf.nn.leaky_relu
        else:
            act_func = tf.nn.sigmoid

        block_depth = unit_block_depth

        if x2 is not None:
            x = tf.concat([x1, x2], axis=-1)
        else:
            x = x1
        print(scope + ' Input: ' + str(x.get_shape().as_list()))

        l = layers.conv(x, scope='init', filter_dims=[3, 3, block_depth], stride_dims=[2, 2], non_linear_fn=act_func)

        downsample_num_itr = downsample_num

        for i in range(downsample_num_itr):
            block_depth = block_depth + unit_block_depth
            l = layers.conv(l, scope='disc_dn_' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[2, 2],
                            non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='disc_dn_norm_' + str(i))
            l = act_func(l)

        last_layer = l
        feature = l

        print(scope + ' Discriminator GP Dims: ' + str(feature.get_shape().as_list()))

        logit = layers.conv(last_layer, scope='conv_pred', filter_dims=[1, 1, 1], stride_dims=[1, 1],
                            non_linear_fn=tf.nn.sigmoid)
        print(scope + ' Discriminator Logit Dims: ' + str(logit.get_shape().as_list()))

    return feature, logit


def train(model_path='None'):
    print('Please wait. Preparing to start training...')

    train_start_time = time.time()

    # Mask Creation
    empty_img = np.ones((input_height, input_width, 3), dtype=np.uint8) * 1.0
    center_x = input_width // 2
    center_y = input_height // 2
    center = np.array((center_x, center_y))
    radius = 1.0 * (input_width // 2) - 10

    for i in range(input_width):
        for j in range(input_height):
            p = np.array((i, j))
            d = np.linalg.norm(p - center)
            if d > radius:
                empty_img[i, j] = empty_img[i, j] * 0
    effective_pixels = np.sum(empty_img)
    mask_img = empty_img

    learning_rate = 2e-4
    min_learning_rate = 1e-5
    sparsity = 2e-5

    X_IN = tf.placeholder(tf.float32, [None, input_height, input_width, num_channel])
    Y_IN = tf.placeholder(tf.float32, [None, input_height, input_width, num_channel])
    MASK = tf.placeholder(tf.float32, [None, input_height, input_width, num_channel])
    B_TRAIN = tf.placeholder(tf.bool)
    LR = tf.placeholder(tf.float32, None)

    # Generator
    query = encoder(X_IN, norm='layer', scope=G_Encoder_scope, b_train=B_TRAIN)
    if use_style is True:
        style = style_encoder(X_IN, norm='layer', scope=Style_Encoder_scope, b_train=B_TRAIN)
    else:
        style = None
    attention_g, latent_g = spatial_memory(query, size=aug_mem_size, dims=representation_dimension, scope=G_M_scope)
    G_X = decoder(latent_g, style, norm='layer', scope=G_Decoder_scope, b_train=B_TRAIN)

    unet_input = tf.concat([G_X, X_IN], axis=-1)
    z_gen, laterals = unet_encoder(unet_input, norm='layer', scope=G_UNet_Encoder_scope, b_train=B_TRAIN)
    U_G_X = unet_decoder(z_gen, laterals, norm='layer', scope=G_UNet_Decoder_scope, b_train=B_TRAIN)
    G_X = tf.math.multiply(G_X, MASK)
    U_G_X = tf.math.multiply(U_G_X, MASK)
    delta_X = tf.abs(U_G_X - X_IN)

    unet_residual_loss = get_residual_loss(Y_IN, U_G_X, type='mix_l1', alpha=0.8, effective_pixel=effective_pixels)
    pseudo_residual_loss = get_residual_loss(Y_IN, G_X, type='mix_l1', alpha=0.8, effective_pixel=effective_pixels)

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

    _, fake_logit = discriminator(U_G_X, x2=None, norm='layer', scope=D_scope, b_train=B_TRAIN)
    _, real_logit = discriminator(Y_IN, x2=None, norm='layer', scope=D_scope, b_train=B_TRAIN)
    d_real_loss = get_discriminator_loss(tf.ones_like(real_logit), real_logit, type='ls') + \
                  get_discriminator_loss(tf.zeros_like(fake_logit), fake_logit, type='ls')
    d_fake_loss = get_discriminator_loss(tf.ones_like(fake_logit), fake_logit, type='ls')
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
    #unet_generator_vars += ae_vars
    #unet_generator_vars += memory_vars
    # Optimizer
    unet_loss = unet_residual_loss
    mnet_loss = pseudo_residual_loss

    unet_generator_optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(unet_loss,
                                                                                 var_list=unet_generator_vars)
    pseudo_generator_optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(mnet_loss,
                                                                                   var_list=pseudo_generator_vars)
    disc_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(d_real_loss, var_list=disc_vars)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        try:
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            print('Model Restored')
        except:
            print('Start New Training. Wait ...')

        te_dir = test_data
        tr_dir = train_data

        mask_batch = []
        for i in range(batch_size):
            mask_batch.append(mask_img)
        mask_batch = np.array(mask_batch)

        cv2.imwrite('mask.jpg', mask_img * 255)

        for e in range(num_epoch):
            tr_files = []
            # Classes
            classes = os.listdir(tr_dir)

            print(' Num classes: ' + str(len(classes)))
            for cls in classes:
                class_path = os.path.join(tr_dir, cls).replace("\\", "/")
                samples = os.listdir(class_path)
                samples = np.random.choice(samples, size=20)  # (1000//len(classes)))
                for s in samples:
                    sample_path = os.path.join(class_path, s).replace("\\", "/")
                    tr_files.append(sample_path)
            print(' Num samples per epoch: ' + str(len(tr_files)))

            tr_files = shuffle(tr_files)
            total_input_size = len(tr_files)
            training_batch = zip(range(0, total_input_size, batch_size),
                                 range(batch_size, total_input_size + 1, batch_size))
            itr = 0

            # Perlin Noise
            #noise = util.generate_fractal_noise_2d((input_width, input_height), (16, 16), 5) # 512 x 512
            noise = util.generate_fractal_noise_2d((input_width, input_height), (64, 64), 5)  # 1024 x 1024
            noise = np.where(noise > np.average(noise), 1.0, 0.0)
            #noise2 = np.where(noise > np.average(noise), 0.0, 1.0) * np.random.normal(size=noise.shape)
            #noise = noise1 + noise2

            mask_noise = np.stack([noise, noise, noise], axis=-1)
            cv2.imwrite('mask_noise.jpg', 255 * mask_noise)

            for start, end in training_batch:
                itr = itr + 1
                outlier_file = np.random.choice(tr_files, size=1)
                batch_imgs, gt_imgs = load_images(tr_files[start:end], base_dir=None, mask=mask_img, cutout=True,
                                                  rotate=True, shift=True, add_eps=False, brightness=False, mask_noise=mask_noise, masking_img=None)

                lr = np.max([min_learning_rate, learning_rate * np.cos((np.pi * 7.0 / 16.0) * (e / num_epoch))])

                _, pseudo_g_loss, g_x_imgs = sess.run([pseudo_generator_optimizer, mnet_loss, G_X],
                                                      feed_dict={X_IN: batch_imgs, Y_IN: gt_imgs, LR: lr, B_TRAIN: True,
                                                                 MASK: mask_batch})
                _, unet_g_loss, u_g_x_imgs = sess.run([unet_generator_optimizer, unet_loss, U_G_X],
                                                      feed_dict={X_IN: batch_imgs, Y_IN: gt_imgs, LR: lr, B_TRAIN: True,
                                                                 MASK: mask_batch})
                _ = sess.run([disc_optimizer],
                             feed_dict={X_IN: batch_imgs, Y_IN: gt_imgs, LR: lr, B_TRAIN: True, MASK: mask_batch})

                print('epoch: ' + str(e) + ', unet_g_loss: ' + str(unet_g_loss) + ', pseudo_g_loss: ' + str(
                    pseudo_g_loss))

                if itr % 10 == 0:
                    #cv2.imwrite(out_dir + '/input_' + str(e) + '_' + str(itr) + '.jpg', batch_imgs[0] * 255.0)
                    cv2.imwrite(out_dir + '/' + str(e) + '_' + str(itr) + '.jpg', u_g_x_imgs[0] * 255.0)
                    # cv2.imwrite(out_dir + '/' + str(e) + '_u_' + str(itr) + '.jpg', u_g_x_imgs[0] * 255.0)
                    print('Elapsed Time at  ' + str(e) + '/' + str(num_epoch) + ' epochs, ' + str(
                        time.time() - train_start_time) + ' sec')
                    try:
                        print('Saving model...')
                        saver.save(sess, model_path)
                        print('Saved.')
                    except:
                        print('Save failed')
                    print('Training Time: ' + str(time.time() - train_start_time))

            te_files = os.listdir(te_dir)
            te_batch = zip(range(0, len(te_files), batch_size),
                           range(batch_size, len(te_files) + 1, batch_size))
            for t_s, t_e in te_batch:
                test_imgs, _ = load_images(te_files[t_s:t_e], base_dir=te_dir, mask=mask_img)
                gx_imgs, u_gx_imgs, delta_imgs = sess.run([G_X, U_G_X, delta_X],
                                                          feed_dict={X_IN: test_imgs, Y_IN: test_imgs, B_TRAIN: False,
                                                                     MASK: mask_batch})

                for i in range(batch_size):
                    cv2.imwrite('out/' + te_files[t_s + i], u_gx_imgs[i] * 255.0)
                    # cv2.imwrite('out/u_' + te_files[t_s + i], u_gx_imgs[i] * 255.0)
                    cv2.imwrite('out/delta_img_' + te_files[t_s + i], delta_imgs[i] * 255.0)

                # for i in range(len(gx_imgs)):
                #    cv2.imwrite('out/' + te_files[t_s + i], gx_imgs[i] * 255.0)
                #    cv2.imwrite('out/delta_img_' + te_files[t_s + i], delta_imgs[i] * 255.0)


def calculate_anomaly_scores(imgs1, imgs2, pixel_max=1.0):
    _, h, w, c = imgs1.get_shape().as_list()

    X1 = imgs1
    X2 = imgs2

    anomaly_score = tf.reduce_mean(tf.abs(tf.subtract(X1, X2)), axis=[1, 2, 3])

    '''
    if c > 1:
        X1 = tf.image.rgb_to_grayscale(X1)
        X2 = tf.image.rgb_to_grayscale(X2)
    if h < 255 or w < 255:
        X1 = tf.image.resize_images(X1, (255, 255))
        X2 = tf.image.resize_images(X2, (255, 255))
    anomaly_score = 1 - tf.image.ssim_multiscale(X1, X2, pixel_max)
    '''
    return anomaly_score


def test(model_path):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, help='train/test', default='train')
    parser.add_argument('--model_path', type=str, help='model check point file path', default='model/m.ckpt')
    parser.add_argument('--train_data', type=str, help='training data directory', default='data/train')
    parser.add_argument('--test_data', type=str, help='test data directory', default='data/test')
    parser.add_argument('--out_dir', type=str, help='output directory', default='imgs')
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

    unit_block_depth = 24
    bottleneck_num = 5
    query_dimension = 256
    style_dimension = 256

    representation_dimension = query_dimension
    enc_conv_num_groups = 1
    dec_conv_num_groups = 1

    upsample_ratio = 16
    downsample_num = int(np.log2(upsample_ratio))
    upsample_num = downsample_num
    aug_mem_size = 2000
    num_channel = 3
    use_style = False
    use_categorical_constraints = True

    if mode == 'train':
        train(model_path)
    elif mode == 'test':
        test(model_path)
    else:
        print('Train or Test?')
