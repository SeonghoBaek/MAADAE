# MAADAE[meidei]: Memory Augmented Adversarial Dual AUto Encoder
# Author: Seongho Baek
# e-mail: seonghobaek@gmail.com

import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.utils import shuffle
import util
import layers
import argparse
import time

# scope
G_Encoder_scope = 'generator_encoder'
G_Decoder_scope = 'generator_decoder'
G_Disc_scope = 'g_discriminator'
D_Encoder_scope = 'discriminator_encoder'
D_Decoder_scope = 'discriminator_decoder'
G_M_scope = 'generator_mem'
D_M_scope = 'discriminator_mem'
GS_M_scope = 'generator_style_mem'
DS_M_scope = 'discriminator_style_mem'


def load_images(file_name_list, base_dir, cutout=False, add_eps=False, rotate=-1):
    try:
        images = []

        for file_name in file_name_list:
            fullname = os.path.join(base_dir, file_name).replace("\\", "/")
            img = cv2.imread(fullname)

            if img is None:
                print('Load failed: ' + fullname)
                return None

            img = cv2.resize(img, dsize=(input_width, input_height), interpolation=cv2.INTER_AREA)

            if img is not None:
                img = np.array(img)
                img = img * 1.0
                n_img = img / 255.0

                if rotate > -1:
                    n_img = cv2.rotate(n_img, rotate)

                if add_eps is True:
                    if np.random.randint(low=0, high=10) < 5:
                        n_img = n_img + np.random.uniform(low=0, high=1/256, size=n_img.shape)

                if cutout is True:
                    num_cutout = input_width // 10

                    # square cut out
                    co_w = 2
                    co_h = co_w
                    padd_w = co_w // 2
                    padd_h = padd_w

                    for _ in range(num_cutout):
                        r_x = np.random.randint(low=padd_w, high=input_width - padd_w)
                        r_y = np.random.randint(low=padd_h, high=input_height - padd_h)

                        for i in range(co_w):
                            for j in range(co_h):
                                n_img[r_x - padd_w + i][r_y - padd_h + j] *= 0

                images.append(n_img)
    except cv2.error as e:
        print(e)
        return None

    return np.array(images)


def get_residual_loss(value, target, type='l1', gamma=1.0):
    if type == 'rmse':
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(target, value))))
    elif type == 'ce':
        eps = 1e-10
        loss = tf.reduce_mean(-1 * target * tf.log(value + eps) - 1 * (1 - target) * tf.log(1 - value + eps))
        #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=value, labels=target))
    elif type == 'l1':
        loss = tf.reduce_mean(tf.abs(tf.subtract(target, value)))
    elif type == 'l2':
        loss = tf.reduce_mean(tf.square(tf.subtract(target, value)))
    elif type == 'entropy':
        eps = 1e-10
        loss = tf.reduce_mean(-1 * value * tf.log(value + eps))

    return loss * gamma


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


def encoder(in_tensor, activation=tf.nn.relu, norm='batch', b_use_style=False, scope='encoder', b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        print(scope + ' encoder input: ' + str(in_tensor.get_shape().as_list()))

        block_depth = unit_block_depth

        if b_use_style is True:
            s = layers.conv(in_tensor, scope='style_init', filter_dims=[7, 7, block_depth],
                            stride_dims=[1, 1], non_linear_fn=activation, padding='REFL', pad=3)

            for i in range(downsample_num):
                s = layers.blur_pooling2d(s, kernel_size=5, scope='style_blur_' + str(i), padding='REFL')
                print(scope + 'Style Blur Pooling Block ' + str(i) + ': ' + str(s.get_shape().as_list()))

                block_depth = block_depth + unit_block_depth
                s = layers.conv(s, scope='style_downsapmple_' + str(i), filter_dims=[1, 1, block_depth],
                                stride_dims=[1, 1], non_linear_fn=None, bias=False)
                s = layers.conv_normalize(s, norm='batch', b_train=b_train, scope='style_norm_' + str(i))
                s = layers.se_block(s, scope='style_se_block_' + str(i))
                s = activation(s)

                print(scope + ' Style Downsample Block ' + str(i) + ': ' + str(s.get_shape().as_list()))

            style_query = layers.global_avg_pool(s, style_dimension, scope='sgap')
        else:
            style_query = None

        l = layers.conv(in_tensor, scope='contents_init', filter_dims=[3, 3, block_depth],
                        stride_dims=[1, 1], non_linear_fn=activation, padding='REFL', pad=1)

        # Downsample stage.
        for i in range(downsample_num):
            l = layers.blur_pooling2d(l, kernel_size=5, scope='blur_' + str(i), padding='REFL')
            print(scope + ' Blur Pooling Block ' + str(i) + ': ' + str(l.get_shape().as_list()))

            block_depth = block_depth + unit_block_depth
            l = layers.conv(l, scope='downsapmple_' + str(i), filter_dims=[1, 1, block_depth],
                            stride_dims=[1, 1], non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm_' + str(i))
            l = layers.se_block(l, scope='content_se_block_' + str(i))
            l = activation(l)

            print(scope + ' Downsample Block ' + str(i) + ': ' + str(l.get_shape().as_list()))

        # Bottleneck stage
        for i in range(bottleneck_num):
            print(scope + ' Bottleneck Block : ' + str(l.get_shape().as_list()))
            l = layers.add_se_residual_block(l, filter_dims=[3, 3, block_depth], act_func=activation,
                                             norm=norm, b_train=b_train, use_dilation=False, scope='bt_block_' + str(i), padding='REFL', pad=1)

        l = layers.global_avg_pool(l, query_dimension, scope='gap')
        print(scope + ' z dimension : ' + str(l.get_shape().as_list()))
        content_query = l

    return content_query, style_query


def decoder(content, style, activation=tf.nn.relu, norm='batch', scope='decoder', b_use_style=False, b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        print(scope + ' decoder input: ' + str(content.get_shape().as_list()))
        l = content
        block_depth = unit_block_depth + (unit_block_depth * upsample_num)

        # Update by image size for your own.
        l = layers.fc(l, decoder_int_filter_size * decoder_int_filter_size, non_linear_fn=activation, scope='fc1', use_bias=True)
        l = tf.reshape(l, shape=[-1, decoder_int_filter_size, decoder_int_filter_size, 1])
        l = layers.conv(l, scope='init', filter_dims=[3, 3, block_depth],
                        stride_dims=[1, 1], non_linear_fn=activation, padding='REFL', pad=1)

        # Bottleneck stage
        if b_use_style is True:
            s = layers.fc(style, block_depth, non_linear_fn=activation, scope='style_linear1', use_bias=True)
            style_mu = layers.fc(s, block_depth, non_linear_fn=None, scope='style_mu')
            style_var = layers.fc(s, block_depth, non_linear_fn=None, scope='style_var')
            style_mu = tf.reshape(style_mu, shape=[-1, 1, 1, block_depth])
            style_var = tf.reshape(style_var, shape=[-1, 1, 1, block_depth])

        for i in range(bottleneck_num):
            if b_use_style is True:
                l = layers.add_se_adain_residual_block(l, style_mu, style_var, filter_dims=[3, 3, block_depth], act_func=activation,
                                                       use_dilation=False, scope=scope + '_bt_block_' + str(i), padding='REFL', pad=1)
            else:
                l = layers.add_se_residual_block(l, filter_dims=[3, 3, block_depth], act_func=activation,
                                                 norm=norm, b_train=b_train, use_dilation=False, scope='bt_block_' + str(i), padding='REFL', pad=1)

            print(scope + ' Bottleneck Block : ' + str(l.get_shape().as_list()))

        # Upsample stage
        for i in range(upsample_num):
            # ESPCN upsample
            block_depth = block_depth - unit_block_depth
            l = layers.conv(l, scope='espcn_' + str(i), filter_dims=[3, 3, block_depth * 2 * 2],
                            stride_dims=[1, 1], non_linear_fn=None, padding='REFL', pad=1)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='espcn_norm_' + str(i))
            l = layers.se_block(l, scope='se_block_' + str(i))
            l = activation(l)
            l = tf.nn.depth_to_space(l, 2)

            print(scope + ' Upsampling ' + str(i) + ': ' + str(l.get_shape().as_list()))

        l = layers.conv(l, scope='last', filter_dims=[3, 3, num_channel], stride_dims=[1, 1],
                        non_linear_fn=None, padding='REFL', pad=1)

        print(scope + ' Output: ' + str(l.get_shape().as_list()))
    return l


def memory(query, scope='aug_mem'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        aug_mem = tf.get_variable('mem_vars', [aug_mem_size, representation_dimension], dtype=tf.float32,
                                  initializer=tf.random_normal_initializer)
        print(scope + ' mem size: ' + str(aug_mem.get_shape().as_list()))
        # query = [B, R], aug_mem = [Q, R]
        norm_q = tf.nn.l2_normalize(query, axis=0)
        print(scope + ' norm_q: ' + str(norm_q.get_shape().as_list()))
        norm_mem = tf.nn.l2_normalize(aug_mem, axis=0)
        print(scope + ' norm mem: ' + str(norm_mem.get_shape().as_list()))
        distance = tf.matmul(norm_q, norm_mem, transpose_b=True)  # [B, Q]
        sm = tf.nn.softmax(distance, axis=-1)  # [B, Q]
        print(scope + ' softmax: ' + str(sm.get_shape().as_list()))
        threashold = tf.constant(1/aug_mem_size, dtype=tf.float32)
        attention = tf.multiply(tf.nn.relu(sm - threashold), sm)
        attention = attention / (1e-4 + tf.abs(sm - threashold))
        print(scope + ' attention: ' + str(attention.get_shape().as_list()))
        l1 = tf.expand_dims(tf.reduce_sum(attention, axis=-1), axis=-1)
        print(scope + ' l1: ' + str(l1.get_shape().as_list()))
        attention = attention / l1

        latent = tf.matmul(attention, aug_mem)  # [B, Q] x [Q, R] = [B, R]
        print(scope + ' attention: ' + str(attention.get_shape().as_list()))
        latent = tf.add(query, latent)

        return attention, latent


def train(model_path='None'):
    print('Please wait. Preparing to start training...')

    train_start_time = time.time()

    X_IN = tf.placeholder(tf.float32, [None, input_height, input_width, num_channel])
    B_TRAIN = tf.placeholder(tf.bool)
    LR = tf.placeholder(tf.float32, None)

    # Generator
    z_gen, s_gen = encoder(X_IN, norm='instance', b_use_style=use_style, scope=G_Encoder_scope, b_train=B_TRAIN)
    attention_g, latent_g = memory(z_gen, scope=G_M_scope)

    if use_style_mem is True:
        st_attention_g, st_latent_g = memory(s_gen, scope=GS_M_scope)
        G_X = decoder(latent_g, st_latent_g, norm='instance', scope=G_Decoder_scope, b_use_style=True, b_train=B_TRAIN)
    else:
        G_X = decoder(latent_g, s_gen, norm='instance', scope=G_Decoder_scope, b_use_style=use_style, b_train=B_TRAIN)

    # Discriminator
    z_disc_gx, s_disc_gx = encoder(G_X, norm='instance', b_use_style=use_style, scope=D_Encoder_scope, b_train=B_TRAIN)
    attention_d_gx, latent_d_gx = memory(z_disc_gx, scope=D_M_scope)

    if use_style_mem is True:
        st_attention_d_gx, st_latent_d_gx = memory(s_disc_gx, scope=DS_M_scope)
        D_GX = decoder(latent_d_gx, st_latent_d_gx, norm='instance', scope=D_Decoder_scope, b_use_style=True, b_train=B_TRAIN)
    else:
        D_GX = decoder(latent_d_gx, s_disc_gx, norm='instance', scope=D_Decoder_scope, b_use_style=use_style, b_train=B_TRAIN)

    z_disc_x, s_disc_x = encoder(X_IN, norm='instance', b_use_style=use_style, scope=D_Encoder_scope, b_train=B_TRAIN)
    attention_d_x, latent_d_x = memory(z_disc_x, scope=D_M_scope)

    if use_style_mem is True:
        st_attention_d_x, st_latent_d_x = memory(s_disc_x, scope=DS_M_scope)
        D_X = decoder(latent_d_x, st_latent_d_x, norm='instance', scope=D_Decoder_scope, b_use_style=True, b_train=B_TRAIN)
    else:
        D_X = decoder(latent_d_x, s_disc_x, norm='instance', scope=D_Decoder_scope, b_use_style=use_style, b_train=B_TRAIN)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    learning_rate = 2e-4

    # Loss Define
    sparsity = 1e-5

    d_loss_x_dx = get_residual_loss(X_IN, D_X, type='l1', gamma=alpha)
    d_loss_gx_dgx = get_residual_loss(G_X, D_GX, type='l1')
    g_loss_x_gx = get_residual_loss(X_IN, G_X, type='l1', gamma=alpha)
    g_loss_gx_dgx = get_residual_loss(G_X, D_GX, type='l1')

    # Simple Balance Mode
    sigmoid_d_loss_gx_dgx = 1 - tf.nn.sigmoid(d_loss_gx_dgx)
    sigmoid_g_loss_gx_dgx = tf.nn.sigmoid(d_loss_gx_dgx)

    disc_loss = (d_loss_x_dx - sigmoid_d_loss_gx_dgx * d_loss_gx_dgx) + sparsity * get_residual_loss(attention_d_x, None, type='entropy')
    gen_loss = (g_loss_x_gx + sigmoid_g_loss_gx_dgx * g_loss_gx_dgx) + sparsity * get_residual_loss(attention_g, None, type='entropy')

    #slope = 2.0
    #d_delta = 1.0 / (1.0 + 1.0 / tf.exp(slope * (d_loss_x_dx - d_loss_gx_dgx)))
    #disc_loss = d_loss_x_dx - d_delta * d_loss_gx_dgx + sparsity * get_residual_loss(attention_d_x, None, type='entropy')

    if use_style_mem is True:
        disc_loss = disc_loss + sparsity * get_residual_loss(st_attention_d_x, None, type='entropy')

    #gen_loss = g_loss_x_gx + (1 - d_delta) * g_loss_gx_dgx + sparsity * get_residual_loss(attention_g, None, type='entropy')

    if use_style_mem is True:
        gen_loss = gen_loss + sparsity * get_residual_loss(st_attention_g, None, type='entropy')

    # Variable Define
    generator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=G_Encoder_scope) + \
                     tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=G_Decoder_scope) + \
                     tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=G_M_scope) + \
                     tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=GS_M_scope)

    discriminator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=D_Encoder_scope) + \
                         tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=D_Decoder_scope) + \
                         tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=D_M_scope) + \
                         tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=DS_M_scope)

    # Optimizer
    discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(disc_loss, var_list=discriminator_vars)
    generator_optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(gen_loss, var_list=generator_vars)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        try:
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            print('Model Restored')
        except:
            print('Start New Training. Wait ...')

        tr_dir = train_data
        tr_files = os.listdir(tr_dir)

        total_input_size = len(tr_files)
        total_steps = (total_input_size * num_epoch * 1.0)
        cur_step = 0
        warmup_epoch = 10 # num_epoch // 10

        for e in range(num_epoch):
            tr_files = shuffle(tr_files)
            training_batch = zip(range(0, total_input_size, batch_size),
                                 range(batch_size, total_input_size + 1, batch_size))
            itr = 0

            for start, end in training_batch:
                itr = itr + 1
                cur_step = cur_step + 1
                batch_imgs = load_images(tr_files[start:end], base_dir=tr_dir)

                if e < warmup_epoch:
                    lr = learning_rate * cur_step / (total_input_size * warmup_epoch)
                else:
                    if e == warmup_epoch:
                        cur_step = 1
                        total_steps = (total_input_size * (num_epoch - warmup_epoch) * 1.0)

                    lr = learning_rate * np.cos((np.pi * 7.0 / 16.0) * (cur_step / total_steps))

                _, d_loss, d_x_imgs, d_x_dx, d_gx_dgx = sess.run([discriminator_optimizer, disc_loss, D_X, d_loss_x_dx, d_loss_gx_dgx], feed_dict={X_IN: batch_imgs, LR: lr, B_TRAIN: True})
                _, g_loss, g_x_imgs, g_x_gx = sess.run([generator_optimizer, gen_loss, G_X, g_loss_x_gx], feed_dict={X_IN: batch_imgs, LR: lr, B_TRAIN: True})

                print('epoch: ' + str(e) + ', ' +
                      'd_loss: ' + str(d_loss) + ', d_x_dx: ' + str(d_x_dx) + ', d_gx_dgx: ' + str(d_gx_dgx) +
                      ', ' + 'g_loss: ' + str(g_loss) +
                      ', g_x_gx: ' + str(g_x_gx))

                if itr % 10 == 0:
                    d_images = d_x_imgs[0] * 255.0
                    g_images = g_x_imgs[0] * 255.0

                    cv2.imwrite(out_dir + '/d_' + tr_files[start], d_images)
                    cv2.imwrite(out_dir + '/g_' + tr_files[start], g_images)
                    print('Elapsed Time at  ' + str(cur_step) + '/' + str(total_steps) + ' steps, ' + str(time.time() - train_start_time) + ' sec')
            try:
                print('Saving model...')
                saver.save(sess, model_path)
                print('Saved.')
            except:
                print('Save failed')
            print('Training Time: ' + str(time.time() - train_start_time))


def calculate_anomaly_scores(imgs1, imgs2, pixel_max=1.0):
    _, h, w, c = imgs1.get_shape().as_list()

    X1 = imgs1
    X2 = imgs2

    #anomaly_score = tf.reduce_mean(tf.square(tf.subtract(X1, X2)), axis=[1, 2, 3])

    if c > 1:
        X1 = tf.image.rgb_to_grayscale(X1)
        X2 = tf.image.rgb_to_grayscale(X2)
    if h < 255 or w < 255:
        X1 = tf.image.resize_images(X1, (255, 255))
        X2 = tf.image.resize_images(X2, (255, 255))
    anomaly_score = 1 - tf.image.ssim_multiscale(X1, X2, pixel_max)

    return anomaly_score


def test(model_path):
    print('Model Loading...')

    X_IN = tf.placeholder(tf.float32, [None, input_height, input_width, num_channel])
    B_TRAIN = tf.placeholder(tf.bool)

    # Generator
    z_gen, s_gen = encoder(X_IN, norm='instance', b_use_style=use_style, scope=G_Encoder_scope, b_train=B_TRAIN)
    attention_g, latent_g = memory(z_gen, scope=G_M_scope)

    if use_style_mem is True:
        st_attention_g, st_latent_g = memory(s_gen, scope=GS_M_scope)
        G_X = decoder(latent_g, st_latent_g, norm='instance', scope=G_Decoder_scope, b_use_style=True, b_train=B_TRAIN)
    else:
        G_X = decoder(latent_g, s_gen, norm='instance', scope=G_Decoder_scope, b_use_style=use_style, b_train=B_TRAIN)

    # Discriminator
    z_disc_gx, s_disc_gx = encoder(G_X, norm='instance', b_use_style=use_style, scope=D_Encoder_scope, b_train=B_TRAIN)
    attention_d_gx, latent_d_gx = memory(z_disc_gx, scope=D_M_scope)

    if use_style_mem is True:
        st_attention_d_gx, st_latent_d_gx = memory(s_disc_gx, scope=DS_M_scope)
        D_GX = decoder(latent_d_gx, st_latent_d_gx, norm='instance', scope=D_Decoder_scope, b_use_style=True,
                       b_train=B_TRAIN)
    else:
        D_GX = decoder(latent_d_gx, s_disc_gx, norm='instance', scope=D_Decoder_scope, b_use_style=use_style,
                       b_train=B_TRAIN)

    z_disc_x, s_disc_x = encoder(X_IN, norm='instance', b_use_style=use_style, scope=D_Encoder_scope, b_train=B_TRAIN)
    attention_d_x, latent_d_x = memory(z_disc_x, scope=D_M_scope)

    if use_style_mem is True:
        st_attention_d_x, st_latent_d_x = memory(s_disc_x, scope=DS_M_scope)
        D_X = decoder(latent_d_x, st_latent_d_x, norm='instance', scope=D_Decoder_scope, b_use_style=True,
                      b_train=B_TRAIN)
    else:
        D_X = decoder(latent_d_x, s_disc_x, norm='instance', scope=D_Decoder_scope, b_use_style=use_style, b_train=B_TRAIN)

    anomaly_score = calculate_anomaly_scores(X_IN, D_GX)
    #anomaly_score = tf.maximum(calculate_anomaly_scores(X_IN, D_GX), calculate_anomaly_scores(X_IN, D_X))

    # tf.image.rgb_to_grayscale
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        try:
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            print('Model Loaded')
        except:
            print('Loading Failed')
            return

        te_dir = test_data
        te_files = os.listdir(te_dir)

        total_input_size = len(te_files)
        test_batch = zip(range(0, total_input_size, batch_size),
                         range(batch_size, total_input_size + 1, batch_size))

        for start, end in test_batch:
            test_imgs = load_images(te_files[start:end], base_dir=te_dir)
            scores, dgx_imgs, gx_imgs, dx_imgs = sess.run([anomaly_score, D_GX, G_X, D_X],
                                                   feed_dict={X_IN: test_imgs, B_TRAIN: False})
            for i in range(batch_size):
                print('Anomaly Score ' + te_files[start + i] + ': ' + str(scores[i]))
                cv2.imwrite(out_dir + '/dgx_' + te_files[start + i], dgx_imgs[i] * 255.0)
                cv2.imwrite(out_dir + '/gx_' + te_files[start + i], gx_imgs[i] * 255.0)
                cv2.imwrite(out_dir + '/dx_' + te_files[start + i], dx_imgs[i] * 255.0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, help='train/test', default='train')
    parser.add_argument('--model_path', type=str, help='model check point file path', default='model/m.ckpt')
    parser.add_argument('--train_data', type=str, help='training data directory', default='data/train')
    parser.add_argument('--test_data', type=str, help='test data directory', default='data/test')
    parser.add_argument('--out_dir', type=str, help='output directory', default='imgs')
    parser.add_argument('--img_size', type=int, help='training image size', default=512)
    parser.add_argument('--epoch', type=int, help='num epoch', default=300)
    parser.add_argument('--batch_size', type=int, help='Training batch size', default=16)
    parser.add_argument('--alpha', type=int, help='AE loss weight', default=10)

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

    unit_block_depth = 32
    decoder_int_filter_size = 16
    downsample_num = int(np.log2(input_width // decoder_int_filter_size))
    upsample_num = downsample_num
    bottleneck_num = 4
    query_dimension = 128
    style_dimension = 128
    representation_dimension = query_dimension
    aug_mem_size = 500
    num_channel = 3
    use_style = True
    use_style_mem = True

    if mode == 'train':
        train(model_path)
    elif mode == 'test':
        test(model_path)
    else:
        print('Train or Test?')
