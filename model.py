# MAADAE[meidei]: Memory Augmented Adversarial Dual AUto Encoder
# Author: Seongho Baek
# e-mail: seonghobaek@gmail.com


import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.utils import shuffle
import layers
import argparse
import time

# scope
G_Encoder_scope = 'generator_encoder'
G_Decoder_scope = 'generator_decoder'
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


def get_residual_loss(value, target, type='l1', alpha=1.0):
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

    return loss * alpha


def get_random_box_residual_loss(img1, img2, num_box=5, num_batches=1, box_size=(12, 12), alpha=1.0):
    boxes = tf.random.uniform(shape=(num_box, 4))
    box_indices = tf.random.uniform(shape=(num_box,), minval=0, maxval=num_batches, dtype=tf.int32)

    out1 = tf.image.crop_and_resize(img1, boxes, box_indices, box_size)
    out2 = tf.image.crop_and_resize(img2, boxes, box_indices, box_size)

    return alpha * tf.reduce_mean(tf.abs(out1 - out2))


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


def encoder(in_tensor, activation=tf.nn.relu, norm='batch', b_use_style=False, scope='encoder', b_train=False, b_disc=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        print(scope + ' encoder input: ' + str(in_tensor.get_shape().as_list()))

        # Style Encoder
        block_depth = unit_block_depth

        if b_disc is True:
            block_depth = block_depth // 2

        if b_use_style is True:
            s = layers.conv(in_tensor, scope='style_init', filter_dims=[7, 7, block_depth],
                            stride_dims=[1, 1], non_linear_fn=activation)

            for i in range(downsample_num):
                block_depth = block_depth * 2

                if use_blurpooling2d is True:
                    s = layers.blur_pooling2d(s, kernel_size=5, scope='style_blur_' + str(i))
                    s = layers.conv(s, scope='style_downsapmple_' + str(i), filter_dims=[3, 3, block_depth],
                                    stride_dims=[1, 1], non_linear_fn=activation)
                else:
                    s = layers.conv(s, scope='style_downsapmple_' + str(i), filter_dims=[3, 3, block_depth],
                                    stride_dims=[2, 2], non_linear_fn=activation)

                print(scope + ' Style Downsample Block ' + str(i) + ': ' + str(s.get_shape().as_list()))

            style_query = layers.global_avg_pool(s, style_dimension, scope='sgap')
            print(scope + ' style latent dimension : ' + str(style_query.get_shape().as_list()))
        else:
            style_query = None

        # Contents Encoder
        block_depth = unit_block_depth
        if b_disc is True:
            block_depth = block_depth // 2

        l = layers.conv(in_tensor, scope='contents_init', filter_dims=[7, 7, block_depth], stride_dims=[1, 1], non_linear_fn=activation)

        # Downsample stage.
        for i in range(downsample_num):
            block_depth = block_depth * 2
            if use_blurpooling2d is True:
                l = layers.blur_pooling2d(l, kernel_size=5, scope='blur_' + str(i))
                l = layers.conv(l, scope='downsapmple_' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[1, 1], non_linear_fn=activation)
            else:
                l = layers.conv(l, scope='downsapmple_' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[2, 2], non_linear_fn=activation)

            print(scope + ' Downsample Block ' + str(i) + ': ' + str(l.get_shape().as_list()))

        # Bottleneck stage
        for i in range(bottleneck_num):
            print(scope + ' Bottleneck Block : ' + str(l.get_shape().as_list()))
            l = layers.add_se_residual_block(l, filter_dims=[3, 3, block_depth], act_func=activation, num_groups=enc_conv_num_groups,
                                             norm=norm, b_train=b_train, use_dilation=False, scope='bt_block_' + str(i), padding='REFL', pad=1)

        content_query = layers.global_avg_pool(l, query_dimension, scope='gap')
        print(scope + ' content latent dimension : ' + str(content_query.get_shape().as_list()))

    return content_query, style_query


def decoder(content, style, activation=tf.nn.relu, norm='instance', scope='decoder', b_use_style=False, b_train=False, b_disc=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        print(scope + ' decoder input: ' + str(content.get_shape().as_list()))
        l = content

        block_depth = unit_block_depth * (2 ** upsample_num)
        decoder_bottleneck_num = bottleneck_num

        if b_disc is True:
            block_depth = block_depth // 2

        print('mem vector shape: ' + str(l.get_shape().as_list()))

        l = tf.reshape(l, shape=[-1, bottleneck_feature_size, bottleneck_feature_size, 1])
        l = layers.conv(l, scope='init', filter_dims=[3, 3, block_depth], stride_dims=[1, 1], non_linear_fn=activation, padding='REFL', pad=1)

        # Style Parmaters
        if b_use_style is True:
            s = layers.fc(style, block_depth, non_linear_fn=activation, scope='style_linear1', use_bias=True)
            s = layers.fc(s, block_depth, non_linear_fn=activation, scope='style_linear2', use_bias=True)
            style_alpha = layers.fc(s, block_depth, non_linear_fn=None, scope='style_mu')
            style_beta = layers.fc(s, block_depth, non_linear_fn=None, scope='style_var')
            style_alpha = tf.reshape(style_alpha, shape=[-1, 1, 1, block_depth])
            style_beta = tf.reshape(style_beta, shape=[-1, 1, 1, block_depth])

        for i in range(decoder_bottleneck_num):
            if b_use_style is True:
                l = layers.add_se_adain_residual_block(l, style_alpha, style_beta, filter_dims=[3, 3, block_depth], act_func=activation, use_dilation=False,
                                                       scope=scope + '_bt_block_' + str(i), padding='REFL', pad=1, num_groups=dec_conv_num_groups)
            else:
                l = layers.add_se_residual_block(l, filter_dims=[3, 3, block_depth], act_func=activation, norm=norm, b_train=b_train, use_dilation=False,
                                                 scope='bt_block_' + str(i), padding='REFL', pad=1, num_groups=dec_conv_num_groups)

            print(scope + ' Bottleneck Block : ' + str(l.get_shape().as_list()))

        l = layers.conv(l, scope='espcn', filter_dims=[3, 3, 3 * upsample_ratio * upsample_ratio], stride_dims=[1, 1], non_linear_fn=None)
        l = tf.nn.depth_to_space(l, upsample_ratio)
        l = tf.nn.sigmoid(l)

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

        # Simple Adaptive Scaling
        scale_alpha = layers.fc(query, aug_mem_size//16, non_linear_fn=tf.nn.relu, scope='mem_linear_alpha1', use_bias=True)
        scale_alpha = layers.fc(scale_alpha, aug_mem_size, non_linear_fn=tf.nn.sigmoid, scope='mem_linear_alpha2', use_bias=True)
        scale_beta = layers.fc(query, aug_mem_size//16, non_linear_fn=tf.nn.relu, scope='mem_linear_beta1', use_bias=True)
        scale_beta = layers.fc(scale_beta, aug_mem_size, non_linear_fn=tf.nn.sigmoid, scope='mem_linear_beta2', use_bias=True)

        distance = scale_alpha * distance + scale_beta  # [B, Q]

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

    learning_rate = 1e-3
    min_learning_rate = 1e-5
    sparsity = 2e-5

    X_IN = tf.placeholder(tf.float32, [None, input_height, input_width, num_channel])
    B_TRAIN = tf.placeholder(tf.bool)
    LR = tf.placeholder(tf.float32, None)

    # Generator
    z_gen, s_gen = encoder(X_IN, norm='instance', b_use_style=use_style, scope=G_Encoder_scope, b_train=B_TRAIN)
    attention_g, latent_g = memory(z_gen, scope=G_M_scope)
    st_attention_g, st_latent_g = memory(s_gen, scope=GS_M_scope)
    G_X = decoder(latent_g, st_latent_g, scope=G_Decoder_scope, b_use_style=True, b_train=B_TRAIN)

    if use_generator_only is True:
        gen_loss = get_residual_loss(X_IN, G_X, type='l1', alpha=alpha)
        sparsity_reg_loss = sparsity * (get_residual_loss(attention_g, None, type='entropy') + get_residual_loss(st_attention_g, None, type='entropy'))

        generator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=G_Encoder_scope) + \
                         tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=G_Decoder_scope) + \
                         tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=G_M_scope) + \
                         tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=GS_M_scope)

        memory_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=G_M_scope) + \
                      tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=GS_M_scope)

        sparsity_reg_vars = [v for v in memory_vars if 'mem_vars' in v.name]

        # Optimizer
        generator_optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(gen_loss, var_list=generator_vars)
        sparsity_reg_optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(sparsity_reg_loss, var_list=sparsity_reg_vars)
    else:
        # Discriminator
        z_disc_gx, s_disc_gx = encoder(G_X, norm='instance', b_use_style=use_style, scope=D_Encoder_scope, b_train=B_TRAIN, b_disc=True)
        D_GX = decoder(z_disc_gx, s_disc_gx, scope=D_Decoder_scope, b_use_style=True, b_train=B_TRAIN, b_disc=True)
        z_disc_x, s_disc_x = encoder(X_IN, norm='instance', b_use_style=use_style, scope=D_Encoder_scope, b_train=B_TRAIN, b_disc=True)
        D_X = decoder(z_disc_x, s_disc_x, scope=D_Decoder_scope, b_use_style=True, b_train=B_TRAIN, b_disc=True)

        d_loss_x_dx = get_residual_loss(X_IN, D_X, type='l1')
        d_loss_gx_dgx = get_residual_loss(G_X, D_GX, type='l1')
        g_loss_x_gx = get_residual_loss(X_IN, G_X, type='l1')
        g_loss_gx_dgx = get_residual_loss(G_X, D_GX, type='l1')

        # Simple Balance Mode
        sigmoid_d_loss_gx_dgx = layers.sigmoid(d_loss_x_dx - 0.5 * d_loss_gx_dgx, slope=2.0)
        sigmoid_g_loss_gx_dgx = layers.sigmoid(g_loss_gx_dgx - g_loss_x_gx)

        disc_loss = alpha * d_loss_x_dx - sigmoid_d_loss_gx_dgx * d_loss_gx_dgx
        gen_loss = alpha * g_loss_x_gx + sigmoid_g_loss_gx_dgx * g_loss_gx_dgx

        lr_weight = layers.sigmoid(d_loss_x_dx - d_loss_gx_dgx, shift=0.05)
        lr_d = LR * lr_weight
        lr_g = LR * (tf.constant(1.0) - lr_weight)

        sparsity_reg_loss = sparsity * get_residual_loss(attention_g, None, type='entropy')
        sparsity_reg_loss = sparsity_reg_loss + sparsity * get_residual_loss(st_attention_g, None, type='entropy')

        # Variable Define
        generator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=G_Encoder_scope) + \
                         tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=G_Decoder_scope) + \
                         tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=G_M_scope) + \
                         tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=GS_M_scope)

        discriminator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=D_Encoder_scope) + \
                             tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=D_Decoder_scope)

        memory_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=G_M_scope) + \
                      tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=GS_M_scope)

        sparsity_reg_vars = [v for v in memory_vars if 'mem_vars' in v.name]

        # Optimizer
        discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=lr_d).minimize(disc_loss, var_list=discriminator_vars)
        generator_optimizer = tf.train.AdamOptimizer(learning_rate=lr_g).minimize(gen_loss, var_list=generator_vars)
        sparsity_reg_optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(sparsity_reg_loss, var_list=sparsity_reg_vars)

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

        tr_dir = train_data
        tr_files = os.listdir(tr_dir)

        te_dir = test_data
        te_files = os.listdir(te_dir)

        total_input_size = len(tr_files)
        total_steps = (total_input_size * num_epoch * 1.0)
        cur_step = 0
        warmup_epoch = 0

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
                    update_steps = 10 * (cur_step // 10) # per 10 steps
                    lr = np.max([min_learning_rate, learning_rate * np.cos((np.pi * 7.0 / 16.0) * (update_steps / total_steps))])

                if use_generator_only is True:
                    _, g_loss, g_x_imgs = sess.run([generator_optimizer, gen_loss, G_X], feed_dict={X_IN: batch_imgs, LR: lr, B_TRAIN: True})
                    _ = sess.run([sparsity_reg_optimizer], feed_dict={X_IN: batch_imgs, LR: lr, B_TRAIN: True})

                    print('epoch: ' + str(e) + ', g_loss: ' + str(g_loss))

                    if itr % 10 == 0:
                        g_images = g_x_imgs[0] * 255.0
                        cv2.imwrite(out_dir + '/g_' + tr_files[start], g_images)
                        print('Elapsed Time at  ' + str(cur_step) + '/' + str(total_steps) + ' steps, ' + str(time.time() - train_start_time) + ' sec')
                else:
                    _, d_loss, d_x_imgs, d_x_dx, d_gx_dgx = sess.run([discriminator_optimizer, disc_loss, D_X, d_loss_x_dx, d_loss_gx_dgx], feed_dict={X_IN: batch_imgs, LR: lr, B_TRAIN: True})
                    _, g_loss, g_x_imgs, g_x_gx = sess.run([generator_optimizer, gen_loss, G_X, g_loss_x_gx], feed_dict={X_IN: batch_imgs, LR: lr, B_TRAIN: True})
                    _ = sess.run([sparsity_reg_optimizer], feed_dict={X_IN: batch_imgs, LR: lr, B_TRAIN: True})

                    print('epoch: ' + str(e) + ', ' +
                          'd_loss: ' + str(d_loss) + ', d_x_dx: ' + str(d_x_dx) + ', d_gx_dgx: ' + str(d_gx_dgx) +
                          ', ' + 'g_loss: ' + str(g_loss) +
                          ', g_x_gx: ' + str(g_x_gx))

                    if itr % 10 == 0:
                        d_images = d_x_imgs[0] * 255.0
                        g_images = g_x_imgs[0] * 255.0

                        #cv2.imwrite(out_dir + '/d_' + tr_files[start], d_images)
                        cv2.imwrite(out_dir + '/g_' + tr_files[start], g_images)
                        print('Elapsed Time at  ' + str(cur_step) + '/' + str(total_steps) + ' steps, ' + str(time.time() - train_start_time) + ' sec')
            try:
                print('Saving model...')
                saver.save(sess, model_path)
                print('Saved.')
            except:
                print('Save failed')
            print('Training Time: ' + str(time.time() - train_start_time))

            total_test_size = len(te_files)
            test_batch = zip(range(0, total_test_size, batch_size),
                             range(batch_size, total_test_size + 1, batch_size))

            for start, end in test_batch:
                test_imgs = load_images(te_files[start:end], base_dir=te_dir)
                if use_generator_only is True:
                    gx_imgs = sess.run([G_X], feed_dict={X_IN: test_imgs, B_TRAIN: False})
                    for i in range(batch_size):
                        cv2.imwrite('out/gx_' + te_files[start + i], gx_imgs[i] * 255.0)
                else:
                    dgx_imgs, gx_imgs, dx_imgs = sess.run([D_GX, G_X, D_X], feed_dict={X_IN: test_imgs, B_TRAIN: False})

                    for i in range(batch_size):
                        #cv2.imwrite('out/dgx_' + te_files[start + i], dgx_imgs[i] * 255.0)
                        cv2.imwrite('out/gx_' + te_files[start + i], gx_imgs[i] * 255.0)
                        #cv2.imwrite('out/dx_' + te_files[start + i], dx_imgs[i] * 255.0)


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
    st_attention_g, st_latent_g = memory(s_gen, scope=GS_M_scope)
    G_X = decoder(latent_g, st_latent_g, norm='instance', scope=G_Decoder_scope, b_use_style=True, b_train=B_TRAIN)

    if use_generator_only is False:
        z_disc_gx, s_disc_gx = encoder(G_X, norm='instance', b_use_style=use_style, scope=D_Encoder_scope, b_train=B_TRAIN, b_disc=True)
        D_GX = decoder(z_disc_gx, s_disc_gx, norm='instance', scope=D_Decoder_scope, b_use_style=True, b_train=B_TRAIN, b_disc=True)
        z_disc_x, s_disc_x = encoder(X_IN, norm='instance', b_use_style=use_style, scope=D_Encoder_scope, b_train=B_TRAIN, b_disc=True)
        D_X = decoder(z_disc_x, s_disc_x, norm='instance', scope=D_Decoder_scope, b_use_style=True, b_train=B_TRAIN, b_disc=True)
        anomaly_score = calculate_anomaly_scores(X_IN, D_GX)
    else:
        anomaly_score = calculate_anomaly_scores(X_IN, G_X)

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
            if use_generator_only is True:
                scores, gx_imgs = sess.run([anomaly_score, G_X],
                                                              feed_dict={X_IN: test_imgs, B_TRAIN: False})
                for i in range(batch_size):
                    print('Anomaly Score ' + te_files[start + i] + ': ' + str(scores[i]))
                    cv2.imwrite(out_dir + '/gx_' + te_files[start + i], gx_imgs[i] * 255.0)
            else:
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

    # small size
    #unit_block_depth = 16
    #bottleneck_num = 4

    # medium size
    unit_block_depth = 32
    bottleneck_num = 12

    # large size
    #unit_block_depth = 64
    #bottleneck_num = 12

    enc_conv_num_groups = 4
    dec_conv_num_groups = 4
    upsample_ratio = 16  # input_width % upsample_ratio = 0
    bottleneck_feature_size = input_width // upsample_ratio
    downsample_num = int(np.log2(upsample_ratio))
    upsample_num = downsample_num
    query_dimension = 1024
    style_dimension = 1024
    representation_dimension = query_dimension
    aug_mem_size = 1000
    num_channel = 3
    use_style = True
    use_style_mem = True
    use_generator_only = False
    use_blurpooling2d = False
    use_grouped_conv = False

    if mode == 'train':
        train(model_path)
    elif mode == 'test':
        test(model_path)
    else:
        print('Train or Test?')
