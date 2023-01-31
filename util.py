# ==============================================================================
# Author: Seongho Baek
# Contact: seonghobaek@gmail.com
# ==============================================================================


import math
import numpy as np
import tensorflow as tf
import errno
import os
import copy
import cv2
import functools


def rotate_image(img, angle, scale=1):
    # img = cv2 image
    if img.ndim > 2:
        h, w, c = img.shape
    else:
        h, w = img.shape

    rad_angle = float(angle) * math.pi / 180
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rad_angle, scale)
    r_img = cv2.warpAffine(img, M, (w, h))

    return r_img


def get_batch(X, X_, size):
    # X, X_ must be nd-array
    a = np.random.choice(len(X), size, replace=False)
    return X[a], X_[a]


def get_sequence_batch(X, seq_length, batch_size):
    # print('input dim:', len(X[0]), ', seq len:', seq_length, ', batch_size:', batch_size)
    # X must be nd-array
    a = np.random.choice(len(X)-seq_length, batch_size, replace=False)
    a = a + seq_length

    # print('index: ', a)

    seq = []

    for i in range(batch_size):
        if a[i] < seq_length - 1:
            s = np.random.normal(0.0, 0.1, [seq_length, len(X[0])])
            seq.append(s)
        else:
            s = np.arange(a[i]-seq_length, a[i])
            seq.append(X[s])

    seq = np.array(seq)

    return X[a], seq


def noise_validator(noise, allowed_noises):
    '''Validates the noise provided'''
    try:
        if noise in allowed_noises:
            return True
        elif noise.split('-')[0] == 'mask' and float(noise.split('-')[1]):
            t = float(noise.split('-')[1])
            if t >= 0.0 and t <= 1.0:
                return True
            else:
                return False
    except:
        return False
    pass


def sigmoid_normalize(value_list):
    list_max = float(max(value_list))
    alist = [i/list_max for i in value_list]
    alist = [1/(1+math.exp(-i)) for i in alist]

    return alist


def swish(logit,  name='swish'):
    with tf.name_scope(name):
        l = tf.multiply(logit, tf.nn.sigmoid(logit))

        return l


def generate_samples(dim, num_inlier, num_outlier, normalize=True):
    inlier = np.random.normal(0.0, 1.0, [num_inlier, dim])

    sample_inlier = []

    if normalize:
        inlier = np.transpose(inlier)

        for values in inlier:
            values = sigmoid_normalize(values)
            sample_inlier.append(values)

        inlier = np.array(sample_inlier).transpose()

    outlier = np.random.normal(1.0, 1.0, [num_outlier, dim])

    sample_outlier = []

    if normalize:
        outlier = np.transpose(outlier)

        for values in outlier:
            values = sigmoid_normalize(values)
            sample_outlier.append(values)

        outlier = np.array(sample_outlier).transpose()

    return inlier, outlier


def add_gaussian_noise(input_layer, mean, std):
    if std < 0.0:
        return input_layer

    noise = tf.random_normal(shape=input_layer.get_shape().as_list(), mean=mean, stddev=std, dtype=tf.float32)
    return tf.add(input_layer, noise)


def mkdirP(path):
    """
    Create a directory and don't error if the path already exists.

    If the directory already exists, don't do anything.

    :param path: The directory to create.
    :type path: str
    """
    assert path is not None

    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def patch_compare(img1, img2, patch_size=[4, 4]):
    img1_h, img1_w = img1.shape
    img2_h, img2_w = img2.shape
    img_w = np.min([img1_w, img2_w])
    img_h = np.min([img1_h, img2_h])

    patch_h, patch_w = patch_size
    num_patch_h = img_h // patch_h
    num_patch_w = img_w // patch_w

    diff_vector = []

    img1 = img1 // 255
    img2 = img2 // 255

    for h in range(num_patch_h):
        for w in range(num_patch_w):
            patch1 = img1[h * patch_h:h * patch_h + patch_h, w * patch_w: w * patch_w + patch_w]
            patch2 = img2[h * patch_h:h * patch_h + patch_h, w * patch_w: w * patch_w + patch_w]
            d = np.abs(patch1 - patch2)
            d = np.sum(d)
            diff_vector.append(d)

    return diff_vector


class ImagePool(object):
    def __init__(self, maxsize=50, threshold=0.5):
        self.maxsize = maxsize
        self.num_img = 0
        self.images = []
        self.threshold = threshold

    def __call__(self, image):
        if self.maxsize <= 0:
            return image

        if self.num_img < self.maxsize:
            self.images.append(image)
            self.num_img += 1
            return image

        if np.random.rand() > self.threshold:
            idx = int(np.random.rand() * self.maxsize)
            tmp = copy.copy(self.images[idx])
            self.images[idx] = image
            return tmp
        else:
            return image


class COLORS:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def create_lr_images(input_dir, output_dir_Y, output_dir_X, ratio=8):
    try:
        images = []
        trX = os.listdir(input_dir)

        for file_name in trX:
            fullname = os.path.join(input_dir, file_name).replace("\\", "/")
            img = cv2.imread(fullname)
            cv2.imwrite(os.path.join(output_dir_Y, file_name.replace('gt', 'lr_gt')), img)

            if img is None:
                print('Load failed: ' + fullname)
                return None

            height, width, channel = img.shape
            img = cv2.resize(img, dsize=(width // ratio, height // ratio), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            tmp_file = os.path.join(output_dir_X, 'tmp.jpg')
            cv2.imwrite(tmp_file, img)

            out = cv2.imread(tmp_file)
            out = cv2.resize(out, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(output_dir_X, file_name.replace('gt', 'lr_input')), out)

    except cv2.error as e:
        print(e)
        return None

    return np.array(images)


def create_patch(img_file1, img_file2=None, ratio=4):
    img1 = cv2.imread(img_file1)

    if img_file2 is None:
        img2 = None
    else:
        img2 = cv2.imread(img_file2)

    img_height, img_width, _ = img1.shape

    patch_height = img_height // ratio
    patch_width = img_width // ratio
    input_img_list = [img_file1, img_file2]

    for img_file in input_img_list:
        if img_file is None:
            break
        patch_x = 0
        patch_y = 0
        img = cv2.imread(img_file)

        for w in range(ratio):
            for h in range(ratio):
                patch = img[patch_x:patch_x + patch_width, patch_y:patch_y + patch_height]
                patch_file_name, ext = os.path.splitext(img_file)
                patch_file_name = patch_file_name + '_patch_' + str(w)+'_'+str(h) + ext
                cv2.imwrite('patch/' + patch_file_name, patch)
                patch_y = patch_y + patch_height
            patch_x = patch_x + patch_width
            patch_y = 0


def _random_integer(minval, maxval, seed):
  """Returns a random 0-D tensor between minval and maxval.
  Args:
    minval: minimum value of the random tensor.
    maxval: maximum value of the random tensor.
    seed: random seed.
  Returns:
    A random 0-D tensor between minval and maxval.
  """
  return tf.random_uniform(
      [], minval=minval, maxval=maxval, dtype=tf.int32, seed=seed)

# TODO(mttang): This method is needed because the current
# tf.image.rgb_to_grayscale method does not support quantization. Replace with
# tf.image.rgb_to_grayscale after quantization support is added.
def _rgb_to_grayscale(images, name=None):
  """Converts one or more images from RGB to Grayscale.
  Outputs a tensor of the same `DType` and rank as `images`.  The size of the
  last dimension of the output is 1, containing the Grayscale value of the
  pixels.
  Args:
    images: The RGB tensor to convert. Last dimension must have size 3 and
      should contain RGB values.
    name: A name for the operation (optional).
  Returns:
    The converted grayscale image(s).
  """
  with tf.name_scope(name, 'rgb_to_grayscale', [images]) as name:
    images = tf.convert_to_tensor(images, name='images')
    # Remember original dtype to so we can convert back if needed
    orig_dtype = images.dtype
    flt_image = tf.image.convert_image_dtype(images, tf.float32)

    # Reference for converting between RGB and grayscale.
    # https://en.wikipedia.org/wiki/Luma_%28video%29
    rgb_weights = [0.2989, 0.5870, 0.1140]
    rank_1 = tf.expand_dims(tf.rank(images) - 1, 0)
    gray_float = tf.reduce_sum(
        flt_image * rgb_weights, rank_1, keep_dims=True)
    gray_float.set_shape(images.get_shape()[:-1].concatenate([1]))
    return tf.image.convert_image_dtype(gray_float, orig_dtype, name=name)


def random_horizontal_flip(image,
                           seed=None):
  """Randomly flips the image and detections horizontally.
  The probability of flipping the image is 50%.
  Args:
    image: rank 3 float32 tensor with shape [height, width, channels].
    seed: random seed
  Returns:
    image: image which is the same shape as input image.
  """

  def _flip_image(image):
    # flip image
    image_flipped = tf.image.flip_left_right(image)
    return image_flipped

  with tf.name_scope('RandomHorizontalFlip', values=[image]):
    do_a_flip_random = tf.random_uniform([], seed=seed)
    do_a_flip_random = tf.greater(do_a_flip_random, 0.5)

    # flip image
    image = tf.cond(do_a_flip_random, lambda: _flip_image(image), lambda: image)

    return image


def random_vertical_flip(image,
                         seed=None):
  """Randomly flips the image and detections vertically.
  The probability of flipping the image is 50%.
  Args:
    image: rank 3 float32 tensor with shape [height, width, channels].
    seed: random seed
  Returns:
    image: image which is the same shape as input image.
  """

  def _flip_image(image):
    # flip image
    image_flipped = tf.image.flip_up_down(image)
    return image_flipped

  with tf.name_scope('RandomVerticalFlip', values=[image]):
    do_a_flip_random = tf.random_uniform([], seed=seed)
    do_a_flip_random = tf.greater(do_a_flip_random, 0.5)

    # flip image
    image = tf.cond(do_a_flip_random, lambda: _flip_image(image), lambda: image)

    return image


def random_rotation90(image,
                      seed=None):
  """Randomly rotates the image and detections 90 degrees counter-clockwise.
  The probability of rotating the image is 50%. This can be combined with
  random_horizontal_flip and random_vertical_flip to produce an output with a
  uniform distribution of the eight possible 90 degree rotation / reflection
  combinations.
  Args:
    image: rank 3 float32 tensor with shape [height, width, channels].
    seed: random seed
  Returns:
    image: image which is the same shape as input image.
  """

  def _rot90_image(image):
    # flip image
    image_rotated = tf.image.rot90(image)
    return image_rotated

  with tf.name_scope('RandomRotation90', values=[image]):
    do_a_rot90_random = tf.random_uniform([], seed=seed)
    do_a_rot90_random = tf.greater(do_a_rot90_random, 0.3)

    # flip image
    image = tf.cond(do_a_rot90_random, lambda: _rot90_image(image),
                    lambda: image)

    return image


def random_image_scale(image,
                       min_scale_ratio=0.5,
                       max_scale_ratio=2.0,
                       seed=None):
  """Scales the image size.
  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels].
    min_scale_ratio: minimum scaling ratio.
    max_scale_ratio: maximum scaling ratio.
    seed: random seed.
  Returns:
    image: image which is the same rank as input image.
  """
  with tf.name_scope('RandomImageScale', values=[image]):
    image_shape = tf.shape(image)
    image_height = image_shape[0]
    image_width = image_shape[1]

    size_coef = tf.random_uniform([], minval=min_scale_ratio, maxval=max_scale_ratio, dtype=tf.float32, seed=seed)

    image_newysize = tf.to_int32(
        tf.multiply(tf.to_float(image_height), size_coef))
    image_newxsize = tf.to_int32(
        tf.multiply(tf.to_float(image_width), size_coef))
    image = tf.image.resize_images(
        image, [image_newysize, image_newxsize], align_corners=True)

    return image


def random_pad_image(image,
                     min_image_size=None,
                     max_image_size=None,
                     pad_color=None,
                     seed=None):
  """Randomly pads the image.
  This function randomly pads the image with zeros. The final size of the
  padded image will be between min_image_size and max_image_size.
  if min_image_size is smaller than the input image size, min_image_size will
  be set to the input image size. The same for max_image_size. The input image
  will be located at a uniformly random location inside the padded image.
  The relative location of the boxes to the original image will remain the same.
  Args:
    image: rank 3 float32 tensor containing 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    min_image_size: a tensor of size [min_height, min_width], type tf.int32.
                    If passed as None, will be set to image size
                    [height, width].
    max_image_size: a tensor of size [max_height, max_width], type tf.int32.
                    If passed as None, will be set to twice the
                    image [height * 2, width * 2].
    pad_color: padding color. A rank 1 tensor of [3] with dtype=tf.float32.
               if set as None, it will be set to average color of the input
               image.
    seed: random seed.
  Returns:
    image: Image shape will be [new_height, new_width, channels].
  """
  if pad_color is None:
    pad_color = tf.reduce_mean(image, axis=[0, 1])

  image_shape = tf.shape(image)
  image_height = image_shape[0]
  image_width = image_shape[1]

  if max_image_size is None:
    max_image_size = tf.stack([image_height * 2, image_width * 2])
  max_image_size = tf.maximum(max_image_size,
                              tf.stack([image_height, image_width]))

  if min_image_size is None:
    min_image_size = tf.stack([image_height, image_width])
  min_image_size = tf.maximum(min_image_size,
                              tf.stack([image_height, image_width]))

  target_height = tf.cond(
      max_image_size[0] > min_image_size[0],
      lambda: _random_integer(min_image_size[0], max_image_size[0], seed),
      lambda: max_image_size[0])

  target_width = tf.cond(
      max_image_size[1] > min_image_size[1],
      lambda: _random_integer(min_image_size[1], max_image_size[1], seed),
      lambda: max_image_size[1])

  offset_height = tf.cond(
      target_height > image_height,
      lambda: _random_integer(0, target_height - image_height, seed),
      lambda: tf.constant(0, dtype=tf.int32))

  offset_width = tf.cond(
      target_width > image_width,
      lambda: _random_integer(0, target_width - image_width, seed),
      lambda: tf.constant(0, dtype=tf.int32))

  new_image = tf.image.pad_to_bounding_box(
      image,
      offset_height=offset_height,
      offset_width=offset_width,
      target_height=target_height,
      target_width=target_width)

  # Setting color of the padded pixels
  image_ones = tf.ones_like(image)
  image_ones_padded = tf.image.pad_to_bounding_box(
      image_ones,
      offset_height=offset_height,
      offset_width=offset_width,
      target_height=target_height,
      target_width=target_width)
  image_color_padded = (1.0 - image_ones_padded) * pad_color
  new_image += image_color_padded

  return new_image


def random_crop_to_aspect_ratio(image,
                                aspect_ratio=1.0,
                                seed=None):
  """Randomly crops an image to the specified aspect ratio.
  Randomly crops the a portion of the image such that the crop is of the
  specified aspect ratio, and the crop is as large as possible. If the specified
  aspect ratio is larger than the aspect ratio of the image, this op will
  randomly remove rows from the top and bottom of the image. If the specified
  aspect ratio is less than the aspect ratio of the image, this op will randomly
  remove cols from the left and right of the image. If the specified aspect
  ratio is the same as the aspect ratio of the image, this op will return the
  image.
  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    aspect_ratio: the aspect ratio of cropped image.
    clip_boxes: whether to clip the boxes to the cropped image.
    seed: random seed.
  Returns:
    image: image which is the same rank as input image.
  Raises:
    ValueError: If image is not a 3D tensor.
  """
  if len(image.get_shape()) != 3:
    raise ValueError('Image should be 3D tensor')

  with tf.name_scope('RandomCropToAspectRatio', values=[image]):
    image_shape = tf.shape(image)
    orig_height = image_shape[0]
    orig_width = image_shape[1]
    orig_aspect_ratio = tf.to_float(orig_width) / tf.to_float(orig_height)
    new_aspect_ratio = tf.constant(aspect_ratio, dtype=tf.float32)
    def target_height_fn():
      return tf.to_int32(tf.round(tf.to_float(orig_width) / new_aspect_ratio))

    target_height = tf.cond(orig_aspect_ratio >= new_aspect_ratio,
                            lambda: orig_height, target_height_fn)

    def target_width_fn():
      return tf.to_int32(tf.round(tf.to_float(orig_height) * new_aspect_ratio))

    target_width = tf.cond(orig_aspect_ratio <= new_aspect_ratio,
                           lambda: orig_width, target_width_fn)

    # either offset_height = 0 and offset_width is randomly chosen from
    # [0, offset_width - target_width), or else offset_width = 0 and
    # offset_height is randomly chosen from [0, offset_height - target_height)
    offset_height = _random_integer(0, orig_height - target_height + 1, seed)
    offset_width = _random_integer(0, orig_width - target_width + 1, seed)

    new_image = tf.image.crop_to_bounding_box(
        image, offset_height, offset_width, target_height, target_width)

    return new_image


def random_pad_to_aspect_ratio(image,
                               aspect_ratio=1.0,
                               min_padded_size_ratio=(1.0, 1.0),
                               max_padded_size_ratio=(2.0, 2.0),
                               seed=None):
  """Randomly zero pads an image to the specified aspect ratio.
  Pads the image so that the resulting image will have the specified aspect
  ratio without scaling less than the min_padded_size_ratio or more than the
  max_padded_size_ratio. If the min_padded_size_ratio or max_padded_size_ratio
  is lower than what is possible to maintain the aspect ratio, then this method
  will use the least padding to achieve the specified aspect ratio.
  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    aspect_ratio: aspect ratio of the final image.
    min_padded_size_ratio: min ratio of padded image height and width to the
                           input image's height and width.
    max_padded_size_ratio: max ratio of padded image height and width to the
                           input image's height and width.
    seed: random seed.
  Returns:
    image: image which is the same rank as input image.
  Raises:
    ValueError: If image is not a 3D tensor.
  """
  if len(image.get_shape()) != 3:
    raise ValueError('Image should be 3D tensor')

  with tf.name_scope('RandomPadToAspectRatio', values=[image]):
    image_shape = tf.shape(image)
    image_height = tf.to_float(image_shape[0])
    image_width = tf.to_float(image_shape[1])
    image_aspect_ratio = image_width / image_height
    new_aspect_ratio = tf.constant(aspect_ratio, dtype=tf.float32)
    target_height = tf.cond(
        image_aspect_ratio <= new_aspect_ratio,
        lambda: image_height,
        lambda: image_width / new_aspect_ratio)
    target_width = tf.cond(
        image_aspect_ratio >= new_aspect_ratio,
        lambda: image_width,
        lambda: image_height * new_aspect_ratio)

    min_height = tf.maximum(
        min_padded_size_ratio[0] * image_height, target_height)
    min_width = tf.maximum(
        min_padded_size_ratio[1] * image_width, target_width)
    max_height = tf.maximum(
        max_padded_size_ratio[0] * image_height, target_height)
    max_width = tf.maximum(
        max_padded_size_ratio[1] * image_width, target_width)

    max_scale = tf.minimum(max_height / target_height, max_width / target_width)
    min_scale = tf.minimum(
        max_scale,
        tf.maximum(min_height / target_height, min_width / target_width))

    scale = tf.random_uniform([], min_scale, max_scale, seed=seed)

    target_height = tf.round(scale * target_height)
    target_width = tf.round(scale * target_width)

    new_image = tf.image.pad_to_bounding_box(
        image, 0, 0, tf.to_int32(target_height), tf.to_int32(target_width))

    return new_image


def random_pixel_value_scale(image,
                             minval=0.9,
                             maxval=1.1,
                             seed=None):
  """Scales each value in the pixels of the image.
     This function scales each pixel independent of the other ones.
     For each value in image tensor, draws a random number between
     minval and maxval and multiples the values with them.
  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 255].
    minval: lower ratio of scaling pixel values.
    maxval: upper ratio of scaling pixel values.
    seed: random seed.
  Returns:
    image: image which is the same shape as input image.
  """
  with tf.name_scope('RandomPixelValueScale', values=[image]):
    color_coef = tf.random_uniform(tf.shape(image), minval=minval, maxval=maxval, dtype=tf.float32, seed=seed)
    image = tf.multiply(image, color_coef)
    image = tf.clip_by_value(image, 0.0, 255.0)

    return image


def random_rgb_to_gray(image,
                       probability=0.1,
                       seed=None):
  """Changes the image from RGB to Grayscale with the given probability.
  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 255].
    probability: the probability of returning a grayscale image.
            The probability should be a number between [0, 1].
    seed: random seed.
  Returns:
    image: image which is the same shape as input image.
  """
  def _image_to_gray(image):
    image_gray1 = _rgb_to_grayscale(image)
    image_gray3 = tf.image.grayscale_to_rgb(image_gray1)
    return image_gray3

  with tf.name_scope('RandomRGBtoGray', values=[image]):
    do_gray_random = tf.random_uniform([], seed=seed)

    image = tf.cond(
        tf.greater(do_gray_random, probability), lambda: image,
        lambda: _image_to_gray(image))

    return image


def random_augment_brightness_cutout(images, probability=0.5, seed=None, batch_size=1):
    with tf.name_scope('RandomAugment', values=[images]):
        random_prob = tf.random_uniform([], minval=0.0, maxval=1.0, dtype=tf.float32, seed=seed)

        image = tf.cond(tf.greater(random_prob, probability),
                        lambda: random_adjust_brightness(image=images),
                        lambda: random_black_patches(image=images, batch_size=batch_size))

        return image


def random_augments_hard(images, probability=0.5, seed=None, batch_size=1):
    with tf.name_scope('RandomAugmentHard', values=[images]):
        random_prob = tf.random_uniform([], minval=0.0, maxval=1.0, dtype=tf.float32, seed=seed)

        image = tf.cond(tf.greater(random_prob, probability),
                        lambda: random_black_patches(image=images, batch_size=batch_size),
                        lambda: random_augment_brightness_constrast(images=images, probability=probability))
        return image


def random_augments(images, probability=0.5, seed=None):
    with tf.name_scope('RandomAugment', values=[images]):
        random_prob = tf.random_uniform([], minval=0.0, maxval=1.0, dtype=tf.float32, seed=seed)

        image = tf.cond(tf.greater(random_prob, probability),
                        lambda: random_augment_brightness_constrast(images=images, probability=probability),
                        lambda: random_rotation90(image=images))
        return image


def random_augment_brightness_constrast(images, probability=0.2, seed=None):
    with tf.name_scope('RandomAugmentBC', values=[images]):
        random_prob = tf.random_uniform([], minval=0.0, maxval=1.0, dtype=tf.float32, seed=seed)

        image = tf.cond(tf.greater(random_prob, probability),
                        lambda: random_adjust_brightness(image=images),
                        lambda: 0.5 * (images + tf.stop_gradient(random_adjust_contrast(image=images))))
        return image


def random_adjust_brightness(image,
                             max_delta=0.3,
                             probability=0.7,
                             seed=None):
  """Randomly adjusts brightness.
  Makes sure the output image is still between 0 and 255.
  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 255].
    max_delta: how much to change the brightness. A value between [0, 1).
    seed: random seed.
  Returns:
    image: image which is the same shape as input image.
    boxes: boxes which is the same shape as input boxes.
  """
  with tf.name_scope('RandomAdjustBrightness', values=[image]):
    delta = tf.random_uniform([], -max_delta, max_delta, seed=seed)
    #image = tf.image.adjust_brightness(image / 255, delta) * 255
    #image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)
    #image = tf.image.adjust_brightness(image, delta)
    random_prob = tf.random_uniform([], minval=0.0, maxval=1.0, dtype=tf.float32, seed=seed)
    image = tf.cond(
        tf.greater(random_prob, probability), lambda: image,
        functools.partial(tf.image.adjust_brightness, image=image, delta=delta))

    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    return image


def random_adjust_contrast(image,
                           min_delta=0.8,
                           max_delta=1.2,
                           probability=0.7,
                           seed=None):
    """Randomly adjusts contrast.
    Makes sure the output image is still between 0 and 255.
    Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 255].
    min_delta: see max_delta.
    max_delta: how much to change the contrast. Contrast will change with a
               value between min_delta and max_delta. This value will be
               multiplied to the current contrast of the image.
    seed: random seed.
    Returns:
    image: image which is the same shape as input image.
    """
    with tf.name_scope('RandomAdjustContrast', values=[image]):
        contrast_factor = tf.random_uniform([], min_delta, max_delta, seed=seed)
        #image = tf.image.adjust_contrast(image / 255, contrast_factor) * 255
        #image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)

        #image = tf.image.adjust_contrast(image, contrast_factor)
        #image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

        random_prob = tf.random_uniform([], minval=0.0, maxval=1.0, dtype=tf.float32, seed=seed)
        image = tf.cond(
            tf.greater(random_prob, probability), lambda: image,
            functools.partial(tf.image.adjust_contrast, images=image, contrast_factor=contrast_factor))
        image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    return image


def random_adjust_hue(image,
                      max_delta=0.02,
                      seed=None):
  """Randomly adjusts hue.
  Makes sure the output image is still between 0 and 255.
  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 255].
    max_delta: change hue randomly with a value between 0 and max_delta.
    seed: random seed.
  Returns:
    image: image which is the same shape as input image.
  """
  with tf.name_scope('RandomAdjustHue', values=[image]):
    delta = tf.random_uniform([], -max_delta, max_delta, seed=seed)
    image = tf.image.adjust_hue(image / 255, delta) * 255
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)

    return image


def random_adjust_saturation(image,
                             min_delta=0.8,
                             max_delta=1.25,
                             seed=None):
  """Randomly adjusts saturation.
  Makes sure the output image is still between 0 and 255.
  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 255].
    min_delta: see max_delta.
    max_delta: how much to change the saturation. Saturation will change with a
               value between min_delta and max_delta. This value will be
               multiplied to the current saturation of the image.
    seed: random seed.
  Returns:
    image: image which is the same shape as input image.
  """
  with tf.name_scope('RandomAdjustSaturation', values=[image]):
    saturation_factor = tf.random_uniform([], min_delta, max_delta, seed=seed)
    image = tf.image.adjust_saturation(image / 255, saturation_factor) * 255
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)

    return image


def random_distort_color(image, color_ordering=0):
  """Randomly distorts color.
  Randomly distorts color using a combination of brightness, hue, contrast and
  saturation changes. Makes sure the output image is still between 0 and 255.
  Args:
    image: rank 3 float32 tensor contains 1 image -> [height, width, channels]
           with pixel values varying between [0, 255].
    color_ordering: Python int, a type of distortion (valid values: 0, 1).
  Returns:
    image: image which is the same shape as input image.
  Raises:
    ValueError: if color_ordering is not in {0, 1}.
  """
  with tf.name_scope('RandomDistortColor', values=[image]):
    if color_ordering == 0:
      image = random_adjust_brightness(
          image, max_delta=32. / 255.)
      image = random_adjust_saturation(
          image, min_delta=0.5, max_delta=1.5)
      image = random_adjust_hue(
          image, max_delta=0.2)
      image = random_adjust_contrast(
          image, min_delta=0.5, max_delta=1.5)

    elif color_ordering == 1:
      image = random_adjust_brightness(
          image, max_delta=32. / 255.)
      image = random_adjust_contrast(
          image, min_delta=0.5, max_delta=1.5)
      image = random_adjust_saturation(
          image, min_delta=0.5, max_delta=1.5)
      image = random_adjust_hue(
          image, max_delta=0.2)
    else:
      raise ValueError('color_ordering must be in {0, 1}')
    return image


def random_black_patches(image,
                         max_black_patches=3,
                         probability=0.5,
                         size_to_image_ratio=0.1,
                         random_seed=None,
                         batch_size=1):
  """Randomly adds some black patches to the image.
  This op adds up to max_black_patches square black patches of a fixed size
  to the image where size is specified via the size_to_image_ratio parameter.
  Args:
    image: rank 3 float32 tensor containing 1 image -> [height, width, channels]
           with pixel values varying between [0, 1].
    max_black_patches: number of times that the function tries to add a
                       black box to the image.
    probability: at each try, what is the chance of adding a box.
    size_to_image_ratio: Determines the ratio of the size of the black patches
                         to the size of the image.
                         box_size = size_to_image_ratio *
                                    min(image_width, image_height)
    random_seed: random seed.
  Returns:
    image
  """
  def add_black_patch_to_image(image, idx, batch_size=1):
    """Function for adding one patch to the image.
    Args:
      image: image
      idx: counter for number of patches that could have been added
    Returns:
      image with a randomly added black box
    """
    #image_shape = tf.shape(image)
    image_shape = image.get_shape().as_list()

    image_height = image_shape[1]
    image_width = image_shape[2]
    image_channel = image_shape[3]

    box_size = tf.to_int32(
        tf.multiply(
            tf.minimum(tf.to_float(image_height), tf.to_float(image_width)),
            size_to_image_ratio))

    normalized_y_min = tf.random_uniform([], minval=0.0, maxval=(1.0 - size_to_image_ratio), seed=random_seed)
    normalized_x_min = tf.random_uniform([], minval=0.0, maxval=(1.0 - size_to_image_ratio), seed=random_seed)

    y_min = tf.to_int32(normalized_y_min * tf.to_float(image_height))
    x_min = tf.to_int32(normalized_x_min * tf.to_float(image_width))
    black_box = tf.ones([batch_size, box_size, box_size, image_channel], dtype=tf.float32)
    mask = tf.ones([batch_size, image_height, image_width, image_channel])
    #print('mask shape: ' + str(mask.get_shape().as_list()))
    mask = mask - tf.image.pad_to_bounding_box(black_box, y_min, x_min,
                                              image_height, image_width)
    image = (image + 1.0)/2.0
    image = tf.multiply(image, mask)
    image = 2.0 * image - 1.0
    #print('mask shape: ' + str(mask.get_shape().as_list()))
    return image

  with tf.name_scope('RandomBlackPatchInImage', values=[image]):
    for idx in range(max_black_patches):
      random_prob = tf.random_uniform([], minval=0.0, maxval=1.0, dtype=tf.float32, seed=random_seed)
      image = tf.cond(
          tf.greater(random_prob, probability), lambda: image,
          functools.partial(add_black_patch_to_image, image=image, idx=idx, batch_size=batch_size))

    return image


def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


# Spherical linear interpolation. low, high are samples are drawn under random normal distribution.
def slerp(val, low, high):
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0 - val) * low + val * high

    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


def interpolate_points(p1, p2, n_steps=10):
    ratios = np.linspace(0, 1, num=n_steps)
    vectors = list()

    vectors.append(p1)
    for ratio in ratios:
        v = slerp(ratio, p1, p2)
        vectors.append(v)
    vectors.append(p2)

    return vectors


def interpolant(t):
    return t*t*t*(t*(t*6 - 15) + 10)


def generate_perlin_noise_2d(shape, res, tileable=(False, False), interpolant=interpolant):
    """Generate a 2D numpy array of perlin noise.
    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            res.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).
    Returns:
        A numpy array of shape shape with the generated noise.
    Raises:
        ValueError: If shape is not a multiple of res.
    """
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]]\
             .transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    if tileable[0]:
        gradients[-1,:] = gradients[0,:]
    if tileable[1]:
        gradients[:,-1] = gradients[:,0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[:-d[0], :-d[1]]
    g10 = gradients[d[0]:, :-d[1]]
    g01 = gradients[:-d[0], d[1]:]
    g11 = gradients[d[0]:, d[1]:]
    # Ramps
    n00 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]  )) * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]  )) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = interpolant(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)


def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5, lacunarity=2,
                              tileable=(False, False), interpolant=interpolant):
    """Generate a 2D numpy array of fractal noise.
    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multiple of lacunarity**(octaves-1)*res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            (lacunarity**(octaves-1)*res).
        octaves: The number of octaves in the noise. Defaults to 1.
        persistence: The scaling factor between two octaves.
        lacunarity: The frequency factor between two octaves.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The, interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).
    Returns:
        A numpy array of fractal noise and of shape shape generated by
        combining several octaves of perlin noise.
    Raises:
        ValueError: If shape is not a multiple of
            (lacunarity**(octaves-1)*res).
    """
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(
            shape, (frequency*res[0], frequency*res[1]), tileable, interpolant
        )
        frequency *= lacunarity
        amplitude *= persistence
    return noise


def cate2pol(img, width, height):
    margin = 0.92  # Cut off black-background
    polar_img = cv2.warpPolar(img, (width, height), (img.shape[0] / 2, img.shape[1] / 2), img.shape[1] * margin * 0.5,
                              cv2.WARP_POLAR_LINEAR)
    polar_img = cv2.rotate(polar_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return polar_img


def pol2cate(img, width, height):
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    margin = 0.92  # Cut off black-background
    polar_img = cv2.warpPolar(img, (width, height), (img.shape[0] / 2, img.shape[1] / 2), img.shape[1] * margin * 0.5,
                              cv2.WARP_INVERSE_MAP)

    return polar_img


def otsu_binarization(img, width, height):
    max_sigma = 0
    max_t = 0
    H = height
    W = width

    for _t in range(1, 256, 2):
        v0 = img[np.where(img < _t)]
        m0 = np.mean(v0) if len(v0) > 0 else 0.
        w0 = len(v0) / (H * W)
        v1 = img[np.where(img >= _t)]
        m1 = np.mean(v1) if len(v1) > 0 else 0.
        w1 = len(v1) / (H * W)
        sigma = w0 * w1 * ((m0 - m1) ** 2)
        if sigma > max_sigma:
            max_sigma = sigma
            max_t = _t

    # Binarization
    print("threshold >>", max_t)
    th = max_t + max_sigma
    img[img < th] = 0
    img[img >= th] = 255

    return img
