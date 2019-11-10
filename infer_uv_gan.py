from __future__ import absolute_import, division, print_function, unicode_literals
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
import cv2
import numpy as np


class UV_GAN():
    def __init__(self, checkpoint_dir):
        self.model = self._generator()
        self.checkpoint = tf.train.Checkpoint(generator=self.model)
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
        self.h = 256
        self.w = 256

    def _handle_input(self, img):
        def _normalize(input_image):
          input_image = (tf.cast(input_image, tf.float32) / 127.5) - 1
          return input_image 

        def _random_jitter(input_image):
          input_image = tf.image.resize(input_image, [self.h, self.w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
          flip_image = tf.image.flip_left_right(input_image)
          input_image = tf.concat([input_image, flip_image], axis=2)
          return input_image
        img = _random_jitter(img)
        img = _normalize(img)
        img = tf.expand_dims(img, 0)
        return img

    def _generator(self):
        OUTPUT_CHANNELS = 3
        def _downsample(filters, size, apply_batchnorm=True):
          initializer = tf.random_normal_initializer(0., 0.02)

          result = tf.keras.Sequential()
          result.add(
              tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                     kernel_initializer=initializer, use_bias=False))

          if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

          result.add(tf.keras.layers.LeakyReLU())

          return result

        def _upsample(filters, size, apply_dropout=False):
          initializer = tf.random_normal_initializer(0., 0.02)

          result = tf.keras.Sequential()
          result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))

          result.add(tf.keras.layers.BatchNormalization())

          if apply_dropout:
              result.add(tf.keras.layers.Dropout(0.5))

          result.add(tf.keras.layers.ReLU())

          return result

        down_stack = [
          _downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
          _downsample(128, 4), # (bs, 64, 64, 128)
          _downsample(256, 4), # (bs, 32, 32, 256)
          _downsample(512, 4), # (bs, 16, 16, 512)
          _downsample(512, 4), # (bs, 8, 8, 512)
          _downsample(512, 4), # (bs, 4, 4, 512)
          _downsample(512, 4), # (bs, 2, 2, 512)
          _downsample(512, 4), # (bs, 1, 1, 512)
        ]

        up_stack = [
          _upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
          _upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
          _upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
          _upsample(512, 4), # (bs, 16, 16, 1024)
          _upsample(256, 4), # (bs, 32, 32, 512)
          _upsample(128, 4), # (bs, 64, 64, 256)
          _upsample(64, 4), # (bs, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                               strides=2,
                                               padding='same',
                                               kernel_initializer=initializer,
                                               activation='tanh') # (bs, 256, 256, 3)

        concat = tf.keras.layers.Concatenate()

        inputs = tf.keras.layers.Input(shape=[None,None,6])
        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
          x = down(x)
          skips.append(x)
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
          x = up(x)
          x = concat([x, skip])
        x = last(x)
        return tf.keras.Model(inputs=inputs, outputs=x)
    def infer(self, img):
        h, w, _ = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        img = self._handle_input(img)
        output = self.model(img, training=False)
        output  = ((np.array(output[0])+1)*127.5).astype('uint8')
        #output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        output = cv2.resize(output, (w, h))
        return output


if __name__ == "__main__":
    import glob 
    imgs = glob.glob("./samples/abc/*.[pj]*")
    a = UV_GAN("./uv_gan_checkpoint_100") 
    for img_path in imgs:
        name = img_path.split('/')[-1]
        img = cv2.imread(img_path)
        b = a.infer(img)
        b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
        cv2.imwrite("./tmp/" + name, b)



    
