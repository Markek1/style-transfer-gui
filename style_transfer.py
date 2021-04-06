import os

import tensorflow as tf

def magenta_v1_256_2(content_image, style_image, resize=False, content_res=None, style_res=None):
    '''Resolution of generated image = resolution of content image.
       Resolution of the style image is 256x256 by default because the net
       was trained on it and it generally works best'''
    if resize:
        if content_res:
            content_image = tf.image.resize(content_image, content_res)

        if style_res:
            style_image = tf.image.resize(style_image, style_res)
        else:
            style_image = tf.image.resize(style_image, (256, 256))
    local_path = 'models/magenta_arbitrary-image-stylization-v1-256_2'
    if os.path.exists(local_path):
        model = tf.saved_model.load(local_path)
    image = tf.squeeze(model(tf.constant(content_image), tf.constant(style_image))[0])
    return image
