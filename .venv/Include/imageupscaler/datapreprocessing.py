import tensorflow as tf
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.AUTOTUNE

def load_div2k_data(scale=4):
    div2k_data = tfds.image.Div2k(config=f"bicubic_x{scale}")
    div2k_data.download_and_prepare()
    train = div2k_data.as_dataset(split="train", as_supervised=True)
    val = div2k_data.as_dataset(split="validation", as_supervised=True)
    return train.cache(), val.cache()

def flip_left_right(lowres_img, highres_img):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(
        rn < 0.5,
        lambda: (lowres_img, highres_img),
        lambda: (
            tf.image.flip_left_right(lowres_img),
            tf.image.flip_left_right(highres_img),
        ),
    )

def random_rotate(lowres_img, highres_img):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lowres_img, rn), tf.image.rot90(highres_img, rn)

def random_crop(lowres_img, highres_img, hr_crop_size=96, scale=4):
    lowres_crop_size = hr_crop_size // scale
    lowres_img_shape = tf.shape(lowres_img)[:2]

    lowres_width = tf.random.uniform(
        shape=(), maxval=lowres_img_shape[1] - lowres_crop_size + 1, dtype=tf.int32
    )
    lowres_height = tf.random.uniform(
        shape=(), maxval=lowres_img_shape[0] - lowres_crop_size + 1, dtype=tf.int32
    )

    highres_width = lowres_width * scale
    highres_height = lowres_height * scale

    lowres_img_cropped = lowres_img[
        lowres_height : lowres_height + lowres_crop_size,
        lowres_width : lowres_width + lowres_crop_size,
    ]
    highres_img_cropped = highres_img[
        highres_height : highres_height + hr_crop_size,
        highres_width : highres_width + hr_crop_size,
    ]

    return lowres_img_cropped, highres_img_cropped

def dataset_object(dataset_cache, training=True, scale=4):
    ds = dataset_cache
    ds = ds.map(
        lambda lowres, highres: random_crop(lowres, highres, scale=scale),
        num_parallel_calls=AUTOTUNE,
    )

    if training:
        ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
        ds = ds.map(flip_left_right, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(16)

    if training:
        ds = ds.repeat()
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds
