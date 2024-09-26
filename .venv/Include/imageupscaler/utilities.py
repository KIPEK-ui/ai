import numpy as np
from PIL import Image
import tensorflow as tf
from imageupscaler.edsr import EDSRModel

def load_model(model_path):
    return tf.keras.models.load_model(model_path, custom_objects={'EDSRModel': EDSRModel})

def upscale_image(model, image_path, scale):
    img = Image.open(image_path).convert('RGB')
    low_res = img.resize((img.width // scale, img.height // scale), Image.BICUBIC)
    low_res = low_res.resize((img.width, img.height), Image.BICUBIC)
    low_res_array = np.array(low_res)
    low_res_array = np.expand_dims(low_res_array, axis=0)
    high_res_array = model.predict(low_res_array)
    high_res = Image.fromarray(np.uint8(high_res_array[0]))
    return high_res
