import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from datapreprocessing import load_div2k_data, dataset_object
from mdsr import make_mdsr_model

def PSNR(super_resolution, high_resolution):
    psnr_value = tf.image.psnr(high_resolution, super_resolution, max_val=255)
    return psnr_value

def train_model(scale, epochs=100, batch_size=16):
    train_cache, val_cache = load_div2k_data(scale)
    train_ds = dataset_object(train_cache, training=True, scale=scale)
    val_ds = dataset_object(val_cache, training=False, scale=scale)

    model = make_mdsr_model(num_filters=64, num_of_residual_blocks=16)
    optim_mdsr = tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=[5000], values=[1e-4, 5e-5]
        )
    )
    model.compile(optimizer=optim_mdsr, loss="mae", metrics=[[PSNR], [PSNR], [PSNR]])

    # Adjust the dataset to provide the correct target shapes for each output
    def adjust_dataset(dataset, scale):
        def adjust_data(x, y):
            y_scaled = [tf.image.resize(y, [tf.shape(y)[1] // s, tf.shape(y)[2] // s]) for s in [2, 3, 4]]
            return x, y_scaled
        return dataset.map(adjust_data)

    train_ds = adjust_dataset(train_ds, scale)
    val_ds = adjust_dataset(val_ds, scale)

    model.fit(train_ds, epochs=100, steps_per_epoch=200, validation_data=val_ds)
    
    # Save the model as a .h5 file
    # model.save('mdsr_model.h5')
    # Save the model in the .keras format
    model.save('mdsr_model.keras')

if __name__ == "__main__":
    train_model(scale=4)
