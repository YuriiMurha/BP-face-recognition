import tensorflow as tf

def setup_gpu():
    """
    Configures TensorFlow to use GPU memory growth to avoid OOM errors.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: 
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Num GPUs Available: {len(gpus)}")
            print("GPUs: ", gpus)
        except RuntimeError as e:
            print(e)
    else:
        print("No GPUs found.")

if __name__ == "__main__":
    setup_gpu()