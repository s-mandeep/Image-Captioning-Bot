import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)
import test

print(test.caption("A.jpeg"))
print("\n")
print(test.caption("B.jpg"))
print("\n")
print(test.caption("C.jpg"))
print("\n")
print(test.caption("D.jpg"))