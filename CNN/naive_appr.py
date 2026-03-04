#import tensorflow as tf
#import numpy as np

#from tensorflow.keras import datasets, layers, models
#import matplotlib.pyplot as plt

# We load our already-made model (see "cnn-tensorflow.py").
#model = tf.keras.models.load_model("cnn-tensorflow_model.keras")

# Getting the test data we want to try to use.
#(_, _), (test_images, test_labels) = datasets.cifar10.load_data()


# Now that we have our model compiled and trained, now we can try some Conformal Prediction methods. For all methods, we want to try to get 90% coverage.

# First the "naive" approach: Get the softmax probability distribution and order them from highest to lowest. Then just add the labels from "highest" to "lowest" until their accumulated softmax score is ~0.9.


# Wrap our model with a softmax/output layer

#model = tf.keras.Sequential(
#    [
#        tf.keras.layers.Dense(5, input_shape=(3,)),
#        tf.keras.layers.Softmax(),
#    ],
#)
#model.save("test_softmax_model.keras")
#prediction_model = tf.keras.models.load_model("test_softmax_model.keras")
#x = tf.keras.random.uniform((10, 3)) # Test data

#predictions = prediction_model.predict(test_images)
#print(predictions[0])  # array of 10 probabilities that sum to 1






import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets

# 1) Load your trained CNN
base_model = tf.keras.models.load_model("cnn-tensorflow_model.keras")

# Optional: sanity check input shape
print("Model input shape:", base_model.input_shape)  # should be (None, 32, 32, 3)

# 2) Load CIFAR-10 test set and normalize to match training preprocessing
(_, _), (test_images, test_labels) = datasets.cifar10.load_data()
test_images = test_images.astype("float32") / 255.0

# 3) Turn logits into probabilities
# Option A: wrap with Softmax layer
probability_model = tf.keras.Sequential([base_model, tf.keras.layers.Softmax()])

# Option B (equivalent): use tf.nn.softmax on the logits
# logits = base_model.predict(test_images, batch_size=256)
# predictions = tf.nn.softmax(logits, axis=-1).numpy()

predictions = probability_model.predict(test_images, batch_size=256)

# 4) Pretty-print one distribution
#np.set_printoptions(precision=4, suppress=True)
print("Probs for test image 0:", predictions[0], "\nsum:", predictions[0].sum())
