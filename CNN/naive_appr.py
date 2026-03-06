import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets


"""
No calibration data. 
Simply get a new test point, have the CNN Classifier get the softmax score for all possible labels, 
order the softmax scores from highest (most likely (according to model)) to lowest.

If we want 90% coverage (we guarantee that, on average, there is a 90% chance that the true label is in the prediction set), 
we just take labels from highest to lowest such that their combined softmax score is ~0.9 (90%).
"""

# NOTE: We could output a list of indexes, such that if we get a list of [2, 5, 6] then we know that labels with index 2, 5 and 6 should be a part of the Prediction Region.

# model = whole CNN model. labels = all possible labels for the input.  test_point = chosen new test point.   test_label = the true label of the chosen new test point.  conf_level = If we want a 90% coverage, then conf_level = 0.9.
def naive_appr(model, labels, test_point, test_label, conf_level):  
    # Get the softmax distribution of the test point
    softmax_dist = model.predict(np.array([test_point]))[0] # (Taken from https://datascience.stackexchange.com/questions/13461/how-can-i-get-prediction-for-only-one-instance-in-keras)

    # We create a dictionary so that we can pair each probability with the corresponding class/label.
    prob_dict = {}

    # For each probability given in the softmax dist, we pair it together with their corresponding label.
    for i, prob in enumerate(softmax_dist):
        prob_dict[labels[i]] = prob     # If we choose to not use string labels but instead the label indexes, then we can just use "i" instead of "labels[i]".

    # We sort the list from the highest probability to the lowest.  (Taken from GeeksForGeeks: https://www.geeksforgeeks.org/python/python-sort-python-dictionaries-by-key-or-value/)
    k = list(prob_dict.keys())
    v = list(prob_dict.values())
    idx = np.argsort(v)[::-1]
    res = {k[i]: v[i] for i in idx}
    print("\nOrdered list of softmax scores:")
    print(res)

    # Now we just add labels from the ordered dictionary until the sum of their probabilities add up to add least the confidence level.
    sum = 0
    pred_region = []
    for i, item in enumerate(res):  # Go through each item in the ordered dictionary
        if sum > conf_level:    # For the naive approach, if the sum of the softmax scores for the labels already added to the prediction region exceeds the confidence level, then we're done.
            break
        else:
            pred_region.append(item)
            sum = sum + res[item]
            #print("sum = " + str(sum))

    print("\nPrediction Region (naive):")
    print(pred_region)

    # Possibly print something about "the true label is not in the prediction region" (using the given argument 'test_label').

    return pred_region
    



#       1) Get a new test point
# Load CNN model + CIFAR-10 test set and normalize to match training preprocessing
base_model = tf.keras.models.load_model("cnn_softmax_model.keras")
(_, _), (test_images, test_labels) = datasets.cifar10.load_data()
test_images = test_images.astype("float32") / 255.0

# All possible labels for the CIFAR10 dataset.
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

image_nr = 10   # One image taken from the test images from the CIFAR10 dataset.

# Run naive approach to CP with the first of the test images and test labels.
naive_appr(base_model, class_names, test_images[image_nr], test_labels[image_nr], 0.9)



