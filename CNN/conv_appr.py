import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets


"""
Use calibration data. (CIFAR10 test images/labels, just a small part of them. We can then use one of the test images that were NOT chosen as calibration data to be used as a "new test point".)
To get the threshold value “q”, we let the model get the softmax score for the (as we know it) true label for each calibration data example 
(for example, if one example is (input, true label) = (image.png, “Dog”), 
then we let the model tell their softmax score for the label “Dog” for that image).
If we want 90% coverage, then we pick a threshold value “q” such that “q” is smaller than ~90% of all the softmax scores from the calibration data.

When we get a new test point (we know the input, not the true label), we simply get the softmax scores for every label, 
order them from highest to lowest, and then we add every label whose softmax score is higher than our threshold value “q”.
"""

# NOTE: We could output a list of indexes, such that if we get a list of [2, 5, 6] then we know that labels with index 2, 5 and 6 should be a part of the Prediction Region.

def score_function(softmax_scores, true_labels):
    scores = []

    # For each example given (in which each example has a set of softmax scores), we get the example's true label and get the softmax score for that true label.
    for i, pred in enumerate(softmax_scores):
        true_label = true_labels[i]     # Get the true label for this calibration data example
        scores.append(pred[true_label])    # Add the softmax score given by the model for the actual true label (for this example) into calib_probs.

    return scores


# model = whole CNN model. labels = all possible labels for the input.  test_point = chosen new test point.   test_label = the true label of the chosen new test point.  alpha = If we want a 90% coverage, then alpha = 0.1 (since coverage = 1 - alpha).
def conv_appr(model, labels, calib_input, calib_label, test_point, alpha, test_label=None):  

                # 1) Get the threshold value

    # We first get the calibration dataset, which typically consists of 1000 sample.
    predictions = model.predict(calib_input, batch_size=32)

    calib_probs = score_function(predictions, calib_label)

    # If we want a 90% coverage, then we want to find the value which is smaller than 90% of the values/ bigger than 10% of the values in calib_probs
    calib_probs = np.array(calib_probs)
    q = np.percentile(calib_probs, alpha*100)

    #n = len(calib_probs)
    #q = np.quantile(calib_probs, np.ceil((n+1)*(1-alpha)/n), method='higher')

    np.set_printoptions(precision=4, suppress=True)
    print("\nThreshold value 'q' is: ")
    print(q)
    
    # We now have our threshold value "q", which is smaller than (1 - conf_level)*100 percent of all the values in calib_probs.


                # 2) Add labels into our prediction region.
    # Get the softmax distribution of the test point
    softmax_dist = model.predict(np.array([test_point]), verbose=0)[0] # (Taken from https://datascience.stackexchange.com/questions/13461/how-can-i-get-prediction-for-only-one-instance-in-keras)

    # Now we add the labels whose softmax scores are higher than the threshold value into the prediction region array.
    pred_region = {}
    for i in range(len(softmax_dist)):  # Go through each softmax score.
        if softmax_dist[i] > q:    # For the conventional approach, we only add labels who've gotten a softmax score that is higher than the threshold value "q".
            pred_region[labels[i]] = softmax_dist[i]

    # Special case: If there are no nonconformity scores that are less than the confidence level, we just add the one with the lowest score.
    if not bool(pred_region):
        i = np.argmin(softmax_dist)
        pred_region[labels[i]] = softmax_dist[i]

    print("\nPrediction Region (conventional):")
    print(pred_region)

    if test_label is not None:  # If we have given some test label, then we can print it out.
        true_label = labels[int(test_label.item())]
        print(f"\nTrue label is: '{true_label}'.\n")

    # Possibly print something about "the true label is not in the prediction region" (using the given argument 'test_label').

    return pred_region



#       1) Get a new test point
# Load CNN model + CIFAR-10 test set and normalize to match training preprocessing
base_model = tf.keras.models.load_model("cnn_softmax_model.keras")
(_, _), (test_images, test_labels) = datasets.cifar10.load_data()
test_images = test_images.astype("float32") / 255.0

# Get the first 1000 images + labels from the test data as calibration data.
calibration_images = test_images[:1000]
calibration_labels = test_labels[:1000]  

# All possible labels for the CIFAR10 dataset.
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

image_nr = 2000  # One image taken from the test images from the CIFAR10 dataset.

# Run conventional approach to CP.
conv_appr(base_model, class_names, calibration_images, calibration_labels, test_images[image_nr], 0.1, test_labels[image_nr])