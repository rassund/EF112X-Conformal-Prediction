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

# NOTE: Not needed in the naive approach below. This is used to evaluate some metrics for this method.
# Given a model, some input test point "x", and the input test point's true label "y", computes the "naive" score s(x, y) for this test point.
def score_function(softmax_dist, true_label):
    """
    Given some input "x" and its true label "y", we get the softmax distribution for the input.
    Instead of sorting the softmax distribution from highest to lowest, 
    we just add together all the softmax scores bigger than the true label's softmax score, 
    and then add the true label's softmax score.
    The score "s(x, y)" for the naive approach is then:
    The sum of all softmax scores higher than the true label's softmax score + the softmax score of the true label.
    """
    
    # Remember the softmax score of the true label.
    softmax_true_label = softmax_dist[true_label]

    score = 0

    # The score "s(input, true_label)" is equal to (the sum of all softmax scores bigger than the true label's softmax score) + (the true label's softmax score).
    for example in softmax_dist:
        if example >= softmax_true_label:
            score = score + example
    
    return score

# model = whole CNN model. labels = all possible labels for the input.  test_point = chosen new test point.   test_label = the true label of the chosen new test point.  alpha = If we want a 90% coverage, then alpha = 0.1 (since coverage = 1 - alpha).
def naive_appr(model, labels, test_point, alpha, test_label=None):  

    scores = np.zeros((len(labels,)))

    softmax_dist = model.predict(np.array([test_point]), verbose=0)[0] # (Taken from https://datascience.stackexchange.com/questions/13461/how-can-i-get-prediction-for-only-one-instance-in-keras)

    # Get the nonconformity scores for every softmax score predicted by the model for this new data point.
    for i in range(len(scores)):
        scores[i] = score_function(softmax_dist, i)

    #print("\nList of softmax scores:")
    #np.set_printoptions(precision=4, suppress=True)
    #print(model.predict(np.array([test_point]))[0])

    #print("\nList of nonconformity scores:")
    #print(scores)

    # Every label with a nonconformity score lower than the wanted coverage level should be a part of the prediction region.
    # This is the same as sorting every softmax score from highest to lowest, adding labels from highest to lowest until their combined softmax score > conf_level.
    conf_level = 1 - alpha
    sum = 0
    pred_region = {}
    # Go through each item in the scores list, add every score smaller than 1 - alpha.
    for i, score in enumerate(scores):
        if score <= conf_level:
            pred_region[labels[i]] = score
            sum = sum + score

    # Special case: If there are no nonconformity scores that are less than the confidence level, we just add the one with the lowest score.
    if sum == 0:
        i = np.argmin(scores)
        pred_region[labels[i]] = scores[i]

    print("\nPrediction Region (naive):")
    print(pred_region)

    if test_label is not None:  # If we have given some test label, then we can print it out.
        true_label = labels[int(test_label.item())]
        print(f"\nTrue label is: '{true_label}'.\n")

    return pred_region
    



#       1) Get a new test point
# Load CNN model + CIFAR-10 test set and normalize to match training preprocessing
base_model = tf.keras.models.load_model("cnn_softmax_model.keras")
(_, _), (test_images, test_labels) = datasets.cifar10.load_data()
test_images = test_images.astype("float32") / 255.0

# All possible labels for the CIFAR10 dataset.
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

image_nr = 3   # One image taken from the test images from the CIFAR10 dataset.

# Run naive approach to CP with the first of the test images and test labels.
naive_appr(base_model, class_names, test_images[image_nr], 0.1, test_labels[image_nr])



