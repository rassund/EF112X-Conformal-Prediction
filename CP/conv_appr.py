import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets
from functions import evaluate_marg_coverage, evaluate_cond_coverage, evaluate_adaptivity, evaluate_efficiency


"""
Use calibration data. (CIFAR10 test images/labels, just a small part of them. We can then use one of the test images that were NOT chosen as calibration data to be used as a "new test point".)
To get the threshold value “q”, we let the model get the softmax score for the (as we know it) true label for each calibration data example 
(for example, if one example is (input, true label) = (image.png, “Dog”), then we let the model tell their softmax score for the label “Dog” for that image),
and we use it to compute the threshold value "q".
Since the threshold value must always be higher if the true label is more "nonconforming" to the given input 
    (i.e if the softmax score of the (as we know it) true label is very small, then the model thinks that label is "very unlikely" to be the true label / that label-input pairing is very "nonconforming"/"unlikely".),
which means that for the conventional approach, we cannot just choose the score to be the softmax score of the true label, we must choose it to be "1 - the softmax score of the true label".

When we get a new test point (we know the input, not the true label), we simply get the softmax scores for every label, get the nonconformity score for each softmax score 
    (for the first softmax score, we pretend that the 1st label is the "true label" and see what nonconformity score we get. For the second softmax score, we pretend the 2nd label is the true label etc...),
and we see which labels give nonconformity scores which are lower than the threshold value. These labels are "likely enough" (according to the model) for them to be included in our prediction region.
"""

# NOTE: We could output a list of indexes, such that if we get a list of [2, 5, 6] then we know that labels with index 2, 5 and 6 should be a part of the Prediction Region.

# Given some softmax score distribution "x" for some data point, and that data point's true label "y", computes the "conventional" score s(x, y) for this test point.
def score_function(softmax_dist, true_label):
    # In several papers, such as pages 5 and 11 in "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification" by Anastasios et al.,
    # it is discussed that the nonconformity score for an example should be high if the chosen true label is very unlikely (according to the model) to be the actual true label.
    # This is why the nonconformity score for the conventional approach to CP must be "1 - softmax score of the true label", such that if the model gives the true label a smaller softmax score,
    # then the nonconformity score must be higher.
    return (1 - softmax_dist[true_label])

# model = whole CNN model. labels = all possible labels for the input.  test_point = chosen new test point.   test_label = the true label of the chosen new test point.  alpha = If we want a 90% coverage, then alpha = 0.1 (since coverage = 1 - alpha).
def conv_appr(softmax_dist, labels, calib_input, calib_label, alpha, test_label=None):  

                # 1) Get the threshold value

    # We first get the calibration dataset, which typically consists of 1000 sample.
    #predictions = model.predict(calib_input, batch_size=32, verbose=0)

    calib_probs = []     # Contains the nonconformity scores given by the score function for each example.

    for i in range(len(calib_input)):  # For each calibration data example...
        # We add the nonconformity score for this calibration data example to the list of all calibration data scores.
        true_label = int(calib_label[i].item())    # Get the true label for this calibration data example.
        calib_probs.append(score_function(calib_input[i], true_label)) 

    #print("Calib_probs softmax scores:")
    #print(calib_probs)

    # If we want a 90% coverage, and we know that the nonconformity score is always higher the more "bad" of a guess for true label the model gives, 
    # then we want the threshold value "q" to be a value higher than 90% of all scores, i.e we want "q" to be in the 10th quantile of scores.
    # We compute this value, the threshold value "q", using the formula presented in Chapter 1 of the paper "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification" (Anastasios et. al)
    n = len(calib_probs)
    q_level = int(np.ceil((n + 1) * (1 - alpha)))
    q = np.quantile(calib_probs, q_level / n, method='higher')

    #np.set_printoptions(precision=4, suppress=True)
    #print("\nThreshold value 'q' is: ")
    #print(q)
    
    # We now have our threshold value "q", which is smaller than (1 - conf_level)*100 percent of all the values in calib_probs.


                # 2) Add labels into our prediction region.
    # Get the softmax distribution of the test point
    #softmax_dist = model.predict(np.array([test_point]), verbose=0)[0] # (Taken from https://datascience.stackexchange.com/questions/13461/how-can-i-get-prediction-for-only-one-instance-in-keras)

    # Now we add the labels whose nonconformity scores are lower than the threshold value "q". (Only the labels which the model deems "too unlikely" are excluded from the prediction region)
    pred_region = {}
    for i in range(len(softmax_dist)):  # Go through each softmax score.
        # We pretend that each possible label is the "true label", so that we can get the nonconformity score if we pretend the 1st softmax score is for the true label, then teh same for the 2nd softmax score, then the same for the 3rd...
        score = 1 - softmax_dist[i]     # Get the nonconformity score for this softmax score/label.
        if score <= q: 
            pred_region[labels[i]] = score
            

    # Special case: If there are no nonconformity scores that are less than the confidence level, we just add the one with the highest score.
    if not bool(pred_region):
        i = np.argmax(softmax_dist)     # We add the "most" probable label (according to the model) as the most likely true label.
        pred_region[labels[i]] = softmax_dist[i]

    #print("\nPrediction Region (conventional):")
    #print(pred_region)

    if test_label is not None:  # If we have given some test label, then we can print it out.
        true_label = labels[int(test_label.item())]
        print(f"\nTrue label is: '{true_label}'.\n")

    # Possibly print something about "the true label is not in the prediction region" (using the given argument 'test_label').

    return pred_region



#       1) Get a new test point
# Load CNN model + CIFAR-10 test set and normalize to match training preprocessing
base_model = tf.keras.models.load_model("CNN/cnn_softmax_model.keras")
(_, _), (test_images, test_labels) = datasets.cifar10.load_data()
test_images = test_images.astype("float32") / 255.0

# Get the first 1000 images + labels from the test data as calibration data.
#calibration_images = test_images[:1000]
#calibration_labels = test_labels[:1000]  

# All possible labels for the CIFAR10 dataset.
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

image_nr = 2000  # One image taken from the test images from the CIFAR10 dataset.
alpha = 0.1

# Run conventional approach to CP.
#conv_appr(base_model, class_names, calibration_images, calibration_labels, test_images[image_nr], alpha, test_labels[image_nr])

softmax_scores = base_model.predict(test_images, batch_size=32) # Get the softmax scores for all test images.
#calib_scores = base_model.predict(calibration_images, batch_size=32, verbose=0) # Get the softmax scores all calibration images.

# Evaluating marginal coverage
# Get the nonconformity scores for all test data, using this method's score function
scores = []
for i, example in enumerate(softmax_scores):
    scores.append(score_function(example, test_labels[i]))  # Get the nonconformity score for this example

n = 9000    # The CIFAR10 dataset contains 10 000 test images/labels. We use 9000 of them as "calibration data" when evaluating marginal coverage.
num_rounds = 10
alpha = 0.1
evaluate_marg_coverage(scores, num_rounds, n, alpha)

# Evaluate adaptivity & conditional coverage
num_of_labels = 10  # In the CIFAR10 dataset, we have 10 possible labels.
n = 5000
calib_input = softmax_scores[:n]  # The first "n" examples of "softmax_scores" (i.e the first "n" examples of "test_images") are calibration data inputs.
calib_label = test_labels[:n]     # The first "n" examples of "test_labels" are the true labels for each calibration data input above.

rest = len(softmax_scores) - n

val_input = softmax_scores[rest:]   # The last examples of "softmax_scores" are used as validation data examples.
val_label = test_labels[rest:]

evaluate_cond_coverage(score_function, calib_input, calib_label, val_input, val_label, alpha)
evaluate_adaptivity(score_function, num_of_labels, calib_input, calib_label, val_input, val_label, alpha)

evaluate_efficiency(conv_appr, softmax_scores, val_input, class_names, alpha, calib_input, calib_label)