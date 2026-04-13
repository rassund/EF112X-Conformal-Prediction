import numpy as np

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

NAME = "conventional"

# NOTE: We could output a list of indexes, such that if we get a list of [2, 5, 6] then we know that labels with index 2, 5 and 6 should be a part of the Prediction Region.

# Given some softmax score distribution "x" for some data point, and that data point's true label "y", computes the "conventional" score s(x, y) for this test point.
def score_function(softmax_dist, true_label):
    # In several papers, such as pages 5 and 11 in "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification" by Anastasios et al.,
    # it is discussed that the nonconformity score for an example should be high if the chosen true label is very unlikely (according to the model) to be the actual true label.
    # This is why the nonconformity score for the conventional approach to CP must be "1 - softmax score of the true label", such that if the model gives the true label a smaller softmax score,
    # then the nonconformity score must be higher.
    return (1 - softmax_dist[true_label])

def threshold(alpha, calib_input, calib_label, score_function):
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
    return np.quantile(calib_probs, q_level / n, method='higher')