import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets
from functions import evaluate_marg_coverage, evaluate_cond_coverage, evaluate_adaptivity, evaluate_efficiency

"""
(The same as the APS approach, just that we add a “penalization term” to the score function.)

Use calibration data. To get the threshold value “q”, for each calibration data example, we let the model get the softmax score for each label, 
and then we order those softmax scores from highest to lowest. 
Then, we see where in this “ranking” the true label for this calibration data example is. 
If the true label is, for example, the “35th most likely label to fit with the input” (according to the model), 
then we add together the softmax scores for the “1st most likely”, the 2nd most likely, the 3rd, the 4th, …, the 34th most likely and lastly the 35th most likely label.

AFTER THAT! We add a “penalization term” to the score, which gives a bigger score increase the “deeper in” the true label is in the ranking.
We get a “penalized accumulative softmax mass”. We do this for every calibration data example.

If we want 90% coverage, then we pick a threshold value “q” such that “q” is bigger than ~90% of all the penalized accumulative softmax masses from the calibration data.

When we get a new test point (we know the input, not the true label), we get the softmax scores for every label, order them from highest to lowest, 
and then we first pretend that the “1st most likely” label is the true label, and use the (above explained) score function to generate its accumulative softmax mass. 
Then we do the same for the 2nd most likely label, where we pretend that IT is the true label and we run the score function (adding together the softmax score for the 1st most likely label and the 2nd most likely label, then adding the penalization term). 
Then we do the same for the 3rd most likely label, then the same for the 4th most likely label, etc…

Once we have all our penalized accumulative softmax masses for the new test point, we just pick the labels with a penalized accumulative softmax mass below the threshold value “q”.
"""


# NOTE: We could output a list of indexes, such that if we get a list of [2, 5, 6] then we know that labels with index 2, 5 and 6 should be a part of the Prediction Region.

# Given some softmax probability distribution and the true label for this example, returns the nonconformity score for the RAPS approach (i.e returns the accumulative softmax mass of this input and softmax dist, including a penalization term.)
def score_function(softmax_dist, true_label):

    score = 0
    rank = 1

    # Remember the softmax score of the true label.
    softmax_true_label = softmax_dist[true_label]

    # 1) Get the sum of all softmax scores for all the labels before the true label in the ranking. 
    # Instead of having to sort the list of softmax scores, we can just add every softmax score that is HIGHER than the softmax score of the true label, into the score.
    for i in range(len(softmax_dist)):  # Go through each softmax score.
        if softmax_dist[i] > softmax_true_label:    # If we find some softmax score that is higher than the true label's softmax score, then that label would be "higher" in the ranking than the true label.
            score = score + softmax_dist[i]
            rank = rank + 1     # To see where in the "ranking" of "highest to lowest softmax scores" the true label would be for this example, we check how many softmax scores are higher than the true label's softmax score.

    # 2) Add the softmax score of the true label * u
    # According to the paper "Uncertainty Sets for Image Classifiers using Conformal Prediction" by Anastasios N. et. al, the accumulative softmax mass for each data point
    #   should follow this formula: "ρ_x(y) + ˆπ_x(y) · u", in which ρ_x(y) = the sum of the softmax scores for every label BEFORE the true label in the ranking, and ˆπ_x(y) = the softmax score of the true label.
    #   In this formula, we also have the argument "u", which is included to allow for randomized procedures. For each data-point, we let u be a i.i.d uniform [0, 1] random variable.
    #   (u seems to be there to help achieve marginal coverage. See https://arxiv.org/pdf/2205.05878 , chapter 2.2  and https://arxiv.org/html/2408.05037v1#bib.bib9 , chapter 2.)
    u = np.random.uniform(0, 1)
    score = score + softmax_true_label*u  

    # For each term, we also add a penalization term. According to the paper "Uncertainty Sets for Image Classifiers using Conformal Prediction" by Anastasios N. et. al, 
    #   the penalization term should be "λ · (o_x(y) − k_reg)^+", i.e λ times the positive part of o_x(y) − k_reg.
    #   λ = determines how strongly we penalize low-ranked labels. If λ is high, then give a big penalty for if the true label is in the "tail" of the softmax scores/probabilities. This often results in smaller prediction sets.
    #   k_reg = essentially says at which place in the ranking we should start adding the penalty. Because we only want the positive part of "o_x(y) − k_reg", that means that if o_x(y) < k_reg then (o_x(y) − k_reg)^+ = 0 meaning the penalty is 0.

    #   For this program, we experiment with different λ and k_reg values. For example, λ = 1 and k_reg = 5 are chosen in chapter 3.4: "Experiment 4: Adaptiveness of RAPS on Imagenet" in the paper "Uncertainty Sets for Image Classifiers using Conformal Prediction".
    penalization_factor = 1
    start_penalty_after_label = 5
    score = score + penalization_factor * max((rank - start_penalty_after_label), 0)    # Add penalization term. If rank - start_penalty_after_label is less than 0, then we instead have the penalty be 0.
            
    # 3) Output the score for this softmax distribution.
    return score


# model = whole CNN model. labels = all possible labels for the input.  test_point = chosen new test point.   test_label = the true label of the chosen new test point.  alpha = If we want a 90% coverage, then alpha = 0.1 (since coverage = 1 - alpha).
def raps_appr(softmax_dist, labels, calib_input, calib_label, alpha, test_label=None):  

                # 1) Get the threshold value

    # We first get the calibration dataset, which typically consists of 1000 sample.
    #calib_softmax_dist = model.predict(calib_input, batch_size=32, verbose=0)

    calib_scores = []     # Contains the accumulates softmax masses (given by the score function) for all of the calibration data examples.

    # For each calibration data example, we get the "accumulative softmax mass" by adding up the softmax score for every label BEFORE we reach the example's true label.
    for i in range(len(calib_input)):  # For each calibration data example...
        # We add the score (accumulative softmax mass) for this calibration data example to the list of all calibration data scores.
        true_label = int(calib_label[i].item())    # Get the true label for this calibration data example.
        calib_scores.append(score_function(calib_input[i], true_label))          

    #print("\nAccumulative softmax mass of the first 10 calib. data examples:")
    #print(calib_scores[10::])


    # If we want a 90% coverage, and we know that the nonconformity score is always higher the more "bad" of a guess for true label the model gives, 
    # then we want the threshold value "q" to be a value higher than 90% of all scores, i.e we want "q" to be in the 10th quantile of scores.
    # We compute this value, the threshold value "q", using the formula presented in Chapter 1 of the paper "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification" (Anastasios et. al)
    n = len(calib_scores)
    q_level = int(np.ceil((n + 1) * (1 - alpha)))
    q = np.quantile(calib_scores, q_level / n, method='higher')
    #print(f"\nThreshold value 'q' is: {q}")


                # 2) Add labels into our prediction region.
    # Get the softmax distribution of the test point
    #softmax_dist = model.predict(np.array([test_point]), verbose=0)[0] # (Taken from https://datascience.stackexchange.com/questions/13461/how-can-i-get-prediction-for-only-one-instance-in-keras)

    #print("\nSoftmax Probability distribution:")
    #print(softmax_dist)

    # Now we get the "accumulative softmax score" for each label in the softmax distribution. In other words, we first pretend that the first label in the ranking would be the "true label" and we get the acc. softmax mass.
    #   Then we do the same for if the second label in the ranking would be the "true label". Then the same for every other label in the ranking.
    scores = []

    # Now we add every softmax score until we get to the example's true label, and we add it to the "acc_softmax" array.
    for i in range(len(softmax_dist)):
        # We pretend the "true label" is first the first label in the ranking, then the second, then the third...
        scores.append(score_function(softmax_dist, i))

        # The first element in the "acc_softmax" array will now be for the first label in the ranking of the softmax_dist array. The second element -||- the second label in softmax_dist. etc...

    #print("\nAccumulative softmax mass for this test point: \n{ ")
    #for i in range(len(softmax_dist)):
    #    print(f"\t Label {i} : {scores[i]}, ")
    #print("}")

    # Now that we've gotten the accumulated softmax masses for the entire softmax distribution for this new test point, 
    # we can now add every such mass that has a lower value than the threshold value "q" into our prediction region.
    pred_region = {}
    for i in range(len(softmax_dist)):  # Go through each accumulated softmax mass for this test point.
        if scores[i] <= q:    # For the APS approach, we only want the labels whose accumulated softmax mass (score) is lower than the threshold value.
            pred_region[labels[i]] = scores[i]

    # Special case: If there are no nonconformity scores that are less than the confidence level, we just add the one with the lowest score.
    if not bool(pred_region):
        i = np.argmin(scores)
        pred_region[labels[i]] = scores[i]

    #print("\nPrediction Region (RAPS):")
    #print(pred_region)

    if test_label is not None:  # If we have given some test label, then we can print it out.
        true_label = labels[int(test_label.item())]
        print(f"\nTrue label is: '{true_label}'.\n")

    return pred_region


#       1) Get a new test point
# Load CNN model + CIFAR-10 test set and normalize to match training preprocessing
base_model = tf.keras.models.load_model("CNN/cnn_softmax_model.keras")
(_, _), (test_images, test_labels) = datasets.cifar10.load_data()
test_images = test_images.astype("float32") / 255.0

# Get the first 1000 images + labels from the test data as calibration data.
calibration_images = test_images[:1000]
calibration_labels = test_labels[:1000]  

# All possible labels for the CIFAR10 dataset.
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

image_nr = 2000  # One image taken from the test images from the CIFAR10 dataset.

# Run RAPS approach to CP.
#raps_appr(base_model, class_names, calibration_images, calibration_labels, test_images[image_nr], 0.1, test_labels[image_nr])


softmax_scores = base_model.predict(test_images, batch_size=32) # Get the softmax scores for all test images.

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

evaluate_efficiency(raps_appr, softmax_scores, val_input, class_names, alpha, calib_input, calib_label)