import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets
from functions import evaluate_marg_coverage, evaluate_cond_coverage, evaluate_adaptivity, evaluate_efficiency


"""
No calibration data. 
Simply get a new test point, have the CNN Classifier get the softmax score for all possible labels, order the softmax scores from highest (most likely (according to model)) to lowest,
and add labels from highest softmax score to lowest until their combined softmax score is about the required coverage (For a 90% coverage, we want the combined softmax score to be ~0.9).

A score function would technically not be needed in this case, but we have one in order to evaluate certain metrics.
The score function for the naive approach is just that we get the softmax score of the given true label, 
and we let the nonconformity score be the sum of all softmax scores HIGHER than the true label's softmax score (including the true label's softmax score).
Thus, a higher nonconformity score would show that the model think it's very unlikely for the given true label to be the actual true label for the given input.

If we want 90% coverage (we guarantee that, on average, there is a 90% chance that the true label is in the prediction set), 
we can think of it as if we just take labels from highest to lowest such that their combined softmax score is ~0.9 (90%).
In practice, we let the model give a softmax score distribution for this new test point (one softmax score for each possible label),
then we get a nonconformity score for each of the softmax scores in that distribution. We do that by first pretending that the 1st softmax score is the softmax score of the "true label", 
and we run the score function to get a nonconformity score for that (we pretend the 1st possible label is the "true label").
Then we pretend that the 2nd softmax score is the softmax score of the "true label", and we do the same thing. We repeat this for every softmax score (where each softmax score corresponds to one label),
which results in us having one nonconformity score for each possible label.
Since the nonconformity score for each possible label is the same as the sum of all softmax scores higher than that label's softmax score, 
then we know that only the labels whose nonconformity scores are higher than 0.9 (if we want 90% coverage) would only be included in the prediction region if all other label's softmax scores higher than it would be in the prediction region too.
For example, if some label had a nonconformity score of 0.98 and we want a coverage of 0.9, then that combined softmax score might be the sum of softmax scores [0.5, 0.4, 0.08]. In that case, we only want to include the labels whose softmax scores are 0.5 and 0.4, NOT this label too!
For that reason, we only have to exclude all labels whose nonconformity scores are higher than the required coverage!
"""

# Given some softmax score distribution "x" for some data point, and that data point's true label "y", computes the "naive" score s(x, y) for this test point.
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
def naive_appr(softmax_dist, labels, alpha, test_label=None):
    """
    
    """ 
    scores = np.zeros((len(labels,)))

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
    q = 1 - alpha

    final_softmax_score = 1
    final_softmax_index = -1
    pred_region = {}
    # Go through each item in the scores list, add every score smaller than 1 - alpha.
    for i, score in enumerate(scores):
        if score <= q:
            pred_region[labels[i]] = score

        # SPECIAL CASE: We add labels whose combined softmax score does NOT exceed the required coverage "1 - alpha". However, there is a high risk that we end up not having actually REACHED the required coverage yet.
        # Therefore, we keep track of which label has the highest softmax score whose nonconformity score still exceeds the required coverage.
        elif score < final_softmax_score:
            final_softmax_index = i
            final_softmax_score = score

    # Special case: If there are no nonconformity scores that are less than the confidence level, we just add the one with the highest score.
    if not bool(pred_region):
        i = np.argmax(softmax_dist)     # We add the "most" probable label (according to the model) as the most likely true label.
        pred_region[labels[i]] = softmax_dist[i]

    # Special case: If the prediction region is NOT empty, but the labels' combined softmax scores do NOT exceed the threshold value "1 - alpha", 
    # then we must include the label among all excluded labels whose softmax score is the highest (since it would be the next label in the "ranking" of softmax scores from highest to lowest)
    elif max(pred_region.values()) < q:
        i = final_softmax_index
        pred_region[labels[i]] = final_softmax_score

    #print("\nPrediction Region (naive):")
    #print(pred_region)

    if test_label is not None:  # If we have given some test label, then we can print it out.
        true_label = labels[int(test_label.item())]
        print(f"\nTrue label is: '{true_label}'.\n")

    return pred_region