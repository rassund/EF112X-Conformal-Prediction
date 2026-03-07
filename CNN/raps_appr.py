import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets
from functions import create_label_softmax_dict, sort_descending_softmax_dict


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

# Given some sorted softmax score distribution (from highest to lowest) and the true label for this example, returns the score for the APS approach (i.e returns the accumulative softmax mass of this input and softmax dist.)
# NOTE: Right now, the "softmax_dist" must be a dictionary, since we map the correct label as a string onto the correct softmax score. 
#   Later on, we can change some stuff such that we create the dictionary here, where the first softmax score gets label "0", the next gets label "1" etc... And then we can just exchange the label index for the actual label name. (i.e label "0" should be label "airplane"...)
def score_function(true_label, ordered_softmax):

    score = 0

    # 1) Get the sum of all softmax scores for all the labels before the true label in the ranking.
    for i, current_label in enumerate(ordered_softmax):  # Go through each item in the ordered dictionary
        # 2) Add the softmax score of the true label * u
        if current_label == true_label:    # Once we find our true label in the softmax score ranking, we add its softmax score to the sum of softmax scores and then we're done.

            # According to the paper "Uncertainty Sets for Image Classifiers using Conformal Prediction" by Anastasios N. et. al, the accumulative softmax mass for each data point
            #   should follow this formula: "ρ_x(y) + ˆπ_x(y) · u", in which ρ_x(y) = the sum of the softmax scores for every label BEFORE the true label in the ranking, and ˆπ_x(y) = the softmax score of the true label.
            #   In this formula, we also have the argument "u", which is included to allow for randomized procedures. For each data-point, we let u be a i.i.d uniform [0, 1] random variable.
            #   (u seems to be there to help achieve marginal coverage. See https://arxiv.org/pdf/2205.05878 , chapter 2.2  and https://arxiv.org/html/2408.05037v1#bib.bib9 , chapter 2.)
            u = np.random.uniform(0, 1)
            score = score + ordered_softmax[current_label]*u  

            # For each term, we also add a penalization term. According to the paper "Uncertainty Sets for Image Classifiers using Conformal Prediction" by Anastasios N. et. al, 
            #   the penalization term should be "λ · (o_x(y) − k_reg)^+", i.e λ times the positive part of o_x(y) − k_reg.
            #   λ = determines how strongly we penalize low-ranked labels. If λ is high, then give a big penalty for if the true label is in the "tail" of the softmax scores/probabilities. This often results in smaller prediction sets.
            #   k_reg = essentially says at which place in the ranking we should start adding the penalty. Because we only want the positive part of "o_x(y) − k_reg", that means that if o_x(y) < k_reg then (o_x(y) − k_reg)^+ = 0 meaning the penalty is 0.

            #   For this program, we choose λ = 1 and k_reg = 5, as is used in chapter 3.4: "Experiment 4: Adaptiveness of RAPS on Imagenet" in the paper "Uncertainty Sets for Image Classifiers using Conformal Prediction".
            penalization_factor = 1
            start_penalty_after_label = 5
            rank = i+1  # If the true label is the first label in the ranking, then the rank should be 1.
            score = score + penalization_factor * max((rank - start_penalty_after_label), 0)    # Add penalization term. If rank - start_penalty_after_label is less than 0, then we instead have the penalty be 0.
            break
        else:
            score = score + ordered_softmax[current_label]

    # 3) Output the score for this softmax distribution.
    return score


# model = whole CNN model. labels = all possible labels for the input.  test_point = chosen new test point.   test_label = the true label of the chosen new test point.  conf_level = If we want a 90% coverage, then conf_level = 0.9.
def raps_appr(model, labels, calib_input, calib_label, test_point, test_label, conf_level):  

                # 1) Get the threshold value

    # We first get the calibration dataset, which typically consists of 1000 sample.
    predictions = model.predict(calib_input, batch_size=32)

    scores = []     # Contains the accumulates softmax masses (given by the score function) for all of the calibration data examples.

    # For each calibration data example, we get the "accumulative softmax mass" by adding up the softmax score for every label BEFORE we reach the example's true label.
    for current_example_index, softmax_scores in enumerate(predictions):  # For each calibration data example...
        # We run the score function for each calibration data example.
        prob_dict = create_label_softmax_dict(labels, softmax_scores)   # Create dictionary which pairs each softmax score with their corresponding label.
        prob_dict = sort_descending_softmax_dict(prob_dict)            # Sorts the probability dictionary in descending order.
        true_label = labels[int(calib_label[current_example_index])]    # Get the true label for this calibration data example.
        scores.append(score_function(true_label, prob_dict))            # Add the score (accumulative softmax mass) for this calibration data example to the list of all calibration data scores.

    #print("\nAccumulative softmax mass of the first 10 calib. data examples:")
    #print(acc_softmax[:10])


    # Now we get the threshold value "q". If we want a 90% coverage, then "q" must be higher than 90% of the values in "acc_softmax".
    # If we want a 90% coverage, then we want to find the value which is smaller than 90% of the values/ bigger than 10% of the values in calib_probs
    scores = np.array(scores)
    q = np.percentile(scores, conf_level*100)
    print("\nThreshold value 'q' is: ")
    print(q)


                # 2) Add labels into our prediction region.
    # Get the softmax distribution of the test point
    softmax_dist = model.predict(np.array([test_point]), verbose=0)[0] # (Taken from https://datascience.stackexchange.com/questions/13461/how-can-i-get-prediction-for-only-one-instance-in-keras)

    # We sort the softmax scores from highest to lowest, just like when we computed the threshold value.
    prob_dict = create_label_softmax_dict(labels, softmax_dist)    # Create dictionary which pairs each softmax score with their corresponding label.
    prob_dict = sort_descending_softmax_dict(prob_dict)            # Sorts the probability dictionary in descending order.
    print("\nProbability distribution (in descending order):")
    print(prob_dict)

    # Now we get the "accumulative softmax score" for each label in the softmax distribution. In other words, we first pretend that the first label in the ranking would be the "true label" and we get the acc. softmax mass.
    #   Then we do the same for if the second label in the ranking would be the "true label". Then the same for every other label in the ranking.
    scores = []

    # Now we add every softmax score until we get to the example's true label, and we add it to the "acc_softmax" array.
    for item in prob_dict:
        true_label = item # We pretend the "true label" is first the first label in the ranking, then the second, then the third...
        scores.append(score_function(true_label, prob_dict))

        # The first element in the "acc_softmax" array will now be for the first label in the ranking of the softmax_dist array. The second element -||- the second label in softmax_dist. etc...

    #print("\nAccumulative softmax score for this test point: \n{ ")
    #for i, item in enumerate(prob_dict):
    #    print("\t" + str(item) + " : " + str(acc_softmax[i]) + ", ")
    #print("}")

    # Now that we've gotten the accumulated softmax masses for the entire softmax distribution for this new test point, 
    # we can now add every such mass that has a lower value than the threshold value "q" into our prediction region.
    pred_region = []
    for i, item in enumerate(prob_dict):  # Go through each accumulated softmax mass for this test point.
        if scores[i] > q:    # For the APS approach, we only want the labels whose accumulated softmax score is lower than the threshold value.
            break
        else:
            pred_region.append(item)

    print("\nPrediction Region (RAPS):")
    print(pred_region)

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

# Run RAPS approach to CP.
raps_appr(base_model, class_names, calibration_images, calibration_labels, test_images[image_nr], test_labels[image_nr], 0.9)