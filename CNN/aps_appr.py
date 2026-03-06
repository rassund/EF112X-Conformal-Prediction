import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets
from functions import create_label_softmax_dict, sort_descending_softmax_dict


"""
Use calibration data. To get the threshold value “q”, for each calibration data example, we let the model get the softmax score for each label, 
and then we order those softmax scores from highest to lowest. 
Then, we see where in this “ranking” the true label for this calibration data example is. 
If the true label is, for example, the “35th most likely label to fit with the input” (according to the model), 
then we add together the softmax scores for the “1st most likely”, the 2nd most likely, the 3rd, the 4th, …, the 34th most likely and lastly the 35th most likely label. 
We get a “accumulative softmax mass”. We do this for every calibration data example.

If we want 90% coverage, then we pick a threshold value “q” such that “q” is bigger than ~90% of all the accumulative softmax masses from the calibration data.

When we get a new test point (we know the input, not the true label), we get the softmax scores for every label, 
order them from highest to lowest, and then we first pretend that the “1st most likely” label is the true label, 
and use the (above explained) score function to generate its accumulative softmax mass. 
Then we do the same for the 2nd most likely label, where we pretend that IT is the true label and we run the score function 
(adding together the softmax score for the 1st most likely label and the 2nd most likely label). 
Then we do the same for the 3rd most likely label, then the same for the 4th most likely label, etc…

Once we have all our accumulative softmax masses for the new test point, we just pick the labels with an accumulative softmax mass below the threshold value “q”.
"""


# NOTE: We could output a list of indexes, such that if we get a list of [2, 5, 6] then we know that labels with index 2, 5 and 6 should be a part of the Prediction Region.

# model = whole CNN model. labels = all possible labels for the input.  test_point = chosen new test point.   test_label = the true label of the chosen new test point.  conf_level = If we want a 90% coverage, then conf_level = 0.9.
def aps_appr(model, labels, calib_input, calib_label, test_point, test_label, conf_level):  

                # 1) Get the threshold value

    # We first get the calibration dataset, which typically consists of 1000 sample.
    predictions = model.predict(calib_input, batch_size=32)

    acc_softmax = []
    prob_dict = {}

    # For each calibration data example, we get the "accumulative softmax mass" by adding up the softmax score for every label BEFORE we reach the example's true label.
    for current_example_index, softmax_scores in enumerate(predictions):  # For each calibration data example...
        #... We first sort the softmax scores from highest to lowest (with their corresponding labels).
        prob_dict = create_label_softmax_dict(labels, softmax_scores)    # Create dictionary which pairs each softmax score with their corresponding label.
        prob_dict = sort_descending_softmax_dict(prob_dict)            # Sorts the probability dictionary in descending order.

        sum = 0

        # Now we add every softmax score until we get to the example's true label, and we add it to the "acc_softmax" array.
        true_label = labels[int(calib_label[current_example_index])]

        for current_label in prob_dict:  # Go through each item in the ordered dictionary
            if current_label == true_label:    # Once we find our true label in the softmax score ranking, we add its softmax score to the sum of softmax scores and then we're done.
                sum = sum + prob_dict[current_label]  
                break
            else:
                sum = sum + prob_dict[current_label]

        # After going through and adding up every softmax score before the true label, we add this "accumulative softmax mass" to the array "acc_softmax".
        acc_softmax.append(sum)

    #print("\nAccumulative softmax mass of the first 10 calib. data examples:")
    #print(acc_softmax[:10])


    # Now we get the threshold value "q". If we want a 90% coverage, then "q" must be higher than 90% of the values in "acc_softmax".
    # If we want a 90% coverage, then we want to find the value which is smaller than 90% of the values/ bigger than 10% of the values in calib_probs
    acc_softmax = np.array(acc_softmax)
    q = np.percentile(acc_softmax, conf_level*100)
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

    # Now we get the "accumulative softmax score" for each label in the ranking. In other words, we first pretend that the first label in the ranking would be the "true label" and we get the acc. softmax mass.
    #   Then we do the same for if the second label in the ranking would be the "true label". Then the same for every other label in the ranking.
    acc_softmax = []

    # Now we add every softmax score until we get to the example's true label, and we add it to the "acc_softmax" array.
    for item in prob_dict:
        true_label = item # We pretend the "true label" is first the first label in the ranking, then the second, then the third...
        sum = 0

        # We then get the accumulative softmax score (if we pretend that the currently-selected "true label" were to be the actual true label of the new test point).
        for current_label in prob_dict:
            if current_label == true_label:
                sum = sum + prob_dict[current_label]
                break
            else:
                sum = sum + prob_dict[current_label]

        acc_softmax.append(sum)

        # The first element in the "acc_softmax" array will now be for the first label in the ranking of the softmax_dist array. The second element -||- the second label in softmax_dist. etc...

    #print("\nAccumulative softmax score for this test point: \n{ ")
    #for i, item in enumerate(prob_dict):
    #    print("\t" + str(item) + " : " + str(acc_softmax[i]) + ", ")
    #print("}")

    # Now that we've gotten the accumulated softmax masses for the entire softmax distribution for this new test point, 
    # we can now add every such mass that has a lower value than the threshold value "q" into our prediction region.
    pred_region = []
    for i, item in enumerate(prob_dict):  # Go through each accumulated softmax mass for this test point.
        if acc_softmax[i] > q:    # For the APS approach, we only want the labels whose accumulated softmax score is lower than the threshold value.
            break
        else:
            pred_region.append(item)

    print("\nPrediction Region (APS):")
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

# Run conventional approach to CP.
aps_appr(base_model, class_names, calibration_images, calibration_labels, test_images[image_nr], test_labels[image_nr], 0.9)