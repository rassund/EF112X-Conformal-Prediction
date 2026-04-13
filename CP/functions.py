import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets
from collections import defaultdict

def cifar10_get_softmax_dists(image_nr): # Is this used?
    # 1) Load your trained CNN
    #base_model = tf.keras.models.load_model("cnn-tensorflow_model.keras")
    base_model = tf.keras.models.load_model("cnn_softmax_model.keras")

    # 2) Load CIFAR-10 test set and normalize to match training preprocessing
    (_, _), (test_images, test_labels) = datasets.cifar10.load_data()
    test_images = test_images.astype("float32") / 255.0

    #predictions = probability_model.predict(test_images, batch_size=256)
    predictions = base_model.predict(test_images, batch_size=256)

    # 3) Show the softmax distribution for this test image
    np.set_printoptions(precision=4, suppress=True)
    print("Probability dist. for test image " + str(image_nr) + ": ", predictions[image_nr], "\nsum:", predictions[image_nr].sum())



    # Just for testing, we can see the probability distribution for the different test images
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Softmax distribution and true label for the given image.
    softmax_dist = predictions[image_nr]
    true_label_index = test_labels[image_nr][0]

    plt.figure(figsize=(14,6))
    
    # Show the chosen test image
    plt.subplot(1,2,1)
    plt.imshow(test_images[image_nr])
    plt.title(f"Test image #{image_nr}\nTrue label: {class_names[true_label_index]}")
    plt.axis("off")


    # Show the probability distribution
    plt.subplot(1,2,2)
    bars = plt.bar(class_names, softmax_dist)
    plt.title("Softmax probabilities")
    plt.ylim([0, 1])


    # Highlight the predicted class
    predicted_label = np.argmax(softmax_dist)
    bars[predicted_label].set_color('red')

    # Show the exact probability above each bar
    for i, p in enumerate(softmax_dist):
        plt.text(i, p + 0.02, f"{p:.2f}", ha='center')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def cifar10_per_class_acc(): # Is this used?
    # 1) Load your trained CNN
    #base_model = tf.keras.models.load_model("cnn-tensorflow_model.keras")
    model = tf.keras.models.load_model("testing.keras")

    # 2) Load CIFAR-10 test set and normalize to match training preprocessing
    (_, _), (test_images, test_labels) = datasets.cifar10.load_data()
    test_images = test_images.astype("float32") / 255.0

    #predictions = probability_model.predict(test_images, batch_size=256)
    predictions = model.predict(test_images, batch_size=256)
    examples_with_correct_guess = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    examples_per_class = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Get the label with the highest softmax score for each test image example.
    highest_softmax_label = np.argmax(predictions, axis=1)

    for i in range(len(predictions)):
        # Get the softmax score for the true label for this example and add it to the true label index in per_class_acc
        true_label = int(test_labels[i])
        examples_per_class[true_label] += 1     # We've found one more example with this true label!
        # Is the label of the highest softmax score for this example the same as the true label?
        if highest_softmax_label[i] == true_label:
            examples_with_correct_guess[true_label] += 1    # Then we know that the model has predicted the true label correctly for this example!

    #print("\nExample statistics:")
    #for i in range(10):
    #    print(f"Label {i} has a total of {examples_per_class[i]} examples, with a total of {examples_with_correct_guess[i]} correct guesses.")

    # Get the average softmax score for each label.
    for i in range(10):
        accuracy_per_class = examples_with_correct_guess[i] / examples_per_class[i]
        print(f"Class {i}: {accuracy_per_class:.4f}")

def create_prediction_set(score_function, threshold, softmax_dist, labels, test_label=None):
    # Add labels into our prediction region.

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
    # we can now add every such mass that has a lower value than the threshold value into our prediction region.
    pred_region = {}
    for i in range(len(softmax_dist)):  # Go through each accumulated softmax mass for this test point.
        if scores[i] <= threshold:    # We only want the labels whose accumulated softmax mass (score) is lower than the threshold value.
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

# Given a set of scores (given by a score function by some conformal prediction method), a number of "rounds" to go through, the number of calibration data samples "n" (among the set of scores), 
# and a value "alpha" (in which the desired coverage is "1 - alpha"), this function computes the marginal coverage for these scores.
def evaluate_marg_coverage(scores, num_rounds, n, alpha):
    '''
    Evaluate marginal coverage (or validity). We check if the method produces prediction sets with a mathematical guarantee of marginal coverage 
    (first check for 90% coverage, then 95% coverage, then 99% coverage)

    On page 15 of "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification", 
        it is discussed how you can check for correct coverage by calculating the "empirical coverage" for several trials 
        (where each trial is a running of the method with new calibration and validation sets)
    Some python code is also given in the text.

    The equation shared on page 15:
        C_j = 1/n_val * (sum from i = 1 to i = n_val): 1{(Y_i,j)^(val)  ∈ C_j ((X_i,j)^(val)) }, for j = 1, ..., R,
    seems to just show that you can calculate marginal coverage by having several sets ("R" sets, in which "n_val" = R) of calibration and validation sets, 
    running the CP method on them, adding up all the times that the true label actually is IN the produced prediction region,
    and then dividing by the number of sets of calibration/validation sets (R, or n_val) to see how which percentage of these sets in which the prediction region has the true label.
    If we want 90% coverage, then we expect the true label to be in the prediction region in about 90% of these R sets.

    The paper then explains how we can do this process realistically on an actual dataset:
    With real datasets, we only have some number of data points, which are divided into "n" regular data points and "n_val" validation data points 
        (meaning we have a total of "n + n_val" data points).
    This makes it hard to draw new data for each of the R "rounds", which is why we instead can just take all the calibration and validation datapoints
    for a real dataset, get the scores of all those datapoints, and then we can redo one process R times:
        1. Shuffle the entire list of scores (gotten from all the calibration and validation data points)
        2. Take the first "n" data points in this new shuffled list of scores as calibration data, and take the remaining scores as validation data.
        3. Get the threshold value "q" for the calibration data scores.
        4. Using the threshold value "q", save the empirical coverage for this round.
    After doing this R times/rounds, we should have R empirical coverage values. Using this, we see what the average coverage is, and we should see that it is about 1 - alpha.



    True label is covered <===> s(x, y_true) <= q
    Let's break down "coverages[r] = (val_scores <= qhat).astype(float).mean()":
        We want to find which percentage of validation data examples produce a prediction region which contains the TRUE LABEL. But we don't include any true labels as an argument to this function, so how can we check if the true label is covered?
        Well, for this function, we accept a set of SCORES. How do we get these scores? We put some input and ITS TRUE LABEL into a score function, and it gives a score which kind of represents how much the model thinks the input and the true label fit together.
        In this way, as long as we use nonconformity scores, we in a way actually DO check which true labels are covered in the prediction region (by checking which of these scores, which actually represent the true label, would be a part of the prediction region).

        To be more technical:
        val_scores <= qhat   returns a True or a False value. If we convert this into floats, we get a "1" for true, and a "0" for false.
        So we go through all the validation scores (i.e the scores which are s(x, y_true) i.e the score for the validation example's true label)
        and check which validation scores (i.e true label scores) would be a part of the prediction region (gives "1"), and which would not (gives "0").
    '''

    # Taken from "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification" by Anastasios et. al.

    # We know that we will get "R" (or in this case, "num_rounds") empirical coverages, so we initiate a list with "R" indexes, all with an initial value of 0.
    coverages = np.zeros((num_rounds,))

    # We go through R rounds, shuffle all the scores, choose some scores as calibration data scores, 
    # get the correct threshold value "q" (1-alpha:th percentile of calibration data scores) and then 
    for r in range(num_rounds):
        np.random.shuffle(scores) # We shuffle the scores to get a "new" set of calibration and validation scores.
        calib_scores, val_scores = (scores[:n],scores[n:]) # We choose the "n" first scores as calibration data scores, the rest are validation scores.

        # NOTE: Corrected? In research paper, we round up "(n + 1) * (1 - alpha)/n" to the nearest integer. From testing, we seem to get more accurate threshold values by rounding up "(n + 1) * (1 - alpha)" and dividing by n.
        q_level = int(np.ceil((n + 1) * (1 - alpha)))
        q = np.quantile(calib_scores, q_level/n, method='higher') # We get the threshold value "q", which is the value at the "1-alpha":th quantile of the calibration data scores.

        # NOTE: Taken right from research paper "A Gentle Introduction..." by Anastasios et. al
        #q = np.quantile(calib_scores, np.ceil((n + 1) * (1 - alpha)/n), method='higher') # We get the threshold value "q", which is the value at the "1-alpha":th quantile of the calibration data scores.

        print(f"\nRound {r+1}: Threshold value: {q}.")
        coverages[r] = (val_scores <= q).astype(float).mean()
        print(f"Coverage: {coverages[r]}.")
    print(f"\nAverage coverage = {coverages.mean()}.\n") # should be close to 1-alpha
    print(f"Expected coverage = {1 - alpha}.\n")

    # NOTE: We should probably make this histogram look nicer...
    plt.hist(coverages) # should be roughly centered at 1-alpha
    plt.show()

def evaluate_efficiency(score_function, threshold, softmax_dist, test_images, labels):
    '''
    Evaluate efficiency, i.e the average set size. The smaller the set sizes on average, the more efficient the method is.

    In "UNCERTAINTY SETS FOR IMAGE CLASSIFIERS USING CONFORMAL PREDICTION" just before chapter 2.2 (on page 5), 
    it is discussed a formula for an "upper bound" for how much coverage a prediction set can have.

    IDEA: We can evaluate marginal coverage, i.e check how much coverage the prediction set actually achieves.
    The closer the actual marginal coverage is to the wanted marginal coverage, the better the efficiency of the method.

    IDEA: Create a bunch of sets, find their mean size for different choices of alpha.
    "UNCERTAINTY SETS FOR IMAGE CLASSIFIERS USING CONFORMAL PREDICTION" does 100 trials of each procedure for two different choices of alpha.
    The median-of-means is then taken to determine the average set size.
    '''
    # For each test image, compute the prediction set and record the size
    # Split the data into 20 groups and take the mean of each group, then return the median of the means
    set_sizes = [len(create_prediction_set(score_function, threshold, softmax_dist[i], labels)) for i in range(len(test_images))]
    # Calculate mean
    print(f"Mean: {np.mean(set_sizes)}")
    # Calculate tail, aka worst-case
    print(f"90th percentile: {np.percentile(set_sizes, 90)}")
    # Calculate median-of-means
    np.random.shuffle(set_sizes)
    groups = np.array_split(set_sizes, 20)
    means = [group.mean() for group in groups]
    median = np.median(means)
    print(f"Median-of-means: {median}")
    return median

# Given a call to a score function, a set of all possible labels, some "alpha" value, 
# some "input & true label" pairs for calibration data and some "input & true label" pairs for validation data,
# this function evaluates conditional coverage for the given data and the given score function using FSC and CovGap metrics.
def evaluate_cond_coverage(score_function, calib_input, calib_label, val_input, val_label, alpha):
    '''
    Evaluate conditional coverage. See how close the prediction sets are towards achieving conditional coverage. 
    NOTE: By asking for the conditional coverage, we can also formalize the adaptivity of each method.

    We measure this by checking the FSC (Feature-stratified coverage metric) for the chosen method.
    On page 13 of "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification", 
        it is discussed how FSC can be used to check how close the method is to achieving conditional coverage.

    From "A Gentle Introduction...", the correct implementation seems to be:
    1. Divide the evaluation data into some K groups (so we have groups G_1, G_2, ..., G_K). Each group in turn contains a set of validation data examples, along with the true label for each such example.
    2. For each group, you get the prediction region for each example and count how many of them contains the example's true label. We then divide this sum with the total amount of examples in this group.
    3. After we have now gotten the empirical coverage for each group, we then return the lowest empirical coverage among all the groups.
    The closer this "minimum group-wise empirical coverage" is to 1 - alpha, the closer the method is to achieving conditional coverage. 


    According to the paper, a first step would be to plot histograms of the prediction set sizes. We want to check for adaptivity: 
    If all prediction sets have about equal sizes, then the CP method might have bad adaptivity since the prediction regions do not get much smaller if the model gives good guesses (and do not get much larger if the model gives bad guesses).
    If all prediction sets have very varying sizes (if we have 10 labels, then some sets might have 2 label, some might have 6, some might have 9, etc...), then the method might have good adaptivity.
    HOWEVER: To check adaptivity, this is often not enough. We can now with this whether or not the prediction sets have dynamic sizes, but we still "need to verify that that large sets occur for hard examples" (taken from paper).


    Good source: https://arxiv.org/html/2512.11779v1

    OBS!: It seems best to use SSC together with FSC to evaluate conditional coverage. 
    In SSC, we divide the examples into groups in which: 
        Group 1. The prediction set contains 1 element.
        Group 2. The prediction set contains 1 < x =< [total number of labels / 2] elements.
        Group 3. The prediction set contains > [total number of labels / 2] elements.
    So we get the prediction regions for each example, group them together, then check how many contains the true label, in order to check adaptivity more easily (so we check "When the method gives me a small (or large) set, can I trust it?").

    In the paper: "https://arxiv.org/pdf/2501.10139",
    the authors say that they use the CovGap method to evaluate conditional coverage. 
    According to the paper, it seems like they group the test data into several "groups" based on some feature 
    (where they do 2 groupings which both start with grouping the test data according to how confident their number 1 "most likely label" is for each example, 
    and 1 grouping where each group consists of examples who share the same true label). 
    For the CovGap method, it seems like they describe it as them first getting the mean empirical coverage of each group, 
    then for each group they check how far off that is from perfect conditional coverage 
    (so if the mean empirical coverage across the group is 91% and we want 90% conditional coverage, then there is a "difference" ^c_b = 1%), 
    then we get the "mean difference" for all groups.

    So we can use SSC along with FSC (where we group examples on confidence, and then group on true labels), then also CovGap (where we group in the same way).
    I.e, we group examples on confidence by selecting:
        Group 1: Highest softmax score for example is >= 0.9.
        Group 2: Highest softmax score for example is 0.9 < x =< 0.5
        Group 3: Highest softmax score for example is < 0.5
    
    Then we group examples on true label by selecting:
        Group 1: Contains only examples whose true label is "label 1".
        Group 2: Contains only examples whose true label is "label 2"
        etc...
    '''

    # Compute FSC and CovGap
    # Do this BEFORE we get any scores. So we want to have a list of softmax scores for each test data example, so we can group them on confidence and true label.

    # For FSC and CovGap, we have 1 set of groups based on the "confidence" of the model, i.e the highest softmax score for the example.
    # Group 1: max(softmax_scores) >= 0.9     Group 2: 0.9 > max(softmax_scores) >= 0.5.     Group 3: 0.5 > max(softmax_scores)
    conf_group = {
        1 : [],
        2 : [],
        3 : []
    }

    # For FSC and CovGap, we also have 1 set of groups based on their true label. 
    # Group 1: all examples who have true label "label 1". Group 2: All examples who have true label "label 2", etc...
    tlabel_group = defaultdict(list)    # As recommended in https://stackoverflow.com/questions/11509721/how-do-i-initialize-a-dictionary-of-empty-lists-in-python

    val_scores = []

    # Get all examples in "val_input" into their respective "conf_group" group and "tlabel_group" group, and get the nonconformity scores for each 
    for i, example in enumerate(val_input):
        highest = max(example)  # Get the highest softmax score for this example.

        # We add the index of this example in "calib_input" to represent this example, into the correct "conf_group" group.
        if highest >= 0.9:      # If this example's highest softmax score is equal to or higher than 0.9, then we add it to the first group.
            conf_group[1].append(i)
        elif highest >= 0.5:    # If this example's highest softmax score is not higher than 0.9 but is higher than 0.5, we add it to the second group.
            conf_group[2].append(i)
        else:                   # If example's -||- is not higher than 0.5, we add it to the second group.
            conf_group[3].append(i)
        
        # We also add the index of this example in "calib_input" into the correct "tlabel_group" group.
        label = val_label[i].item() # Might be a temporary fix, since specifically the CIFAR10 dataset gives val_label examples as shape (N, 1) (and not (N,)), so we need to extract the "value" of the label first.
        tlabel_group[label].append(i)

        # We also get the nonconformity score for each validation example.
        true_label = int(val_label[i].item())
        val_scores.append(score_function(val_input[i], true_label))


    # We can also get the threshold value for the given score function.
    calib_scores = []
    for i in range(len(calib_input)):  # For each calibration data example...
        # We add the nonconformity score for this calibration data example to the list of all calibration data scores.
        true_label = int(calib_label[i].item())    # Get the true label for this calibration data example.
        calib_scores.append(score_function(calib_input[i], true_label))

    # See "evaluate_marg_coverage()"
    n = len(calib_scores)
    q_level = int(np.ceil((n + 1) * (1 - alpha)))
    threshold = np.quantile(calib_scores, q_level/n, method='higher') # We get the threshold value "q", which is the value at the "1-alpha":th quantile of the calibration data scores.

    
    print("\nNumber of examples in each group:")
    for i, group in enumerate(conf_group):
        print(f"Group {i+1} in conf_group has {len(conf_group[group])} examples.\n")
    for i, group in enumerate(tlabel_group):
        print(f"Group {i+1} in tlabel_group has {len(tlabel_group[group])} examples.\n")

    # Now that we have all our groups, we can evaluate FSC and CovGap.
    print(f"\nConfidence-based FSC gives a lowest mean empirical coverage of {evaluate_fsc(conf_group, threshold, val_scores)}.\n")
    print(f"Class-based FSC gives a lowest mean empirical coverage of {evaluate_fsc(tlabel_group, threshold, val_scores)}.\n")

    # NOTE: Here I should write my calls to "evaluate_covgap()" once I'm done with them
    print(f"\nConfidence-based CovGap gives an average of differences of {evaluate_covgap(conf_group, threshold, val_scores, alpha)}.\n")
    print(f"Class-based CovGap gives an average of differences of {evaluate_covgap(tlabel_group, threshold, val_scores, alpha)}.\n")
    return

# Given a call to a score function, a set of all possible labels, some "alpha" value, 
# some "input & true label" pairs for calibration data and some "input & true label" pairs for validation data,
# this function evaluates the adaptivity for the given data and the given score function by giving a histogram of the different prediction set sizes, and through the SSC metric.
def evaluate_adaptivity(score_function, threshold, num_of_labels, calib_input, calib_label, val_input, val_label, alpha):
    # After we've gotten the threshold value, we can to find out which validation data examples should be in which group.
    val_scores = []

    '''
        Group 1. The prediction set contains 1 element.
        Group 2. The prediction set contains 1 < x =< [total number of labels / 2] elements.
        Group 3. The prediction set contains > [total number of labels / 2] elements.
    '''
    size_group = {
        1 : [],
        2 : [],
        3 : []
    }

    # To see how many prediction sets of each size there is, we use the below array.
    pred_set_sizes = np.zeros(num_of_labels)   # A prediction set can have a maximum of "len(labels)" labels inside it, and a minimum of 1 label (or technically 0, but that should never happen)

    for i, example in enumerate(val_input):

        # We assemble the prediction set to find out how many elements would be in the example's prediction set.
        # We first get the nonconformity score for each label.
        scores = np.array([score_function(val_input[i], label) for label in range(len(example))])
        #print(scores)

        # After we've gotten a nonconformity score for each label for this example, we see which ones would be in the produced prediction region.
        # We get the amount of scores that are below the threshold value, i.e the scores for all labels that would be in the prediction set.
        num_of_labels_in_pred_set = (scores <= threshold).sum() # Goes through all scores in "scores". Every score that is lower than "threshold" makes it so the expression = True = 1. Otherwise, the expression = False = 0.
        #print(num_of_labels_in_pred_set)
        # In order to show a histogram of the prediction set sizes for all examples, we add "+1" to the "pred_set_sizes" array, with the index being the length of the prediction set.
        pred_set_sizes[num_of_labels_in_pred_set - 1] += 1    # If we have 1 label in the prediction set, that would be put into index "0" in the "pred_set_sizes" array.

        # Now we can put this example into the correct group in "size_group".
        if num_of_labels_in_pred_set == 1:  # If the pred. set only contains only 1 label, we put it into group 1.
            size_group[1].append(i)

        elif num_of_labels_in_pred_set <= np.ceil(num_of_labels/2):   # If the pred. set contains from 2 to [total number of labels / 2] labels, we put it into group 2.
            size_group[2].append(i)

        else:
            size_group[3].append(i) # Otherwise, if the pred. set contains more than [total number of labels / 2] labels, we put it into group 3.
        
        # We also save the softmax score of the actual true label into "val_scores".
        true_label = int(val_label[i].item())
        val_scores.append(scores[true_label])

    print("\nNumber of examples in each group:")
    for i, group in enumerate(size_group):
        print(f"Group {i+1} in size_group has {len(size_group[group])} examples.\n")

    # Once we've gone through every example and added them into the correct group, we can now get the SSC metric for this data (which is the same as the FSC metric, just that we use "size_group"...)
    print(f"\nSSC gives a lowest mean empirical coverage of {evaluate_fsc(size_group, threshold, val_scores)}.\n")

    # We can also show a histogram with the sizes of all validation example prediction sets, to see if there are varying sizes of prediction sets (which is required for good adaptivity)
    # NOTE: pred_set_sizes gives the amount of examples with prediction sets of different lengths. 
    #   pred_set_sizes[0] gives the amount of examples whose pred. sets contain only 1 label. 
    #   pred_set_sizes[1] gives the amount of examples whose pred. sets contain 2 labels.
    #   pred_set_sizes[2] gives -||- 3 labels.
    #   etc....

    # We create a figure which shows how many of each possible prediction set size we have (how many examples have only 1 label in their prediction set, how many have 2 labels, etc...)
    # We first get an array of values from "1" to "num_of_labels+1" so that we can have the x-axle in our matplotlib plot be "1, 2, 3, 4, ..., num_of_labels" (so if num_of_labels = 5, then we get the x-axle to be "1  2  3  4  5".)
    set_sizes = []
    for i in range(num_of_labels):
        set_sizes.append(i+1)

    plt.figure()
    plt.bar(set_sizes, pred_set_sizes)  # x-axle: set_sizes. y-axle: the number of examples in pred_set_sizes for each set_sizes value.
    plt.xlabel("Prediction set size")
    plt.ylabel("Number of validation data examples")
    plt.title("Number of examples with each prediction set size")
    plt.show()



    # NOTE: Here are my previous notes for "evaluate_ssc()", which is a function not needed since the only difference between FSC and SSC is that in SSC, each group is divided on how many labels there are in the example's prediction set.
        # Given ..., we evaluate the "size-stratified coverage" for the given prediction sets.
        # We divide the given prediction sets into several groups based on how many elements are in them.
        # We then return the lowest coverage among all groups.

        # This is to find out more on the adaptiveness of the CP method: 
        # If all different sizes of prediction sets give at least the required coverage, then the CP method's allocation of set sizes strongly supports adaptivity.


# Given ....,
# we evaluate "feature-stratified coverage" for the given data.
# We calculate the mean empirical coverage for each group and then return the lowest coverage among all groups.

# This is to know if coverage remains approximately correct across these certain characteristics/features. We check if, for the chosen groupings, there is any under-coverage.
# In our case, we check "confidence-wise" FSC and "class-wise" FSC.
#   * For "confidence-wise" FSC, we check if the desired coverage is more or less achieved regardless of how "confident" the model is in its guesses.
#     This can be used to prove adaptivity, since the desired coverage is achieved for all different confidence levels (for the tested data).
#   * For "class-wise" FSC, we check if the desired coverage is more or less achieved for all possible classes/labels.
#     If proved, it would mean the given CP method does not violate class-wise conditional coverage too much (though it does not PROVE group-wise conditional coverage per-se).    
def evaluate_fsc(groups, threshold, val_scores):
    '''
    From "A Gentle Introduction...", the correct implementation seems to be:
    1. Divide the evaluation data into some K groups (so we have groups G_1, G_2, ..., G_K). Each group in turn contains a set of validation data examples, along with the true label for each such example.
    2. For each group, you get the prediction region for each example and count how many of them contains the example's true label. We then divide this sum with the total amount of examples in this group.
    3. After we have now gotten the empirical coverage for each group, we then return the lowest empirical coverage among all the groups.
    The closer this "minimum group-wise empirical coverage" is to 1 - alpha, the closer the method is to achieving conditional coverage (?).
    '''
    is_in_pred_region = {}
    mean_empirical_coverages = []
    

    # For each group...
    for group in groups:
        is_in_pred_region[group] = 0    # Initialize this group's "is_in_pred_region" value.
        # ...We go through each example in that group.
        for example in groups[group]:
            # We now get the mean empirical coverage for this group of validation data examples. 
            # We do this by finding all validation scores corresponding to each example in this group, 
            # and see if the prediction set created from this validation data example would contain the true label.
            # We check that by seeing if the nonconformity score given to the actual true label for this example (which is what val_scores[i] is) is less than the threshold value,
            # which would mean that the true label WOULD be included in this example's prediction region.
            
            # In other words, we go through all the validation scores (i.e the scores which are s(x, y_true) i.e the score for the validation example's true label) for this group,
            # and check which validation scores (i.e true label scores) would be a part of the prediction region (gives "1"), and which would not (gives "0").

            is_in_pred_region[group] += (val_scores[example] <= threshold)    # If this example's true label would be in the prediction region, then we add "+1" to this group's "is_in_pred_region" list. If not, then "+0" is added.
        
        # When we've gotten the number of examples in this group that would contain the true label, we can then make that into a percentage of how many examples contain the true label.
        if len(groups[group]) > 0:  # Special case: If there are NO examples which fit into group nr. "group", then we skip adding that mean empirical coverage.
            mean_empirical_coverage = (is_in_pred_region[group] / len(groups[group])).astype(float)
            mean_empirical_coverages.append(mean_empirical_coverage)
        else:
            print(f"\nGroup number {group} contains no examples.")

    # After doing this for all groups, we can now return the lowest mean empirical coverage among all groups.

    '''
    print("\nTESTING\n")
    for i, cov in enumerate(mean_empirical_coverages):
        print(f"Group {i} gives mean emp. cov. of {cov}.\n")
    '''


    return min(mean_empirical_coverages)


# Given ..., we evaluate the "average coverage gap" across all the data.
# We divide the given data into several groups based on a certain characteristic/feature, 
# and for each group we compute how much the "mean empirical coverage" differs from the desired coverage.
# We then return the average of these "differences".

# This is to know on average how far off the CP method is from achieving group-wise conditional coverage.
def evaluate_covgap(groups, threshold, val_scores, alpha):
    # NOTE: I'm a bit tired right now, so I should write some comment on how this works later (maybe after testing though)
    is_in_pred_region = {}
    differences = []
    
    # For each group...
    for group in groups:
        is_in_pred_region[group] = 0    # Initialize this group's "is_in_pred_region" value.
        # ...We go through each example in that group.
        for example in groups[group]:
            # We now get the mean empirical coverage for this group of validation data examples. 
            # We do this by finding all validation scores corresponding to each example in this group, 
            # and see if the prediction set created from this validation data example would contain the true label.
            # We check that by seeing if the nonconformity score given to the actual true label for this example (which is what val_scores[i] is) is less than the threshold value,
            # which would mean that the true label WOULD be included in this example's prediction region.
            
            # In other words, we go through all the validation scores (i.e the scores which are s(x, y_true) i.e the score for the validation example's true label) for this group,
            # and check which validation scores (i.e true label scores) would be a part of the prediction region (gives "1"), and which would not (gives "0").

            is_in_pred_region[group] += (val_scores[example] <= threshold)    # If this example's true label would be in the prediction region, then we add "+1" to this group's "is_in_pred_region" list. If not, then "+0" is added.
        
        # When we've gotten the number of examples in this group that would contain the true label, we can then make that into a percentage of how many examples contain the true label.
        if len(groups[group]) > 0:  # Special case: If there are NO examples which fit into group nr. "group", then we skip adding that mean empirical coverage.
            mean_empirical_coverage = (is_in_pred_region[group] / len(groups[group])).astype(float)
            
            # We then calculate and save how far off this group's coverage is from perfect group-wise coverage.
            differences.append(abs(  (1 - alpha) - mean_empirical_coverage  ))
        else:
            print(f"\nGroup number {group} contains no examples.")
    
    # After doing this for all groups, we can now return the average of all these differences.
    return (sum(differences) / len(differences))



