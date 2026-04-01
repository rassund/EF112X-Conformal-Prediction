import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets


def cifar10_get_softmax_dists(image_nr):
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


def cifar10_per_class_acc():
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
    plt.hist(coverages) # should be roughly centered at 1-alpha
    plt.show()

def evaluate_efficiency():
    '''
    Evaluate efficiency, i.e the average set size. The smaller the set sizes on average, the more efficient the method is.

    In "UNCERTAINTY SETS FOR IMAGE CLASSIFIERS USING CONFORMAL PREDICTION" just before chapter 2.2 (on page 5), 
    it is discussed a formula for an "upper bound" for how much coverage a prediction set can have.

    IDEA: We can evaluate marginal coverage, i.e check how much coverage the prediction set actually achieves.
    The closer the actual marginal coverage is to the wanted marginal coverage, the better the efficiency of the method.
    '''
    return

def evaluate_cond_coverage():
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
    HOWEVER: To check adaptivity, this is often not enough. We can now with this whether or not the prediction sets have dynamic sizes, but we still "need to verify that that large sets occur for hard example" (taken from paper).
    '''
    return