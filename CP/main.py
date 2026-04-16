# %%
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets
import naive_appr as naive, conv_appr as conv, daps_appr as daps, aps_appr as aps, raps_appr as raps
from functions import evaluate_marg_coverage, evaluate_cond_coverage, evaluate_adaptivity, evaluate_efficiency

def evaluate(cp_appr, has_calib_data=True):
    #       1) Get a new test point
    # Load CNN model + CIFAR100 test set and normalize to match training preprocessing
    base_model = tf.keras.models.load_model(r"C:\Users\rasmu\Documents\GitHub\EF112X-Conformal-Prediction\CNN\cnn_softmax_model.keras")
    (_, _), (test_images, test_labels) = datasets.cifar100.load_data(label_mode="fine")
    test_images = test_images.astype("float32") / 255.0

    # All possible labels for the CIFAR10 dataset.
    #class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # All possible labels for the CIFAR100 dataset.
    class_names = [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
        'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
        'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
        'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
        'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
        'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
        'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
        'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
        'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
    ]
    
    softmax_scores = base_model.predict(test_images) # Get the softmax scores for all test images.

    print(f"Evaluating {cp_appr.NAME}:")

    # Evaluating marginal coverage
    # Get the nonconformity scores for all test data, using this method's score function
    scores = []
    for i, example in enumerate(softmax_scores):
        scores.append(cp_appr.score_function(example, test_labels[i]))  # Get the nonconformity score for this example

    n = 9000    # The CIFAR100 dataset contains 10 000 test images/labels. We use 9000 of them as "calibration data" when evaluating marginal coverage.
    num_rounds = 10
    alpha = 0.1
    evaluate_marg_coverage(scores, num_rounds, n, alpha)

    # Evaluate adaptivity & conditional coverage
    num_of_labels = 100  # In the CIFAR100 dataset, we have 100 possible labels.
    n = 5000
    calib_input = softmax_scores[:n]  # The first "n" examples of "softmax_scores" (i.e the first "n" examples of "test_images") are calibration data inputs.
    calib_label = test_labels[:n]     # The first "n" examples of "test_labels" are the true labels for each calibration data input above.

    rest = len(softmax_scores) - n

    val_input = softmax_scores[rest:]   # The last examples of "softmax_scores" are used as validation data examples.
    val_label = test_labels[rest:]

    if has_calib_data:
        threshold = cp_appr.threshold(alpha, calib_input, calib_label, cp_appr.score_function)
    else:
        threshold = cp_appr.threshold(alpha)

    evaluate_cond_coverage(cp_appr.score_function, calib_input, calib_label, val_input, val_label, alpha)
    evaluate_adaptivity(cp_appr.score_function, threshold, num_of_labels, calib_input, calib_label, val_input, val_label, alpha)
    evaluate_efficiency(cp_appr.score_function, threshold, softmax_scores, val_input, class_names)

#evaluate(naive, False)
evaluate(conv)
#evaluate(daps)
#evaluate(aps)
#evaluate(raps)
# %%
