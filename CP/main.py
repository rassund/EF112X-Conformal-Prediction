# %%
import tensorflow as tf
from tensorflow.keras import datasets
import naive_appr as naive, conv_appr as conv, aps_appr as aps, raps_appr as raps
from functions import evaluate_marg_coverage, evaluate_cond_coverage, evaluate_adaptivity, evaluate_efficiency

def evaluate(cp_appr, name):
    #       1) Get a new test point
    # Load CNN model + CIFAR-10 test set and normalize to match training preprocessing
    base_model = tf.keras.models.load_model("CNN/cnn_softmax_model.keras")
    (_, _), (test_images, test_labels) = datasets.cifar10.load_data()
    test_images = test_images.astype("float32") / 255.0

    # All possible labels for the CIFAR10 dataset.
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    softmax_scores = base_model.predict(test_images, batch_size=32) # Get the softmax scores for all test images.

    print(f"Evaluating {name}:")

    # Evaluating marginal coverage
    # Get the nonconformity scores for all test data, using this method's score function
    scores = []
    for i, example in enumerate(softmax_scores):
        scores.append(cp_appr.score_function(example, test_labels[i]))  # Get the nonconformity score for this example

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

    evaluate_cond_coverage(cp_appr.score_function, calib_input, calib_label, val_input, val_label, alpha)
    evaluate_adaptivity(cp_appr.score_function, num_of_labels, calib_input, calib_label, val_input, val_label, alpha)

    if name == "naive":
        evaluate_efficiency(cp_appr.score_function, cp_appr.threshold(alpha), softmax_scores, val_input, class_names)
    else:
        evaluate_efficiency(cp_appr.score_function, cp_appr.threshold(alpha, calib_input, calib_label, cp_appr.score_function), softmax_scores, val_input, class_names)

#evaluate(naive, "naive")
#evaluate(conv, "conventional")
#evaluate(aps, "APS")
evaluate(raps, "RAPS")
# %%
