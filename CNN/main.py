import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets
from naive_appr import *    # Import functions defined in "naive_appr.py"


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


cifar10_get_softmax_dists(100)