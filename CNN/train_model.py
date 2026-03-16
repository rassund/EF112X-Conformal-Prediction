import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, callbacks

def train_model(dataset, num_classes):

    if "CIFAR10" in dataset:
        (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0   #NOTE: I don't think this is standardized... Will check after testing another training data set.
    # Why do this? What does it do? Well, when deciding the color of a chosen pixel in the image, we look at the color channels for that pixel (for example, for RGB, we look at the Red channel, the Green channel and the Blue channel).
    # The values for each color channel ranges from 0 to 255, and by instead having them be numbers between 0 and 1, it seems like it helps improve training stability (from what we could find).

    # Finding out the width and height of images and the color channels
    image_height = train_images.shape[1]
    image_width = train_images.shape[2]
    color_channels = train_images.shape[3]

    # We create the "convolutional base" by having a stack of convolution layers and pooling layers.

        # The convolutional layers (Conv2D) also include an activation layer/function, 
        # introducing "non-linearity" into the pattern searching done on the layer. This in turn
        # makes it possible for the model to find more complex patterns such as curves and shapes.
        # After each convolution + activation layer, there is a pooling layer which reduces the size of the generated feature maps by
        # essentially "removing" all the unnecessary details/patterns found (thus also reducing the risk of overfitting).

    model = models.Sequential()
    # For the first convolution layer, we give a 3-channel RGB image, and we let the convolution layer output a tensor of size 32 (32 "versions" of the same image, since we say that we want to use 32 filters for this conv. layer)
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, color_channels))) 
    model.add(layers.MaxPooling2D((2, 2)))
    # For the second convolution layer, we don't give a 3-channel RGB image: We instead give a tensor with 32 "channels" (different versions of the same image).
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    # Since the second conv. layer uses 64 filters, the third conv. layer receives a tensor with 64 channels.
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))


    # To explain:
    # We create a sequential model, since CNNs are neural network architectures in which we basically just send some data (for example, images) through several layers, one after another in a sequence (i.e, sequentially).
    # In this first part, we mix between convolutional layers and pooling layers. 

    # For the first convolutional layer we add to the model/architecture, we receive the input data: Images of (image_height, image_width, color_channels) = [For the CIFAR10 dataset:] (32, 32, 3) (RGB = 3 color channels), therefore we have "input_shape=(32, 32, 3)".
    # Also, we say that we first want to use 32 filters (each filter looks for different things) where each filter looks over a 3 x 3 pixel area searching for their pattern (then moving to another part of the image to do the same). Here, we check for the most simple features.
    #   (Also, we introduce an activation function called "ReLU" in order to introduce "non-linearity" into the pattern recognition filters, meaning we can find more complex patterns).
    # After this first convolution layer, we use a "pooling layer", in which we select the "pooling window" as being a 2 x 2 square. This means that we basically "summarize" the features found in each image for each filter by "summarizing" each 2 x 2 square of the image.
    #   This in turn causes the size of the data that we have to examine to be halved (instead of having 32 x 32 pixel images to examine for further patterns, where there are 32 different versions of each image (one for each filter), we instead have 32 different versions of 16 x 16 images.)

    # Now that we have found some simpler features, we can do another convolution layer to find more complex features (since we "summarized" the simple patterns found from the first conv. layer, we can now try to find patterns IN THESE SIMPLE PATTERNS, thus resulting in more complex patterns being found), 
    #   in which we now use 64 filter instead of just 32 (more filters = more computation, but since we have smaller images to work with, this is generally deemed to be alright).
    # We then reduce these images again using a pooling layer (make the model focus more on the "important"/most prominent features for each image, ignoring the not-so important features. Also reduces risk of overfitting).
    # We continue this one more time to find even more complex patterns, before sending our findings to the flattening layer and then to the dense/ "fully formed" layer for classification.


    # To perform "classification" (i.e let the model look at all the features/patterns we've found in the chosen image to choose which label fits best),
    # we need to feed the feature maps (the "channels", the different versions of an image in which each conv. filter creates one such version, saying where in the image that filter's specific pattern is most/least recognized) 
    # that are created through the convolutional + activation + pooling layers into a "dense"/"fully functional layer".
    # First of all, we need to "flatten" the feature maps (we essentially want to give all the features found in all feature maps to the dense layer in the form of one big list of features, instead of a list of feature maps which contain patterns).
    model.add(layers.Flatten())

    # Now that we have added a flattening layer after all the convolutional + activation + pooling layers, we can do classification using dense layers.
    model.add(layers.Dense(64, activation='relu'))
    # We take ALL of the features we've gathered from our images (we have 64 "versions" of each image / 64 feature maps, in which each image has a specific amount of pixels (height x width), meaning we have "height * width * # of feature maps" of "patterns" too look at for each image we have)
    # and we try to combine these into 64 large complex patterns, such as "cat-like", "feather-like" etc... (that's why we introduce an activation layer, so that we can try to group together all our found patterns into 64 complex, non-linear patterns).
    #   NOTE: It seems like it isn't important that we summarize the patterns into exactly 64 complex patterns: We CAN summarize them into 32 complex patterns, or 500, or... It just takes more space and might introduce more/less complexity.

    model.add(layers.Dense(10))
    # After we've "summarized" all the patterns we've found into 64 complex patterns (for example, "cat-like", "feather-like" etc...),
    # we can now use these "summarized" features to give a "score"/logit value to each of the 10 classes for this dataset


    # With GeeksForGeeks implementation
    model.add(layers.Dense(num_classes, activation='softmax'))


    early_stop = callbacks.EarlyStopping(monitor='loss', patience=1)    # Add an "early stop" for training the model: If the training loss stops improving for 1 epoch, then we stop running more epochs.

    # Now we can compile the model (we essentially tell the model HOW it should be training itself)
    model.compile(
        optimizer='adam',   # We use a common optimizer using the "Adam algorithm".
        loss='sparse_categorical_crossentropy',     # We use the "sparse categorical cross-entropy" loss function.
        metrics=['accuracy']    # Just for us to read. We let the program show how much the accuracy of the model improves for each epoch completed during model training.
)
    #       Explanation for the "optimizer" parameter:
    # When a model is trained, it gets some training data, tries to predict the true class for that training data, and then it checks how wrong its prediction was to then change the model's "weights" to make better predictions.
    # The "optimizer" is the algorithm which, given some data on how wrong its predictions were, changes the weights in an "optimal" way such that the model can create better predictions in the future.
    # The "Adam algorithm" seems to be one of the most popular optimizers for image classification, hence why it's used here.
    
    #       Explanation for the "loss" parameter:
    # Categorical cross-entropy is a loss function used for multi-label classification (such as for most image classification datasets).
    # Essentially, for each training image, we let the model create a softmax probability distribution for each class. We then check how "accurate" the model was in its guess by checking the probability given by the model to the (given by the dataset) true label for the training image.
    #   If the model was very confident that that label WAS the true label for the training image, then the loss function gives a small penalty. If the model was NOT confident (didn't guess well), then the loss function gives a big penalty. Thus, we can lead the model towards making better guesses.
    # "Sparse" categorical cross-entropy is the exact same as the above, just that with the "sparse" version, we check the true label with an integer representation (the true label can be represented as a number), 
    #   instead of with a "one-hot vector" representation (if the true label is represented as "label 3" and we have 10 possible labels (0-9), then the "one-hot vector" version of that would be "[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]").


    # We then train the model.
    history = model.fit(train_images, train_labels,
                        epochs=50,
                        batch_size=64,
                        validation_split=0.2,
                        verbose=1,
                        callbacks=[early_stop])

    plt.figure(figsize=(12,5))


    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()


    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


    #       Explanation of the "epochs" parameter:
    # When training a model, you go through some given training data and let the model predict the true labels (then check how correct its predictions was) for small batches of that training data at a time.
    # If we selects "epochs = 15", then we go through the entirety of the training data a total of 15 times. So after the model has tried to predict the last of the training data (after which we update the "weights" thus hopefully getting better predictions), 
    #   we then go back to the start of the training data and let the model try to predict the true labels AGAIN. As we let the model go through this training data again and again, it gets better and better at guessing the correct true label ON AVERAGE throughout the ENTIRE training dataset.


    #       Explanation of the "batch_size" parameter:
    # When we let the model predict the true label for the training data, we do this in "batches". If we set "batch_size = 64", then we only predict the true labels for 64 inputs/images at a time (we also check how wrong these predictions are in 64-image batches). 
    #   Only after each batch (not after each training image) do we update the model's "weights" (i.e update how the model predicts which label is the true label).

    #       Explanation of the "validation_split" parameter:
    # By setting "validation_split = 0.2", we say that we want 20% of the training data to be "validation data", so that we can check the accuracy increase for each epoch. 
    #   So if we have 50 000 training images and we set 20% of that as "validation data", then we use only 40 000 of these training images to actually train the model.


    # We now save this model so that we don't have to train it again after every time we want to try something new.
    #model.save("cnn-tensorflow_model.keras")
    model.save("cnn_softmax_model.keras")




train_model("CIFAR10", 10)