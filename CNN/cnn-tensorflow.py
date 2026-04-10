import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Using the CIFAR10 dataset to get 50 000 training images and 10 000 testing/verification images. There are 10 possible classes for these images.
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
# Why do this? What does it do? Well, when deciding the color of a chosen pixel in the image, we look at the color channels for that pixel (for example, for RGB, we look at the Red channel, the Green channel and the Blue channel).
# The values for each color channel ranges from 0 to 255, and by instead having them be numbers between 0 and 1, it seems like it helps improve training stability (from what we could find).


    # To verify that the dataset looks correct, we can display the first 25 images from the dataset and their true label/class.
#class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
#plt.figure(figsize=(10,10))
#for i in range(25):
#plt.subplot(5,5,i+1)
#plt.xticks([])
#plt.yticks([])
#plt.grid(False)
#plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
#plt.xlabel(class_names[train_labels[i][0]])
#plt.show()


# We create the "convolutional base" by having a stack of convolution layers and pooling layers.

    # The convolutional layers (Conv2D) also include an activation layer/function, 
    # introducing "non-linearity" into the pattern searching done on the layer. This in turn
    # makes it possible for the model to find more complex patterns such as curves and shapes.
    # After each convolution + activation layer, there is a pooling layer which reduces the size of the generated feature maps by
    # essentially "removing" all the unnecessary details/patterns found (thus also reducing the risk of overfitting).

model = models.Sequential()
# For the first convolution layer, we give a 3-channel RGB image, and we let the convolution layer output a tensor of size 32 (32 "versions" of the same image, since we say that we want to use 32 filters for this conv. layer)
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))    #OBS!!!! To standardize, we need to adapt the "input_shape" to always fit the input data (the image dimensions).
model.add(layers.MaxPooling2D((2, 2)))
# For the second convolution layer, we don't give a 3-channel RGB image: We instead give a tensor with 32 "channels" (different versions of the same image).
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
# Since the second conv. layer uses 64 filters, the third conv. layer receives a tensor with 64 channels.
model.add(layers.Conv2D(64, (3, 3), activation='relu'))


# To explain:
# We create a sequential model, since CNNs are neural network architectures in which we basically just send some data (for example, images) through several layers, one after another in a sequence (i.e, sequentially).
# In this first part, we mix between convolutional layers and pooling layers. 

# For the first convolutional layer we add to the model/architecture, we receive the input data: Images of (image_height, image_width, color_channels) = (32, 32, 3) (RGB = 3 color channels), therefore we have "input_shape=(32, 32, 3)".
# Also, we say that we first want to use 32 filters (each filter looks for different things) where each filter looks over a 3 x 3 pixel area searching for their pattern (then moving to another part of the image to do the same). Here, we check for the most simple features.
#   (Also, we introduce an activation function called "ReLU" in order to introduce "non-linearity" into the pattern recognition filters, meaning we can find more complex patterns).
# After this first convolution layer, we use a "pooling layer", in which we select the "pooling window" as being a 2 x 2 square. This means that we basically "summarize" the features found in each image for each filter by "summarizing" each 2 x 2 square of the image.
#   This in turn causes the size of the data that we have to examine to be halved (instead of having 32 x 32 pixel images to examine for further patterns, where there are 32 different versions of each image (one for each filter), we instead have 32 different versions of 16 x 16 images.)

# Now that we have found some simpler features, we can do another convolution layer to find more complex features (since we "summarized" the simple patterns found from the first conv. layer, we can now try to find patterns IN THESE SIMPLE PATTERNS, thus resulting in more complex patterns being found), 
#   in which we now use 64 filter instead of just 32 (more filters = more computation, but since we have smaller images to work with, this is generally deemed to be alright).
# We then reduce these images again using a pooling layer (make the model focus more on the "important"/most prominent features for each image, ignoring the not-so important features. Also reduces risk of overfitting).
# We continue this one more time to find even more complex patterns, before sending our findings to the flattening layer and then to the dense/ "fully formed" layer for classification.





# Just to display the architecture of the model thus far.
#model.summary()


# To perform "classification" (i.e let the model look at all the features we've found in the chosen image to choose which label fits best),
# we need to feed the feature maps created through the convolutional + activation + pooling layers into a "dense"/"fully functional layer".
# First of all, we need to "flatten" the feature maps (we essentially want to give all the features found in all feature maps to the dense layer in the form of one big list of features, instead of a list of feature maps which contain patterns).
model.add(layers.Flatten())

# Now that we have added a flattening layer after all the convolutional + activation + pooling layers, we can do classification using dense layers.
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

# Now we can compile the model...
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
#... And train the model.
history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))


# We now save this model so that we don't have to train it again after every time we want to try something new.
model.save("CNN/cnn-tensorflow_model.keras")

    # Evalutate the model
#plt.plot(history.history['accuracy'], label='accuracy')
#plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
#plt.xlabel('Epoch')
#plt.ylabel('Accuracy')
#plt.ylim([0.5, 1])
#plt.legend(loc='lower right')

#test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

#print("Test accuracy: " + test_acc)
