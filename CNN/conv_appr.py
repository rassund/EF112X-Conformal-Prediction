import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets


"""
Use calibration data. 
To get the threshold value “q”, we let the model get the softmax score for the (as we know it) true label for each calibration data example 
(for example, if one example is (input, true label) = (image.png, “Dog”), 
then we let the model tell their softmax score for the label “Dog” for that image).
If we want 90% coverage, then we pick a threshold value “q” such that “q” is smaller than ~90% of all the softmax scores from the calibration data.

When we get a new test point (we know the input, not the true label), we simply get the softmax scores for every label, 
order them from highest to lowest, and then we add every label whose softmax score is higher than our threshold value “q”.
"""