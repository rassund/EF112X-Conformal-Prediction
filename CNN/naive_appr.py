import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets


"""
No calibration data. 
Simply get a new test point, have the CNN Classifier get the softmax score for all possible labels, 
order the softmax scores from highest (most likely (according to model)) to lowest.

If we want 90% coverage (we guarantee that, on average, there is a 90% chance that the true label is in the prediction set), 
we just take labels from highest to lowest such that their combined softmax score is ~0.9 (90%).
"""