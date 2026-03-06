import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets
from functions import create_label_softmax_dict, sort_descending_softmax_dict


"""
(The same as the APS approach, just that we add a “penalization term” to the score function.)

Use calibration data. To get the threshold value “q”, for each calibration data example, we let the model get the softmax score for each label, 
and then we order those softmax scores from highest to lowest. 
Then, we see where in this “ranking” the true label for this calibration data example is. 
If the true label is, for example, the “35th most likely label to fit with the input” (according to the model), 
then we add together the softmax scores for the “1st most likely”, the 2nd most likely, the 3rd, the 4th, …, the 34th most likely and lastly the 35th most likely label.

AFTER THAT! We add a “penalization term” to the score, which gives a bigger score increase the “deeper in” the true label is in the ranking.
We get a “penalized accumulative softmax mass”. We do this for every calibration data example.

If we want 90% coverage, then we pick a threshold value “q” such that “q” is bigger than ~90% of all the penalized accumulative softmax masses from the calibration data.

When we get a new test point (we know the input, not the true label), we get the softmax scores for every label, order them from highest to lowest, 
and then we first pretend that the “1st most likely” label is the true label, and use the (above explained) score function to generate its accumulative softmax mass. 
Then we do the same for the 2nd most likely label, where we pretend that IT is the true label and we run the score function (adding together the softmax score for the 1st most likely label and the 2nd most likely label, then adding the penalization term). 
Then we do the same for the 3rd most likely label, then the same for the 4th most likely label, etc…

Once we have all our penalized accumulative softmax masses for the new test point, we just pick the labels with a penalized accumulative softmax mass below the threshold value “q”.
"""