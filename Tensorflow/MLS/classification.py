from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from IPython.display import clear_output
import tensorflow._api.v2.compat.v2.feature_column as fc
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

CSV_COLUMNS = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
#dict of what we want to predict
predict = {}

train_path = tf.keras.utils.get_file("iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file("iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMNS, header=0)
y_train = train.pop('Species')
test = pd.read_csv(test_path, names=CSV_COLUMNS, header=0)
y_test = test.pop('Species')

#train.head()
def input_fcn(features, labels, training = True, batch_size = 256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    
    if training:
        dataset = dataset.shuffle(1000).repeat()
    
    return dataset.batch(batch_size)

feature_columns = []
#returns all the keys (column names) in the dataframe
for key in train.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=key))

classifier = tf.estimator.DNNClassifier(
    feature_columns = feature_columns,
    #2 hidden layers with 30 and 10 neurons
    hidden_units = [30, 10],
    #we have to choose between 3 groups of flowers
    n_classes = 3
)

classifier.train(
    input_fn = lambda: input_fcn(train, y_train, training=True),
    steps = 5000
)

eval_res = classifier.evaluate(
    input_fn = lambda: input_fcn(test, y_test, training=False)
)
print("accuracy: {accuracy:0.3f}".format(**eval_res))

#predicting for the user
for feature in features:
    valid = True
    while valid:
        val = input(feature + ": ")
        if not val.isdigit(): 
            valid = False

    predict[feature] = [float(val)]

for pred in classifier.predict(input_fn = lambda: input_fcn(predict, labels=None, training=False)):
    class_id = pred['class_ids'][0]
    probability = pred['probabilities'][class_id]
    print('Prediction is "{}" ({:.1f}%)'.format(SPECIES[class_id], 100 * probability))