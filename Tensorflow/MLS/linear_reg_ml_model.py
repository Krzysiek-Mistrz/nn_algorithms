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

#load dataset
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

#some basic statistical operations to preview our data
#print(dftrain.head())
#print(dftrain.shape)
#dftrain.age.hist(bins=20)
#plt.savefig('/home/krzychu/Documents/Programowanie/python_neural_networks/Tensorflow/age_histogram.png')
#dftrain.sex.value_counts().plot(kind='barh')
#plt.savefig('/home/krzychu/Documents/Programowanie/python_neural_networks/Tensorflow/sex_count.png')
#dftrain['class'].value_counts().plot(kind='barh')
#plt.savefig('/home/krzychu/Documents/Programowanie/python_neural_networks/Tensorflow/class_count.png')
#pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
#plt.savefig('/home/krzychu/Documents/Programowanie/python_neural_networks/Tensorflow/survive_percentage.png')

categorical_columns = ["sex", "n_siblings_spouses", "parch", "class", "deck", "embark_town", "alone"]
numeric_columns = ["age", "fare"]
feature_columns = []

for feature_name in categorical_columns:
    vocabulary = dftrain[feature_name].unique()
    #from categorical to one-hot encoding for linear estimator
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in numeric_columns:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)

#data_df is our dataframe, label_df is our eval dataframe
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)

        return ds
    
    return input_function

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

clear_output()
print(result['accuracy'])

result = list(linear_est.predict(eval_input_fn))
#propability of surviving for certain person
print(dfeval.loc[0])
print(result[0]['probabilities'])
#whether this person survived for real
print(y_eval.loc[0])