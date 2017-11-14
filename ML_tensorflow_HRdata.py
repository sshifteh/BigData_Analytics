
import tensorflow as tf

import numpy as np

import pandas as pd

#matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns

from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold, train_test_split




# Les inn datasettet i en Pandas DataFrame
hr_data = pd.read_csv('HR_comma_sep.csv')

hr_data = hr_data.sample(frac=1).reset_index(drop=True)
#print hr_data.describe()
#print hr_data.info()


# fra info om dataenen ser vi at sales og salary har string verdier.
# Vi mapper de om til int verdier.
hr_data['sales'].unique() # er alle kategoriene featuren kan ta.
hr_data['sales'].replace(['sales', 'accounting', 'hr', 'technical', 'support', 'management',
        'IT', 'product_mng', 'marketing', 'RandD'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], inplace = True)

# Det samme for salary features
hr_data['salary'].unique()
hr_data['salary'].replace(['low', 'medium', 'high'], [0, 1, 2], inplace = True)


# Korrelasjon Matrise
corr = hr_data.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
sns.plt.title('Heatmap of Correlation Matrix')
#sns.plt.show()


# Ser her kun paa left dvs. left the company, det er targeten
# Korrelasjonmatrisen viser kun lineare forhold mellom "left" og de andre avhengige variablene.
corr_left = pd.DataFrame(corr['left'].drop('left'))
#print corr_left.sort_values(by = 'left', ascending = False)






# Tensor flow neural network model

from sklearn.model_selection import train_test_split

label = hr_data.pop('left')
data_train, data_test, label_train, label_test = train_test_split(hr_data, label, test_size = 0.2, random_state = 42)

x_train = data_train.values
x_test = data_test.values
y_train = label_train.values
y_test_cls  = label_test.values

print ("Shape X-train: ", data_train.shape)
print ("Shape Y-train: ", label_train.shape)
print ("Shape X-test: ", data_test.shape)
print ("Shape Y-test: ", label_test.shape)

label_train = pd.DataFrame(label_train.values.reshape((11999,1)))
label_test = pd.DataFrame(label_test.values.reshape((3000,1)))
print ("Shaper new label train: {}".format(label_train.shape))
print ("Shaper new label test: {}".format(label_test.shape))



# Placeholders
# 9 input_units. 9 hidden layers.
ANTALL_FEATURES = 9
input_units = 9
HIDDEN_UNITS = 9
OUTPUT_UNITS = 2

x_placeholder = tf.placeholder(tf.float32, shape=[None, ANTALL_FEATURES])
y_placeholder = tf.placeholder(tf.float32, shape=[None, OUTPUT_UNITS])
y_true_cls_placeholder = tf.placeholder(tf.int64, [None])


def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot

y_hot_train = dense_to_one_hot(y_train, 2)
y_hot_test = dense_to_one_hot(y_test_cls, 2)



# Variables
weights = {
    'hidden': tf.Variable(tf.random_uniform((ANTALL_FEATURES, HIDDEN_UNITS), -1, 1), dtype=tf.float32),
    'output': tf.Variable(tf.random_normal( (HIDDEN_UNITS, OUTPUT_UNITS), -1, 1), dtype=tf.float32)
}

biases = {
    'hidden': tf.Variable(tf.random_normal([HIDDEN_UNITS]), dtype=tf.float32) ,
    'output': tf.Variable(tf.random_normal([OUTPUT_UNITS]), dtype=tf.float32)
}



# Operations
hidden_multi = tf.matmul(x_placeholder, weights['hidden'])
hidden_layer = tf.add(hidden_multi, biases['hidden'])
hidden_layer_h = tf.nn.relu(hidden_layer)
output_layer = tf.matmul(hidden_layer_h, weights['output']) + biases['output']



#softmax
y_pred = tf.nn.softmax(output_layer)
# Finner ekte predikert klasse. 1 eller 0.
y_pred_cls = tf.argmax(y_pred, dimension=1)




#cost
cross_e_cost = tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y_placeholder)
cost = tf.reduce_mean(cross_e_cost)



#optimization
# Learning Rate er også valgt litt tilfeldig innfor rimelighet
learning_rate = 0.05
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(cost)
#Accuracy stuff
correct_prediction = tf.equal(y_pred_cls, y_true_cls_placeholder)
# Regner accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

epochs = 5
batch_str = 200
