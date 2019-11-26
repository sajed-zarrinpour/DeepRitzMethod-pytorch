
import argparse, sys

parser=argparse.ArgumentParser()

parser.add_argument('--test', help='Path to the csv file containing the test data set')
parser.add_argument('--train',help='Path to the csv file containing the train dataset')
parser.add_argument('--PREFIX',help='Path in which the model generated file shall be store')

args=parser.parse_args()

#usage :  args.option

import numpy as np

# Load the dataset
test = np.loadtxt(args.test, delimiter=',')
train = np.loadtxt(args.train, delimiter=',')
# split both sets into X (input / data) and Y (output / lables) parts
data = np.array(train)[:,0:2]
labels = np.array(train)[:,2]

d_test = np.array(test)[:,0:2]
l_test = np.array(test)[:,2]

#Declaring custom loss, activation and metrics

import keras.backend as K

def euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

# For custom metrics (this is mean squarred error)
def mean_pred(y_true, y_pred):
    return K.mean(np.square(np.subtract(y_true,y_pred)))

#custom activation function
from keras.layers import Activation
#from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

def custom_activation(x):
    return  K.relu(x**3) 

get_custom_objects().update({'custom_activation': Activation(custom_activation)})


from keras.layers import Input, Dense, add
from keras.models import Model


#Actual DR Network
# Data manipulation and model settings
#number of neurons per layer
m=10
#padding zeros so that the input dimension can match our desired dimension:
data = np.hstack((data, np.zeros((data.shape[0], m - data.shape[1] ))))
data_test = np.hstack((d_test, np.zeros((d_test.shape[0], m - d_test.shape[1] ))))

number_of_hidden_layers_plus_input_layer = 5

#Declaring The Model

stack = []
inputTensor = Input(shape=(m,))

for i in range(number_of_hidden_layers_plus_input_layer):
    # First Layer Of the i-th Block
    if(i==0):
        stack.append(Dense(m, input_dim = (None,m,),activation='custom_activation')(inputTensor))
    else:
        stack.append(Dense(m, activation='custom_activation')(stack[len(stack)-1]))
    # Second Layer of the i-th Block
    stack.append(Dense(m,activation='custom_activation')(stack[len(stack)-1]))
    # Third Layer of the i-th Block
    # (adding the out put of the first layer and second layer of the block together)
    stack.append(add([stack[len(stack)-2],stack[len(stack)-1]]))

#adding the final Layer
finalOutput = Dense(1)(stack[len(stack)-1])


#Running The Model
model = Model(inputTensor,finalOutput)
#model.compile(optimizer='adam', loss=euclidean_distance_loss, metrics=[mean_pred])
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mse', 'acc'])
model.fit(data, labels, epochs=1, batch_size=10)

loss, MSE, acc = model.evaluate(data_test, l_test)
print('Loss Funcion Minima: %.8f' % (MSE))
print('Mean Square Error: %.8f' % (MSE))
print('Accuracy: %.8f' % (MSE))
#print(model.metrics_names)
import os.path
#completeName = os.path.join(save_path, name_of_file+".txt") 
with open(os.path.join(os.path.abspath(""),args.PREFIX,'model_evaluation_report.log'), 'w') as f:
    f.write('Loss Funcion Minima: %.8f' % (MSE))
    f.write('Mean Square Error: %.8f' % (MSE))
    f.write('Accuracy: %.8f' % (MSE))

#making predictions and save it!

p = model.predict(data_test)
out = np.hstack((d_test, p))

np.savetxt(args.PREFIX+'/predictions.csv', train, delimiter=',')


# Open the file
with open(args.PREFIX+ '/model_summary.log','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))
model.summary()
