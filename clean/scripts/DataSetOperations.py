# In this module we will devide our main dataset, 
# which is the out put of the FEM Software in GNUPlot format,
# into two distict sets.
# To ensure that the final model will be generalized over the entire Domain,
# we choose uniformly random 25% of the data points distributed acros the domain.

import argparse, sys

parser=argparse.ArgumentParser()

parser.add_argument('--filename', help='Path to the csv dataset')

args=parser.parse_args()

#usage :  args.option

import numpy as np


# Load the dataset
dataset = np.loadtxt(args.filename+"/"+"data.csv", delimiter=',')
test_percentage = 0.25

# to randomly select k items from a stream of items 
#reference : https://www.geeksforgeeks.org/reservoir-sampling/
import random 

# This function randomly selects k items from stream[0..n-1]. 
def selectKItems(stream, n, k): 
    i=0; # index for elements in stream[] 
    
    # Initialize the output array with first k elements from stream[] 
    reservoir = [0]*k; 
    for i in range(k): 
        reservoir[i] = stream[i]; 

    # Iterate from the (k+1)th element to nth element 
    while(i < n): 
        # Pick a random index from 0 to i. 
        j = random.randrange(i+1); 

        # If the randomly picked  index is smaller than k, 
        # then replace the element present at the index 
        # with new element from stream 
        if(j < k): 
            reservoir[j] = stream[i]; 
        i+=1; 
    return reservoir


#creating an array of indices of the points
stream = range(dataset.shape[0]); 
n = len(stream);
# we took 75% of the data as the train set and 25% of the data set as the test set
k = (int) (dataset.shape[0] * test_percentage);
#here we actually generating indices of those 25% of the data set which is uniformly distributed across the domain
reservoir = selectKItems(stream, n, k); 

#split the data into train and test sets :
def partition_on_index(it, indices):
    indices = set(indices)   # convert to set for fast lookups
    l1, l2 = [], []
    l_append = (l1.append, l2.append)
    for idx, element in enumerate(it):
        l_append[idx in indices](element)
    return l1, l2

train, test = partition_on_index(dataset,reservoir)

np.savetxt(args.filename+"/"+'test.csv', test, delimiter=',')
np.savetxt(args.filename+"/"+'train.csv', train, delimiter=',')

# usage :
# np.loadtxt('test.csv', delimiter=',')
