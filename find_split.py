import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

def find_split(x_train, y_train):
    """
    Find the split point that maximises info gain.
    HOW FIND_SPLIT WORKS
    1. FINDS PARENT INFORMATION
        - this is the information of the sample before any split occurs
        - this is stored in the variable parentInformation
        - parent information is used in the equation to find information gain (entropy before any split(parrent information) - average of information after split)

    2. ITERATE THROUGH EACH ATTRIBUTE, AND FIND INFORMATION GAIN AT EACH POSSIBLE THRESHOLD
        - for each attribute, create a sorted list, going upwards, and sort y so that each value matches its class
        - for each value in the sorted list, set the value as the threshold, and find the information gain using the function "threshold_information")
        - constantly update the maximum information,and store the threhsold and attribute number of any maximum information gain
    """

    def entropy(y):
        numberDataPoints = len(y)
        unique, counts = np.unique(y, return_counts=True)
        information = 0
        for x in counts:
            information -= (x / numberDataPoints) * np.log2(x / numberDataPoints)
        return information
        
    # correct:
    def threshold_information(x_data, y_data, threshold):
        # get labels of indices of values in x_data s.t. values < threshold -> left
        # ... >= threshold -> right
        (l_indices, ) = np.where(x_data < threshold)
        (r_indices, ) = np.where(x_data >= threshold)

        l_arr = y_data[l_indices]
        r_arr = y_data[r_indices]
        
        #entropy of left side
        left_info = entropy(l_arr)
        #entropy of right side
        right_info = entropy(r_arr)

        left_size = len(l_arr)
        right_size = len(r_arr)
        
        res = left_info*(left_size/len(y_data))+right_info*(right_size/len(y_data))

        return (res, l_arr, r_arr)

    maxInformationGain = -float("inf")

    #finding the amount of information before the split   
    parentInformation = entropy(y_train)

    numberAttributes = len(x_train[0])

    for i in range(numberAttributes):
        attribute = x_train[:,[i]]
        sorted_indices = np.argsort(attribute, axis=0)
        sorted_attribute = attribute[sorted_indices]
        sorted_attribute = sorted_attribute.ravel()
        y_data = y_train[sorted_indices]
        for index, threshold in enumerate(sorted_attribute):
            information, leftSplit, rightSplit = threshold_information(sorted_attribute, y_data, threshold) 
            res = parentInformation - information
            
            if res > maxInformationGain and len(leftSplit) > 0 and len(rightSplit) > 0 : #can probably remove these length checks
                maxInformationGain = res
                output_attribute = i
                output_threshold = threshold
                output_index = index
                
    return (output_attribute, output_threshold)