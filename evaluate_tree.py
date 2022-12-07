import numpy as np
from numpy.random import default_rng

def predict(test_instance, node):
  """
  A function to return the predicted class for a specific instance
  Arguments:
    - the test instance, an array with 7 values
    - the current node of the tree to recursively evaluate
  Returns:
    - the predicted room for this reading
  """
  
  # if the node is at a leaf, we can return the room of that leaf
  if node.leaf:
    return node.value

  # else figure out which side to travel down
  else:
    # take out the parameters of the node
    value = node.value
    attribute = node.attribute

    # the left hand side is for x[attribute] < value
    if test_instance[attribute] < value:
      # recursively travel down the left side until reaching a leaf
      return predict(test_instance, node.left)

    # the right hand side is for x[attribute] >= value
    else:
      # recursively travel down the right sideuntil reaching a leaf
      return predict(test_instance, node.right)

def construct_confusion_matrix(y_test, y_prediction, class_labels):
  """
  Build the confusion matrix for these predictions.
  Arguments:
    - The test (gold) dataset
    - The predictions made on these tests
    - The labels of all classes
  Returns:
    - The confusion matrix
  """
  
  # set up confusion matrix for the four rooms
  confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)

  # for each correct class, put the numbers of predicted class_labels in the confusion matrix
  for (i, label) in enumerate(class_labels):
    # get predictions
    indices = (y_test == label)
    actual = y_test[indices]
    predictions = y_prediction[indices]

    # get counts per label
    (unique_labels, counts) = np.unique(predictions, return_counts=True)

    # convert counts to a dictionary
    frequency_dict = dict(zip(unique_labels, counts))

    # fill up the confusion matrix for the current row
    for (j, class_label) in enumerate(class_labels):
      confusion[i, j] = frequency_dict.get(class_label, 0)

  # return the confusion matrix
  return confusion

def evaluate_tree(test_db, trained_tree):
  """
  Returns accuracy of tree
  Arguments:
    - the testing dataset: a tuple of (x_test, y_test)
    - the trained tree to test on
  Returns:
    - Confusion Matrix
    - Accuracy
    - Precision/Recall/F1 per class
  """
  
  # split testing dataset into data and labels
  x_test, y_test = test_db

  # set up predicted array to fill in
  y_prediction = np.zeros((len(y_test), ))

  # test each instance
  for idx, instance in enumerate(x_test):
    y_prediction[idx] = predict(instance, trained_tree)
    
  # get the unique class labels in the dataset for the confusion matrix
  class_labels = np.unique(np.concatenate((y_test, y_prediction)))

  # construct the confusion matrix
  confusion_matrix = construct_confusion_matrix(y_test, y_prediction, class_labels)

  # compute the accuracy from the confusion matrix: accuracy = (TP + TN) / (TP + FP + TN + FN)
  accuracy = np.sum(confusion_matrix.diagonal()) / np.sum(confusion_matrix.flatten())

  # compute the precision and recall for each class label in the confusion matrix
  p = np.zeros((len(confusion_matrix), ))
  r = np.zeros((len(confusion_matrix), ))
  f = np.zeros((len(confusion_matrix), ))

  for c in range(confusion_matrix.shape[0]):
    # precision = TP / (TP + FP)
    if np.sum(confusion_matrix[:, c]) > 0:
      p[c] = confusion_matrix[c, c] / np.sum(confusion_matrix[:, c])

    # recall = TP / (TP + FN)
    if np.sum(confusion_matrix[c, :]) > 0:
      r[c] = confusion_matrix[c, c] / np.sum(confusion_matrix[c, :])

    # f1 = 2 * p * r / (p + r)
    if p[c] + r[c] > 0:
      f[c] = 2 * p[c] * r[c] / (p[c] + r[c])
  
  # return the metrics as a tuple of (confusion, accuracy, precision, recall, f1)
  return (confusion_matrix, accuracy, p, r, f)
