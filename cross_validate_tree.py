import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from prune_tree import prune_tree
from build_tree import decision_tree_learning, calculate_max_depth
from evaluate_tree import evaluate_tree
from draw_tree import draw_tree

def k_fold_split(n_instances, k=10, random_generator=default_rng()):
  """
  Split the dataset into k equally sized folds.
  Arguments:
    - the number of instances
    - k: the number of folds
    - a random generator to shuffle the indices
  Returns:
    - an array of k folds, each being a unique set of shuffled indices
  """ 

  # shuffle the indices of the elements in the dataset
  shuffled_indices = random_generator.permutation(n_instances)

  # split the indices into k folds
  split_indices = np.array_split(shuffled_indices, k)

  # return the set of split indices
  return split_indices

def cross_validation(dataset, datatype, k=10, random_generator=default_rng()):
  """
  Split the tree into a test fold, then perform cross validation on the remaining k-1 folds
  Then repeat k times with sequential test folds, and average the results.
  Arguments:
    - the full dataset
    - k: the number of folds
    - a random generator to shuffle the dataset
  Returns:
    - a tuple of averages of:
      - confusion matrix
      - accuracy
      - precision
      - recall
      - F1-measure
  """

  # split the dataset into x and y components
  (x_data, y_data) = dataset
  n_instances = len(y_data)

  # split into k folds
  split_indices = k_fold_split(n_instances, k, random_generator)

  # make arrays to store loop metrics
  depths_unpruned, depths_pruned = [], []
  confusions_unpruned, confusions_pruned = [], []
  accuracies_unpruned, accuracies_pruned = [], []
  precisions_unpruned, precisions_pruned = [], []
  recalls_unpruned, recalls_pruned = [], []
  f1s_unpruned, f1s_pruned = [], []

  # for every fold, separate a test set and do nested cross-validation on the rest of the folds
  for fold in range(k):
    # choose one of the 10 test array folds as the test set
    test_indices = split_indices[fold]
    x_test = x_data[test_indices, :]
    y_test = y_data[test_indices]

    # for the remaining folds in the set, do nested cross-validation
    for i in range(fold + 1, k + fold):
      # get the index of the validate set
      j = i % k

      # get the indices of the validate set
      validate_indices = split_indices[j]

      # we set train to all the folds in split_indices that aren't in the test or validate sets
      train_indices = np.setdiff1d(np.setdiff1d(split_indices, validate_indices), test_indices)
      
      # set up the validate set
      x_validate = x_data[validate_indices, :]
      y_validate = y_data[validate_indices]

      # set up the training set
      x_train = x_data[train_indices, :]
      y_train = y_data[train_indices] 

      # aggregate the datasets
      train_dataset = (x_train, y_train)
      validate_dataset = (x_validate, y_validate)
      test_dataset = (x_test, y_test)

      # print to initiate this nested cross-validation iteration
      print("test fold = ", fold, " validate fold = ", j)

      # build tree on train set
      unpruned_tree, unpruned_depth = decision_tree_learning(train_dataset)

      # evaluation of the normal tree
      unpruned_evaluation = evaluate_tree(test_dataset, unpruned_tree)
      
      # show the evaluation of the unpruned tree
      print("Unpruned tree has depth ", unpruned_depth,": ", unpruned_evaluation)

      # store the results in the metrics arrays
      confusion, accuracy, precision, recall, f1 = unpruned_evaluation
      depths_unpruned.append(unpruned_depth)
      confusions_unpruned.append(confusion)
      accuracies_unpruned.append(accuracy)
      precisions_unpruned.append(precision)
      recalls_unpruned.append(recall)
      f1s_unpruned.append(f1)

      # prune the tree based on the validation set
      pruned_tree = prune_tree(unpruned_tree, unpruned_tree, validate_dataset)
      pruned_depth = calculate_max_depth(pruned_tree)
      
      # evaluate the pruned tree and we can compare with the unpruned
      pruned_evaluation = evaluate_tree(test_dataset, pruned_tree)
      
      # show the evaluation of the pruned tree
      print("Pruned tree has depth ", pruned_depth, ": ", pruned_evaluation)

      print("Depth reduction of ", unpruned_depth - pruned_depth, " -> accuracy improvement of ", pruned_evaluation[1] - unpruned_evaluation[1])

      # store the results in the metrics arrays
      confusion, accuracy, precision, recall, f1 = pruned_evaluation
      depths_pruned.append(pruned_depth)
      confusions_pruned.append(confusion)
      accuracies_pruned.append(accuracy)
      precisions_pruned.append(precision)
      recalls_pruned.append(recall)
      f1s_pruned.append(f1)
    
  # plot accuracy against depth
  fig, ax = plt.subplots()
  ax.set_xlabel("Depth of tree (nodes)")
  ax.set_ylabel("Accuracy")
  ax.set_xlim([min(depths_pruned) - 2, max(depths_unpruned) + 2])
  ax.set_ylim([min(accuracies_unpruned + accuracies_pruned) - 0.1, 1])
  plt.scatter(depths_unpruned, accuracies_unpruned, s=10, c='red')
  plt.scatter(depths_pruned, accuracies_pruned, s=10, c='blue')
  plt.tight_layout()
  
  # after the outer loop has completed, average those results
  depth_mean_unpruned, depth_mean_pruned = np.mean(depths_unpruned, axis=0), np.mean(depths_pruned, axis=0)
  confusion_mean_unpruned, confusion_mean_pruned = np.rint(np.mean(confusions_unpruned, axis=0)), np.rint(np.mean(confusions_pruned, axis=0))
  accuracy_mean_unpruned, accuracy_mean_pruned = np.mean(accuracies_unpruned, axis=0), np.mean(accuracies_pruned, axis=0)
  precision_mean_unpruned, precision_mean_pruned = np.mean(precisions_unpruned, axis=0), np.mean(precisions_pruned, axis=0)
  recall_mean_unpruned, recall_mean_pruned = np.mean(recalls_unpruned, axis=0), np.mean(recalls_pruned, axis=0)
  f1_mean_unpruned, f1_mean_pruned = np.mean(f1s_unpruned, axis=0), np.mean(f1s_pruned, axis=0)

  # plot the average results
  plt.scatter(depth_mean_unpruned, accuracy_mean_unpruned, s=40, c='red')
  plt.scatter(depth_mean_pruned, accuracy_mean_pruned, s=40, c='blue')
  plt.savefig('accuracies_depths_' + datatype + '.pdf')
  
  # form tuples of (unpruned, pruned) results for the metrics
  depth_mean = (depth_mean_unpruned, depth_mean_pruned)
  confusion_mean = (confusion_mean_unpruned, confusion_mean_pruned)
  accuracy_mean = (accuracy_mean_unpruned, accuracy_mean_pruned)
  recall_mean = (recall_mean_unpruned, recall_mean_pruned)
  precision_mean = (precision_mean_unpruned, precision_mean_pruned)
  f1_mean = (f1_mean_unpruned, f1_mean_pruned)

  # return a tuple of (depth, confusion, accuracy, precision, recall and f1)
  return (depth_mean, confusion_mean, accuracy_mean, precision_mean, recall_mean, f1_mean)
