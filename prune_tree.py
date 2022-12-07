import numpy as np
from evaluate_tree import evaluate_tree

def prune_tree(tree_root, tree_node, validate_dataset):
  """
  Prune the tree by checking if tree performs better without certain leaves.
  Arguments:
    - the root of the tree, which gets overwritten with new pruned trees
    - the current node of the tree that we are operating on
    - the data to test the tree against
  Returns:
    - the root of the newly pruned tree
  """

  # get the training data that is valid for this node and its subtrees
  (x_train, y_train)  = tree_node.train

  # if the tree has been pruned such that the tree node doesn't exist any more, we can't do anything
  if not tree_node:
    return tree_root

  # if connected to two leaf nodes, try to compare accuracies
  if tree_node.left.leaf and tree_node.right.leaf:
    # evaluate tree as it is
    unchanged_accuracy = evaluate_tree(validate_dataset, tree_root)[1]

    # store left and right leaf nodes in temporary nodes
    temp_left, temp_right = tree_node.left, tree_node.right

    # store all the features of the current tree node
    temp_value = tree_node.value
    temp_attribute = tree_node.attribute

    # store the values of the left and right leaf nodes
    temp_value_left, temp_value_right = tree_node.left.value, tree_node.right.value

    # (perhaps temporarily) 'remove' left and right nodes and make current node a leaf
    tree_node.left = None
    tree_node.right = None
    tree_node.leaf = True

    # calculate majority element between two leaf values for pruning    
    leaf_labels = y_train[(y_train==temp_value_left) | (y_train==temp_value_right)]

    unique_labels, counts = np.unique(leaf_labels, return_counts=True)
    ind = np.argmax(counts)
    majority_element = unique_labels[ind]

    # try out majority element instead
    tree_node.value = majority_element
    changed_accuracy = evaluate_tree(validate_dataset, tree_root)[1]
    
    # # if there's an accuracy improvement, set value to best of both
    if changed_accuracy >= unchanged_accuracy:
      return prune_tree(tree_root, tree_node.parent, validate_dataset)

    # else, revert back to original node like nothing happened
    else:
      tree_node.left, tree_node.right = temp_left, temp_right
      tree_node.value, tree_node.attribute = temp_value, temp_attribute
      tree_node.leaf = False
      return tree_root

  # while the current node isn't connected to two leaf nodes, travel down
  else:
    if tree_node.left:
      if not tree_node.left.leaf:
        tree_root = prune_tree(tree_root, tree_node.left, validate_dataset)

    if tree_node.right:    
      if not tree_node.right.leaf:
        tree_root = prune_tree(tree_root, tree_node.right, validate_dataset)

    return tree_root