import numpy as np
from find_split import find_split

class Node:
  """
  A class to handle everything to do with nodes
  """
  
  def __init__(self, attribute=None, value=None, leaf=False, parent=None, train=[]):
    self.attribute = attribute
    self.value = value
    self.left = None
    self.right = None
    self.leaf = leaf
    self.parent = parent
    self.train = train

def read_data(filepath):
    """
    Function reads data from the files and puts it into arrays
    Arguments:
      - filepath to read from
    Returns:
      - tuple of (x, y, classes) such that x and y form the dataset, and classes is all the labels
    """

    data = np.loadtxt(filepath, dtype=float)
    x = data[:, :-1]
    y_labels = np.array(data[:, -1], dtype=int)
    [classes, y] = np.unique(y_labels, return_inverse=True)
    x = np.array(x)
    y = np.array(y)
    return (x, y, classes)

def decision_tree_learning(train_dataset, depth=1, parent=None):
  """
  Build the decision tree.
  Arguments:
    - matrix containing the dataset: a tuple of (x_train, y_train)
    - depth variable: maximal depth for plotting purposes
  Returns:
    - the node to implement in this position of the tree
  """

  # separate the dataset into x and y
  (x_train, y_train) = train_dataset

  # find all the labels in the training set
  training_labels = np.unique(y_train)

  # if all samples have same label
  if (len(training_labels) == 1):
    # construct a leaf node with this value
    node = Node(value=training_labels[0], leaf=True, parent=parent)

    # the leaf node and the depth
    return (node, depth)

  # else build recursively
  else:
    # find split point, which is a tuple of (attribute, value) representing: x[attribute] < value
    (attribute, value) = find_split(x_train, y_train)

    # make a new decision tree with root as split value
    # this node has the attribute and value calculated in find_split, a reference to its parent and all the training data that is valid from this node down the tree
    node = Node(attribute=attribute, value=value, parent=parent, train=train_dataset)
    
    # find left and right branches recursively by splitting into two, where the left side is x[attribute] < value and the right side is x[attribute] >= value
    (l_indices, ) = np.where(x_train[:, attribute] < value)
    l_x = x_train[l_indices]
    l_y = y_train[l_indices]

    (r_indices, ) = np.where(x_train[:, attribute] >= value)
    r_x = x_train[r_indices]
    r_y = y_train[r_indices]

    # form the datasets and build the tree recursively
    if len(l_indices) > 0:
      l_dataset = (l_x, l_y)
      (l_branch, l_depth) = decision_tree_learning(l_dataset, depth + 1, node)
      node.left = l_branch
    else:
      l_depth = 0

    if len(r_indices) > 0:
      r_dataset = (r_x, r_y)
      (r_branch, r_depth) = decision_tree_learning(r_dataset, depth + 1, node)
      node.right = r_branch
    else:
      r_depth = 0

    # return the node and the maximum depth of the tree
    return (node, max(l_depth, r_depth))

def calculate_max_depth(node):
  """
  Recursively calculate the depth of a tree from a given node
  Arguments:
    - the current node
  Returns:
    - the depth of the tree from a node in the tree
  """ 

  # if the current 'node' isn't actually a node, you're done
  if node is None:
    return 0

  # else recursivley travel down the left and right nodes of the tree until you find the end
  else:
    return 1 + max(calculate_max_depth(node.left), calculate_max_depth(node.right))