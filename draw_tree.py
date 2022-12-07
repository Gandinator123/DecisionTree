import matplotlib.pyplot as plt

def draw_tree(root, max_depth, id=None, type=None):
  """
  A function to draw the tree using matplotlib.
  Arguments:
    - the root of the tree
    - the depth of the tree
    - the id and type for file writing purposes
  Returns nothing.
  """

  # set up a new figure large enough to show the graph
  fig, ax = plt.subplots()
  ax.set_xlim([0, 100])
  ax.set_ylim([0, 10 * max_depth])

  # these are matplotlib.patch.Patch properties for showing the text boxes
  props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

  # a recursive function that goes down the tree and draws each node
  def draw_tree_recursive(node, parent_coord, child_coord, left, right):
    """
    Draw the tree recursively.
    Arguments:
      - the node to draw
      - the coordinates of the parent and child to connect with a line
      - the left and right coords of the plottable area, so that the graph builds in the correct half
    Returns nothing.
    """

    # if the node actually exists
    if node is not None:
      # get the coordinates of the child
      child_x, child_y = (child_coord)

      # if the node is a leaf, show Leaf: {value}
      if node.leaf:
        text = f"Leaf: {node.value}"

      # else, show {attribute} < {value}
      else:
        text = f"x{node.attribute} < {node.value}"
      
      # place a text box at the coordinates of the child to represent the node
      t = ax.text(child_x * 100, child_y * 10 * max_depth, text, fontsize=8, verticalalignment='top', bbox=props, ha='center', va='bottom')
    
      # for all nodes other than the root, draw a line connecting parent and child
      if node.parent is not None:
        # get parent and child coordinates
        parent_x, parent_y = (parent_coord)
        point1 = [parent_x * 100, parent_y * 10 * max_depth]
        point2 = [child_x * 100, child_y * 10 * max_depth]
        x_values = [point1[0], point2[0]]
        y_values = [point1[1], point2[1]]

        # draw the line
        plt.plot(x_values, y_values, linestyle="-")

      # define coords of next nodes
      left_coord = ((child_x + left) / 2, child_y - 0.07)
      right_coord = ((child_x + right) / 2, child_y - 0.07)

      # recursively draw the left and right sides
      draw_tree_recursive(node.left, (child_x, child_y), left_coord, left, (right + left) / 2)
      draw_tree_recursive(node.right, (child_x, child_y), right_coord, (right + left) / 2, right)
      
  # start recursion at the root node of the tree, at the top of the plot and centered horizontally
  draw_tree_recursive(root, None, (0.5, 1), 0, 1)

  # finalise the plot and show it
  plt.tight_layout()
  plt.axis('off')

  # if necessary, save the figure at image_{id}_{type}.png, e.g. image_0_unpruned.png
  plt.savefig('image_' + str(id) + '_' + type + '.pdf')

  # plt.show()
