from build_tree import read_data, decision_tree_learning
from cross_validate_tree import cross_validation
from numpy.random import default_rng
from draw_tree import draw_tree

def main():
  # set up the random generator
  seed = 69
  rg = default_rng(seed)

  # load the clean dataset
  (ds_x, ds_y, ds_classes) = read_data("wifi_db/clean_dataset.txt")

  # build the full clean dataset
  dataset = (ds_x, ds_y)

  # build a tree trained on the entire clean dataset for visualisation purposes
  clean_tree_root, clean_tree_depth = decision_tree_learning(dataset)
  draw_tree(clean_tree_root, clean_tree_depth, id="clean", type="unpruned")

  # run cross-validation on the clean dataset
  depth, confusion, accuracy, precision, recall, f1 = cross_validation(dataset, datatype="clean", random_generator=rg)

  # write results of evaluation
  with open('result_clean.txt', 'w') as f:
    f.write(
      """
      CLEAN DATASET\n
      Unpruned:\n
      Average depth: %s\n
      Average confusion matrix: %s\n
      Average accuracy: %s\n
      Average precision: %s\n
      Average recall: %s\n
      Average f1: %s\n
      \n
      Pruned:\n
      Average depth: %s\n
      Average confusion matrix: %s\n
      Average accuracy: %s\n
      Average precision: %s\n
      Average recall: %s\n
      Average f1: %s\n
      \n
      Average depth reduction: %s\n
      Average accuracy improvement: %s\n
      """
      % (
        str(depth[0]),
        str(confusion[0]),
        str(accuracy[0]),
        str(precision[0]),
        str(recall[0]),
        str(f1[0]),

        str(depth[1]),
        str(confusion[1]),
        str(accuracy[1]),
        str(precision[1]),
        str(recall[1]),
        str(f1[1]),

        str(depth[0] - depth[1]),
        str(accuracy[1] - accuracy[0])
      )
    )

  # load the noisy dataset
  (ds_x, ds_y, ds_classes) = read_data("wifi_db/noisy_dataset.txt")

  # run cross-validation on the noisy dataset
  dataset = (ds_x, ds_y)
  depth, confusion, accuracy, precision, recall, f1 = cross_validation(dataset, datatype="noisy", random_generator=rg)

  with open('result_noisy.txt', 'w') as f:
    f.write(
      """
      NOISY DATASET\n
      Unpruned:\n
      Average depth: %s\n
      Average confusion matrix: %s\n
      Average accuracy: %s\n
      Average precision: %s\n
      Average recall: %s\n
      Average f1: %s\n
      \n
      Pruned:\n
      Average depth: %s\n
      Average confusion matrix: %s\n
      Average accuracy: %s\n
      Average precision: %s\n
      Average recall: %s\n
      Average f1: %s\n
      \n
      Average depth reduction: %s\n
      Average accuracy improvement: %s\n
      """
      % (
        str(depth[0]),
        str(confusion[0]),
        str(accuracy[0]),
        str(precision[0]),
        str(recall[0]),
        str(f1[0]),

        str(depth[1]),
        str(confusion[1]),
        str(accuracy[1]),
        str(precision[1]),
        str(recall[1]),
        str(f1[1]),
        
        str(depth[0] - depth[1]),
        str(accuracy[1] - accuracy[0])
      )
    )

if __name__ == "__main__":
  main()
