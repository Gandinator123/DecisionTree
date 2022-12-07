# WifiDecisionTree

Developed by Rohan Gandhi, Anthony Jones, Chris Myers, Luca Chammah

The code for Introduction to Machine Learning: Coursework 1 (Decision Trees).

To run the code, type:

```
python3 main.py
```

To edit the path to load the dataset, edit the path provided to the read_data functions inside main.py, for example (on line 69 of main.py):

```
(ds_x, ds_y, ds_classes) = read_data("wifi_db/clean_dataset.txt")
```

The program displays the tree trained on the clean dataset, as well as several graphs to demonstrate the accuracy improvement of pruning. These are saved as PDFs.
