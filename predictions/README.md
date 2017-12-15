
## Description
This folder contains:
 
* Several subfolders, each containing:     
  - `config.json`: a configuration file with the exact hyper-parameter values
  - `predictions.epochN`: a file with predictions by the model (highest score on dev set)
  - `scores.csv`: exact scores a system obtained at each epoch of the training procedure
  
- `aggregate.py`: 
a Python script which aggregates the scores, contained in the subfolders

- `model-t_predictions.txt`:
predictions of the template-based system on the development set

## Usage
In order to aggregate evaluation results from multiple
runs of the NN model with different random seed values, run the following command:

```
$ python aggregate.py */scores.csv
```

This will print all the necessary statistics regarding the performance of the model.
