## Latent Feature-based Data Splits to Improve Generalisation Evaluation: A Hate Speech Detection Case Study

This repository contains the code corresponding to the paper *Latent Feature-based Data Splits to Improve Generalisation Evaluation: A Hate Speech Detection Case Study*.


With the ever-growing presence of social media platforms comes the increased spread of harmful content and the need for robust hate speech detection systems.
Such systems easily overfit to specific targets and keywords, and evaluating them without considering distribution shifts that might occur between train and test data overestimates their benefit.
We challenge hate speech models via new train-test splits of existing datasets that rely on the clustering of models' hidden representations.
We present two split variants (SUSBET-SUM-SPLIT and CLOSEST-SPLIT) that, when applied to two datasets using four pretrained models, reveal how models catastrophically fail on blind spots in the latent space.
This result generalises when developing a split with one model and evaluating it on another.
Our analysis suggests that there is no clear surface-level property of the data split that correlates with the decreased performance, which underscores that task difficulty is not always humanly interpretable.

In the following, we detail the steps for obtaining latent representations, creating the data splits, and training and evaluating a model.


## Requirements
The requirements can be found in `requirements.txt`.
To install a virtual environment with all the depencencies, run the following commands:

Create a virtual environment:

```
python3 -m venv venv
```

Activate the environment:
```
source venv/bin/activate
```

Install the requirements:
```
python3 -m pip install -r requirements.txt
```

## 1. Retrieve Latent Representations
Our data split is based on the latent representations of a language model. The following steps describe how the
representations can be obtained.
### 1. Specify your parameters in a yaml file
An example yaml for retrieving latent representations can be found in `yamls/example_obtain_hiddens.yml`.
Additional to the usual parameters, make sure to set `save_hiddens: True`. If a bottleneck is desired to reduce 
the dimensionality of the hidden representations set `bottleneck: 50`, otherwise comment out this line.

Note that to obtain the hidden representations for all examples, the test set and train set need to be the same,
as the hidden representations are obtained when testing the model. 
The hidden representations and labels are then saved to a folder called `hiddens` in the destination path.


### 2. Run the model
```
python scripts/model_scripts/run.py <your_yaml.yaml>
```

## 2. Split the data with the proposed splits

Run `scripts/split_dataset/split.py` using the following command with your parameters:
```
python scripts/split_dataset/split.py
--hidden_files <path_to_hiddens.pkl>      # path to the hidden files, saved as pkl
--hidden_labels <path_to_labels.pkl>      # path to the labels for the data, corresponding to the hidden file, saved as pkl
--folder_to_save <folder_path>            # folder where to save the indices for the new data split
--split <closest/subset>                  # splitting method: either "closest" or "subset"
--test_ratio <0.1>                       # desired test set ratio', default=0.1
--n_classes <2>                          # number of classes considered in the dataset
--desired_hate_ratio <0.33>               # desired hate ratio in test set
--desired_offensive_ratio <0.33>          # desired offensive ration in test set, if applicable, default=None
```

Other, optional parameters can be added:
```
--max_clusters <50> # maximum number of clusters considered in the algorithm', default=50
--seed <42> # cluster seed, default=42
--umap <50> # desired dimension for dimensionality reduction for the hiddens with umap, default=False
```


## 3. Train a model on the new data split
### 1. Specify your parameters in a yaml file
Example files to train a model on the obtained data split can be found in `yamls/example_closest_split.yml`.
Make sure to include `split_seed` in the yaml file with the path to the indices for the train test split 
(if you did not change its name, it should end with `indices.txt`).

### 2. Run the model
```
python scripts/model_scripts/run.py <your_yaml.yaml>
```


## 4. Evaluate a model:
### If you want to test a model from a hugging face checkpoint
#### 1. Specficy your parameters in a yaml file as explained above
#### 2. Run the model 
Add `test` after the yaml file:
```
python scripts/model_scripts/run.py <your_yaml.yaml> test
```
### If you want to test a model from a local checkpoint
#### 1. Specify your parameters in a yaml file 
Importantly, you need to specify `load_from` in the embeddings section - this is the path to the checkpoint that you want to load your model from. 
For example:
```
classifier:

  embeddings:
    load_from: "model_folder_with_checkpoints/best.pt"
    embeddings: "roberta-base"                           # roberta-base or bert-base-cased
    tokenizer: "roberta-base"                            # roberta-base or bert-base-cased
```
Make sure you specify a folder that includes a checkpoint (ending with .pt).

### 2. Run the model 
Add `test` to the command after specifying the yaml file:
```
python scripts/model_scripts/run.py <your_yaml.yaml> test
```

## 5. Different Data Split Scenarios
You can specify different data split scenarios in the .yaml file, some of them have been mentioned above.
### 1. Train, Dev, and Test sets given (as indices files)
An example can be found in `yamls/example_standard_split.yml`.
Add to the yaml file:
  ```
  data:
    path: "data/"           # path to data folder
    train_file: "train.csv" # path to train set in csv format
    dev_file: "dev.csv"     # path to dev set in csv format
    test_file: "test.csv"   # path to test set in csv format
  ```

### 2. Dataset given, train, dev, and test sets created by a random split  
Add to the yaml file:
  ```
  data:
    path: "data/"               # path to data folder
    train_file: "all_data.csv"  # path to train set in csv format
    split_seed: 42              # seed used to split dataset
  ```
### 3. Dataset given, train and test sets determined by indices file, dev set split randomly from the train set  
Add to the yaml file:
  ```
  data:
    path: "data/"                                       # path to data folder
    train_file: "train.csv"                             # path to train set in csv format
    split_seed: "data/hatexplain/new_split_indices.txt" # path to file with indices for train and test set
  ```

# 4. Data
The data used in this project are hate speech detection datasets. Specifically, this work makes use of the following two datasets:
## Reddit Data
The Reddit dataset from "A Benchmark Dataset for Learning to Intervene in Online Hate Speech" (Qian et al. 2019) is used as a development dataset. The original data can be found at https://github.com/jing-qian/A-Benchmark-Dataset-for-Learning-to-Intervene-in-Online-Hate-Speech.

The dataset is licensed under the Creative Commons Attribution-NonCommercial 4.0 International Public License. To view a copy of this license, visit https://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

## Gab and Twitter Data
The test dataset used in this work is the HateXplain Dataset, introduced in "HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection" by  Mathew et al., 2020.
The oringinal data can be found at https://github.com/hate-alert/HateXplain. 

The data is published under the MIT license.
