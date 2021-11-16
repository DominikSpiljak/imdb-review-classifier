# imdb-review-classifier

## Setup
1. Download the dataset from [Kaggle IMDBDataset](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
2. Run `pip install -r requirements.txt`
3. Run the script: 
```
usage: train.py [-h] [--dataset DATASET] [--train-to-test-ratio TRAIN_TO_TEST_RATIO] [--tfidf-max-features TFIDF_MAX_FEATURES] [--device {cpu,gpu}] [--batch-size BATCH_SIZE]
                [--num-workers NUM_WORKERS] [--learning-rate LEARNING_RATE] [--num-epochs NUM_EPOCHS] [--save-path SAVE_PATH]

optional arguments:
  -h, --help            show this help message and exit

data:
  --dataset DATASET     Path to dataset file
  --train-to-test-ratio TRAIN_TO_TEST_RATIO
                        Ratio of train to test size when splitting

model:
  --tfidf-max-features TFIDF_MAX_FEATURES
                        Max features to be used during tfidf vectorization

training:
  --device {cpu,gpu}    Device for running the training
  --batch-size BATCH_SIZE
                        Batch size for training
  --num-workers NUM_WORKERS
                        Number of workers for parallel data loading
  --learning-rate LEARNING_RATE
                        Learning rate for training
  --num-epochs NUM_EPOCHS
                        Number of epochs
  --save-path SAVE_PATH
                        Path to save models
```

## Repo structure

### Data
Contains all the scripts used for data loading, transformation and dataset definition

### Models
Contains models and modules written using PyTorch

### argument_parser.py
Module used for parsing command line arguments

### train.py
Entrypoint used for training the model
