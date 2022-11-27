# How Powerful are Graph Neural Networks in Diagnosing Psychiatric disorders?

This is the original implementation of the paper [Benchmarking Graph Neural Networks for fMRI analysis](www.ss.com)



## Requirements
- python 3.7+
- see `requirements.txt`


## Data Preparation

### Step1: Download the pre-processed Datasets

### Step2: Use an atlas to segment the brain int ROIs and calculate the timecourses and Pearson correlations of the regions.

### Step3: Prepare csv files with demographics and files path. see examples at '/csvfiles'.

## Running models

## Refer to main_{static/dynamic/baseline }.py sto train the respective model from each group. you can add custom models or custom datasets easily to the data.py or networks folders and use the same pipeline to run the new experiments.


## For hyperparameters tuning we recommend using the weights&biases sweep tool.
