# Benchmarking Graph Neural Networks for fMRI analysis

This is the original implementation of the paper [Benchmarking Graph Neural Networks for fMRI analysis](https://arxiv.org/abs/2211.08927)
<p align="center">
  <img src=img.png>
</p>
## Requirements
- python 3.9+
- see `requirements.txt`

### Data

The Abide dataset can be requested from [here](http://preprocessed-connectomes-project.org/abide/)
The Mddrest dataset can be requested from [here](http://rfmri.org/REST-meta-MDD)

The Mddrest provide the pre-processed timecourses directly. For ABIDE you can generate the time-courses using nilearn.
You can see an example of how to parcellate the data and extract the timecourses  in `parcellate_example.ipynb`

For ease of use, you can organize the your data into a csv file as provided in the examples in `csvfiles/`
All logic for creating and loading datasets to the framework is in `data.py`.


## Training and Evaluation

Refer to main_{static/dynamic/baseline }.py sto train the respective model from each group. you can add custom models or custom datasets easily to the data.py or networks folders and use the same pipeline to run the new experiments.
For hyperparameters tuning we recommend using [weights&biases](https://docs.wandb.ai/guides/sweeps) sweep tool.

If you find use this work useful, kindly cite our paper:

```
@misc{https://doi.org/10.48550/arxiv.2211.08927,
  doi = {10.48550/ARXIV.2211.08927},
  url = {https://arxiv.org/abs/2211.08927},
  author = {ElGazzar, Ahmed and Thomas, Rajat and van Wingen, Guido},
  title = {Benchmarking Graph Neural Networks for FMRI analysis},
  publisher = {arXiv},
  year = {2022}}
```
