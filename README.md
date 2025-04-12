<h1 align="center">Generating Negative Samples for Multi-Modal Recommendation</h1>

## Setup
### Requirements

The following environment has been tested and recommended, you can use either pip or conda to create it:
```txt
python==3.10
pytorch==2.1.2
tqdm==4.66.1
numpy==1.24.4
transformers==4.46.1
wandb==0.16.2 (if used)
```
### Datasets
The Amazon datasets can be downloaded at https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews

## Usage
### Train
To train the models, you should first run the proprocessing script: 
```python
python src/preprocess.py --dataset baby 
```

Then you should train the base recommender model:
```python
python src/lightgcn.py --dataset baby
```

Finally, you can train the NegGen model by running following command:
```python
python src/neggen.py --dataset baby --lambda 0.4 --alpha 1e-2 --tau 0.2 
```
