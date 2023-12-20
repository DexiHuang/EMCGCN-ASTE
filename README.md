# COMP7607
The training and testing instructions for the model have remained largely unchanged compared to the source code.

- Model Training:
```
python main.py --mode train --dataset res14 --batch_size 16 --epochs 100 --model_dir savemodel/ --seed 42 --pooling avg --prefix ../data/D2/
```

- Model Testing: 
```
- python main.py --mode test --dataset res14 --batch_size 16 --epochs 100 --model_dir savemodel/ --seed 42 --pooling avg --prefix ../data/D2/
```
We have used a different random seed, which is 42.

- Ablation Study:  

In the provided code in the features branch, we have implemented the ablation study. By modifying the data dimensions of the input model and the corresponding model parameters, we adjust the linguistic features of the input model. You can adjust the input linguistic features 
```
by modifying the command parameters such as --post, --deprel, --postag, and --synpost, setting them to true or false.
```
```
By setting the --relation_constraint parameter to true or false
```
, you can adjust the calculation method of the loss function.
```
By setting the --refine parameter to true or false
```
, you can adjust the network structure to decide whether to use the refine strategy.


# EMC-GCN

Code and datasets of our paper "[Enhanced Multi-Channel Graph Convolutional Network for Aspect Sentiment Triplet Extraction](https://aclanthology.org/2022.acl-long.212/)" accepted by ACL 2022.

## Requirements

- python==3.7.6

- torch==1.4.0
- transformers==3.4.0
- argparse==1.1

pip install scikit-learn
pip install protobuf==3.20.0
## Training

To train the EMC-GCN model, run:

```
cd EMC-GCN/code
sh run.sh
```
or
```
python main.py --mode train --dataset res14 --batch_size 16 --epochs 100 --model_dir savemodel/ --seed 1000 --pooling avg --prefix ../data/D2/
```

## Inference

To test the performance of EMC-GCN, you only need to modify the --mode parameter.
```
python main.py --mode test --dataset res14 --batch_size 16 --epochs 100 --model_dir savemodel/ --seed 1000 --pooling avg --prefix ../data/D2/
```

## Acknowledge

We appreciate all authors from this paper: "Grid Tagging Scheme for Aspect-oriented Fine-grained Opinion Extraction", because the code in this repository is based on their work [GTS](https://github.com/NJUNLP/GTS).

