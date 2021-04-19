## NePTuNe: Neural Powered Tucker Network.

This codebase contains PyTorch implementation of the paper:

> NePTuNe: Neural Powered Tucker Network.
> Shashank Sonkar, Arzoo Katiyar, and Richard G. Baraniuk.
> [[Paper]](http://arxiv.org/abs/2104.07824)

The codebase is inspired from [TuckER's github repository](https://github.com/ibalazevic/TuckER).
TuckER is the PyTorch implementation of the paper:

> TuckER: Tensor Factorization for Knowledge Graph Completion.
> Ivana Balažević, Carl Allen, and Timothy M. Hospedales.
> Empirical Methods in Natural Language Processing (EMNLP), 2019.
> [[Paper]](https://arxiv.org/pdf/1901.09590.pdf)

### Link Prediction Results

Dataset | MRR | Hits@1 | Hits@3 | Hits@10
:--- | :---: | :---: | :---: | :---:
FB15k-237 | 0.366 | 0.272 | 0.404 | 0.547 
WN18RR | 0.491 | 0.455 | 0.507 | 0.557 

### Running the NePTuNE model

To run the model on FB15k-237 dataset, execute the following command:

```
 cd neptune-fb15k237/
 CUDA_VISIBLE_DEVICES=0 python -u main.py 
                --dataset FB15k-237 
                --num_iterations 1500 
                --batch_size 128 
                --lr 0.0005 --dr 1.0 
                --edim 200 --rdim 150 
                --input_dropout 0.3
                --hidden_dropout1 0.2
                --hidden_dropout2 0.4 
                --label_smoothing 0.1
```

To run the model on WN18RR dataset, execute the following command:

```
cd neptune-wn18rr/
CUDA_VISIBLE_DEVICES=0 python -u main.py
                --dataset WN18RR 
                --num_iterations 2500
                --batch_size 128
                --lr 0.003 --dr 1.0
                --edim 500 --rdim 30
                --input_dropout 0.1
                --hidden_dropout1 0.2
                --hidden_dropout2 0.7
                --label_smoothing 0.1
```
Note that WN18RR uses inverse relations, while FB15k-237 does not.

### Requirements

The codebase is implemented in Python 3.6.6. Required packages are:

    numpy      1.15.1
    pytorch    1.0.1
