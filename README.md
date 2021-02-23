# RippleNet

This repository is a **PyTorch** implementation of a model  that combines RippleNet([arXiv](https://arxiv.org/abs/1803.03467)) and KGCN(https://arxiv.org/abs/1904.12575) together:
> RippleNet: Propagating User Preferences on the Knowledge Graph for Recommender Systems  
Hongwei Wang, Fuzheng Zhang, Jialin Wang, Miao Zhao, Wenjie Li, Xing Xie, Minyi Guo  
The 27th ACM International Conference on Information and Knowledge Management (CIKM 2018)

> KGCN:Wang H, Zhao M, Xie X, et al. Knowledge graph convolutional networks for recommender systems[C]
The world wide web conference. 2019: 3307-3313.

For the authors' official TensorFlow implementation, see [hwwang55/RippleNet](https://github.com/hwwang55/RippleNet).
and [hwwang55/KGCN].https://github.com/hwwang55/KGCN


RippleNet is a deep end-to-end model that naturally incorporates the knowledge graph into recommender systems.
Ripple Network overcomes the limitations of existing embedding-based and path-based KG-aware recommendation methods by introducing preference propagation, which automatically propagates users' potential preferences and explores their hierarchical interests in the KG.



### Files in the folder

- `data/`
  - `book/`
  - `movie/`
    - `item_index2entity_id.txt`: the mapping from item indices in the raw rating file to entity IDs in the KG;
    - `kg_part1.txt` and `kg_part2.txt`: knowledge graph file;
    - `ratrings.dat`: raw rating file of MovieLens-1M;
- `src/`: implementations of this model.



### Required packages
The code has been tested running under Python 3.6, with the following packages installed (along with their dependencies):
- pytorch >= 1.0
- numpy >= 1.14.5
- sklearn >= 0.19.1


### Running the code
```
$ cd src
$ python preprocess.py --dataset movie (or --dataset book)
$ python main.py --dataset movie (note: use -h to check optional arguments)
```
result movie: auc 0.9255 acc 0.8521 better than ripplenet(auc 0.9204 acc 0.844) on same data
