# FastAI

## Creating world class classifiers
1. Enable data augmentation
2. (Imp) Use **LR-finder** to get the highets LR for which loss still improves
3. Train last layer for 1-2 epochs
4. (Imp) Train last layer _with data augmentation_ (`precompute=False`) for 2-3 epochs with `cycle_len=1`
5. (Imp) Unfreeze all the layers
6. (Imp) Set earlier layers to $3x-10x$ lower LR than next higher layer. (10x : for ImageNet-similar data, 3x: for diff data) 
7. Use **LR-finder** again
8. (Imp) Train full network with `cycle_mult=2` until over-fitting.

- **Summary** : 
  1. `lr_find`
  2. Train last layer with n/w frozen
  3. Train unfrozen network with differential-LRs.


## Setup
#### Options
- crestle.com
- paperspace.com
- AWS

#### Instance Setup
- Run the following:
```
$ curl http://files.fast.ai/setup/paperspace | bash
$ cd fastai
$ git pull
$ jupyter notebook
```

#### Download data
```
$ wget http://files.fast.ai/data/dogscats.zip
```

#### Forums
- forums.fast.ai
- Use extensively.

#### Jupyter notebooks tricks
1. Shift+Tab :
    - ONCE : function signature
    - TWICE : detailed documentation in pop-up dialog
    - THRICE : documentation in overlay window
2. ?func : documentation in overlay window
3. ?? func : src-code overlay window
4. !cmd : running bash-script


## Class 1 (CNNs)
- order --> 2 class, multi-class, structured data (sales forecasting), language, collaborative filtering, where object is (heatmap)

### Deep Learning - advantages
- Infinitely flexible functions
- all purpose parameter fitting
- fast and scalable
- **Universal approximation theorem** : network with several linear and non-linear units can approximate any mathematical function as long as we add enough parameters -- _this can provably be shown_. (several hidden layers)
- **Gradient descent** used to compute the parameters to fit the unknown function. (parameter learning)

### Novel application areas
- Fraud detection, sales forecasting, product failure prediction, prcing, credit risk, customer retention/churn, recommendation systems, ad optimization, anti-money laundering, resume screening, sales prioritization, call centre routing, store layout, store location optimization, staff scheduling.



## Class 2
==========

### Improving models
#### Data augmentation
- Perform transforms on 4-crops of the I/P image.
  - geometry (scale, shear, translate, rotate, reflect)
  - contrast

#### Transfer Learning
- Retain network till penultimate layer
- **Finetune** : Learn this pre-trained network + new linear layer.

#### SGRD (stochastic GD with restarts)
- LR annealing
  - Reduce the LR once reaching minima.
  - Traditional annealing used samples on a linear curve.
- Better : **cosine annealing** - ensures we move out of narrow valley (and thus, spikes) and end up in a wide valley.
  - Change LR every mini-batch `learn.save` to snapshot the current model


#### Differential LRs
`learn.unfreeze()` low ($1e-4$) rate for earlier layers, mid-LR for middle layers $1e-3$ and high LR ($1e-2$) for FC/fine-tuning layers.


### Handling new tasks

#### I/P data
- data could be as images, or csv file containing name and class
- Use `pandas` to get a sense of the data (`data.pivot_table(index=class, aggfunc=len).sort_values('id', ascending=True)`
- Split data into **train**, **validation** (approx. 20%) and **test**. 
- get a sense of how data looks like by :
  1. seeing images, 
  2. size, 
  3. histogram of image sizes
- **dictionary comprehension** : extremely useful!!! Also, `zip`

#### Initial model
- get some data (~10%), use batch_size=64 (or, keep halving on CUDA-OOM error).
- Run the pre-trained network

#### Increase the data size (helps remove overfitting)
- VGG cannot handle arbitrary sizes
- Recent fully convoltuional models can (ResNet, etc.)

#### Go through standard process

#### Change to using a recent model
- ResNet
- ResNext


#### O/P activation
- Multi-label classification : sigmoid (softmax used for single-label case only)
- Single-label classification : softmax (takes 1 I/P, gives 1 O/P)


## Good reads
- **Neural doodle : Alex Champandard**, Semantic style transfer and turning 2-bit doodles into fine artworks, 2016.
- **Zeiler Fergus 2014**, Visualizing and understanding CNNs.
- **Best learning rate : Lewis 2015** : Cyclical Learning Rates for training neural networks.
  - Keep doubling LR from an initial point.
  - Loss will start diverging after a particular LR value.
  - `plot : LR vs loss` Choose LR for which loss still improves. Use `learn.sched.plot()` 
gives plot of LR (log-scale) vs loss.
  - run every time there's a change in the network (add/mute, etc. the layers)
- **Snapshot ensembles**
  - Use cosine annealing
  - save weights at every cosine minima and average the results
- **TTA**
    - Test time augmentation
- **Unbalanced datasets, recent paper**
  - Make copies of the rare classes
- workshop before Part1 containing basic ML libraries
- **feather** format in pandas - RAM dump

## Good links
- (Decoding ResNet) [https://teleported.in]
- (Medium) [Apil Tamang] Summary of L2
- (Medium) [surmenok] Deep dive of `lr_find` -- very popular!
- (FastAI, AWS, Kaggle basics) [https://github.com/reshamas/fastai_deeplearn_part1]
- (LR vs batch size) [https://miguel-data-sc.github.io/2017-11-05-first] Visualizing learning rate vs batch size
- (Visual CNN understanding) [https://www.youtube.com/watch?v=Oqm9vsf_hvU]

## Challenges
- text : Rossman
- image : dog breed, satellite
