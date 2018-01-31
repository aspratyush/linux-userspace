# FastAI

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
  - Change LR every mini-batch
- ```learn.save``` to snapshot the current model

## Good reads
- **Neural doodle : Alex Champandard**, Semantic style transfer and turning 2-bit doodles into fine artworks, 2016.
- **Zeiler Fergus 2014**, Visualizing and understanding CNNs.
- **Best learning rate : Lewis 2015** : Cyclical Learning Rates for training neural networks.
  - Keep doubling LR from an initial point.
  - Loss will start diverging after a particular LR value.
  - ```plot : LR vs loss```. Choose LR for which loss still improves. Use ```learn.sched.plot()```. Gives plot of LR (log-scale) vs loss.
  - run every time there's a change in the network (add/mute, etc. the layers)
- **Snapshot ensembles**
  - Use cosine annealing
  - save weights at every cosine minima and average the results
