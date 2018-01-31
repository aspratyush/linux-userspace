## ML Strategy
- Analyzing ML problems to point direction about whats best to try to improve performance.
- ML strategy is changing in DL era.

### General ideas to improve performance
- More data
- More diverse data
- train longer with GD
- try Adam optimizer
- try bigger N/W
- try smaller N/W
- try dropout
- Add L2 regularization
- N/W archietcture:
  - activation functions
  - # of hidden units

### Orthogonalization
- Break the problem into distinct steps, and address each separately.
- Be clear about what to tune
- Always prefer orthogonals over a parameter that affects multiple things
- **Chain of assumption in ML**:
  - Fit training set well (orthogonals : bigger N/W, Adam, etc.)
  - Fit dev-set well (orthogonals : regularization, bigger training set)
  - Fit test set well (orthogonals : bigger dev-set)
  - Performs well in real world (orthogonals : change dev-set, change cost function)

### Evaluating performance
- Set up a single real number evaluation metric.
- Problem with Precision - Recall:
  - PRECISION : TP/(TP+FP) - % of total predictions which is actually True
  - RECALL : (TP/TP+FN) - % of actual total which is actually True
  - Often there is a trade-off between Precision and Recall
  - Since these are 2 params, how to decide which model to use!
- **Use F1-SCORE** :
  - Harmonic mean of Precision and Recall

### Satisficing and optimizing metrics
- If we have $N$ metrics to decide upon, chose 1 optimizing and N-1 Satisficing (reach some threshold).
  - for e.g : maximize accuracy subject to running time < 100ms
  - another e.g : maximize accuracy s.t. $\le 1$ FP every 24 hours.

### Train-dev-test set distributions
- Iterate on dev set till happy with the classifier, and then run on the test set
- dev and test set should come from the same distribution.
- Going by orthogonalization, below should be separate concerns:
  - Define a metric
  - Solve the metric
- dev/test set should do well on the application.

### Human-level performance
- Havinf estimate of 'human-level performance' helps understand what to address (bias / variance).
- **Bayes optimal error** : best possible accuracy for a function mapping X to Y
- **NOTE** : Human performance is proxy for Bayes error for CV
- Knowing how human performance works may help in better results
- Following may help achieve human-like or just better performance:
  - labelled data from humans
- **Avoidable bias** : (Human - Train) error
- **Variance** : (Train - dev) error

- Examples:
  1. Human - 1%, Train - 8%, Dev - 10% : Address 'avoidable bias'
  2. Human - 7.5%, Train - 8%, Dev - 10% : Address 'variance'.

### Surpassing human-level performance
- E.g.:
  1. Team - 0.5%, human = 1%, train - 0.6%, dev - 0.8% : avoidable bias = 0.5%
  2. Team - 0.5%, human = 1%, train - 0.3%, dev - 0.4% : making progress is very UNCLEAR as of now

- ML doing better (structured data available, not natural perception scenario, lots of data exists):
  1. Online advt
  2. product Recommendation
  3. Logistics (predict  transit time)
  4. Loan approvals

- Given enough data, ML algos have started outperforming human-performance:
  1. Speech Recognition
  2. Some image Recognition
  3. Medical imaging, etc.

### Improving model performance
1. Avoidable bias :
  - Train bigger model
  - Train longer
  - Better optimization algorithm (momentum/RMSProp/Adam)
  - NN architecture (RNN/CNN)
  - Hyperparamaters

2. Variance:
  - More data
  - regularization (L2, dropout, data augmentation)
  - NN architecture
  - Hyperparamaters search

### Error Analysis

#### Ceiling on class performance
- Get $\approx$ 100 mislabelled dev-set examples.
- Count up how many are the class we are interested in.
  - Say 5/100 are getting mislabelled. No gain in working on this class.
  - Say 50/100 are getting mislabelled. Major gains possible working on this class.
- **Make a table** of these 100 dev-set examples as rows, classes as columns and remarks column.
- Sum up and find mislabelled contribution per class.

#### Incorrectly labelled I/P data
- DL algorithms are quite robust to errors in the training set.
- Add 1 more column in the table "Incorrectly labelled".
- Then :
  - x1 = Get 'overall dev-set error'
  - x2 = Get 'Error due to incorrect labels'
  - x3 = Get 'Error due to other causes'
    - if x1 >> x2 and x3 >> x2, no point in working on incorrect labels
    - if x1 > x2 and x3 > x3, we may need to work on these.

- **Advice** :
  - Setup dev-test set and metrics
  - Build you 1st system soon, and then iterate!
  - Use **Bias-Variance analysis** and **Error analysis** to prioritize next steps.

#### Mismatched training and dev-test set
- Suggested :
  - TRAIN : 98% train-set, 50% dev-set
  - DEV : 25% dev-set
  - TEST : 25% dev-set

#### Diagnosing problems
- E.g#1 :
  - Train error : 1%
  - Training-dev error : 9%
  - Dev-set error : 10%
  - $\Rightarrow$ Variance problem

- E.g#2 :
  - Train error : 1%
  - Training-dev error : 1.5%
  - Dev-set error : 10%
  - $\Rightarrow$ Data mismatch problems

- E.g#3 :
  - Train error : 10%
  - Training-dev error : 11%
  - Dev-set error : 12%
  - $\Rightarrow$ Avoidable bias problem

- E.g#4 :
  - Train error : 10%
  - Training-dev error : 11%
  - Dev-set error : 20%
  - $\Rightarrow$ Bias and data mismatch problem

- **NOTE** : Human performance <AVOIDABLE BIAS> Training Error <VARIANCE> Training-dev error <DATA MISMATCH> Dev error <OVERFIT TO DEV-SET> Test error

- Synthesize training data to make training data match dev-test distribution.


### Transfer Learning
- When a lot of data exists for domain, but lesser data for specific case being addressed.
- Knowledge of the domain (images / audio, etc.) can be transferred from Pre-trained networks.

- From a PRE_TRAINED network, remove last layer, add new last layers and FINE-TUNE.
  - retrain ONLY last layer if dataset is less.
  - retrain larger (potentially full network) if enough data available.

### Multi-task Learning
- multi-labels provided by the last layer
- loss is the usual logistic loss:
\[
\mathcal{L} = \frac{1}{m} \sum_{i=1} \sum_{j=1} \mathbf{L}(\hat{y}_j^{(i)}, {y}_j^{(i)}) \mathcal{I}(\hat{y} \ne ?) \tag{1}
\]
- Unlike softmax regression which assigned single label to each image, this assigns multiple labels per image.
- Sometimes multi-task learning is better than ensemble learning.
  - benefit comes from having similar lower-level features
  - generally works better than ensemble when we have enough deep network

### End-to-end learning
- avoids multiple processing blocks
- works better in much HIGHER amount of data
- generally, we have enough data for individual tasks available. End-to-end labels not available well.
- E.g#1 (Speech Rec):
\[
audio \xrightarrow{MFCC} features \xrightarrow{ML} phonemes \xrightarrow words \xrightarrow transcript (y) \tag{2}
\]

\[
audio \xrightarrow{end-to-end \ deep-learning} transcript (y) \tag{3}
\]

- E.g#2 (Face rec):
\[
image\ with\ humans \xrightarrow{\text{end-to-end deep learning}} \text{Face Recognition} \tag{4}
\]

\[
\text{image} \xrightarrow{\text{Face detection}} \text{zoom face} \rightarrow \text{Identity Recognition} \tag{5}
\]

- E.g#3 (Self-drive):
\[
image (radar/lidar) \xrightarrow{DL} cars, pedestrians \xrightarrow{motion\ planning} route \xrightarrow{control} steering \tag{6}
\]
- Use DL for individual components. Carefully choose X to Y depending on data that we have.

- Pros:
  - lets the data speak
  - less hand-designing features / intermediate components
- Cons:
  - May need large amount of data
  - excludes useful hand-designed components (some understanding can be injected using these features)

- **How to decide**:
  - Do we have **enough** data regarding complexity of to map x to y.
