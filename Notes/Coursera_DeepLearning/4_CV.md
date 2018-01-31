# Lecture#1
## Edge Detection
* Instead of hand-picking the numbers in the kernel, have _"back-prop"_ learn the params - such that conv of the filter with the image gives edge detection.
* The learned filter could be a Sobel filter / Scharr filter / a filter thats even better than these hand-picked filters.
* Aim is to let the data and the objective determine the best filter to use.

#### Output size
* I/P size = $n$
* Filter size = $f$
* O/P size = $n-f+1$

## Padding
* Padding ensures :
(a) Same sized O/P,
(b) non-missing edge values in the next activation.

#### Output size with padding
* Padding = $p$
* O/P size = $n+2p-f+1$

#### How much to pad
* **Same Conv** : O/P size is same as I/P size
\[
n+2p-f+1 = n;
p = \frac{f-1}{2}
\]

## Strided Convolutions
* $s$ allows skipping few columns/rows when performing convolutions.
\[
\text{O/P size} = \lfloor \frac{n+2p-f}{s}+1\rfloor
\]

#### Convolutions over volumes
\[
(n \times n \times n_{c}) \times (f \times f \times n_{c} \times n_{c}')
\]
* no. of channels $n_{c}$ should be same for the I/P and the filter
* $n_{c}'$ is the no. of filters being used.

#### One layer of convolutions
- I/P * kernel $\rightarrow$ ReLU(Conv-O/P + bias) $\rightarrow$ O/P
- Comparing with the conv-layer equation:
\[
z^{l} = w^{l}a^{l-1} + b^{l};
a^{l} = g(z^{l})
\]
- where : $w^{l}$ is the conv-filter weights, $b^{l}$ is the bias for the current layer, $z^{l}$ is the linear output, $a^{l}$ is the non-linear activation (which also serves as the I/P to the next layer).

#### Summary of notations
- I/P : $n_{H}^{[l-1]} \times n_{W}^{[l-1]} \times n_{c}^{[l-1]}$
- O/P : $n_{H}^{[l]} \times n_{W}^{[l]} \times n_{c}^{[l]}$
- $p^{[l]}$ : padding, $f^{[l]}$ : filter size, $s^{[l]}$ : stride
- $n_{c}^{[l]}$ : # of filters, $n_{c}^{[l-1]}$ : channels
\[
\Rightarrow n_{H/W}^{[l]} = \lfloor{\frac{n_{H/W}^{[l-1]} + 2p^{[l]} -f^{[l]}}{s} + 1}\rfloor
\]
- **Each filter** : $f^{[l]} \times f^{[l]} \times n_{c}^{[l-1]}$
- **Activations** : $a^{[l]} \rightarrow n_{H}^{[l]} \times n_{W}^{[l]} \times n_{c}^{[l]}$
- **Batch activations** : $A^{[l]} \rightarrow m \times n_{H}^{[l]} \times n_{W}^{[l]} \times n_{c}^{[l]}$
- **weights** : $f^{[l]} \times f^{[l]} \times n_{c}^{[l-1]} \times n_{c}^{[l]}, \text{where: } n_{c}^{[l]} \rightarrow \text{\# of filters}$
- **bias** : $n_{c}^{[l]}$

## Example Conv Network
- "include image"
- Generally, $n_c^{[l]} = 2*n_c^{[l-1]}$, i.e., no. of filters in $l$-layer will be double of filters in $l-1$ layer.
- Generally, last layer involves unrolling the volume of activations into a flat layer (FC), followed by cost computation.

### Maxpool
- Advantages:
(a) reduces computations.
(b) makes feature detector invariant to its position.

- Preserve and extract the most important feature from a local region
- involves no learning, once $f$ (kernel size) and $s$ (stride) are fixed.
- usually, padding used = 0
- NOTE : **average pooling** used sometimes in very deep networks, to collapse a $n \times n \times n_c^{[l-1]}$ to $1 \times 1 \times n_c^{[l]}$.
- Common values : f=2, s=2 and f=3, s=2

### Conv vs FC
- Advantages : param sharing and sparsity of connections
  - Param sharing : If a filter is useful as a feature detector at a location, its probbaly useful in other location too. FC cannot be applied across spatial locations.
  - Sparsity of connections : each O/P in a layer is dependant on a sparse set of connections from the I/P, as against the whole I/P domain in case of FC.

# Lecture#2
## Classical networks

### LeNet - 5
- __LeCunn 98__, Graident based learning applied to document recognition.
- Section 2 and 3 from paper
- I/P(32 x 32 x 1) $\xrightarrow[f=5, s=1]{CONV}$ (28 x 28 x 6) $\xrightarrow[f=2, s=2]{POOL}$ (14 x 14 x 6) $\xrightarrow[f=5, s=1]{CONV}$ (10 x 10 x 16) $\xrightarrow[f=2, s=2]{POOL}$ (5 x 5 x 16) $\xrightarrow[-]{Flatten}$ (1 x 400) $\xrightarrow[400 \times 120]{FC}$ (1 x 120) $\xrightarrow[120 \times 84]{FC}$ (1 x 84) $\xrightarrow[-]{softmax}$ $\hat{y} (1 \times 10)$
- CONV + POOL form 1 layer
- Therefore, L1 --> L2 --> FC --> FC --> softmax : **5 layers**
- Around **60K params**.

### AlexNet
- **Krizhevsky 2012**, ImageNet classification with deep CNNs.
- I/P(227 x 227 x 3) $\xrightarrow[f=11, s=5]{CONV}$ (55 x 55 x 96) $\xrightarrow[f=3, s=2]{POOL}$ (27 x 27 x 96) $\xrightarrow[f=5, \text{same}]{CONV}$ (27 x 27 x 256) $\xrightarrow[f=3, s=2]{POOL}$ (13 x 13 x 256) $\xrightarrow[f=3, \text{same}]{CONV}$ (13 x 13 x 384) $\xrightarrow[f=3, \text{same}]{CONV}$ (13 x 13 x 384) $\xrightarrow[f=3, \text{same}]{CONV}$ (13 x 13 x 256) $\xrightarrow[f=3, s=2]{POOL}$ (6 x 6 x 256 ~ 9216) $\xrightarrow[-]{flatten}$ (1 x 9216) $\xrightarrow[f=3, \text{same}]{FC}$ (1 x 4096) $\xrightarrow[f=3, \text{same}]{FC}$ (1 x 4096) $\xrightarrow[-]{softmax}$ (1 x 1000)
- L1 --> L2 --> L3 --> L4 --> L5 --> FC --> FC --> softmax : **8 layers**.
- Around **60M params**
- Used ReLUs instead of softmax / tanh
- NOTE : AlexNet also used local response normalization after L2, but this is not used anymore.
- AlexNet convinced CV community to use CNNs

### VGG-16
- **Simonyan & Zisserman 2015**, Very deep CNNs for large-scale image recognition
- highly simplified network, CONV(f=3,s=1,same); POOL(f=2,s=2)
- focusses on using CONV + POOL only, instead of complex networks.
- I/P(224 x 224 x 3) $\xrightarrow[]{CONV64 \times 2}$ (224 x 224 x 64) $\xrightarrow[]{POOL}$ (112 x 112 x 64) $\xrightarrow[]{CONV128 \times 2}$ (112 x 112 x 128) $\xrightarrow[]{POOL}$ (56 x 56 x128) $\xrightarrow[]{CONV256 \times 3}$ (56 x 56 x 256) $\xrightarrow[]{CONV512 \times 3}$ (28 x 28 x 512) $\xrightarrow[]{POOL}$ (14 x 14 x 512) $\xrightarrow[]{CONV512 \times 3}$ (14 x 14 x 512) $\xrightarrow[]{POOL}$ (7 x 7 x 512) $\xrightarrow[4096]{FC}$ $\xrightarrow[4096]{FC}$ $\xrightarrow[1000]{softmax}$ $\hat{y}$
- VGG16 since there are 16 layers to learn params in.
- Around **138M params**
- pattern : height and width go down by 2, and channels go up by 2.
- Also, VGG19 is used sometimes.

#### Problem in training very deep networks
- vanishing gradients
- high no. of params may cause over-fitting

### ResNet
- **He 2015**, Deep residual networks for image recognition.
- Use residual blocks containing skip connections.
- The skip connection feeds activations of a previous layer $(l)$ to ReLU of next layer ($l+2$).
- Stacking residual blocks allows forming very deep networks.
- Converting a plain network to residual network involves adding skip connections in every $(l)$ to $(l+2)$ residual block.
- ResNets work because its easy for the network to learn Identity-function in every residual block. Anything learnt by the blocks over identity-function adds to the learning.

### 1x1 convolutions / bottleneck layer
- **Lin 2013**, Network in network
- Can be used to shrink the no. of channels
- Adds complexity by adding extra params

### Inception network
- **Szegedy 2014**, Going deeper with convolutions.
- Instead of picking filter size, use all $f \in \{$(1x1x64), (3x3x128), (5x5x32)$\}$ and maxpool (s=1, same) in the same layer.
- (1 x 1) reduces computations by almost 10 times.
- Each layer is called the _Inception Module_.
- Prev-Activations --> (1x1) || (1x1) (3x3) || (1x1) (5x5) || (POOL) (1x1) --> ConCat layer
- cost computed at intermediate layers also, has a regularizing effect on the network.

## Pratical Advice
### Transfer learning
- always better to start with pre-trained network, instead of training one from scratch from randomized weights.
- transfer knowledge from existing networks to current task.
- Strategy#1 (low dataset)
  - Freeze all layers, and retrain only classification layer
    - done by ```freeze=1``` or ```trainableParameters=0```
  - Pass all I/Ps through frozen network and store the activations to disk.
- Strategy#2 (medium dataset)
  - Unfreeze last few layers based and train those.
- Strategy#3 (large dataset)
  - Retrain the whole network.

### Data augmentation
- Geometric transforms/distortion
  - **Mirroring**
  - **Random cropping**
  - **Rotation**
  - **Shearing**
- Color shifting/distortion - makes n/w invariant to color changes
  - Add / Subtract random fixed values from each channel. E.g :  (+20,-20,+10)
  - PCA color augmentation : keeps color distribution similar across channels.
- NOTE : keep distortion thread, and training thread separate!

### State of CV
- Less data :
  - focus on hand-engineering / network arch,
  - use transfer learning.
- More data --> simpler algorithms.
- Amount of data:
  - Object detection < Image recognition < Speech recognition
- Tips for benchmarks / winning competitions
  - Ensemble (train 3-15 CNNs randomly, and average their O/Ps ($\hat{y}$))
  - Multi-crop at test time (**10-crop**)
    - Use central crop and 4 corners crop for both normal & mirror image, and average the results.
  - These suggestions not good for production systems as they increase the runtime.

### Keras
- create --> compile --> fit(train) --> evaluate(test)


### ResNet-50 using Keras and TensorFlow (identity_block)
- Very deep networks, though can model highly complex functions, face the problem of vanishing / exploding gradients.
- Norm of gradient in shallower layers falls rapidly towards zero.
- ResNet uses "Residual blocks" containing skip-connections of two types:
  - _identity block_ : unchanged I/P added to the next layer.
  I/P --->    CONV2D -->     BN -->       ACT --->  CONV2D -->         BN ---> ACT
   |------------------------------------------------------------------------>|

```
def identity_block(X, f, filters, stage, block):
  # X : I/P tensor (m,n,w,c)
  # f : integer, no. of filters
  # filters : filter size
  # stage : string, stage in ResNet
  # block : string, naming

  # defining name basis
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  # Retrieve Filters
  F1, F2, F3 = filters

  # Save the input value. You'll need this later to add back to the main path.
  X_shortcut = X

  # 1st component
  X = Conv2D(filters=F1, kernel_size=(1,1), stride=(1,1), padding='valid', name= conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis=3, name=bn_name_base+'2a')(X)
  X = Activation('relu')(X)

  # 2nd component
  X = Conv2D(filters=F2, kernel_size=(f,f), stride=(1,1), padding='same', name= conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis=3, name=bn_name_base+'2b')(X)
  X = Activation('relu')(X)

  # 3rd component
  X = Conv2D(filters=F1, kernel_size=(1,1), stride=(1,1), padding='valid', name= conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
  X = BatchNormalization(axis=3, name=bn_name_base+'2a')(X)

  #Skip connection
  X = Add()([X, X_shortcut])
  X = Activation('relu')(X)

  #return
  return X
```

- test using mixture of TensorFlow and Keras
```
tf.reset_default_graph()
with tf.Session() as session:
  np.random.seed(1)
  # I/P activation
  A_prev = tf.placeholder("float", [1,2,3,4])
  # O/P
  A = identity_block(A_prev, f=2, filters=[2,4,6], stage = 1, block = 'a')
  # compute using TF
  X = np.random.randn(3,4,5,6)
  session.run(tf.global_variables_initializer())
  out = session.run([A], feed_dict={A_prev:X, K.learning_phase(): 0})
  print("out = " + str(out[0][1][1][0]))  
```


  - _convolution block_ : passed through conv-block if size in next layer changes.
  I/P --->    CONV2D -->     BN -->       ACT --->  CONV2D -->    BN ---> ACT
   |-------------------------------CONV2D --> BN --> -------------->|


# Lecture 3

## Object detection
- Image classification ----------- Image Localization ----- Image detection
  (single object classification) ;  (localize 1 object)  ;  (multi-object Localization)

### Localization
- Target label : $\hat{y} = [p \ b_x \ b_y \ b_h \ b_w \ c_1 \ c_2 \cdots c_n]$
  - $p$ : probability, presence / absence of the object
  - $b_x b_y b_h b_w$ : bounding box co-ordinates
  - $c_1 c_2 \cdots c_n$ : one-hot encoded vector indicating object class

- Loss :
\[
\mathbf{L} = (\hat{p} - p)^2 + (\hat{b_x} - b_x)^2 + (\hat{b_y} - b_y)^2 + (\hat{b_w} - b_w)^2 + (\hat{b_h} - b_h)^2 + (\hat{c_1} - c_1)^2 + (\hat{c_2} - c_2)^2 + \cdots + (\hat{c_n} - c_n)^2, if \hat{p} = 1
\]\[
\mathbf{L} = (\hat{p} - p)^2, if \hat{p} = 0
\]

- Alternative (better) loss terms :  $c$ : log-likelihood loss, $b$ : squared-loss, $p$ : logistic regression

- Similar examples:
  - Landmark detection in faces
  - Pose detection

### Sliding window
- **Sermanet 2014** : Overfeat : Integrated recogntion, localization and detection using CNNs.
- FC through conv layer
  - Case 1 : 5 x 5 x 16 ---> FC (1 X 400) --> FC (1 x 400) --> softmax (1 x 4)
  - Case 2 : 5 x 5 x 16 ---> Conv2D (5 x 5 x 400) --> Conv2D (1 x 1 x 400) --> softmax (1 x 4)
- Conv layers allow shared computations across 4-crop sliding window in a single pass. (see lecture video)

### Bounding Box predictions
- **Redmon 2015**, YOLO : You only look once : Unified real-time object detection - _NOTE : harder to understand_.
- Output label = $[p \ b_x \ b_y \ b_h \ b_w \ c_1 \ c_2 \cdots c_n]$
- Assuming 3 X 3 grid on the I/P, O/P volume is 3 x 3 x 8 (8-d label per grid).
- **IoU : Intersection over union** (preferred b/w 0.5-0.7)
- **NMS : non-maximum supression** used to get best O/P from multiple proposals.
  - For every class:
    - discard all boxes with $p \le 0.6$
    - pick the box with largest $p$
    - discard all other boxes with IoU $\ge 0.5$

### Anchor boxes
- multiple objects per grid cell.
- $\hat{y}$ is now **n-times** the size, repeating the feature vector n-times (1/anchor box)
- Assuming 2 anchor boxes : $\hat{y} = [p^{[1]} \ b_x^{[1]} \ b_y^{[1]} \ b_h^{[1]} \ b_w^{[1]} \ c_1^{[1]} \ c_2^{[1]} \cdots c_n^{[1]} \ p^{[2]} \ b_x^{[2]} \ b_y^{[2]} \ b_h^{[2]} \ b_w^{[2]} \ c_1^{[2]} \ c_2^{[2]} \cdots c_n^{[2]}]$
- **NOTE** : Object in training set is assigned to the grid that contains the objects mid-point, and also the anchor box for the grid cell with highest IoU.

### Region proposals
- **Girshik 2013**, R-CNN : segmentations --> CNN on blobs --> label + bounding box.
- **Girshik 2015**, Fast R-CNN : Conv implementation of sliding window.
- **Ren 2016**, Faster R-CNN : Faster region proposals.


### YOLO
<TODO>


# Lecture 4

## Face Recognition
- identifying a face from a set of $K$ faces in a database.

### One-shot learning for Face Verification
- Learn from SINGLE example to achieve the objective.
- Learn a **similarity function** which captures the difference between the images.
\[
d(\text{img}_1, \text{img}_2) = \tau
\]\[
\text{if}: d(\text{img}_1, \text{img}_2) \le \tau \rightarrow \text{same}
\]\[
\text{else}: d(\text{img}_1, \text{img}_2) \gt \tau \rightarrow \text{different}
\]

#### Triplets loss
- **Schraff 2015**, FaceNet : A unified embedding for face recogntion and clustering
- At every moment, network looks at 3 images to compute the loss : Anchor(A), Positive(P) and Negative(N).
- tries to *push* the encodings of two images of the same person closer, while *pulling* the encodings of images of different persons further apart.
- Objective :
\[ \textit{given: } d(A,P) = \| f(A) - f(P) \|^{2}, \text{and}, d(A,N) = \| f(A) - f(N) \|^{2}
\] \[
d(A,P) + \alpha \le d(A,N)
\] \[
\Rightarrow d(A,P) + \alpha - d(A,N) \le 0
\] where : $\alpha$ is the **margin** that ensures trivial encodings are not learned, i.e., $f(A) = f(P) = f(N)$ or $f(A) = f(P) = f(N) = 0$
- Therefore, loss function:
\[
\mathcal{L}(A,P,N) = \max{(\|f(A) - f(P)\|^{2} - \|f(A) - f(N)\|^{2} + \alpha, 0)}
\]
- NOTE : choose triplets that for which $d(A,P) \approx d(A,N)$ (difficult examples)

#### Alternative : Image-pair based Learning face similarity function
- Siamese networks --> respective encodings --> logistic regression --> binary classifier.
\[
\hat{y} = \sigma{(\Sigma_{k=1}^{n} w_{k} |f_k({x^{(i)}}) - f_k({x^{(j)}}) | + b)}
\]where : $w_k$ is $k^{th}$ weight param.
- Chi-square similarity could also be used ($\frac{L_{2}}{sum}$)
- NOTE : precompute the database encodings, and compute only the current encoding.

### Siamese network
- **Taigman 2014**, DeepFace : closing the gap to human level performance
- Running the same CNN for 2 different I/P images, and comparing their encodings.
  - encoding is the O/P of the last FC layer.
  - $ d(x^{(1)}, x^{(2)}) = \| f(x^{(1)}) - f(x^{(2)})\|_{2} $ _........(1)_
- **Goal of learning** :
  - if $x^{(i)}$ and $x^{(j)}$ are _same_, $\| f(x^{(i)}) - f(x^{(j)})\|$ is **small**
  - if $x^{(i)}$ and $x^{(j)}$ are _different_, $\| f(x^{(i)}) - f(x^{(j)})\|$ is **large**

### Visualizing what DNN is learning
- **Zeiler and Fergus 2013**, Visualizing and understanding convolutional networks

## Neural Style Transfer
- **Gatys 2015**, A neural algorithm of artistic style. - (_easy to read_)
- Content = C, Style = S, Generated = G.
- Two cost functions:
  - $J_{\text{content}}(C,G)$, measures how close **content** of C and G are.
  - $J_{\text{style}}(S,G)$, measures how close **style** of S and G are.
  - Final cost : $J(G) = \alpha J_{\text{content}}(C,G) + \beta J_{\text{style}}(S,G)$
- Algorithm:
  - Initialize $G$ randomly (_size_ : 100 x 100 x 3)
  - Use gradient descent to minimize $J(G)$.
    - $G = G - \alpha \frac{\delta J(G)}{\delta G}$

### Content cost function
- choose $l$ (hidden layer) somewhere in between.
- use pre-trained network (say, VGG)
- $a^{[l](C)}$ : activation of $l^{th}$-layer for content.
- $a^{[l](G)}$ : activation of $l^{th}$-layer for generated.
  - if $a^{[l](C)} \approx a^{[l](G)}$, both images have similar content
  - $\Rightarrow J_{\text{content}}(C,G) = \frac{1}{2}\| a^{[l](C)} - a^{[l](G)} \|^{2}$

### Style cost function
- captures the difference in correlation between the activations of $l^{th}$-layer for style and generated image.
- Let $a_{i,j,k}^{[l]}$ be the activation at (i,j,k). If $G^{[l]} \in \mathit{R^{n_{c} \times n_{c}}}$ is the correlation matrix / **Gram-matrix** :
  - $G_{kk'}^{[l](S)} = \Sigma_{i=1} \Sigma_{j=1} a_{i,j,k}^{[l](S)} a_{i,j,k'}^{[l](S)}$
  - $G_{kk'}^{[l](G)} = \Sigma_{i=1} \Sigma_{j=1} a_{i,j,k}^{[l](G)} a_{i,j,k'}^{[l](G)}$
- Then, style-cost function :
  - $J_{\text{style}}(S,G) = \| G_{kk'}^{[l](S)} - G_{kk'}^{[l](G)} \|_{F}^{2} $ _.......(2)_
  - $\Rightarrow J_{\text{style}}(S,G) = \Sigma_{k} \Sigma_{k'} ( G_{kk'}^{[l](S)} - G_{kk'}^{[l](G)} )$
