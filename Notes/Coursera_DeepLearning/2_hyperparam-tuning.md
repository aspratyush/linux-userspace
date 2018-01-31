## L2 : Hyperparameter tuning

### Train / dev / test segment
- Applied ML is a highly iterative process
- **several parameters** : #layers, #hidden units, LR, activation function
- **ICE : Iterate - Code - Execute** cycle gets faster if good train / dev / test set available
- Traditional dataset size : 60-20-20 or 70-15-15 ( < 1 million data)
- DL case : 90-5-5% or 95-3-2%
- Mismatched train/test distribution
  - Train set may be high res images, dev-test set maybe low res, blurry images
- NOTE :
  - dev and train set should come from **same distribution**.
  - not having test set maybe ok, only dev set may suffice : **BUT may overfit!**

#### Bias and Variance
- less of bias-variance trade-off there in DL
- **Train-set error and Dev-set error** becomes important in DL
- Human performance (also called, Optimal / Bayes error)
  1. LOW Train-set error, HIGH dev-set error : _overfit case_, HIGH VARIANCE
  2. LOW Tran-set and LOW dev-set error, but not as good as humans : _under-fit case_, HIGH BIAS
  3. LOW Train-set error, VERY HIGH  dev-set error : HIGH BIAS and HIGH VARIANCE
  4. VERY LOW train-set error, VERY LOW dev-set error : LOW BIAS and LOW VARIANCE
- HIGH BIAS : caused by linearity of the classifier, could use higher order curves
- HIGH VARIANCE : caused by very high order of the data

#### Basic recipes for ML
- High bias? _Train-set performance_ $\rightarrow$ 1) Bigger N/W, 2) Train longer, 3) NN architecture
- High Variance? _Dev-set performance_ $\rightarrow$ 1) More data, 2) Regularization, 3) NN architecture

#### Regularization
- Ensures $w$ doesnt change by a large amount
- $\lambda$ is the regularization parameter
- For logistic regression:
\[
J(w,b) = \frac{1}{m} \Sigma_{i}\mathbf{L}(\hat{y}^{(i)} - y^{(i)}) + \frac{\lambda}{2m}\|w\|^{2}_{p} \tag{1} \\
\text{where:}\\
p = 1 \rightarrow \text{L1-regularization} \\
p = 2 \rightarrow \text{L2-regularization}
\]

- For neural networks:
\[
J(w^{[0]},b^{[0]},w^{[1]},b^{[1]},\dots,w^{[L]},b^{[L]}) = \frac{1}{2}\Sigma_{i}\mathbf{L}(\hat{y}^{(i)},y^{(i)}) + \frac{\lambda}{2m} \Sigma_{l}\|W^{[l]}\|^2_{F} \tag{2} \\
\text{where:}\\
\|W^{[l]}\|^2_{F} = \Sigma^{n^{[l-1]}}_{i=0} \Sigma_{j=0}^{n^{[l]}}(W_{i,j}^{[l]})^{2}, \ W \in \mathbb{R}^{(n^{[l-1]},n^{[l]})}, \ {\|.\|^{2}_{F}} \text{is Frobenius norm}
\]

- L2-norm regularization is also called "Weight decay"

#### Dropout
- Cant rely on any one feature, and thus, helps to spread out the weights
- Can apply different dropout to different layers
- Has an effect similar to L2-regularization
- Implemented using "inverted dropout"
```
# Say for 3rd layer, for every training iteration:
keep_prob = 0.8
d3 = np.random.randn(a3.shape[0], a3.shape[1]) < keep_prob
a3 = np.multiply(a3,d3)
a3 /= keep_prob
```

#### Other regularization techniques
1. Augment data with subtle distortions
  - As training iterations increase, _train-error_ goes down, but _dev-error_ goes up.
2. Normalize trainining data, and test data with the same $\mu$ and $\sigma$.

#### Speeding up training
- **Weight initialization** - avoids gradients exploding / decaying too quickly
  - Relu (**He initialization**): $W^{[l]} = \text{np.random.randn(shape)} * \text{np.sqrt}(\frac{2}{n^{[l-1]}})$
  - tanh : $\text{np.sqrt}(\frac{1}{n^{[l-1]}})$ : Xavier initialization
  - other : $\text{np.sqrt}(\frac{2}{n^{[l-1]} * n^{[l]}})$

#### Gradient checking
\[
\text{2-sided difference: } \ f'(\theta) = \lim_{\epsilon \rightarrow 0} \frac{f(\theta+\epsilon) - f(\theta-\epsilon)}{2\epsilon} \tag{3a}
\]
\[
\text{1-sided difference: } \ f'(\theta) = \lim_{\epsilon \rightarrow 0} \frac{f(\theta+\epsilon) - f(\theta)}{\epsilon} \tag{3b}
\]
- Eqn. $3a$ is better than eqn. $3b$, since error in eqn. $3a \approx$ $O(\epsilon^{2})$, while eqn. $3b \approx$ $O(\epsilon)$.
- **How to check**:
  1. Append all params $\{w^{[0]},b^{[0]},w^{[1]},b^{[1]},\dots,w^{[L]},b^{[L]}\}$ into a giant vector $\theta$. Thus, cost function becomes : $J(\theta) = J(\theta_{1}, \theta_{2}, \dots)$
  2. Calculate $d\theta_{approx}[i]$ using numerical gradients.
  3. Calculate $d\theta[i]$ using analytical gradients.
  4. Check:
    $\omega = \frac{\|d\theta_{approx} - d\theta\|_{2}}{\|d\theta_{approx}\|_{2} + \|d\theta\|_{2}}$.
      - if :  $\omega \approx 10^{-7} \rightarrow \text{GREAT}, \ \ \omega \approx 10^{-5} \rightarrow \text{OK}, \ \ \omega \approx 10^{-3} \rightarrow \text{BAD!}$
  5. Remember:
      - Use only to debug, not during training
      - If check fails, look at individual components ($W$ or $b$) to figure out which calculation going wrong.
      - Remember regularization.
      - Doesn't work with dropouts.

### Optimization algorithms

#### Batch vs Mini-batch gradient descent
- EPOCH : 1-complete pass through the training set.
- _Batch_ : GD update performed after going through full dataset. 1 GD update per epoch.
- _Mini-batch_ : Split data into smaller mini-batches, and perform GD udpate after every mini-batch. Several GD updates per epoch.
  - $X^{\{t\}}$ and $Y^{\{t\}}$ correspond to the $t^{th}$ mini-batch, each comprising $p$ samples. Thus, $X^{\{t\}} \in \mathbb{R}^{n \times p}$, and $Y^{\{t\}} \in \mathbb{R}^{1 \times p}$. Generally, $p \approx 1000$.
  - Plot of Cost vs mini-batch no. is quite noisy, but average trend is cost reduction. This is because $Y^{\{i\}}$ may happen to be a simpler set than $Y^{\{i+1\}}$.
- _Stochastic GD_ : When each example itself serves as the mini-batch. The cost plot will be very noisy and algorithm never converges.
- **Rules**:
  - small dataset : batch GD
  - Typical Mini-batch $p$ size = $\{64, 128, 256, 512, 1024\}$ (powers of 2)
  - Make sure mini-batch size chosen fits in the memory.

#### Moving average
- $V_{t} = \beta V_{t-1} + (1-\beta) \theta_{t}$
  - effect similar to averaging data over $\frac{1}{1-\beta}$ observations instances. Higher the $\beta$, smoother the curve.
- may need _bias correction_ while moving average is still warming up : $\frac{V_{t}}{1-\beta^{t}}$. Could be used for first few values $t \in \{0,1,2,3\}$.

#### GD with momentum
- use exponentially weighted average of gradients and use this in GD update.
- Weighting term $V$, is similar to velocity, $dW$ and $db$ are the acceleration and $\beta$ is friction.
- tends to dampen vertical oscillations (which causes GD to slow down), and fasten horizontal motion.
- **works better** compared to normal GD.
- **Common value :** $\beta = 0.9$.
\[
\text{V: } V_{dW} = \beta V_{dW} + (1-\beta)dW \\
V_{db} = \beta V_{db} + (1-\beta)db \\
\text{update step:} W = W - \alpha V_{dW} \\
b = b - \alpha V_{db}
\]

#### RMSProp
- Root-mean-square prop : square the gradients and divide by square-root.
- Add small $\epsilon$ to sqrt for numerical stability.
- tends to dampen vertical oscillations (which causes GD to slow down), and fasten horizontal motion.
\[
\text{S: } S_{dW} = \beta_{2} V_{dW} + (1-\beta_{2})dW^{2} \\
S_{db} = \beta_{2} V_{db} + (1-\beta_{2})db^{2} \\
\text{update step:} W = W - \alpha \frac{dW}{\sqrt {S_{dw} + \epsilon}} \\
b = b - \alpha \frac{db}{\sqrt {S_{db} + \epsilon}}
\]

#### Adam Optimization
- combine momentum and RMSProp
- $\beta_{1} \approx 0.9, \beta \approx 0.999, \epsilon \approx 10^{-8}, \alpha \rightarrow \text{tune}$

#### LR decay
- slowly decrease LR as training progresses.
\[
\alpha_{j} = \frac{1}{1+\text{decay_rate} \times \text{epoch_number}} \alpha_{j-1} \tag{5a}
\]
\[
\text{or, } (0.95)^{epoch_number} \alpha_{j-1} \rightarrow \text{exponential decay} \tag{5b}
\]
\[
\text{or, } \frac{k}{\sqrt{\text{epoch_number}}} \alpha_{j-1} \tag{5c}
\]
\[
\text{or, } \text{discrete stairs : reduce by half after $t$ mini-batches} \tag{5d}
\]


### Hyperparameter tuning

- Several hyperparameters exist :
  - **(Rank-1)** LR : $\alpha$
  - **(Rank-2a)** momentum : $\beta$
  - **(Rank-2b)** no. of hidden units
  - **(Rank-2c)** mini-batch size : $t$
  - **(Rank-3a)** no. of layers : $L$
  - **(Rank-3b)** LR decay
  - Adam params : $\beta_{1}, \beta_{2}, \epsilon \rightarrow \text{avoid tuning}$

#### Hyperparameter search
- Random works better than grid search, since we will sample a much richer set
- Use coarse-to-fine search process
- Use log-scale for random search
  - for e.g : ``` r = -4*np.random.rand(); alpha = 10^r```
- $\beta$, the momentum friction, is very sensitive to small changes, esp around 1. So, prefer log-scale sampling, instead of uniform sampling.
- **Panda approach** (Babysitting one model over days - if constrained by hardware) vs **Caviar approach** (parallel & several - needs several GPUs). Panda approach more common.

#### Batch Normalization
- normalizing the activations
- makes params in deeper layers more robust to changes due to training of lower layers.
- intuition : we normalize inputs to converge faster in GD. What if we normalize activations to train next layer faster.
- Recommended to normalize $z$ (linear output) instead of $a$ (activations).
\[
\mu_{z} = \frac{1}{m} \Sigma_{i} z^{(i)} \\
\sigma^{2} = \frac{1}{m} \Sigma_{i} (z^{(i)} - \mu_{z}) \\
z^{(i)}_{\text{norm}} = \frac{z^{(i)} - \mu}{\sqrt{(\sigma^{2} + \epsilon)}} \\
\hat{z}^{(i)} = \gamma z^{(i)}_{\text{norm}} + \beta \\
\]
- BIG ADVANTAGE : $\gamma$ and $\beta$ are learnable params.
- Ensures hidden params have standardised mean and variance.
- Replace ${z}^{(i)}$ by $\hat{z}^{(i)}$ while calculating $a^{(i)}$. If:
\[
\gamma = \sqrt{(\sigma^{2} + \epsilon)} \ \text{and } \ \beta = \mu \\
\hat{z}^{(i)} = {z}^{(i)}
\]
- **Params** : $W^{[l]}, b^{[l]}, \beta^{[l]}, \gamma^{[l]}, \ b^{[l]}, \beta^{[l]}, \gamma^{[l]} \in \mathbb{R}^{n^{[l] \times 1}}$
- Since any constant gets removed off during mean-normalization, no point having $b^{[l]}$.

**Working with mini-batches**:
- Look only within the current mini-batch
\[
X^{\{i\}} \xrightarrow[b^{[1]}]{W^{[1]}} z^{[1]} \xrightarrow[\gamma^{[1]}, BN]{\beta^{[1]}} \hat{z^{[1]}} \rightarrow g^{[1]}(\hat{z}^{[1]}) = a^{[1]} \xrightarrow[b^{[2]}]{W^{[2]}} z^{[2]} \dots
\]
- At test time, use a exponentially weighted running average of the values.

#### Why batch normalization works***
1. Learning on shifting I/P distribution
  - Robust to covariance shifts in train and test distributions
  - Especially important for numerical  stability of the deeper layers, as they get robust against the covariance shifts of the activations in the lower layers.


### Softmax Layer
- used for genralization of logistic regression over $c$-classes.
- similar to a normal layer, has a linear output and an activation output
\[
t = \exp^{z^{[L]}} \\
a^{[L]} = \frac{\exp^{z^{[L]}}}{\Sigma_{i} t_{i}}
\]

- $t$ is similar to confidence
- $a^{[L]}$ is the normalized version -- probability
- unlike logistic regression, softmax layer outputs a vector. If $c=2$, softmax reduces to logistic regression.
- HARDMAX - one-hot encoding from $z$.
- SOFTMAX - gentle mapping from $z$ to probability-like values.
- **Loss** :
\[
\mathcal{L}(\hat{y} - y) = - \Sigma_{j} y_{j} \log{\hat{y}_{j}}
\]
- NOTE : $dz^{[L]} = \hat{y} - y$, for softmax loss.

### TensorFlow
- builds computational graph to solve the F/W propagation.
- auto-calculates the B/W propagation values.
- everything happens inside of a session.
- sample code:

```
import numpy as np
import tensorflow as tf

# values to feed in at train time - could be replaced by I/P data
coefficients = np.array([[1., 2., 3.]])

w = tf.Variable([0], dtype=tf.float32)      # define a variable to be **MINIMIZED**
x = tf.placeholder(tf.float32, [3,1])       # run-time variable / I/P data

# cost (F/W computational graph)
cost = x[0][0]*w**2 + x[0][1]*w**1 + x[0][2]

# GD
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)  # optimizer, LR and fn. to minimize

# idiomatic to do the following 5 lines
init = tf.global_variables_initializer()    # tf initialization
session = tf.Session()                      # prefer : with tf.Session() as session:
session.run(init)
print(session.run(w))

# run the training
for i in range(1000):
    session.run(train, feed_dict={x:coefficients})
print session.run(w)
```
