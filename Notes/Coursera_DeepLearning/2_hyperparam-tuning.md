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
  - Relu : $W^{[l]} = \text{np.random.randn(shape)} * \text{np.sqrt}(\frac{2}{n^{[l-1]}})$
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
