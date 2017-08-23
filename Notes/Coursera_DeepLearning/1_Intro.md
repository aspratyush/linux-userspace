## Neural Networks and Deep Learning

### L1: Introduction
- AI is the new electricity
- Sequence of the specialization:
  - Neural Networks and Deep Learning
  - Practical aspects of improving performance (hyperparameters, optimization)
  - Structuring the ML Project
  - CNNs
  - NLP : Sequence models (RNN, LSTM)

#### Neural Networks
- Only I/P and O/P needs to be given
- They will figure out the inter-relations, given enough data
- Scale drives Deep Learning progress
- Traditional ML algorithms (SVM, logistic regression, etc.) plateau out as data increases
- Recent times we have been accumulating massive data
- **Idea - code - experiment cycle** till we get best results. Faster xomputation has helped speed up the cycle.

### L2 : Basic Neural network
- Stack data from different examples in different columns
\[
X = \begin{bmatrix}
\vdots & \vdots & \vdots & \vdots \\
x^{(1)} & x^{(2)} & \dots & x^{(m)} \\
\vdots & \vdots & \vdots & \vdots
\end{bmatrix} \ , \ X \in \mathbf{R}^{n_{x} \times m} \ , \ x \in \mathbf{R}^{n_{x}} \\
Y = \begin{bmatrix}
y^{(1)} & y^{(2)} & \dots & y^{(m)}
\end{bmatrix} \ , \ Y \in \mathbf{R}^{1 \times m}
\]

#### __Logistic Regression__
- Linear regression with output confined to binary
\[
\hat{y} = \sigma(w^{T}x + b), \ \text{where:} \ \sigma(z) = \frac{1}{1+e^{-z}}
\]
- Problem statement:
\[
\text{Given :} \{(x^{1}, y^{1}), \dots, (x^{m}, y^{m}) \}, \ \hat{y_{i}} \approx y_{i}
\]
- Introduce a **loss function** to solve this: $\mathcal{L}(\hat{y}, y) = -(y \log(\hat{y}) + (1-y)\log(1-\hat{y})$
- Analysis of the cost function:
  - If $y = 0, \ \mathcal{L(\hat{y},y)} = -\log(1-\hat{y}) \ \leftarrow$ want params to adjust to make $\hat{y}$ small
  - If $y = 1, \ \mathcal{L(\hat{y},y)} = -\log(\hat{y}) \ \leftarrow$ want params to adjust to make $\hat{y}$ large
- **Cost function** : averaged loss function over all examples: $J(w,b) = \frac{1}{m}\mathcal{L}(\hat{y},y)$

#### Gradient descent
- Optimization algorithm to solve the cost function.
\[
w = w - \alpha \frac{d(w,b)}{dw}
\]
- NOTE : Use $d()$ when differentiating w.r.t 1 variable, and $\delta()$ for multivariable scenario
- Use ```assert``` to check matrix size in numpy

### L3 : Shallow Neural Networks
- Notations : ${x}^{(i)}$ : i-th training example, $w^{[j]}$ : j-th hidden layer

#### Activation functions
1. Without non-linearity, the network of neural nets collapses into a linear function. Non-linearity introduces capability to learn much more complex models.
1. $\sigma(z) = \frac{1}{1+e^{-z}}$ : sigmoid function adds non-linearity to the network
1. $g(z) = \tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$: always better than sigmoid, zero mean!
    - $\sigma(z)$ still used for last layer, since getting O/P $\in [0,1]$ is better than getting in $[-1,1]$.
3. $g(z) = \text{Relu}$ : Derivative is constant, and doesnt die out as with $\sigma(z)$ and $\tanh()$.

- Try all activation functions against a validation set, and use the best suited.

#### Formulae:
- $d\text{Var}$ and $\text{Var}$ have the same dimensions
- Forward propagation:
\[
\text{Hidden layer#1}: \ Z^{[1]} = W^{[1]}X + b^{[1]} \\
\text{Activation}: \ A^{[1]} = g^{[1]}(Z^{[1]}) \\
\text{Hidden layer#2}: \ Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]} \\
\text{Activation}: \ A^{[2]} = g^{[2]}(Z^{[2]})
\]

- Back propagation:
\[
\textbf{Layer#2}: dZ^{[2]} = A^{[2]} - Y \\
dW^{[2]} = \frac{1}{m}dZ^{[2]}A^{[1]} \\
db^{[2]} = \frac{1}{m}\text{np.sum}(dZ^{[2]}, \text{axis}=1, \text{keepdims=True}) \\
\textbf{Layer#1}: dZ^{[1]} = W^{[2]^{T}} dZ^{[2]} * g{[1]}'(Z^{[1]}) \\
dW^{[1]} = \frac{1}{m}dZ^{[1]}X^{T} \\
db^{[1]} = \frac{1}{m}\text{np.sum}(dZ^{[1]}, \text{axis}=1, \text{keepdims=True})
\]

#### Random initialization
1. we want different units to compute different functions
    - Thus, breaking symmetry is important
2. large $W$ may result in $Z$ being in the squashed region, leading to slow/no gradient descent.


### L4 : Deep Neural Networks
- Simple to complex hierarchial features extracted by the layers of the neural nets.
- DL can find complex mappings from $X \rightarrow Y$
- E.g (speech processing): Audio $\rightarrow$ waveform $\rightarrow$ phonemes $\rightarrow$ words $\rightarrow$ sentences
- Deep networks can compute complex functions, for which exponentially more hidden units would be required by shallower networks.
- Notations:
  - $L$ : No. of hidden layers
  - $n^{[l]}$ : no. of units in layer $l$
  - $a^{[l]}$ : activations in layer $l$, $W^{[l]}$ : weights in layer $l$

- **Always check** the dimensions of the matrices we're working with:
  - $W^{[l]} \in \mathbf{R}^{n^{[l]} \times n^{[l-1]}}$
  - $b^{[l]} \in \mathbf{R}^{n^{[l]}}$
  - $Z^{[l]}$ and $A^{[l]} \in \mathbf{R}^{n^{[l]} \times m}$
  - $d\text{Var}$ and $\text{Var}$ should be of same dimension.

- **hyperparameters**:
  - $\alpha$, $L$, $n^{[l]}$, $g(z)$
  - may need to be changed over time with changing data and algorithms

#### Practical tips
- Data preprocessing:
  - flatten data into $(n_{x},m_{\text{train}})$ matrix
  - divide by 255 to standardize the data
- Building NN model:
  - define model structure : $L, n^{[l]}$, etc.
  - Initialize model parameters
  - Loop:
    - calculate loss (F/W propagation)
    - Calculate gradient (B/W propagation)
    - Update parameters (Gradient descent)
  - **model()** consists of **initialize(), propagate(), optimize()**
- **Sanity checks**:
  - Training accuracy close to 100 implies model has **enough capacity** to fit the training data
