Intro to NN
=================

### Machine Learning
- Linear Regression  
    - Metric : sq. distance of points from the line (instead of distance)

- Logistic Regression  
    - Metric : log-loss (instead of count)
    - Large loss for miss-labelling

### Perceptron
- Linear combination of the inputs, weighted by the learned weights.
- linear activation function

### Neural Network
- Linear combination of inputs, passed through a __non-linear, but continuously differentiable__ activation function.
- Training via gradient descent
- __Deep Learning__ : Deep stacks of hidden layers in a neural network.

#### SSE (Sum of squared errors)
\[
\text{Error} = E = \frac{1}{2}\sum_{\mu} \sum_{j} \| y_{j}^{\mu} - \hat{y}_{j}^{\mu} \|_{2} , \ \ \ni \ \ \hat{y}_{j} = f(w_{j}x_{j}) \ \text{and} \ \hat{y} = f(w^{T}x)
\]

- __where__: $y_{j}$ represents all output units in a neural network, $\mu$ represents all the examples, and f() is the non-linear activation function.
- makes the errors positive, and penalizes large errors more.
- the error curve is convex, thus making optimization simpler.
- weights are the tap that tune the output.
- we use gradient descent to figure out the weights that produce lowest error.

#### Gradient descent
- Calculate the change in weights required to reduce the overall error.
\[
\text{gradient} = \Delta{w_{j}} = - \eta \frac{\delta(E)}{\delta(w_{j})}
\]
- __where__ : $\eta$ is the ```LEARNING RATE``` that scales the gradient step.
- refer : Multivariable Calculus : Khan Academy

\[
\text{say} \ \colon \delta = (y-\hat{y})\ \hat{y}' \\
\implies \Delta w_{j} = \eta \delta_{j} x_{j}
\]

__Implementing with ```numpy```__
- It is important to initialize the weights randomly so that they all have different starting values and diverge, __breaking symmetry__

### Multilayer Neural Network
- $w_{ij}$ represents the weight between the I/P unit $i$ and O/P unit $j$.
- __row__ represents weights going out of an I/P unit.
- __col__ represents weights entering into a hidden unit.

\[
\begin{bmatrix}
w_{11} & w_{12} \\
w_{21} & w_{22} \\
w_{31} & w_{32}
\end{bmatrix}
\]

#### Backpropagation
- Derivative of the error at output is scaled by weights between hidden units and the output, and so on.
- The hidden unit contributing more to the output sees more Backpropagating error.
\[
\delta^{h}_{j} = \sum W_{j k} \delta_{k}^{0} f'(h_{j}) \\
\Delta w_{i j} = \eta \delta_{k}^{0} x_{i}
\]
