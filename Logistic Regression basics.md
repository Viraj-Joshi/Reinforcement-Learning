# Building up to Multinomial Logistic Regression
This hack aims to show how the loss function and their derivatives are derived.
## Logistic Regression

![](https://i.imgur.com/TToaVTP.png)

Logistic Regression is a classification model of two classes.

It takes the form of $h_\mathbf w(\mathbf x) = \mathbf \sigma(\mathbf w^T \mathbf x +b)$

The posterior is 
\begin{equation}
P(y|\mathbf x) = 
\left\{
    \begin{array}{lr}
        h_\mathbf w(\mathbf x), & \text{if } y=1\\
        1-h_\mathbf w(\mathbf x), & \text{if } y=0
    \end{array}
\right\}
\end{equation}

A more compact way of writing this is that 
$$P(y|\mathbf x) = h_\mathbf w(\mathbf x)^y+(1-h_\mathbf w(\mathbf x))^{(1-y)}$$



### Loss Function

Using MLE, one maximizes the likelihood of data.

This is \begin{equation}
\begin{split}
\max_w\prod\limits_{i}P(y_i|x_i)&=\max_w\prod\limits_{i}h_\mathbf w(\mathbf x)^{y_i}+(1-h_\mathbf w(\mathbf x))^{(1-y_i)} \\
& \equiv \max_w\sum_{i} y_i \log h_\mathbf w(x_i) + (1-y_i) \log (1- h_\mathbf w(x_i))\\
\end{split}
\end{equation}

It is easier to maximize the log due to floating point error introduced by maximizing the original.

Because most optimizer libraries use gradient descent, we minimize the negative of our objective to obtain,

$$\min_\mathbf w-\sum_{i} y_i \log h_\mathbf w(x_i) + (1-y_i) \log (1- h_\mathbf w(x_i))$$

### Logits

Generally, logits are simply the inputs to the last neurons layer.

In the context of logistic regression, this is simply $z=w^Tx$.

### Binary Cross Entropy (BCE)

**BCE is exactly negative log likelihood**

$$BCE(y,x|\theta) = -\sum_{i} y_i \log p_\theta(y|x_i) + (1-y_i) \log (1- p_\theta(y|x_i))$$

So maximizing the log likelihood(or minimizing the negative log likelihood) is equivalent to minimizing BCE.

In PyTorch, BCEWithLogitsLoss expects the logits as input.

## Multinomial Logistic Regression / Softmax Regression
The model is a 1-layer Neural Network
without          |  w/softmax
:-------------------------:|:-------------------------:
![](https://i.imgur.com/w1itLnJ.png) |  ![](https://i.imgur.com/dt6kJ6b.png)

One can visualize the left handside as training $h$ logistic regression models. Each logistic regression model has a weight vector $w \in \mathbb{R}^{m}$, so stacking them together row-wise,  we have a weight matrix $W \in \mathbb{R}^{h \times m}$

Each of the activations $a_1,..,a_h$ produced by the lefthand side model do not sum to 1. Because we want an activation $a_i$ to be the probability that an example belongs to each class, we apply the softmax function to the logits of the network to make the activations sum to 1.

This is the shown by the right-hand side, where the sigmoid at each of the last layers neuron is removed, and replaced by a softmax layer as the final layer. The prediction of $x_i$ is the $\arg\max_i a_i$

---
### Softmax Function
The softmax activation function $P(y=t|z_t^i)=\sigma(z_t^i)$ where $t\in \{1,\dots,h\}$ is the class label and $i$ indexes the training example.

$$\sigma(z_t^i) = \frac{e^{z_t^i}}{\sum_{j=1}^he^{z_j^i}}$$
### One-hot Encoding
If we have the following training labels $y$ with 4 classes, the one hot encoding is

\begin{equation*}
\begin{pmatrix}
0 \\
1 \\
3 \\
2 \\
0 \\
\end{pmatrix}
\rightarrow
\begin{pmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 1 & 0 \\
1 & 0 & 0 & 0 \\
\end{pmatrix}
\end{equation*}

where each row of the one hot encoding matrix represents the $i$th label of $y$

### Cross Entropy

This is the generalization of BCE.

$$\mathcal{L} = \sum_{i=1}^n \sum_{j=1}^h -y_j^i \log(a_j^i)$$ assuming one-hot encoded labels and each activation has been passed through softmax.
Each multiplication in the inner loop is element wise because the onehot encoding row is $1 \times h$

If h=2 (binary case), the CE term is BCE 
\begin{equation}
\begin{split}
CE &= -\sum_{i=1}^n y_1^i \log a_1^i-y_2^i\log a_2^i \\
& = -\sum_{i=1}^n y_1^i \log a_1^i- y_2^i\log (1-a_1^i) & \text{by $a_1^i + a_2^i$ = 1}\\
& = -\sum_{i=1}^n y_1^i \log a_1^i+ (1-y_1^i)\log (1-a_1^i) & &\text{by $y_1^i$ =1 implies $y_2^i$ = 0}\\
\end{split}
\end{equation}

Ex: The softmax output is the rowwise softmax of the feature matrix multiplied by the weights $XW$

![](https://i.imgur.com/KKcjz4N.png)

See how each $\mathcal{L^i}$ is the cross entropy of the $i$th training example

See how Negative Log Likelihood is CE

---
In PyTorch, cross entropy takes logits as input and returns the mean over the CE of each example. One should be careful to read documentation on what input the loss function expects.

### Loss Function Derivative

Let $X \in \mathbb{R}^{n \times m}$, $Y \in \mathbb{R}^{n \times h}$, and $W \in \mathbb{R}^{m \times h}$

To optimize our model, our desired gradients are $\frac{\partial\mathcal{L}}{\partial W}$ and $\frac{\partial\mathcal{L}}{\partial b}$

For example, $\frac{\partial\mathcal{L}}{\partial w_i}$ is a vector describing how a small nudge in the entries of the $\mathbf w_i$ vector would change $\mathcal{L}$.

$$\frac{\partial\mathcal{L}}{\partial w_i} = \frac{\partial\mathcal{L}}{\partial a}
\frac{\partial\mathcal{a}}{\partial z}
\frac{\partial\mathcal{z}}{\partial w_i}$$

$$\frac{\partial\mathcal{L}}{\partial b} = \frac{\partial\mathcal{L}}{\partial a}
\frac{\partial\mathcal{a}}{\partial z}
\frac{\partial\mathcal{z}}{\partial b}$$

**Assume we have one example to make notation cleaner. If there are multiple examples, we simply average or add the gradients of the examples**


---

The first part of our expression, $\frac{\partial\mathcal{L}}{\partial a}$, is taking a derivative of a scalar w.r.t a vector $a$.

Each entry of that gradient is

\begin{equation}
\begin{split}
\frac{\partial\mathcal{L}}{\partial a_i} & = \frac{\partial}{\partial a_i}
\left[\sum_{j=1}^h -y_j \log a_j\right]\\
& = \frac{\partial}{\partial a_i}[-y_i \log a_i] \\
& = -y_i/a_i
\end{split}
\end{equation}


---

The second part of our expression, $\frac{\partial a}{\partial z}$, is taking the derivative of a vector w.r.t another vector. This is a matrix.
When one takes the derivative of $a_i$ with its corresponding entry $z_i$
\begin{equation}
\begin{split}
\frac{\partial\mathcal{a_i}}{\partial z_i} & = \frac{\partial}{\partial z_i}
\left[\frac{e^{z_i}}{\sum_{j=1}^h e^{z_j}}\right]\\
& = 
\left[\frac{(\sum_{j=1}^h e^{z_j}) \frac{\partial}{\partial z_i}e^{z_i}-e^{z_i} \frac{\partial}{\partial z_i}\sum_{j=1}^h e^{z_j}}{(\sum_{j=1}^h e^{z_j})^2}\right]\\
& = 
\left[\frac{(\sum_{j=1}^h e^{z_j}) e^{z_i} - e^{z_i}e^{z_i}}{(\sum_{j=1}^h e^{z_j})^2}\right] \\
& = \frac{e^{z_i}(\left[\sum_{j=1}^h e^{z_j}\right]-e^{z_i})}{(\sum_{j=1}^h e^{z_j})^2} \\
& = \frac{e^{z_i}}{\sum_{j=1}^h e^{z_j}}\cdot
\frac{\left[\sum_{j=1}^h e^{z_j}\right]-e^{z_i}}{\sum_{j=1}^h e^{z_j}}\\
& = a_i (1-a_i)\\
\end{split}
\end{equation}

Similarly, for $i \neq k$ :
\begin{equation}
\begin{split}
\frac{\partial\mathcal{a_i}}{\partial z_k} & = \frac{\partial}{\partial z_k}
\left[\frac{e^{z_i}}{\sum_{j=1}^h e^{z_j}}\right]\\
& = 
\left[\frac{(\sum_{j=1}^h e^{z_j}) \frac{\partial}{\partial z_k}e^{z_i}-e^{z_i} \frac{\partial}{\partial z_k}\sum_{j=1}^h e^{z_j}}{(\sum_{j=1}^h e^{z_j})^2}\right]\\
& = 
\left[\frac{(\sum_{j=1}^h e^{z_j}) 0 - e^{z_i}e^{z_k}}{(\sum_{j=1}^h e^{z_j})^2}\right] \\
& = \frac{0-e^{z_i}e^{z_k}}{(\sum_{j=1}^h e^{z_j})^2} \\
& = \frac{-e^{z_i}}{\sum_{j=1}^h e^{z_j}}\cdot
\frac{e^{z_k}}{\sum_{j=1}^h e^{z_j}}\\
& = -a_i a_k\\
\end{split}
\end{equation}

Using the two derivations, we can write the Jacobian matrix $\mathbf A \in \mathbb{R}^{h\times h}$ of activations.


\begin{equation*}
A_{n,n} = 
\begin{pmatrix}
\frac{\partial a_1}{\partial z_1} & \frac{\partial a_1}{\partial z_2} & \cdots & \frac{\partial a_1}{\partial z_h} \\
\frac{\partial a_2}{\partial z_1} & \frac{\partial a_2}{\partial z_2} & \cdots & \frac{\partial a_2}{\partial z_h} \\
\vdots  & \vdots  & \ddots & \vdots  \\
\frac{\partial a_h}{\partial z_1} & \frac{\partial a_n}{\partial z_2} & \cdots & \frac{\partial a_h}{\partial z_h} 
\end{pmatrix}
\end{equation*}

This can be more efficiently written, by first observing

\begin{equation}
\frac{\partial a_i}{\partial z_k} = 
\begin{cases} 
      a_i (1-a_k) & i = k \\
      -a_i a_k & i \neq k \\
   \end{cases}
  = (a_i)(\delta_{i,k}-a_k)
\end{equation}

Which in matrix form is, 

\begin{equation}
\begin{split}
\frac{\partial \mathbf a}{\partial \mathbf z} & = \mathbf a1_k^T \circ(\mathbf I - 1_k \mathbf a^T) \\
& =\begin{pmatrix}
a_{1} \\
\vdots \\
a_{h}\\
\end{pmatrix}
\begin{pmatrix}
1 \cdots 1
\end{pmatrix}
\circ
\left[\begin{pmatrix}
1 & 0 & \cdots & 0 \\
0 & 1 & \cdots & 0 \\
\vdots  & \vdots  & \ddots & \vdots  \\
0 & 0 & \cdots & 1 
\end{pmatrix}-
\begin{pmatrix}
1 \\
\vdots \\
1\\
\end{pmatrix}
\begin{pmatrix}
a_1 \cdots a_h
\end{pmatrix}\right] \\
& = \begin{bmatrix} 
        a_1 & \cdots & a_1 \\ 
        a_2 & \cdots & a_2 \\ 
        \vdots & \ddots & \vdots 
        \\ a_h & \cdots & a_h 
    \end{bmatrix} \circ
\left (\mathbf I -
            \begin{bmatrix} 
                a_1 & \cdots & a_1 \\ 
                a_2 & \cdots & a_2 \\ 
                \vdots & \ddots & \vdots \\ 
                a_h & \cdots & a_h 
            \end{bmatrix}^\top
\right ) \\
& = \begin{bmatrix} 
        a_1 (1-a_1) & a_1 (-a_2)  & \cdots & a_1 (-a_h) \\
        a_2 (-a_1)  & a_2 (1-a_2) & a_2 (-a_3) & \vdots \\
        a_3 (-a_1)  & a_3 (-a_2) & \ddots & \vdots\\
        a_h (-a_1) & a_h(-a_2)& \cdots &a_h (1-a_h)
    \end{bmatrix}\\
& = \mathbf A
\end{split}
\end{equation}
where $1_k$ is the ones column vector of (h by 1), $\circ$ is the element wise product, and $\mathbf I$ is $h\times h$ identity

---

The third part of our expression, $\frac{\partial z}{\partial w_i}$, is taking the derivative of a scalar w.r.t a vector.

The $i$th logit $z_i$ has derivative w.r.t to its $j$th weight

\begin{equation}
\begin{split}
\frac{\partial\mathcal{z_i}}{\partial w_j} & =
\frac{\partial}{\partial w_j} [w^T x]\\
& = x_j
\end{split}
\end{equation}

So, $$\frac{\partial\mathcal{z_i}}{\partial w}=x$$

---

Putting it all together,

Assume we have a model with 3 classes and each example has 2 features.
![](https://i.imgur.com/654dRYV.gif)

Then, our chain rule is summing contributions e.g
\begin{equation}
\begin{split}
\frac{dL}{dw_{1,1}} & = \sum_{i=1}^3 \left (\frac{dL}{da_i} \right) \left (\frac{da_i}{dz_1} \right) \left(\frac{dz_1}{dw_{1,1}} \right ) \\
& = -(y_1/a_1)(a_1(1-a_1))x_1 + -(y_2/a_2)(-a_2a_1)x_1 + -(y_3/a_3)(-a_3a_1)x_1
\end{split}
\end{equation}


We can vectorize this last equation to obtain:

$$\nabla_W\mathcal{L} = - (X^T(Y-A))$$








