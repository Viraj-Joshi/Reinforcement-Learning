---
tags: RL
---
# On-Policy Prediction with Approximation

We can use function approximation to estimate the value function from on-policy data generated using policy $\pi$.

That is, the approximation $\hat{v}(s,\mathbf w) \approx v_\pi(s)$ is the goal. We seek the dimensionality of $\mathbf w$ to be $\ll |\mathcal{S}|$. Updating any weight will change the estimated values of many states because there are far more states than weights. This introduces challenges but also introduces the power of function approximation, in that it generalizes the value of states never seen before.

We update the weights our function approximator using examples of the form $s \mapsto u$ where this means the estimated value of state $s$ should be more like the target $u$. As a result, our training examples become the examples $s\mapsto u$.

## Prediction Objective

In the tabular case, there was no need for an objective function because the learned value function could equal true value function through backups (each update only effected one state). 
 
In the approximation case, because there are far more states than weights, making one state's estimate more accurate will makes others less accurate. We specify a state distribution $\mu(s) \geq 0, \sum_s \mu(s)=1$, representing how much the error(MSE) at state $s$ matters.

The mean square value error is,

$$\overline{VE} = \sum\limits_{s\in\mathcal{S}} \mu(s) \left[v_\pi(s)-\hat{v}(s,\mathbf w)\right]^2$$

$\mu(s)$ is often chosen to be the fraction of time spent in $s$ under the on-policy distribution. In continuing tasks, the on-policy distribution is the stationary distribution of $\pi$.

With a linear function approximator, the global optimum $\mathbf w^*$ of $\overline{VE}$ is sometimes possible to find but rarely found for complex function approximators like DNNs or decision trees.  Instead, RL methods seeks to converge to a local optimum and often is enough.


## SGD

Let $\mathbf w = (\mathcal w_1,\cdots \mathcal w_d)^T$ and $\hat{v}(s,\mathbf w)$ is differentiable for all $s\in\mathcal S$.

On each timestep $t$, we observe a new example $S_t \mapsto v_\pi(S_t)$. Assuming states appear according to $\mu(s)$, the SGD update to minimze $\overline{VE}$ using one example is,

\begin{equation}
\begin{split}
\mathbf w_{t+1} & = w_t - \frac{1}{2} \alpha \nabla\left[v_\pi(S_t) - \hat{v}(s,\mathbf w)\right]^2 & (1) \\
& = w_t -\alpha \left[v_\pi(S_t) - \hat{v}(s,\mathbf w)\right] \nabla \hat{v}(s,\mathbf w) & \space &(2)\\
\end{split}
\end{equation}

According to SGD convergence results, if $\alpha$ decreases as to satisfy stochastic approximation condtions, SGD converges to a local optimum.

Of course, $v_\pi$ is unknown, so our examples during training are actually $S_t \mapsto U_t$ where $U_t$ is a noise-corrupted version of $v_\pi(S_t)$ or a bootstrap estimate.

Our general SGD update becomes

$$ w_{t+1} = w_t -\alpha \left[U_t - \hat{v}(s,\mathbf w)\right] \nabla \hat{v}(s,\mathbf w)$$

**If $U_t$ is unbiased estimate of $v_\pi$, then SGD converges to a local optimum under stochastic approximation conditions.**

---

Let $U_t = G_t$, then $U_t$ is an unbiased estimator and the MC prediction with a linear function approximator converges to a locally optimal solution.

![](https://i.imgur.com/HDyrveq.png)
---

### Semi-gradient Methods
Bootstrapping targets like the n-step returns $G_{t:t+n}$ or the DP target $\sum_{a,s'} \pi(a|S_t) p(s',r|s,a)[r+\gamma \hat{v}(s,\mathbf w)]$ as $U_t$ do not guranteed SGD to converge to a local optimum.

#### Why?
This is apparent because the derivation of SGD in $(1)$ depends on the target $U_t$ being independent of $\mathbf w_t$. If one substitutes a bootstrap target as the estimate, then the gradient of the error takes into account the effect of changing the weight on the estimate but not changing the weight on the target. In other words, moving from (1) to (2) would not hold. Thus, when $U_t$ is a bootstrap estimate(which depends on $\mathbf w$), the update only includes part of the gradient and it is a semi-gradient method.

However, semi-gradient methods enable faster learning because they use bootstrapping, and as a result, updates can be made online without waiting for the end of the epsiode.

When $U_t = R_{t+1} + \gamma \hat{v}(S_{t+1},\mathbf w)$, we have the following TD(0) prediction algorithm
![](https://i.imgur.com/DQsFKD4.png)

### State-Aggregation
Group the states together with one weight for each group. The value of a state is its group's value and when the value of a state is updated, its component alone is updated. State Agrreation is a special case of SGD where $\nabla \hat{v}(S_{t+1},\mathbf w)$ is 1 for the $S_t$'s group and 0 otherwise.

## Linear Methods
Let $\hat{v}(S_{t+1},\mathbf w)$ be linear in the weights and for every state $s$, there is a feature vector $\mathbf x(s)=(\mathcal x_1(s) \cdots \mathcal x_d(s))^T$. 

Then, the state-value function is of the form:

$$\hat{v}(S_{t+1},\mathbf w) = \mathbf w^T  \mathbf x(s)$$

Each feature $x_i(s)$ is a basis function and the set of them form a basis for the set of approximate functions we can represent.

In the linear case, the SGD update is:

$$ w_{t+1} = w_t -\alpha \left[U_t - \hat{v}(s,\mathbf w)\right] \mathbf x(S_t)$$

SGD is guranteed to converge to a near/at local optimum under stochastic approximation conditions. But there is only one local optima here. Then, the gradient Monte Carlo algorithm converges to the global optimum of  $\overline{VE}$ under linear function approximation.

### Convergence of Semi-gradient TD(0) under linear function approximation

The update is 
\begin{equation}
\begin{split}
\mathbf w_{t+1} & = w_t - \alpha\left(R_{t+1}+\gamma \mathbf w_t^T \mathbf x_{t+1} - \mathbf w_t^T \mathbf x_{t}\right)\mathbf x_{t} & \space & \text{where} \space \mathbf x_{t}=\mathbf x(S_t)\\
& = w_t - \alpha\left(R_{t+1}\mathbf x_{t}-\mathbf x_{t}(\mathbf x_{t}-\gamma \mathbf x_{t+1})^T \mathbf w_{t}\right) \\
\end{split}
\end{equation}

Proof:

Linear semi-gradient TD(0) converges to the TD fixed point, which obeys:

$$\overline{VE}(\mathbf w_{TD}) \leq \frac{1}{1-\gamma} \min\limits_{\mathbf w} \overline{VE}(\mathbf w)$$

In other words, the asymptotic error of the TD method is upper bounded by the $1/(1-\gamma)$ times the smallest error obtained by the MC method. Because $\gamma$ is usually near 1, the asymptotic performance is worse than MC. 

However, TD methods are much lower variance, so the best method really depends on the problem and the length of the learning.

We can show TD methods retain their advantage through the following example using n-step TD.

![](https://i.imgur.com/5DGeXfO.png)

Code:

```
def semi_gradient_n_step_td(
    env, #open-ai environment
    gamma:float,
    pi:Policy,
    n:int,
    alpha:float,
    V:ValueFunctionWithApproximation,
    num_episode:int,
):
    """
    implement n-step semi gradient TD for estimating v

    input:
        env: target environment
        gamma: discounting factor
        pi: target evaluation policy
        n: n-step
        alpha: learning rate
        V: value function
        num_episode: #episodes to iterate
    output:
        None
    """
    for _ in range(num_episode):
        terminal = False
        traj = []
        T = math.inf
        s,info = env.reset()

        a = pi.action(np.array([s,0]))
        s_prime,r,terminated,_ = env.step(a)
        # env.render()
        traj.append((np.array([s,0]),a,r,s_prime))
        # print(traj[0])
        s=s_prime
        t = 1
        while not terminal:
            a = pi.action(np.array(s))
            s_prime,r,terminated,_ = env.step(a)
            # env.render()
            traj.append((s,a,r,s_prime))
            # print(traj[-1])
            if terminated:
                terminal = True
                T = t+1
            tau = t-n+1
            if tau >= 0:
                G = sum([(gamma**(j-tau))*traj[j][2] for j in range(tau,min(tau+n,T))])
                if tau + n < T:
                    G = G + (gamma**n) * V(traj[tau+n-1][3]) # accessing S_{tau+n} because the last entry of traj[tau+n-1] is S_{tau+n-1+1}
                V.update(alpha,G,traj[tau][0])
            t+=1
            s=s_prime
    return V
```
## Feature Construction for Linear Methods

Above, we defined how to estimate $\hat{v}(S_{t+1},\mathbf w) = \mathbf w^T  \mathbf x(s)$, but not how to choose $\mathbf x(s)$.

The quality of these estimates depends heavily on the feature construction.

If one hand-designs these features, e.g if we are valuing states of the robot using location, battery power, sonar readings, etc., this can add prior domain knowledge to our problem. However, capturing the interaction between features is crucial for most problems and is not reflected in a simple linear form.

For example, the pole-balancing task implies that high angular velocity can be good or bad depending on angle,another state feature. In short, we need to capture these interactions.

### Polynomials

Interaction terms can be captured by introducing them into the features vectors. For example, given $s=[s_1,s_2]$,then we could have $x(s) = (1,s_1,s_2,s_1s_2)$ or a higher dimensional vector like $x(s) = (1,s_1,s_2,s_1s_2,s_1^2,s_2^2,s_1^2s_2,s_1s_2^2,s_1^2s_2^2)$.

Higher order polynomials basis allow for more accurate approximations but the number of features in an order-n polynomial basis grows exponentially with the dimension of the state space $k$. That is, $(n+1)^k$ features are generated and must be pruned from to be practical.

### Fourier Basis

Recall the definition of periodicity: $f(x) = f(x+\tau) \forall x$ and some period $\tau$

Any periodic function can be expressed as a Fourier series which is a linear combination of sine and cosine basis functions (features) of different frequencies as accurately as desired.


However, one can approximate an aperiodic function defined over a bounded interval by using Fourier basis features with $\tau$ set to the length of the interval. Then, one period of the periodic linear combination is your approximation.

Furthermore, if you set $\tau$ to twice the length of the interval of interest, then your approximation is just the half interval $[0,\tau/2]$ and one can just use cosine features. This is because any even function (symmetric around origin) can be represented by a cosine basis.

Letting $\tau=2$, the one-dimensional order-$n$ Fourier cosine basis consists of $n+1$ features, $\mathcal x_i = \cos (i \pi s)$ for $i=0,1,2,3,4$, shown below (first basis is constant function)

![](https://i.imgur.com/1idpSEr.png)


In the multidimensional case, 
Suppose each state $s$ is a vector of $k$ numbers, $s=[s_0,\dots, s_k]$ with $s_i\in [0,1]$. The $ith$ feature of an order-$n$ Fourier cosine basis is $$ \mathcal x_i = \cos(\pi \mathbf s^T \mathbf c^i)$$ where $c_i = (\mathcal c_i \cdots \mathcal c_k)^T$ with $c_j^i \in \{0,...n\}$ for $j = 1,...,k$ and $i = (n+1)^k$

The inner product has the effect of assigning an integer in $\{0,\dots ,n\}$ that represents the feature's frequency in that dimension, to each dimension of $s$.

Picture

The number of features in the order-n Fourier basis grows exponentially with the dimension of the state space, but if that dimension is small enough (e.g., k < 5), then
one can select n so that all of the order-n Fourier features can be used.

### Coarse Coding
Consider a two-dimensional continuos state space. Then, overlay circles in that space and consider each circle as a feature. If the given state $s$ lies inside a circle, then that feature has value $1$ and is 'active', otherwise $0$ and 'inactive'.

![](https://i.imgur.com/8OxzD6e.png)

As a result, we can encode any state as a binary vector of features. This is known as *coarse coding*. 

Because we are using linear approximation, there is a weight corresponding to each feature. Then, the gradient-descent update at a state $s$ affects all weights of the features active at $s$. Geometrically, the value of the approximate value function will be affected at all states within the union of the circles, with a greater effect on the points whose circles are common to those with the updated state. 

As the circle size changes from one coarse coding to another, the generalization of states acts over shorter or longer distances depending on the dimension.

![](https://i.imgur.com/b7ZFb3K.png)


---

### Tile Coding

Tile coding is a form of coarse coding for multi-dimensional continuous spaces that is flexible and computationally efficient.

It works by covering a continuous space with tiles, where each tile has a corresponding index in a vector. The tiles can be any arbitrary shape, but are typically n-dimensional hyperrectangles for computational convenience. The binary feature vector for a point in the space would have a 1 at the indices of the tiles intersected by the point, and a 0 everywhere else:

![](https://i.imgur.com/oj5c4Gg.png)


Tile coding lays tiles over the continuous space through the use of tilings. A **tiling** can be thought of as an n-dimensional grid of **tiles** with potentially different scales of values along each dimension. Several offsetted tilings are then placed over the space to create regions of overlapping tiles. Typically, each tiling is offset by a fraction of a tile width.

A useful property of laying tiles this way is that the number of tiles intersected will always be the number of tilings used, as a point can't intersect two tiles within the same tiling:

![](https://i.imgur.com/OP9U9bp.png)

Again, each tile for each tiling has a weight associated with it and is updated using stochastic gradient descent.  

The approximate value function is simply computed by computing the indices of the active features and then adds up the corresponding components of the weight vector. Note we can do this beause we our feature vector is a binary vector as opposed to computing a dot product between our weight vector and an explicit binary feature vector.

#### Generalization of Tile Codings
As in coarse coding, generalization occurs to states other than the one trained when they fall within any of the same tiles, proportional to the number of tiles in common.

This generalization is affected by how the tilings are offset from each other (1), the shape of the tiles(2), and the number of tilings.


Addressing (1),
![](https://i.imgur.com/LfRym93.png)


Addresing (2), the tiles that are elongated one dimension will promote more generalization along that dimension. Examining the bottom half of the figure above shows generalization roughly equal in each dimension.

The 2nd and 3rd tilings below are biased towards certain dimensions. The first tiling is irregular and is rare in practice.

![](https://i.imgur.com/JgkBMFK.png)

Lastly, one can use *different shaped tiles* in *different tilings* to get the best of all worlds (e.g vertical stripe tilings, horizontal stripe tilings, and normal square tilingss)

### Radial Basis Functions (RBF)

RBFs are the generalization of coarse coding to continous-valued features, that is each feature is not 0/1, but a value in the interval [0,1]. Each feature $\mathcal x_i$ is the Gaussian centered at $c_i$ and has width $\sigma_i$.

$$\mathcal x_i = exp\left(-\frac{||s-c_i||^2}{2\sigma_i^2}\right)$$

## Least Squares TD







