---
tags: RL
---
# Policy Gradient Methods I

So far, we have only dealt with action-value methods - that is, our policy to select actions is based on estimated action-values. 

Here, we learn a policy parameterized by $\theta$ directly, without consulting action-value estimates. Optimizing a policy based on the gradient of a performance measure are a class of methods called policy gradient methods.

### Why Policy Optimization

With learning a state-value function $V$, one needs a model of the world to perform the one-step lookahead.

With learning an action-value function $Q$, one needs to compute an $\arg\max$ at each state, which can be challenging in large action spaces.

Policy Gradient methods avoid those issues.

### Advantages

To ensure that exploration occurs, the policy $\pi(a|s,\theta)$ is stochastic.

Then, a natural parameterization is to estimate preferences $h(s,a,\theta) \in \mathbb{R}$ for all state, action pairs. For example, $h$ can be estimated by tile coding or a DNN. The actions with the highest preference are given the highest probability of being selected according to the exponential softmax distribution:

$$\pi(s,a,\theta) = \frac{e^{h(s,a,\theta)}}{\sum_b e^{h(s,b,\theta)}}$$

It is important to distinguish that action preferences are not action-values. Instead, action preferences are driven to produce optimal stochastic policy. *That is, policy gradient methods provide the flexibility that the optimal policy could be stochastic or deterministic.* E.g in the face of imperfect information, stochastic policy is best (ex: poker)

Even if you apply softmax to action preferences,in $\epsilon$-greedy selection, the action probabilities can change dramatically for a small change in the estimated action values whereas the action probabilities change smoothly, with respect to the action preferences with policy gradient methods. 

Examine the following:

![](https://i.imgur.com/8q48nUC.png)

We see above that action-value methods using $\epsilon$-greedy action selection(as often is the case) can do worse because there is always an $\epsilon$ probability of selecting a random action(**can never approach a deterministic policy**) whereas the the policy gradient method can exactly chose the optimal stochasitc policy.


- a policy parameterization may be a simpler function to approximate than an action-value parameterization.
- learning a value function directly may take longer to converge, but the policy is not changing (wasting compute)
- represent continuous actions, scales to high dimensional state-spaces
- **Finally, we note that the choice of policy parameterization is sometimes a good way of injecting prior knowledge about the desired form of the policy. This is often the most important reason for using a policy-based learning method.**

## Vanilla Policy Gradient Method

As usual, our performance measure $J(\theta)$ depends on the setting - episodic or continuous. In the episodic case, define the trajectory $\tau$ as
$$\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \ldots, s_{T-1}, a_{T-1}, r_{T}, s_T)$$

We seek to maximize $J(\theta)$ by Stochastic Gradient Ascent. Consequently, we need $\nabla J(\theta)$. 

$$J(\theta) =  \mathbb{E}_{\pi_{\theta}}\left[\sum_{t=1}^{T}\gamma^{t-1} r_t\right] = \mathbb{E}_{\pi_{\theta}}\left[R(\tau)\right] = \sum_{\tau} P(\tau;\theta) R(\tau)
$$
 - Notice how the expectation is taken w.r.t to the policy. This means the rewards are computed from the trajectory induced by the policy
 - $P(\tau;\theta)$ is just the probability of experiencing that trajectory under the current policy parameters 

Note, our performance measure depends on both the action selection and the distribution of states in which those selections are made. Both of these are affected by the policy parameter. We know the effect on the action selection (and reward) by our policy parameters. We do not know the effect of the policy changes to the state distribution?.

Q: Then, how can we estimate the gradient (because $\theta$ affects the state distribution)? 
A: **Policy Gradient Theorem**, which provides an analytic expression for the gradient of the performance measure w.r.t $\theta$ without the derivative of the state distribution w.r.t. $\theta$
 

First, observe what is called the likelihood ratio trick using the key fact $\nabla \log f(x) = \frac{\nabla f(x)}{f(x)}$

\begin{align}
\nabla_\theta \mathbb{E}[f(x)] &= \nabla_\theta \int p_\theta(x)f(x)dx \\
&= \int \frac{p_\theta(x)}{p_\theta(x)} \nabla_\theta p_\theta(x)f(x)dx \\
&= \int p_\theta(x)\nabla_\theta \log p_\theta(x)f(x)dx \\
&= \mathbb{E}\Big[f(x)\nabla_\theta \log p_\theta(x)\Big]
\end{align}

Using this trick, we already have $\nabla J(\theta)$.
\begin{align}
\nabla_\theta \mathbb{E}[J(\theta)] &= \nabla_\theta \sum_\tau P(\tau;\theta) R(\tau) \\
&= \sum_\tau P(\tau;\theta)\nabla_\theta \log P(\tau;\theta) R(\tau) \\
&= \mathbb{E}\Big[\nabla_\theta\log P(\tau;\theta) R(\tau)]
\end{align}

This trick allows us to simply approximate the gradient by taking an empirical estimate of $m$ rollouts.
$$\nabla_\theta J(\theta) \approx \sum_{i=1}^m \nabla_\theta\log P(\tau;\theta) R(\tau)$$

However, we do not have the model $P$, so we cannot compute this.
#### Intuition
The gradient tries to increase probability of paths with positive R and decrease probability with negative R.

---

To that end, observe how to compute $\nabla_\theta \log p_\theta(\tau;\theta)$. 

\begin{align}
\nabla_\theta \log p_\theta(\tau) &= \nabla \log \left(\mu(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t|s_t)P(s_{t+1}|s_t,a_t)\right) \\
&= \nabla_\theta \left[\log \mu(s_0)+ \sum_{t=0}^{T-1} (\log \pi_\theta(a_t|s_t) + \log P(s_{t+1}|s_t,a_t)) \right]\\
&= \nabla_\theta \sum_{t=0}^{T-1}\log \pi_\theta(a_t|s_t)
\end{align}

- $\mu(s_0)$ is just the starting state distribution
- Again, $P(\tau;\theta)$ was just decomposed into the chain of probabilities given by the MDP assumption and the policy
- Notice how the dynamics cancel out!

Finally, we can combine these two tricks to compute the gradient of the performance measure $J(\theta)$.

$$\nabla_\theta J(\theta)=\nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \mathbb{E}_{\tau \sim
\pi_\theta} \left[R(\tau) \cdot \nabla_\theta \left(\sum_{t=0}^{T-1}\log
\pi_\theta(a_t|s_t)\right)\right]$$

Now, a naive approach is to gather a set of trajectories $\hat{\tau} = \{\tau_{i}\}_{i=1}^n$ and peform stochastic gradient ascent updates  $\theta \leftarrow \theta + \alpha\nabla_\theta \mathbb{E}_{\tau \in \hat{\tau}}[R(\tau)]$ using the empirical expectation of our gradient. 

![](https://i.imgur.com/M4OqKrF.png)

However, learning with these updates is slow and unreliable as the gradient estimator is high-variance. We can reduce the variance of this estimator by introducing a baseline subtracted from the return.

---

### Baseline

Before we introduce a baseline, we can leverage the temporal structure of the MDP to simplify the reward
\begin{align}
\nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}\Big[R(\tau)\Big] \;&{\overset{(i)}{=}}\; \mathbb{E}_{\tau \sim \pi_\theta} \left[\left(\sum_{t=0}^{T-1}r_t\right) \cdot \nabla_\theta \left(\sum_{t=0}^{T-1}\log \pi_\theta(a_t|s_t)\right)\right] \\
&{\overset{(ii)}{=}}\; \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1}\nabla_\theta \log \pi_\theta(a_t|s_t)\left(\sum_{t'=0}^{t-1} r_{t'} + \sum_{t'=t}^{T-1} r_{t'}\right)\right] \\
&{\overset{(iii)}{=}}\; \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) \left(\sum_{t'=t}^{T-1}r_{t'}\right) \right]\\
&{\overset{(iv)}{=}}\; \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t|s_t) G_t \right]
\end{align}

- (i) is what we derived previously
- (iii) is simply leveraging that the reward-to-go at time $t$ is only relevant to the action being taken at time $t$. The reward experienced in the past for an action you are taking does not matter. So we can throwaway rewards from the past.
- Let $f_t = \nabla_\theta \log \pi_\theta(a_t|s_t)$. Then, (iii) takes the expectation over the rowwise summation of below. (iv) takes the expectation over the columnwise summation.
     - $\begin{align}
            r_0&f_0 + \\
            r_1&f_0 + r_1f_1 + \\
            r_2&f_0 + r_2f_1 + r_2f_2 + \\
            &\cdots \\
            r_{T-1}&f_0 + r_{T-1}f_1 + r_{T-1}f_2 \cdots + r_{T-1}f_{T-1}
        \end{align}$  
    - The first column is $f_0 \cdot \left(\sum_{t'=0}^{T-1}r_{t'}\right)$, the second column is $f_1 \cdot \left(\sum_{t'=1}^{T-1}r_{t'}\right)$
- (iv) notices that reward to go is just the return $G_t$
    
    
Finally, lets insert a baseline $b(s_t)$ which is a function of $s_t$.

$$\nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] =
\mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log
\pi_\theta(a_t|s_t) \left(G_t - b(s_t)\right) \right]$$

1. Does this make the estimator biased?
2. Does this reduce variance?

#### Let us show our estimator remains unbiased.
First, simplify.
$$\nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] =
\mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log
\pi_\theta(a_t|s_t) \left(\sum_{t'=t}^{T-1}r_{t'}\right) - \sum_{t=0}^{T-1}
\nabla_\theta \log \pi_\theta(a_t|s_t) b(s_t) \right]$$

Then, due to linearity of expectation, we just need to show the second term inside the expectation at timestep $t$ is zero.

\begin{align}
\mathbb{E}_{\tau \sim \pi_\theta}\Big[\nabla_\theta \log \pi_\theta(a_t|s_t) b(s_t)\Big] &=  \mathbb{E}_{s_{0:t},a_{0:t-1} \sim \pi_\theta}\Big[ \mathbb{E}_{s_{t+1:T},a_{t:T-1}} [\nabla_\theta \log \pi_\theta(a_t|s_t) b(s_t)]\Big] \\
&= \mathbb{E}_{s_{0:t},a_{0:t-1}}\Big[ b(s_t) \cdot \underbrace{\mathbb{E}_{s_{t+1:T},a_{t:T-1}} [\nabla_\theta \log \pi_\theta(a_t|s_t)]}_{E}\Big] \\
&= \mathbb{E}_{s_{0:t},a_{0:t-1}}\Big[ b(s_t) \cdot \mathbb{E}_{a_t \sim \pi_\theta }[\nabla_\theta \log \pi_\theta(a_t|s_t)]\Big] \\
&= \mathbb{E}_{s_{0:t},a_{0:t-1}}\Big[ b(s_t) \cdot \sum_{a_t} \pi_{\theta}(a_t|s_t) \frac{\nabla_\theta
\pi_\theta(a_t|s_t)}{\pi_{\theta}(a_t|s_t)}\Big]\\
&= \mathbb{E}_{s_{0:t},a_{0:t-1}}\Big[ b(s_t) \cdot \nabla_\theta \sum_{a_t}\pi_{\theta}(a_t|s_t)\Big]\\
&= \mathbb{E}_{s_{0:t},a_{0:t-1}}\Big[ b(s_t) \cdot \nabla_\theta 1 \Big] \\
&= \mathbb{E}_{s_{0:t},a_{0:t-1}}\Big[ b(s_t) \cdot 0 \Big] = 0
\end{align}

The expectation can be split up as we in line 2.
\begin{align}
E &= \sum_{a_t\in \mathcal{A}}\sum_{s_{t+1}\in \mathcal{S}}\cdots \sum_{s_T\in \mathcal{S}} \underbrace{\pi_\theta(a_t|s_t)P(s_{t+1}|s_t,a_t) \cdots P(s_T|s_{T-1},a_{T-1})}_{p((a_t,s_{t+1},a_{t+1}, \ldots, a_{T-1},s_{T}))} (\nabla_\theta \log \pi_\theta(a_t|s_t)) \\
&= \sum_{a_t\in \mathcal{A}} \pi_\theta(a_t|s_t)\nabla_\theta \log \pi_\theta(a_t|s_t) \sum_{s_{t+1}\in \mathcal{S}} P(s_{t+1}|s_t,a_t) \sum_{a_{t+1}\in \mathcal{A}}\cdots \sum_{s_T\in \mathcal{S}} P(s_T|s_{T-1},a_{T-1})\\
&= \sum_{a_t\in \mathcal{A}} \pi_\theta(a_t|s_t)\nabla_\theta \log \pi_\theta(a_t|s_t) 
\end{align}

The third lines comes from the fact that pushing all the sums back results in probability densities that sum to 1 (variable elimination!)

#### Let us show this is a variance reducer.

\begin{align}
& \text{VAR}_{\tau \sim \pi} \bigg[\sum_{t=0}^{T-1} \nabla_\theta \log
\pi_\theta(a_t|s_t) \big(G_t - b(s_t)\big)\bigg]
\\
& {\overset{(i)}{\approx}}\; \mathbb{E}_{\tau \sim \pi}\bigg[\bigg(\sum_{t=0}^{T-1} \nabla_\theta \log
\pi_\theta(a_t|s_t) \big(G_t - b(s_t)\big)\bigg)^2\bigg] - \mathbb{E}_{\tau \sim \pi}\bigg[\sum_{t=0}^{T-1} \nabla_\theta \log
\pi_\theta(a_t|s_t) \big(G_t - b(s_t)\big)\bigg]^2\\
&{\overset{(ii)}{\approx}}\; \sum_{t=0}^{T-1} \mathbb{E}_\tau \left[\Big(\nabla_\theta \log \pi_\theta(a_t|s_t)\Big)^2\right]\mathbb{E}_\tau\left[\Big(R_t(\tau) - b(s_t))^2\right]
\end{align}

- (i) is an approximation because the variance of a sum is not the sum of variances.
- (ii) is because we assume independence

We know the second term is zero (because introducing baseline is still unbiased), so we are left with the first term.

Then, it is revealing that the second term in (ii) is just the least squares problem. This is error is minimized when $b(s_t) = \mathbb{E}[G_t]$, which is $v(s_t)$!

---
What exactly should our baseline $b(s_t)$ be? 
- Constant - With gradient bandits, we used the average of rewards 
- One natural choice of a baseline is the value function, another function approximator

#### Minimum variance baseline $b_t^*$

Look at (i) in the variance reducer section
Let $g = \sum_{t=0}^{T-1}\nabla_\theta \log
\pi_\theta(a_t|s_t)$

\begin{align}
& 0 =\frac{d\text{VAR}}{db} =\frac{d}{db}\mathbb{E}\big[g^2(G_t-b)^2\big]=\frac{d}{db}\mathbb{E}\big[-2g^2G_tb+g^2b^2\big]=-2\mathbb{E}\big[g^2G_t\big]+2\mathbb{E}\big[g^2\big]b \\
& \implies b^*(s) = \frac{\mathbb{E}\bigg[\big(\sum_{t=0}^{T-1} \nabla_\theta \log
\pi_\theta(a_t|s_t)\big)^2 G_t\bigg]}{\mathbb{E}\bigg[\sum_{t=0}^{T-1} \big(\nabla_\theta \log
\pi_\theta(a_t|s_t)\big)^2\bigg]}
\end{align}

Though in practice, this is not used :)


---
Here, REINFORCE With Baseline, which is simply stochastic gradient ascent with our gradient estimator. Note how we must generate trajectories before updating the weights of our baseline and policy.

![](https://i.imgur.com/sDotX80.png)

## Actor-Critic

In REINFORCE and REINFORCE with baseline, there is only an actor, the agent using the parameterized policy. These methods do not use bootstrapping, so despite our best efforts with the baseline, they are high-variance. 

In actor methods, the state-value function estimates only the value of the first state in each transition. In actor-critic methods, we bootstrap through value functions and reduce variance in exchange for biasing our estimate. E.g The state-value function can be applied to the second state of the transition ($R_{t+1} + \gamma\hat v(S_{t+1})$, telling the agent how good it was to take the action *(the critic).*

Consider the one-step actor-critic method, the analog of TD methods like TD(0), SARSA(0) and Q-learning, that replace the full return of REINFORCE with the one-step return.
\begin{align}
\theta_{t+1} & = \theta_t + \alpha_t \left[ G_{t:t+1} - \hat v(s_t,\mathbf w) \right] \nabla_{\theta_t} \log \pi_{\theta_t}(a_t | s_t)\\
& = \theta_t + \alpha \delta_t \nabla_\theta \log \pi_{\theta_t}(a_t | s_t)
\end{align}

We could even extend this to $n$-step methods or backwards-view eligibility trace that trade off more bias for less variance. **But while high variance strategies necessitates using more samples, bias is more
pernicious—even with an unlimited number of samples, bias can cause the algorithm to fail to converge, or to converge to a poor solution that is not even a local optimum**
![](https://i.imgur.com/aMYpusJ.png)

