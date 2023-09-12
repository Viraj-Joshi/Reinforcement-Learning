---
tags: RL
---

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

It is important to distinguish that action preferences are not action-values. Instead, action preferences are driven to produce optimal stochastic policy. A deterministic policy is formed when preferences of the optimal actions will be driven infinitely higher than all suboptimal actions. **That is, parameterizing policies according to the soft-max in action preferences provide the flexibility that the optimal policy could be stochastic or deterministic.** E.g in the face of imperfect information, stochastic policy is best; like poker

In contrast, applying the softmax over action-values may result in a stochastic but not deterministic policy. This is because the action-values will converge with some finite differences and thus translate to probabilties other than 0 or 1.


Examine the following:

![](https://i.imgur.com/8q48nUC.png)

We see above that action-value methods using $\epsilon$-greedy action selection(as often is the case) can do worse because there is always an $\epsilon$ probability of selecting a random action(**can never approach a deterministic policy**) whereas the the policy gradient method can exactly chose the optimal stochastic policy.

- a policy parameterization may be a simpler function to approximate than an action-value parameterization.
- learning a value function directly may take longer to converge, but the policy is not changing (wasting compute)
- represent continuous actions, scales to high dimensional state-spaces
- **the choice of policy parameterization is sometimes a good way of injecting prior knowledge about the desired form of the policy. This is often the most important reason for using a policy-based learning method.**

## Vanilla Policy Gradient Method

One last benefit of policy parameterization is theoretical. With continuous policy parameterization, the action probabilities change smoothly as a function of the learned parameters, whereas applying softmax over action values and using $\epsilon$-greedy selection over these action probabilities may change dramatically
for an arbitrarily small change in the estimated action values. This would contribute to very high-variance in the learning.

As usual, our performance measure $J(\theta)$ depends on the setting - episodic or continuous. In the episodic case, define the trajectory $\tau$ as
$$\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \ldots, s_{T-1}, a_{T-1}, r_{T}, s_T)$$

and the trajectory distribution i.e the probability distribution of experiencing trajectories under the current policy parameters as

$$P(\tau;\theta) = p_\theta(s_1,a_1,\dots,s_T,a_T) = p(s_1)\prod_{t=1}^{T} \pi_\theta(a_t|s_t)p(s_{t+1}|s_t,a_t)$$ where the subscript $\theta$ emphasizes that the distribution depends on the policy parameters.

We seek the optimal parameterization $\theta^*= \arg\max\limits_\theta$ $J(\theta)$ by Stochastic Gradient Ascent. 

$$J(\theta) =  \mathbb{E}_{\tau \sim P(\tau;\theta)}\left[\sum_{t=1}^{T}\gamma^{t-1} r_t\right] = \mathbb{E}_{\tau \sim P(\tau;\theta)}\left[R(\tau)\right] = \sum_{\tau} P(\tau;\theta) R(\tau)
$$
 - Notice the randomness in the rewards are induced by trajectories sampled from the policy

### Evaluating $J(\theta)$

Sample $N$ rollouts to create an unbiased estimate:

$$J(\theta) = \mathbb{E}_{\tau \sim P(\tau;\theta)}\left[R(\tau)\right] \approx \frac{1}{N} \sum_i \sum_t r(s_{i,t},a_{i,t})$$

---

### Direct Policy Differentiation

\begin{align}
\nabla_\theta \mathbb{E}_{\tau \sim P(\tau;\theta)}[R(\tau)] &= \nabla_\theta \int \pi_\theta(\tau)R(\tau)d\tau \\
&=  \int \nabla_\theta\space \pi_\theta(\tau)R(\tau)d\tau 
\end{align}

Note that this integral is not tractable because we cannot integrate over all trajectories. To remedy this, we will turn this integral into an expectation.

Somehow we want $\pi_\theta$ to be a term in the integral times the gradient of some quantity.

We can do this with what is called the likelihood ratio trick: $\nabla \log f(x) = \frac{\nabla f(x)}{f(x)}$

\begin{align}
\nabla_\theta \mathbb{E}[f(x)] &= \nabla_\theta \int p_\theta(x)f(x)dx \\
&= \int \frac{p_\theta(x)}{p_\theta(x)} \nabla_\theta p_\theta(x)f(x)dx \\
&= \int p_\theta(x)\nabla_\theta \log p_\theta(x)f(x)dx \\
&= \mathbb{E}\Big[f(x)\nabla_\theta \log p_\theta(x)\Big]
\end{align}

Apply this trick to $\nabla J(\theta)$.
\begin{align}
\nabla_\theta J(\theta) &= \nabla_\theta \sum_\tau P(\tau;\theta) R(\tau) \\
&{\overset{(i)}{=}}\; \sum_\tau \nabla_\theta P(\tau;\theta) R(\tau)\\
&{\overset{(ii)}{=}}\;\sum_\tau P(\tau;\theta)\nabla_\theta \log P(\tau;\theta) R(\tau) \\
&= \mathbb{E}_{\tau \sim P(\tau;\theta)}\Big[\nabla_\theta\log P(\tau;\theta) R(\tau)]
\end{align}

* (i) moves the gradient inside the integral because differentiation is linear
* (ii) applies the likelihood ratio trick

This trick allows us to simply approximate the gradient by taking an empirical estimate of $m$ rollouts.
$$\nabla_\theta J(\theta) \approx \sum_{i=1}^m \nabla_\theta\log P(\tau;\theta) R(\tau)$$

However, we do not have the model $P$, so we cannot compute this.
To that end, observe how to compute $\nabla_\theta \log p_\theta(\tau;\theta)$. 

\begin{align}
\nabla_\theta \log p_\theta(\tau) &= \nabla \log \left(\mu(s_0) \prod_{t=1}^{T} \pi_\theta(a_t|s_t)P(s_{t+1}|s_t,a_t)\right) \\
&= \nabla_\theta \left[\log \mu(s_0)+ \sum_{t=1}^{T} (\log \pi_\theta(a_t|s_t) + \log P(s_{t+1}|s_t,a_t)) \right]\\
&{\overset{(iii)}{=}}\; \nabla_\theta \sum_{t=1}^{T}\log \pi_\theta(a_t|s_t) \\
&{\overset{(iv)}{=}}\;  \sum_{t=1}^{T}\nabla_\theta\log \pi_\theta(a_t|s_t)
\end{align}

- $\mu(s_0)$ is just the starting state distribution
- Again, $P(\tau;\theta)$ was just decomposed into the chain of probabilities given by the MDP assumption and the policy
- (iii) Notice how the derivatives of the dynamics are zero!
- (iv) moves the gradient inside the summation

This gives us

$$\nabla_\theta J(\theta)=\nabla_\theta \mathbb{E}_{\tau \sim P(\tau;\theta)}[R(\tau)] = \mathbb{E}_{\tau \sim
P(\tau;\theta)} \left[R(\tau) \cdot \left(\sum_{t=1}^{T}\nabla_\theta \log
\pi_\theta(a_t|s_t)\right)\right]$$

Finally, we can compute the gradient of the performance measure $J(\theta)$ because the both of the terms in the expectation, the log probs of the policy and the return, are known.

The unknown quantities, the starting state distribution and the transition function, occur only in distribution from which the expectation is taken. 

This means we can gather a set of trajectories $\hat{\tau} = \{\tau_{i}\}_{i=1}^n$ and calculate the empirical expectation of our gradient 
$$\nabla_\theta J(\theta) \approx \frac{1}{N}\sum_{i=1}^N R(\tau_i) \cdot \left(\sum_{t=1}^{T}\nabla_\theta \log
\pi_\theta(a_{t,i}|s_{t,i})\right)$$


The gradient step tries to increase probability of paths with positive R and decrease probability with negative R.

#### Similarity to maximum likelihood
$$\frac{1}{N}\sum_i^N \sum_t \nabla \log \pi_\theta(a_t^i|s_t^i)$$
In fact, the policy gradient is exactly equivalent to maximum likelihood when the reward of all state action pairs in each trajectory $\tau_i$ are 1 and 0 otherwise.

In the below algorithm, we estimate $\nabla_\theta J(\theta)$ from one trajectory.

![](https://i.imgur.com/M4OqKrF.png)

However, learning with these updates is slow and unreliable as the gradient estimator is high-variance. We can reduce the variance of this estimator by introducing a baseline subtracted from the return.

---

### Variance Reduction

We leverage the temporal structure of the MDP to simplify the reward term

\begin{align}
\nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}\Big[R(\tau)\Big] \;&{\overset{(i)}{=}}\; \mathbb{E}_{\tau \sim \pi_\theta} \left[\left(\sum_{t=1}^{T}\nabla_\theta \log \pi_\theta(a_t|s_t)\right)\left(\sum_{t=1}^{T}r_t\right) \right] \\
&{\overset{(ii)}{=}}\; \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=1}^{T}\nabla_\theta \log \pi_\theta(a_t|s_t)\left(\sum_{t'=1}^{t-1} r_{t'} + \sum_{t'=t}^{T} r_{t'}\right)\right] \\
&{\overset{(iii)}{=}}\; \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \left(\sum_{t'=t}^{T}r_{t'}\right) \right]\\
&{\overset{(iv)}{=}}\; \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) G_t \right]
\end{align}

- (i) is what we derived previously
- (ii) leverages that (a+b+c)(x+y+z) = a(x+y+z) + b(x+y+z) + c(x+y+z)
- (iii) leverages that the reward-to-go at time $t$, $\sum_{t'=t}^{T-1}r_{t'}$, is the only relevant reward to the action being taken at time $t$. This is causality: the reward experienced in the past for an action you are taking now is not going to change as a result of that action. So we can throwaway rewards from the past.
- (iv) notices that reward to go is just the return $G_t$
    
Removing these rewards for finite samples changes the estimator, but we will see that in expectation the estimator remains unbiased. In addition, intutively, we expect this estimator to be lower variance because we are removing information that has no bearing on how the policy affects our performance measure.
    
#### Baseline
Now, let's insert a baseline $b(s_t)$ which is a function of $s_t$.

$$\nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] =
\mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log
\pi_\theta(a_t|s_t) \left(G_t - b(s_t)\right) \right]$$

Q: Why have a baseline?

A: This really is the trick that makes PG work. Let's say all our rewards are positive. Then,a policy gradient step increases the log prob of actions that lead to mediocre return as well as those that lead to high return. Instead, we should strive to increase (decrease) the log prob of actions that lead to better than (worse than) average return. As a result, the baseline is just centering the return by subtracting out the average return, or in general, some quantity.

Some questions:

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
Here, REINFORCE With Baseline, which is simply stochastic gradient ascent with our gradient estimator. Note how we must generate trajectories before updating the weights of our baseline and policy (on-policy method)

![](https://i.imgur.com/sDotX80.png)

## Actor-Critic

We can still improve the on-policy gradient estimator with a class of methods called Actor-Critic. 

$$\nabla_\theta J(\theta) \approx
\sum_{i=1}^N \sum_{t=1}^{T} \nabla_\theta \log
\pi_\theta(a_{i,t}|s_{i,t}) \left(\sum_{t'=t}^T r(s_{i,t'},a_{i,t'})-b(s_{i,t'})\right)$$

Examine the reward-to-go  $\widehat Q_{i,t} := \sum_{t'=t}^T r(s_{i,t'},a_{i,t'})$. This is a single sample (monte-carlo) estimate, computed from that trajectory $\tau_i$, of the expected reward-to-go if we take action $a_t$ in state $s_t$. This is a high-variance estimate of the reward-to-go due to the randomness in the policy and dynamics (a lot of different outcomes can occur from that state,action pair leading to (potentially) much different reward-to-gos). We can get a lower variance estimator by computing the exact expectation of reward at each timestep. *As a result, we can get a lower variance gradient estimator by plugging in a better rewards-to-go estimator.*

This is known as the Q-function, which is the expected reward-to-go from taking $a_t$ in $s_t$ and then following $\pi_\theta$. 

$$Q^\pi(s_t,a_t) = \sum_{t'=t}^T\mathbb E_{\pi_\theta}[r(s_{t'},a_{t'})|s_t,a_t]$$

Similarly, we can define the total expected reward under policy $\pi_\theta$ from $s_t$ as 
$$V^\pi(s_t) = \mathbb E_{a_t\sim\pi_\theta(\cdot|s_t)}[Q^\pi(s_t,a_t)] = \sum_{t'=t} \mathbb E_{\pi_\theta}[r(s_{t'},a_{t'})|s_{t}]$$
Here, the expectation is taken over the probabilties of actions prescribed by our policy at time $t$.


Substituting the reward-to-go as the Q function and baseline as the Value function:

$$\nabla_\theta J(\theta) \approx
\sum_{i=1}^N \sum_{t=1}^{T} \nabla_\theta \log
\pi_\theta(a_{i,t}|s_{i,t}) \left(Q^\pi(s_{i,t},a_{i,t})-V^\pi(s_{i,t})\right)$$

The intuitive meaning of the difference $Q^\pi(s_{i,t},a_{i,t})-V^\pi(s_{i,t})$ is how much better taking the action $a_{i,t}$ is better on average than taking the average action you would take in $s_{i,t}$. Multiplying by log probabilties, we see that the gradient estimator increases (decrease) the probabilities of  actions better (worse) than average in that state.

The quantity $Q^\pi(s_{i,t},a_{i,t})-V^\pi(s_{i,t})$ is known as the advantage function $A^\pi(s_{i,t},a_{i,t})$.

### Estimating the Advantage Function $A^\pi(s,a)$

In reality, we do not have access to $Q^\pi,V^\pi,A^\pi$ and need to estimate these quantities. Instead of estimating $Q^\pi$ and $V^\pi$, we only need to estimate $V^\pi$ (which is easier to learn than Q due to only depending on state) to obtain an estimate of $A^\pi$.

Unlike before, where our changes always kept the policy gradient estimators unbiased, this introduces bias in exchange for lower variance.

To see why:

\begin{align}
Q^\pi (s_t,a_t) & = \sum_{t'=t}^T\mathbb E_{\pi_\theta}[r(s_{t'},a_{t'})|s_t,a_t]\\
&{\overset{(ii)}{=}}\; r(s_t,a_t) + \sum_{t'=t+1}^T\mathbb E_{\pi_\theta}\big[r(s_{t'},a_{t'})|s_{t},a_t\big] \\
&{\overset{(iii)}{=}}\; r(s_t,a_t) + \mathbb E_{s_{t+1} \sim p(\cdot|s_t,a_t)} \bigg[V^\pi(s_{t+1})\bigg] \\
& {\overset{(iv)}{\approx}}\; r(s_t,a_t) + V^\pi(s_{t+1})
\end{align}

- (ii) $(s_t,a_t)$ are not random variables
- (iii) by definition of Value function
- (iv) this is an approximation because the actual $s_{t+1}$ seen in the trajectory is being used instead of taking an expectation over the possible next states 

Now our advantage function is simple:

$$A^\pi(s_{t},a_{t}) \approx r(s_t,a_t) + V^\pi(s_{t+1}) - V^\pi(s_t) $$

This is an approximation because $Q^\pi(s_t,a_t)$ was approximated, but now we only need to fit $V^\pi$ to compute $A^\pi$

### Fitting $V^\pi(s)$ (policy evaluation)

One way is to just use Monte Carlo policy evaluation: $V^\pi(s_t) \approx \sum_i \sum_{t'=t}^T r(s_{t'},a_{t'})$. However, we cannot do this in the model-free setting because generally we cannot repeatedly rollout trajectories from each state, only the initial state.

Instead, we just perform supervised learning with targets $y_{i,t}$. A function approximator will generalize that similar states should have similar values even if its never seen it before.
1. parameterize a network $\widehat{V}^\pi_\phi(s)$ with parameters $\phi$. 
2. The dataset is $\{\big(s_{i,t},y_{i,t}\big)\}$
3. The loss is MSE: $\mathcal L(\phi) = \frac{1}{2} \sum_i ||\widehat{V}^\pi_\phi(s) - y_i||_2$

For example, the targets could be the single sample (high variance) MC estimates $\sum_{t'=t}^T r(s_{i,t'},a_{i,t'})$ of the expected reward for each rollout $i$. We hope that the network will generalize to unseen states. Can we do better?

In general, the ideal target $y_{i,t}$ is the expected reward-to-go from state $s_t$, $\sum_t\mathbb E_{a_t \sim \pi_\theta} \big[r(s_{i,t},a_{i,t})\big]$. 

We know the full return can be approximated by $n$-step returns. For example, from what we saw before,

\begin{split}
y_{i,t} &= \sum_{t'=t}^T\mathbb E_{\pi_\theta}[r(s_{i,t'},a_{i,t'})|s_t] \\
& {\overset{(ii)}{\approx}}\; r(s_{i,t},a_{i,t})+ \sum_{t'=t+1}^TE_{\pi_\theta}[r(s_{i,t'},a_{i,t'})|s_{t+1}]\\
& {\overset{(iii)}{\approx}}\;r(s_{i,t},a_{i,t})+V^\pi(s_{i,t+1})\\
& {\overset{(iv)}{\approx}}\;r(s_{i,t},a_{i,t})+V^\pi_\phi(s_{i,t+1})
\end{split}

- (ii) is an approximation because it states the reward-to-go at $t$ is just the reward  $r(s_{i,t},a_{i,t})$ using $a_t$ encountered in $\tau_i$ plus the expected reward-to-go using $s_{t+1}$ encountered in $\tau_i$.
- (iv) is an approximation because it is a bootstrap estimate

The dataset collected in 2) becomes $\{\big(s_{i,t}, r(s_{i,t},a_{i,t}) + \hat V^\pi_\phi(s_{t+1})\big)\}$

We could even extend this to $n$-step methods or backwards-view eligibility trace that trade off more bias for less variance. But while high variance strategies necessitates using more samples, bias is more pernicious—even with an unlimited number of samples, bias can cause the algorithm to fail to converge, or to converge to a poor solution that is not even a local optimum

### An online actor critic algorithm

1. do $a\sim \pi_\theta(a|s)$, get $(s,a,r,s')$
2. fit $\widehat V^\pi_\phi$ to desired targets like $r + V^\pi_\phi(s')$
3. $\widehat A^\pi (s_i,a_i) := r(s_i,a_i) + \widehat V^\pi_\phi(s') - \widehat V^\pi_\phi(s)$
4. $\nabla_\theta J(\theta) \approx
\nabla_\theta \log
\pi_\theta(a_{i}|s_{i}) \left(\widehat A^\pi (s,a))\right)$
5. $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$

#### Architectural Design

In the above, we imply two seperate networks. In practice, this is stable and simple. If we share the networks between the policy and value function with two heads for the value and policy outputs, perhaps we could benefit from shared representations. However, this is less stable.

![](https://hackmd.io/_uploads/B1jeIW6An.png)

### Offline Actor Critic

Say we used the adapted the previous algorithm just by sampling from a replay buffer

1. do $a\sim \pi_\theta(a|s)$, get $(s,a,r,s')$ and store in buffer $\mathcal R$
2. sample batch of $N$ tuples $\{s_i,a_i,r_i,s'_i\}$ from $\mathcal R$
3. fit $\widehat V^\pi_\phi$ to using targets like $r_i + V^\pi_\phi(s'_i)$ for each $s_i$
4. $\widehat A^\pi (s_i,a_i) := r(s_i,a_i) + \widehat V^\pi_\phi(s'_i) - \widehat V^\pi_\phi(s_i)$ for each $i$
5. $\nabla_\theta J(\theta) \approx
\frac{1}{N}\sum\limits_{i=1}^N \nabla_\theta \log
\pi_\theta(a_{i}|s_{i}) \widehat A^\pi (s_i,a_i)$
6. $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$

There are two problems with this.
1. In step 3, the actions in the transitions are taken by old actors, so $s'_i$ is not the result of taking the action with the latest $\pi_\theta$ in $s_i$ and as a result, the target $r_i + V^\pi_\phi(s')$ is not correct for the value function *under the current policy*. 
2. In step 5, for the same reason, $a_i$ is not the action the current policy would have taken in $s_i$. Recall that the derivation of policy gradient relied on it being an expectation over $\pi_\theta$

We can fix 1) by fitting $Q^\pi_\phi$ instead of $V^\pi_\phi$. Since we cannot assume that the action $a_i$ came from the latest policy $\pi_\theta$, we use $Q^\pi_\phi$ that takes any $s,a$ pair from any version of our policy (crucial to the use of a replay buffer) and give us the expected return. In addition, if we use $Q^\pi_\phi$, the target changes to $r_i + Q^\pi_\phi(s'_i,a'_i)$ where $a'_i \sim \pi_\theta(\cdot|s'_i)$ i.e $a'_i$ is the action the policy would have taken in $s'_i$

We can fix 2) in an analogous manner. Instead of using $a_i$, we query the policy for what it would have done in the replay buffer $s_i$, that is, $a_i^\pi \sim \pi_\theta(\cdot|s_i)$

Fixed Algorithm:
1. do $a\sim \pi_\theta(a|s)$, get $(s,a,r,s')$ and store in buffer $\mathcal R$
2. sample batch of $N$ tuples $\{s_i,a_i,r_i,s'_i\}$ from $\mathcal R$
3. fit $\widehat Q^\pi_\phi$ to using targets like $r_i + Q^\pi_\phi(s'_i,a'_i)$ where $a'_i \sim \pi_\theta(\cdot|s'_i)$ for each $(s_i,a_i)$
4. $\nabla_\theta J(\theta) \approx
\frac{1}{N}\sum\limits_{i=1}^N \nabla_\theta \log
\pi_\theta(a_{i}^\pi|s_{i}) Q^\pi_\phi(s_i,a_i)$
5. $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$

Lastly, we dropped estimating the advantage. While using $Q^\pi_\phi(s_i,a_i)$ is slightly higher variance, we can just run more examples. In fact, we can do this cheaply without a simulator just by sampling more $a_i^\pi \sim \pi_\theta(\cdot|s_i)$




