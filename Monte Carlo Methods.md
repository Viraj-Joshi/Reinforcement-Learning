---
tags: RL
---
# Monte Carlo Methods

Monte Carlo (MC) methods are methods to solve RL problems by estimating the value of a state, as the empirical mean of sample returns from that state, *with no model of the environment $p$*. As a result, MC methods only need experience - a trajectory of states, actions, and rewards from actual or simulated experience. It turns out it is often easy to simulate sample episodes, but hard to construct an explicit model. The term 'Monte Carlo' is used is any estimation method which relies on a random component.

MC methods require that the tasks are episodic - meaning the episodes terminate no matter what - so that the returns of each episode are defined - and as a result, the values.

**They do not bootstrap.**

Note how MC methods evokes problems associated with the bandit problem. We will encounter problems of exploration and non-stationarity.

We will also adapt ideas from DP - learning the value functions from sample returns instead of the MDP knowledge as well as policy improvement.

## MC Prediction

Prediction is the computation of estimating $v_\pi$ as $V$ and $q_\pi$ as $Q$ for some policy $\pi$

Since the value of a state $s$ is the expected return $\mathbb{E}_\pi[G_t|S_t = s]$ starting from $s$, one can estimate it by simply averaging the returns observed after visits to that state. More returns => average is closer to expected value for that state. However, this is **high variance** strategy, **one is estimating the value from a sequence of random transitions and actions.**

The *first visit* MC method estimates $v_\pi(s)$ as the average of the returns following first visits to s, whereas the *every visit* MC method averages the returns following all visits to $s$. Both methods converge to $v_\pi$ as the number of (first) visits $\rightarrow \infty$. 

The return from state $s$ from each episode is a random variable that is i.i.d estimate of $v_\pi(s) \space \forall s$ with fixed variance. By the law of large numbers, the sequence of averages of returns from state $s$ converges to the expected value as $n \rightarrow \infty$. Each average is an unbiased estimate, $\mathbb{E}[V(s)] = v_\pi(s)$ and the standard deviation of the sample averages falls as $1/\sqrt{n}$.
![](https://i.imgur.com/pIfB0Ya.png)

To modify the above for every-visit, simply remove the check before appending and averaging returns. Note how the algorithm proceeds backward through the episode for an ease of calculation

An incremental implementation *per-episode* for every-visit MC is 
$$V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)]$$

The estimate for each state $s$ is independent of each other because no state uses another estimate for its own estimate. That is, MC methods do **not bootstrap**.

As a result, the computation required for estimating the value of a state is independent of the total number of states. If one was interested in the value of some subset of states, this is efficient.

#### Backup Diagrams

![](https://i.imgur.com/gAu8f2l.png)

For MC estimation of $v_\pi$, the backup diagram is the root is a state node and the trajectory of transistions that follow in that episode.

The difference in DP versus MC can be summarized by their horizon and bootstrapping. MC backup diagram depicts only the sampled transitions for the entire episode whereas DP backup diagrams will show all possible transitions for one step. The value backed up in MC is from the total return while DP will backup values from the next state to the current state.

**Example**
![](https://i.imgur.com/SF7AT9Z.png)
These are estimated value functions for blackjack.

For a policy that stops hits on a player sum of 20 or 21, and hits otherwise, we can estimate the value function using every-visit MC method.

Notice how in the usable ace state case, even after 10000 episodes, the estimate is noisy because the state is rare.

## Monte Carlo Estimation for Action Values

Without a model and just a value function, one cannot do a one-step lookahead and choose the action leading to the best combination of reward and next state precisely because choosing the actions/maximizing over rewards is impossible without a model.

As a result, MC methods are particulary useful in Policy Evaluation of the action value function $q_\pi$, the expected return when starting in state $s$, taking action $a$ and then following $\pi$

The MC method for estimating $q_\pi$ is very similar to that of $v_\pi$ except we now record returns from visits to state-action pairs. There are 'every-visit' and 'first-visit' algorithms defined analogusly to the value function estimation case.

We see the issue of exploration arise here as we did for the bandit problem. If $\pi$ is deterministic, then following $\pi$ will observe returns only from one action from each state. The other state-action pairs will not be estimated correctly because they are never encountered.

A solution is 'exploring starts' - where episodes start in a state action pair such that each pair has nonzero probability of being selected. Of course, in the real world, placing the agent into all such pairs at the start of the episode may not be possible.

Another solution is that to only use policies that have non-zero probability of selecting all actions in each state.

## Monte Carlo Control

There is a MC version of Policy Iteration according to the idea of GPI. That the value function is moved toward $v_\pi$ and the policy is improved w.r.t the current value function.
![](https://i.imgur.com/sf6v7Kp.png)
PE is done using MC estimation of action-values where the approximate $Q$ approaches $q_{\pi_k}$ in the limit, just as the iterative bellman expectation approached $v_{\pi_k}$ in the limit in the DP case.

Policy Improvement is done by making the policy greed w.r.t the action-value function, which is just an $\arg\max$ rather than a one-step lookahead with a value function done in the DP case.

$$\pi(s) = \arg\max_a q(s,a)$$

The Policy Improvement theorem applies to successive policies $\pi_k$ and $\pi_{k+1}$ because

\begin{equation}
\begin{split}
q_{\pi_k}(s,\pi_{k+1}(s)) & = q_{\pi_k}(s,\arg\max_a q_{\pi_k}(s,a)) \\
& = \max_a q_{\pi_k}(s,a) \\
& \geq q_{\pi_k}(s,\pi_k(s)) \\
& \geq v_{\pi_k}(s)
\end{split}
\end{equation}

As a result, this theorem tell us Policy Iteration will eventually lead to the optimal value function and policy as before. So we can use MC method to find optimal policies *without model dynamics.* However, we made two crucial assumptions that underlie our convergence guarantee - exploring starts and infinite episodes during policy evaluation (guranteeing $Q$ converges to $q_\pi$)

We write the following algorithm following MC PI w/exploring starts
![](https://i.imgur.com/ZJhx7JC.png)

This algorithm makes the practical assumption of not completing PE before returning to Policy Improvement (like how Value iteration performs one step of PE before returning to Policy Improvement). This how one gets around the requirement of needing an exact $q_\pi$ for policy improvement to hold.

Notice how it accumulates returns regardless of the policy it was using at the time. 


## Monte Carlo Control with $\epsilon$-soft policies

Because exploring starts is an unrealistic assumption to guarantee convergence of $V$ to $v_\pi$, we introduce a *soft* policy such that $\pi(a|s) > 0 \space \forall s,a$ but gradually shift closer and closer to a deterministic optimal policy.

An example of $\epsilon$-soft policies is $\epsilon$-greedy policies where the policy selects the type of action with the following probabilities
\begin{equation}
\pi(a|s)=
\begin{cases} 
\frac{\epsilon}{|\mathcal{A}(s)|} & a \neq a^* \\
1-\epsilon + \frac{\epsilon}{|\mathcal{A}(s)|} & a^* = \arg\max Q(s,a) \\
\end{cases}
\end{equation}

![](https://i.imgur.com/8PdcULS.png)
Still in the GPI framework, we obtain the above algorithm that modifies the Policy Improvement step. Instead of taking the policy all the way to a greedy policy, it moves toward the greedy policy. Instead of computing an exact $q_\pi$, it moves towards the value function for the policy.

The Policy Improvement theorem holds because the prerequisites hold for any $\epsilon$-greedy policy $\pi'$
Proof:
\begin{equation}
\begin{split}
q_\pi(s,\pi'(s)) & = \sum_a \pi'(a|s) q_\pi(s,a)\\
& = \frac{\epsilon}{|\mathcal{A}(s)|} \sum_a q_\pi(s,a) + (1-\epsilon) \sum_a \max_a q_\pi(s,a) \\
& \geq \frac{\epsilon}{|\mathcal{A}(s)|} \sum_a q_\pi(s,a) + (1-\epsilon) \sum_a \frac{\pi(a|s)-\frac{\epsilon}{|\mathcal{A}(s)|}}{1-\epsilon} q_\pi(s,a) \\
& = \frac{\epsilon}{|\mathcal{A}(s)|} \sum_a q_\pi(s,a) -\frac{\epsilon}{|\mathcal{A}(s)|} \sum_a q_\pi(s,a)+ \sum_a \pi(a|s)q_\pi(s,a) \\
& = v_\pi(s)
\end{split}
\end{equation}

In the third to last step the weighted average of nonnegative weights summing to 1 of q values is less than the max q value

So by Policy Improvement Theorem, $\pi' \geq \pi$ because $v_{\pi'} \geq v_\pi$. $\pi'=\pi$ occurs when they are both optimal policies, which occurs when they are better than all other $\epsilon$-soft policies.

Proof:....

This assumes the action-value functions are computed exactly.

## Off-Policy Prediction with Importance Sampling
On-policy methods attempt to evaluate or improve the policy that is used to make decisions. The methods above were on-policy.

In the above methods, we face a dilemma. An agent needs to behave non-optimally to explore all the action, to find the optimal action, but must also learn the action values conditioned on subsequent optimal behavior. 

One can resolve this dilemma by using two policies. One is the *target* policy, which is the policy being learned about and becomes the optimal policy. Another is the behavior policy, which is used to generate trajectories that can explore. **off-policy methods evaluate or improve a policy different from that used to generate the data**

Consider the prediction task of estimating $v_\pi$ or $q_\pi$ given episodes drawn from a behavior policy $\beta$ where $\beta \neq \pi$, where $\pi$ is the target policy.

In addition, we assume the concept of coverage, that every action under $\pi$ is also taken under $\beta$; $\pi(a|s)>0 \rightarrow \beta(a|s)>0$. This is because using episodes drawn from $\beta$ to compute values following $\pi$ is obviously required. Then,if $\pi(a|s)$ is stochastic, then $\beta(a|s)$ must be as well. 

Often, the target policy is the deterministic greedy policy and the behavior policy is stochastic and exploratory(e.g $\epsilon$-greed).

To estimate the expected values under one distribution (target) given samples from another(behavior), we apply importance sampling. By weighting the expected return of the behavior policy, we obtain the expected return under the target policy.

This weighting is the importance-sampling ratio $\rho_{t:T-1}$ or the ratio of probability of the trajectory, starting at time t, drawn from the target policy to the probability of the trajectory, starting at time t, drawn from the behavior policy.

$$\rho_{t:T-1} = \frac{\prod_{k=t}^{T-1} \pi(A_k|S_k)p(S_{k+1}|S_k,A_k)}{\prod_{k=t}^{T-1} \beta(A_k|S_k)p(S_{k+1}|S_k,A_k)} = \frac{\prod_{k=t}^{T-1}\pi(A_k|S_k)}{\prod_{k=t}^{T-1}\beta(A_k|S_k)}$$

Notice how the MDP dynamics function cancels out. 

This makes sense logically too, because if a return is common under $\pi$ but not $\beta$, we would observe a large ratio - upweighting that return - and downweighting vice-versa.

Transforming the expected return $\mathbb{E}[G_t|S_t=s]=v_\beta(s)$ using importance sampling, we have, which is what we desired
$$\mathbb{E}[\rho_{t:T-1}\space G_t|S_t=s]=v_\pi(s)$$

We can estimate the above using a MC algorithm averaging return from episodes drawn under $\beta$.

First, some notation.
let the timestep t not reset at every episode, e.g if episode 1 ends a t=100, then episode 2 starts at t=101

let $\mathcal{J}(s)$ be the set of time steps in which state $s$ is (first) visited
let $T(t)$ be the first time of termination following time t
let $G_t$ be the return from time $t$ up through $T(t)$
let $\{G_t\}_{t\in \mathcal{J}(s)}$ be the set of returns pertaining to state $s$
let $\{\rho_{t:T(t)-1}\}_{t \in \mathcal{J}(s)}$ be the set of importance sampling ratios corresponding to the return starting at $t$ and terminating at $T(t)$

Finally, the ordinary importance-sampling estimate of $v_\pi$ is 
$$V(s) = \frac{\sum_{t \in \mathcal{J}(s)}\rho_{t:T(t)-1}G_t}{|\mathcal{J}(s)|}$$

An alternative formulation that normalizes the importance ratios, the weighted importance sample, is
$$V(s) = \frac{\sum_{t \in \mathcal{J}(s)}\rho_{t:T(t)-1}G_t}{\sum_{t \in \mathcal{J}(s)}\rho_{t:T(t)-1}}$$


---

#### Exercise
What is analogous equation for Q-values?
$$Q(s,a) = \frac{\sum_{t \in \mathcal{T}(s,a)}\rho_{t+1:T(t)-1}G_{t}}{\sum_{t \in \mathcal{T}(s,a)}\rho_{t+1:T(t)-1}}$$


---

The difference between importance sampling methods of first-visit methods are defined by differences in bias and variance of $V(s)$
1. Bias - weighted importance sampling is biased (see that the ratio $\rho$ will cancel in the numerator and denominator when there is only one episode). This bias falls asymptotically to zero as the number of samples increases. It is unbiased in the ordinary case.
2. Variance
    - ordinary importance sampling is unbounded in variance because $\rho$ is unbounded(e.g action very likely in target policy but very unlikely in berhavior policy), so the scaled returns have infinite variance, so the off-policy estimator V(s) is a high-variance method. This is an issue as we will see below.
    - variance of weighted importance sampling is converges to zero (proof omitted, but its easy to see normalizing the importance weights bounds what we multiply $G_t$ by to (0,1))

Using the every-visit method for ordinary and weighted importance sampling are both biased. The bias falls asymptotically to zero as the number of samples increase.

Ex: If s is that the dealer shows a deuce, the sum of the player's card is 13, and the player has a usable ace; the value of the state under the target policy calculated by averaging returns of 1e8 episodes was -.27726
![](https://i.imgur.com/s8DJp0M.png)

It turns out in practice, despite that  *every-visit weighted importance sampling* is biased, it is preferred due to the much lower variance and not needing to maintain what states have been visited.

#### Infinite Variance of Ordinary Importance Sampling
![](https://i.imgur.com/8MAOl2M.png)
Here is a simple MDP with behavior policy $b$ and target policy $\pi$.

In the target policy, all episodes under this policy would be some number of transitions back to s, and then termination with a reward of +1. Then, $v_\pi(s)=1$ with $\gamma=1$. Suppose we estimate this value from off-policy data using the behavior policy b.

Note how the estimate never reaches 1 using ordinary importance sampling. Think why is the case.

In contrast, the weighted importance-sampling algorithm would give an estimate of exactly 1 forever after the first episode that ended with the left action. All returns not equal to 1 (that is, ending with the right action) would be inconsistent with the target policy and thus would have a $\rho_{t:T(t)-1}$ of zero (numerator is 0) and contribute neither to the numerator nor denominator of the $V(s)$. The weighted importance sampling algorithm produces a weighted average of only the returns consistent with the target policy, and all of these would be exactly 1

#### Why the variance of Importance Sampled Returns is infinite in the example?

Given a random variable X, it is known
$$VAR[X] = \mathbb{E}[X^2] - \bar{X}^2$$

So because the mean of our random variable $\rho_{0:T-1} G_0$ is finite, the variance is infinite if the expected value of its square is infinite



\begin{equation}
\mathbb{E}\left[\left(\prod\limits_{t=0}^{T-1} \frac{\pi(A_t|S_t)}{b(A_t|S_t)}\right)^2 G_0\right] \stackrel{?}{=} \infty
\end{equation}

Note that we only consider episodes which take the left action and transition to the terminal state with reward +1 after some amount of transitioning to the non-terminal state. Episodes taking the right action have return 0 so they can be ignored. Episodes taking the left action have return 1, so $G_0$ can be ignored. 

The expectation is the probability of each episode's occurence according to its length multiplied by the importance sample ratio squared.

$$\frac{1}{2} * .1 (\frac{1}{.5})^2+\left[\frac{1}{2} * .9 * \frac{1}{2} * .1 (\frac{1}{.5}\frac{1}{.5})^2\right]+\left[\frac{1}{2}*.9*\frac{1}{2} * .9 * \frac{1}{2} * .1 (\frac{1}{.5}\frac{1}{.5}\frac{1}{.5})^2\right]+\cdots=\infty$$
TLDR: Importance Sampling is a high-variance technique, and we need some additional tricks to reduce this.

## Safety in Off-Policy Evaluation

If one wants to evaluate a policy $\pi$ that could potentially be dangerous(e.g robot breaks) without rolling out episodes using $\pi$, then one could use an off-policy evaluation of $\pi$ using some *safe* behavior policy $b$

One wants high confidence that your new policy $\pi$ is at least as good as your old one $b$. Formally, we want a lower bound $V_\pi^{lb}$ such that $V_\pi > V_\pi^{lb}$ w.prob $1-\delta$ 

Using the Chernoff-Hoeffding Inequality,

$$\mu \geq \frac{1}{n} \sum_i x_i - b \sqrt{\frac{\log 1/\delta}{2n}}$$

where $0\leq x_i \leq b$. 

This is saying the true mean is greater than equal to the sample average of some collection of random variables minus some term which loosens our constraint. 

In the context of importance sampling

$$v_\pi \geq \frac{1}{n} \sum_i G_i \rho_{i:T(i)} - G_{max} \sqrt{\frac{\log 1/\delta}{2n}}$$

There are tighter bounds with distributional assumptions.

## Incremental Implementation

MC methods,off-policy and on-policy, can use incremental methods to compute $V(s)$ like in the bandit case, in an episode by episode update.

For weighted importance sampling, given a set of n-1 returns $\{G_t\}_{t\in \mathcal{J}(s)}$, for each episode, define
1. the weight of each return be $W_i = \rho_{t_i:T(t_i)-1}$
2. the cumulative sum $C_i$ of weights given to the returns

The estimate is,
$$V_n = \frac{\sum_{k=1}^{n-1}W_k G_k}{\sum_{k=1}^{n-1}W_k}$$
And the incremental update rule receiving the nth return $G_n$ is,
$$V_{n+1} = V_n + \frac{W_n}{C_n}[G_n - V_n]$$

![](https://i.imgur.com/mnvKjTq.png)
$Q$ converges to $q_\pi$ as we desired.

If $\pi=b$, the algorithm applies to the on-policy case as well.

## Off Policy MC Control

Using Off Policy MC prediction and GPI, if we let the target policy $\pi$ be the greedy policy *w.r.t* to Q and the behavior policy $b$ be $\epsilon$-soft (so that every state-action pair is selected an infinite number of times), we obtain an optimal policy $\pi_*$ and optimal $q_*$.
![](https://i.imgur.com/C35RKUH.png)
**Exercise**
You may have been expecting the W update to have involved the importance-sampling ratio $\pi/b$ , but instead it involves $1/b$ . Why is this nevertheless correct?

In the off-policy MC control algorithm,$\pi$  is a deterministic policy. Therefore, for the action actually taken, its probability of being taken is always 1.



## Reducing Variance of Off-Policy Estimators
The core idea is by taking advantage of the structure of the return as the sum of rewards to reduce variance of importance sampling.

### Per Decision Importance Sampling

In the numerator of ordinary and weighted importance sampling, we have  
\begin{equation}
\begin{split}
\rho_{t:T-1} G_t  & = \rho_{t:T-1} (R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{T-t-1}R_T) \\
& = \rho_{t:T-1} R_{t+1} + \gamma \rho_{t:T-1} R_{t+2} + \cdots + \gamma^{T-t-1}\rho_{t:T-1}R_T \\
\end{split}
\end{equation}

This is weighting each reward by everything that happened before it, which did influence the likelihood of seeing that random reward, and by everything after, which obviously did not. Lets fix that.

let $\tilde{G_t} = \rho_{t:t} R_{t+1} + \gamma \rho_{t:t+1} R_{t+2} + \gamma^2 \rho_{t:t+2} R_{t+3}+\cdots + \gamma^{T-t-1}\rho_{t:T-1}R_T$

So an alternative to estimating $v_\pi$ as before, is to take the per-decision importance sampling idea above.

Proof:

That is, $V_\pi=\mathbb{E}[\rho_{t:T-1}G_t]=\mathbb{E}[\tilde{G_t}]$. PDIS is unbiased estimator

Then, the lower-variance ordinary importance-sampling estimator of $v_\pi$ is 

$$V(s) = \frac{\sum_{t \in \mathcal{J}(s)}\tilde{G_t}}{|\mathcal{J}(s)|}$$

Intuitively, the less importance weights the less variance

An aside: this is a familar pattern to reduce variance we will see in policy gradients

The estimators proposed for per-decision weighted importance sampling are not consistent.



