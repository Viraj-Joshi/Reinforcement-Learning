# Bandits

RL is different from other types of learning in that it is provided 'evaluative feedback' during training - indicating how good an action is, but not if it was the best or the worst. This drives exploration by the agent because it must determine the best behavior.

Here, we examine a degenerate case of RL, where the learner must find the single best action when the task is stationary or non-stationary. The actions only determine the immediate reward given.

If the situations are tagged with some sort of 'context', then the learner must learn a policy mapping situation to actions. If actions are allowed to affect the next situation as well as the reward, then we have the full reinforcement learning problem. This is the most challenging setting. 

## K-Armed Bandit Definition

This is a sequential game between an agent and the environment, which is simply K choices.

![](https://i.imgur.com/pXFOZn7.png)


In each round, the agent chooses between k 'arms' and depending on the arm chosen, receives a reward sampled from an unknown distribution.

The agent's goal is to maximize its reward over some number of timesteps $T$

The value of an action $a$ is the expected reward of selecting that action, called the action value.
$$q^* (a) = \mathbb{E}[R_t|A_t = a]$$

Obviously, if we knew $q^*(a)$ for all $a$, then we would take the argmax $q^*(a)$ every round, but we do not.

Because, these values are unknown at the start of the game, we define an estimator $Q_t(a)$, which is the estimate at time step $t$.

## Exploitation v Exploration

When you maintain estimators $Q_t(a)$, there is an estimate which is the highest across all actions. When the agent takes such an action, we say the agent is *greedy* and is *exploiting* its current knowledge.

Otherwise, the agent is exploring to improve its estimates of non-greedy actions.

Exploitation gives you greater reward in the short-term, but exploration may produce greater total reward in the long run. Thus, we have a central conflict when deciding what action to take.

There are many sophisticated methods for balancing exploration and exploitation for particular mathematical formulations of the k-armed bandit and related problem, but often these assumptions are *not* realistic in the full RL problem.

Instead, we focus on some balancing methods that work better than simply always exploiting. These are Action Value Methods.

## Action Value Methods

We will our estimates of the value of actions $q^*(a)$ to make action selection decisions.

### Sample Average Method

$$Q_t(a) = \frac{\text{sum of rewards when a taken prior to t}}{\text{number of times a taken prior to t}} = \frac{\sum_{i=1}^{t-1}R_i * 1_{A_i = a}}{\sum_{i=1}^{t-1} 1_{A_i = a}}$$

If the denominator is zero, set $Q_t(a)$ to some default.

$Q_t(a)$ is an unbiased estimator of $q^*(a)$ because it is the ***sample average*** of the relevant rewards.

Action-selection rules
- The agent is greedy every timestep - that is the agent would never spend any time sampling other seemingly inferior actions

$$ \DeclareMathOperator*{\argmax}{argmax} A_t = \argmax_a Q_t(a)$$
- $\epsilon$-greedy - that is, with some probability $\epsilon$ the agent uniformly picks at random from $m$ actions, and otherwise, acts greedily.
    - in the limit, every action will be sampled an infinite number of times, so $Q_t(a) \rightarrow q^*(a)$
    - $$ p(a)=\begin{cases} 
          \text {greedy action} & (1-\epsilon)+\epsilon/m \\
           \text {explore} & \epsilon \\
      \end{cases}$$
    
### Example
    
![](https://i.imgur.com/d6Dcimt.png)

Here is a bandit problem where the true value $q^*(a)$ was selected from $\mathcal{N} \sim (0,1)$ and the rewards from selecting an action $a$ were drawn from $\mathcal{N} \sim (q^*(a),1)$.

![](https://i.imgur.com/s1ZqJi0.png)

After the averaging the performance of $\epsilon$-greedy and greedy methods on 2000 bandit problems, we obtain the above figure. 

The ratio of optimal actions is calculated as the number of times the bandit took an action that was actually optimal at that timestep divided by the total number of runs

We can see how the greedy method performs worse in the long run because it locks into choosing a suboptimal action.

The $\epsilon$-greedy method performs better because by exploring, it has a better chance to recognize the optimal action. When $\epsilon = .1$, you explore very fast to find the best arms but eventually, you don't exploit enough (because you keep taking random actions). Whereas when $\epsilon = .01$, you explore more slowly, but then exploit very strongly far in the future. The $\epsilon = .01$ method would overtake the $\epsilon = .1$ method eventually. 

Of course, to get the best of both worlds, one could decay $\epsilon$ over time on a schedule to get the best of exploring strongly in the beginning and exploiting more as time goes on.

### Nonstationarity

Suppose the reward variance was large - like 10. Then, a$\epsilon$-greedy method fares well versus a greedy approach because exploration is needed to find an optimal action among noisy rewards.

Suppose the reward variance was 0 (deterministic arms), then the greedy approach is optimal because the true value of the action is revealed after trying it once.

Then, what if the bandit task became nonstationary, that is, $q^*(a)$ changes over time. In this case, exploration is needed in *both* the deterministic case and non-deterministic case because the one of the non-greedy actions could have its value shift.

## Incremental Implementation

How can we compute sample averages in a computationally-efficient manner?

$$ Q_n(a) = \frac{R^a_1+\cdots+R^a_{n-1}}{n-1}$$

The obvious implementation is to maintain a record of all rewards and then calculate an average when needed. However, then memory and time complexity would grow as time passes to calculate the numerator of $Q_n(a)$

We can do better by just storing two numbers and incrementally updating our sample average. Given $Q_n(a)$ and the *n*th reward $R^a_n$, the new average $Q_{n+1}$ can be calculated.
\begin{align*}
Q_{n+1}(a) & =  \frac{1}{n(a)} \sum_{i=1}^{n} R^a_i\\&= \frac{1}{n(a)} \left(R^a_n+\sum_{i=1}^{n-1} R_i\right)\\&= \frac{1}{n} \left(R^a_n+\frac{n-1}{n-1}\sum_{i=1}^{n-1} R^a_i\right) \\&=
\frac{1}{n(a)}\left(R^a_n + (n-1)Q_n(a)\right) \\&=
\frac{1}{n(a)}\left(R^a_n + nQ_n(a)-Q_n(a)\right) \\&=
Q_n(a)+\frac{1}{n(a)}\left[R^a_n-Q_n(a)\right]
\end{align*}

Notice how in the final expression, $[R^a_n-Q_n(a)]$ is an error in the estimate. This error in the current estimate $Q_n(a)$ is reduced by taking a step of some proportion towards the target $R^a_n$.

The step size here is $1/n$ and determines how much we change our current estimate. More generally, step size can be on a schedule, $\alpha_n(a)$

## Bandit Algorithm

![](https://i.imgur.com/IXhP96R.png)

## Step Size Convergence

In nonstationary problems, one should weight the more recent rewards more than long-past rewards. One way is to use a constant step-size parameter.

\begin{align*}
Q_{n+1}(a) & = Q_n(a) + \alpha\left[R^a_n-Q_n(a)\right) \\&= 
\alpha R^a_n + (1-\alpha)Q_n(a) \\&=
\alpha R^a_n + (1-\alpha)\left[\alpha R^a_{n-1}+(1-\alpha)Q_{n-1}(a)\right] \\&=
\alpha R^a_n + (1-\alpha)\alpha R^a_{n-1}+(1-\alpha)^2Q_{n-1}(a) \\&=
\alpha R^a_n + (1-\alpha)\alpha R^a_{n-1}+(1-\alpha)^2 \left[\alpha R^a_{n-2}+(1-\alpha)Q_{n-2}(a)\right] \\&=
\alpha R^a_n + (1-\alpha)\alpha R^a_{n-1}+(1-\alpha)^2 \alpha R^a_{n-2}+(1-\alpha)^3Q_{n-2}(a)\\&=
\alpha R^a_n + (1-\alpha)\alpha R^a_{n-1}+(1-\alpha)^2 \alpha R^a_{n-2} + \cdots + (1-\alpha)^n Q_1(a) \\&=
(1-\alpha)^n Q_1(a) + \sum_{i=1}^n (1-\alpha)^{n-i} \alpha R^a_{i}
\end{align*}

The weight $\alpha(1-\alpha)^{n-i}$ given to $R_i^a$ decays exponentially as a function of how many rewards ago, $n-i$, that $R^a_i$ was observed because the quantity $(1-\alpha) < 1$.

If $(1-\alpha)=0$, then all the weight goes on the very last reward $R_n$.

This is the *exponential recency-weighted average*.

For all sequences ${\alpha_n(a)}$, we are not guranteed to converge. To gurantee convergence w.p 1, we need 
1. $\sum_{i=1}^{\infty} \alpha_n(a) = \infty$
2. $\sum_{i=1}^{\infty} \alpha_n^2(a) < \infty$

The first condition ensures the steps are large enough to overcome bad initial estimates or random fluctuations. E.g we have $Q_t(a)=1000000$ and then $R_t = 1$, we need a large enough learning rate to pull our estimate back to the true value. This could happen with a bad initialization of $Q_t$ or a fluke random reward in the tail of a distribution.

The second condition ensures that our learning rate gets small fast enough so we converge.

In the sample average case, where $\alpha_n(a) = 1/n$, we meet both conditions.

In the case of constant step-size parameter $\alpha_n(a) = \alpha$, the first condition is met because this the harmonic series diverges, but the second condition is not met($\infty * \alpha^2 \nless \infty$), indicating that the estimates $Q_t(a) \forall a$ continue to vary in response to new estimates. This is *desirable* in the non stationary environment. 

- Or in other words, because the step-sizes are getting smaller and smaller(and so less and less weight gets assigned to the most recent rewards),$Q_t(a)$ is unable to adapt to nonstationarity, so it cannot be the case that the second condition is met. 

In full-RL, it is often the case the problems are non stationary.
The sequences ${\alpha_n(a)}$ that meet these conditions are not sample efficient because they converge very slowly.

## Optimistic Initialization

Set $Q_t(a)$ to some high value, like in our case, 5. Because $q^*(a)$ is drawn from $\mathcal{N}(q^*(a),1)$, this is wildy optimistic. However, this encourages action-value methods to explore. Whichever action was selected at the start will most likely have a disappointing reward (<5), so the learner will switch to other actions when that estimate falls as a result. Now, all actions will be selected several times before the value estimates converge.

![](https://i.imgur.com/TIRmeJi.png)

However, this trick really only works on stationary problems. In a nonstationary problem, the drive for exploration is temporary, so this technique cannot really help.

## UCB

When an $\epsilon$-greedy action selection chooses a non-greedy action, it does so randomly. Some could be nearly greedy or very uncertain in their estimate. It would be better to select among the non-greedy actions according to their potential for being optimal,meaning how close the estimate is to the maximum and the uncertainity of the estimate.

UCB chooses actions at time $t$ according to

$$A_t = \argmax_a \left[Q_t(a) + c \sqrt\frac{\ln t}{N_t(a)}\right]$$

where $N_t(a)$ is the number of times action $a$ has been selected prior to time $t$. If $N_t(a) = 0$, then a is considered to be a maximizing action.

The square-root term is a measure of uncertainity or variance in the action value's estimate. Each time $a$ is selected the uncertainity is reduced: $N_t(a)$ increments and because it is in the denominator, the uncertainity term decreases. Otherwise, every time an action other than $a$ is selected, $t$ increases but $N_t(a)$ does not; because $t$ appears in the numerator, the uncertainity estimate increases. The use of $\ln$ means that the increases get smaller over time, but are unbounded.

Actions with low value estimates (small $Q_t(a)$) or actions that have been selected frequently (big denominator and relatively smaller numerator) will be selected less frequently over time.

![](https://i.imgur.com/Qoj620m.png)

Up to and including t=10, there is always 1 action with $N_t(a)=0$ and so actions are selected being selected at random without replacement.

At t=11, $Q_t(a)$ comes into play and as a result, UCB will pick the action with highest value estimate. The reward shoots up. 

The reward subequently after t=11 decreases because the uncertainty estimate of the action selected at t=11 will be less than the others ($N_t(a)=2$ for this action versus 1 for all others). This action will thus be at a disadvantage at the next step. If c is large, then this effect dominates and the action that performed best in the first 10 steps is ruled out on step 12.


## Gradient Bandit

Instead of basing action selection on estimates of action values, we learn a real-valued *action preference* for each each action $a$, denoted $H_t(a) \in \mathbb{R}$. A higher number means the action is taken more often but has no interpretation with respect to reward.

When applied to a softmax distribution, we have the probabilties of selecting action a.

$$ \pi_t(a) = P(A_t = a) = \frac{e^{H_t(a)}}{\sum_{b=1}^{K} e^{H_t(b)}}$$

Initially, all action preferences are zero, so all actions have equal probability of being selected.

As the game progresses, we would want to iteratively update our action preferences using *exact* gradient ascent. 

$$H_{t+1}(a) = H_{t}(a) + \alpha * \frac{\partial{\mathbb{E}}[R_t]}{\partial{H_t(a)}}$$

$$\mathbb{E}[R_t] = \sum_x \pi_t(x) q^*(x)$$

This is increasing the action preference of the selected action in proportion to how a perturbation in the action preferences value changes our measure of performance, $\mathbb{E}[R_t]$. However, we cannot compute this because $q^*(x)$ is unknown. Instead, we can derive an update that is equal in expected value to that of the exact gradient update. This is an instance of **SGD**.

Looking at the "performance gradient"
\begin{align*}
\frac{\partial{\mathbb{E}[R_t]}}{\partial{H_t(a)}} & =
\frac{\partial{}}{\partial{H_t(a)}} \left[  \sum_x \pi_t(x) q^*(x) \right] \\&=
\sum_x q^*(x) \frac{\partial \pi_t(x)}{\partial{H_t(a)}} \\&=
\sum_x (q^*(x)-B_t) \frac{\partial \pi_t(x)}{\partial{H_t(a)}}
\end{align*}

#### Why are we allowed to subtract a baseline
Notice what $\sum_x \frac{\partial \pi_t(x)}{\partial{H_t(a)}}$ is saying. How does the probability of taking each action $x$ change as we nudge the preference of taking action $a$? Because $\sum_x \pi_t(x) = 1$ (as it is a probability distribution), then changing $\pi_t(x)$ by some amount means the change has to be distributed among all other $\pi_t(x')$. So the gradient is zero sum! So adding $B_t$ in $(q^*(x)-B_t)$ does not change the bias the expectation because $B_t$ is being multiplied by something that adds up to zero.


Next, we multiply by $\pi_t(x)/\pi_t(x)$

$$\frac{\partial{\mathbb{E}[R_t]}}{\partial{H_t(a)}}=
\sum_x \pi_t(x) (q^*(x)-B_t) \frac{\partial \pi_t(x)}{\partial{H_t(a)}}/\pi_t(x)$$

Notice, how the RHS is an expectation now

$$=\mathbb{E} \left[(q^*(x)-B_t) \frac{\partial \pi_t(x)}{\partial{H_t(a)}}/\pi_t(x)\right]=
\mathbb{E} \left[(R_t-\bar{R}_t) \frac{\partial \pi_t(x)}{\partial{H_t(a)}}/\pi_t(x)\right]$$

where the in RHS, we have chosen baseline $B_t=\bar{R}_t$ and substituted $R_t$ for $q^*(A_t)$ because the $\mathbb{E}[R_t|A_t] = q^*(a)$

Next, the derivative $\frac{\partial \pi_t(x)}{\partial{H_t(a)}}$ must be derived.

$$=\frac{\partial}{\partial{H_t(a)}} \left(\frac{e^{H_t(x)}}{\sum_{b=1}^{K} e^{H_t(b)}}\right)
=\frac{(\sum_{b=1}^{K} e^{H_t(b)}) e^{H_t(x)}1_{x=a} - e^{H_t(x)}e^{H_t(a)}}{(\sum_{b=1}^{K} e^{H_t(b)})^2}
$$
$$
=\frac{(\sum_{b=1}^{K} e^{H_t(b)}) e^{H_t(x)}1_{x=a}}{{(\sum_{b=1}^{K} e^{H_t(b)})^2}} - \frac{e^{H_t(x)}e^{H_t(a)}}{(\sum_{b=1}^{K} e^{H_t(b)})^2}
=\frac{e^{H_t(x)}1_{x=a}}{{(\sum_{b=1}^{K} e^{H_t(b)})}} - \frac{e^{H_t(x)}e^{H_t(a)}}{(\sum_{b=1}^{K} e^{H_t(b)})^2}
$$
$$
=1_{x=a} \pi_t(x) - \pi_t(x) \pi_t(a) = \pi_t(x)(1_{x=a} - \pi_t(a))
$$

Then, substituting $\frac{\partial \pi_t(x)}{\partial{H_t(a)}}$ into 

$$\mathbb{E} \left[(R_t-\bar{R}_t) \pi_t(x)(1_{x=a} - \pi_t(a))/\pi_t(x)\right]
=\mathbb{E} \left[(R_t-\bar{R}_t)(1_{x=a} - \pi_t(a))\right]$$

Finally, substituting a sample of the expectation of $\frac{\partial{\mathbb{E}[R_t]}}{\partial{H_t(a)}}$ into our interative updates

$$H_{t+1}(a) = H_{t}(a) + \alpha * \left[(R_t-\bar{R}_t)(1_{a=A_t} - \pi_t(a))\right] \forall a$$

### Why does the Variance of Gradient estimates matter

Our gradient estimator is based off rewards so large rewards means large gradients (and greatly adjust our preferences) according to the update rule (imagine no baseline term)above. However, subtracting off a baseline **and** multiplying by some number, we lower the variance of gradient estimates. In other words, it keeps us from moving too far from our current $\pi$ in one step.

It is important to note that simply subtracting a number from a random variable does not reduce variance. The fact that we multiplied by another term is what reduced variance.
- subtracting/addition does not change variance of samples, only mean
- multiplication does change variance of samples

E.g Say if $\pi(A) = .9$, $\pi(B)=.1$ and this is optimal selection strategy. Then, in a high-variance update when B gives high reward, the gradient estimate is a large number times $\frac{\partial \pi_t(x)}{\partial{H_t(a)}}$. This large change in preference would pull the current $\pi$ far away from its current probabilities. With a baseline, this change would NOT be so drastic because you subtract off that high reward from a more typical reward and multiply a smaller number by $\frac{\partial \pi_t(x)}{\partial{H_t(a)}}$. This results in smaller gradients, and so the variance was reduced!

![](https://i.imgur.com/zJLOBVl.png)

See how in the left hand column, no baseline causes $\pi$ to swing in extremes. In the right hand column, the updates are more reasonable.

In the above example, when the learner adjusts its probabilities so wildy in the high variance case, it leaves no room that other actions could have been better. Now, when a better action comes along in the future, it will wildy swing towards that one.



#### Why use the mean as the baseline?

Noting that we want a small number to multiply aganist $\frac{\partial \pi_t(x)}{\partial{H_t(a)}}$, choosing the mean by definition is close to all samples and subtracting it from sample rewards would give you that.




## Thompson Sampling
meh










