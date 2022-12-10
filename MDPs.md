---
tags: RL
---
# Markov Decision Process (MDP)

MDPs are an idealized mathematical formalization of the full RL problem, where actions not only influence immediate reward, like bandits, but also subsequent states. As a result, the agent must trade off between immediate reward and delayed reward.

## Agent-Environment Interface
The learner who learns from interaction with the *environment* is the *agent*.

The agent and environment interact at each timestep t=0,1,2,3. . ..At each timestep $t$, the agent recieves a representation of the environment's state $S_t \in \mathcal{S}$. On that basis, the agent selects an action, $A_t \in \mathcal{A}(s)$. At timestep, $t+1$, the environment transistions stochastically to a new state $S_{t+1}$ and the agent receives scalar reward $R_{t+1} \in \mathcal{R}$ from the environment.

A *finite* MDP is one whose sets $(\mathcal{S},\mathcal{A},\mathcal{R})$ are finite.

![](https://i.imgur.com/NaqJZEP.png)

For $s' \in \mathcal{S}$ and $r \in \mathcal{R}$, let the dynamics of the MDP at each timestep be $p: \mathcal{S} \times \mathcal{R} \times \mathcal{S} \times \mathcal{A} \rightarrow [0,1]:$

$$p(s',r|s,a)=Pr\{S_t=s',R_t=r|S_{t-1}=s,A_{t-1}=a)\}$$

This is a conditional probability distribution, so 

$$\sum_{s' \in \mathcal{S}} \sum_{r \in \mathcal{R}}p(s',r|s,a) = 1 \space \forall s,a$$

We can compute many things from $p$.

Such as 
1. the state transistion probabilities,

\begin{equation}
p(s'|s,a) = \sum_{r \in \mathcal{R}} p(s',r|s,a)
\end{equation}

2. expected reward from (s,a)

\begin{equation}
r(s,a) = \mathbb{E}[R_t | S_{t-1}=s,A_{t-1}=a]=\sum_{r \in \mathcal{R}} r \sum_{s' \in \mathcal{S}}p(s',r|s,a)
\end{equation}

3. expected reward from (s,a,s')

<!-- \begin{equation}
r(s,a.s') = \mathbb{E}[R_t | S_{t-1}=s,A_{t-1}=a,S_t=s']=\sum_{r \in \mathcal{R}} r 
\end{equation} -->


### Markov Property
$p$ completely characterizes the environment dynamics given only the immediately preceding state and action. That is, the state fully captures all aspects of the past agent-environment interaction that make a difference for the future. This the **Markov Property**.

In other words, the future is independent of the past given the present. This assumption allows us to discard the long history of all previous interaction, to just focus on the current observation.

Mathematically, 
$$p(s_t | s_0,a_0,s_1,a_1,...,s_{t-1},a_{t-1}) = p(s_t|s_{t-1},a_{t-1})$$

Even if an environment doesn't fully satisfy the Markov property we still treat it as if it is and try to construct the state representation to be approximately Markov.
Ex:
![](https://i.imgur.com/EirAWJB.png)
Say if when the agent visits state A, action $i$ flips in its direction. Then, clearly, with just the state (x,y),once the agent visits A, the transition probabilities change drastically. This means markov property on $p$ won't hold. However,if we just add a boolean to the state if we have vistited A or not, then we have a complete state representation and markov will hold.

### State Representations

It is important to have a **minimal state representation**.
High dimensional states make it exponentially harder to learn good policies. Extraneous information can lead to poor generalization because the policy attempts to fit to states , where the extraneous information changes a lot, as new situations.

One can reduce the state space by abstracting it.

However, it is not always clear what pieces of information in a problem are extraneous...Designing the state space is more of an art than science.

## Reward and Goals

The agent's goal is to maximize the cumulative reward received. 

The designer of the environment must provide rewards in such a way that in maximizing them the agent will also achieve our goals. The use of a reward signal to formalize the idea of a goal is one of the most distinctive features of reinforcement learning

Ex:  In making a robot learn how to escape from a maze, the reward is -1 for every time step that passes prior to escape; this encourages the agent to escape as quickly as possible

It is important to note that the reward signal is say what you want the agent to do, not *how* you want it. There are some problems
1. Reward-hacking: The agent takes reward very literally, and find ways to do tasks that you didn't intend 
2. Reward-Sparsity: the agent receives only a reward at end of the epsiode and 0 otherwise. If the feedback is sparse, how can the agent ever be expected to learn? One needs to provide some sense incremental progress along the way.

Just as state design, shaping the reward signal functions can be tricky!

Inverse RL: find a reward function through demonstrations

## Returns and Episodes

In the episodic case, the cumulative reward, or return, is defined as

$$G_t = R_{t+1} + R_{t+2}+\cdots+R_T$$

At t=T, the episode is ending at a special terminal state. The set of all non-terminal states plus the terminal state is $\mathcal{S}^+$.

We can discount future reward by the discounting factor, $\gamma$.

$$G_t = R_{t+1} + R_{t+2}+\cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

More importantly, the return can be defined recusively
\begin{equation}
\begin{split}
G_t & = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3}+\cdots 
\\
& = R_{t+1}+\gamma G_{t+1}
\end{split}
\end{equation}

If $\gamma < 1$, then $G_t$ is bounded. If reward is 1, then $G_t$ is equal to $\frac{1}{1-\gamma}$

Suppose you treated pole-balancing as an episodic task but also used discounting, with all rewards zero except for -1 upon failure. What then would the return be at each time? How does this return differ from that in the discounted, continuing formulation of this task?

For the continuing case, let $K_i$ be the steps where the pole falls over(failure) for ith time. Then, the return would be $-\sum\gamma^{K_i-1}$.

Ex: If we have reward sequence 0,0,0,-1,0,-1,0; then the discounted sum is $0+\gamma 0 + \gamma^2 0 + \gamma^3(-1) + \gamma^4 0 + \gamma^5 (-1) + \gamma^6 0$

For the episodic case, we have $T$ timesteps where failure occurs at T. So $K = T$ and the return is $-\gamma^{K-1}$.

## Unified Notation for Continuing Case

We have defined return over an infinite number of terms and a number of terms. We can unify these by considering episodic termination to be entering a special absorbing state that generates reward of zero.

![](https://i.imgur.com/mExcOLc.png)

It easy to see we get the same return summing over the first $T$ steps or the full infinite sequence. 

$$G_t = \sum_{k=t+1}^{\infty} \gamma^{k-t-1} R_{k}$$

**Exercise**

The equations in Section 3.1 are for the continuing case and need to be modified (very slightly) to apply to episodic tasks (Assume that the random experiment is terminated after reaching the terminal state, rather than the absorbing state interpretation used in the book). What is the modified version of $\sum_{s' \in \mathcal{S}} \sum_{r \in \mathcal{R}}p(s',r|s,a) = 1 \space \forall s,a$

$$\sum_\limits{s'\in \mathbb{S^{+}}} \sum_\limits{r \in \mathbb{R}} p(s',r|s,a) = 1
$$


## Policies and Value Functions

Some RL algorithms estimate value functions - how 'good' the given state is or equivalently, the expected return for the agent to be in a given state. However, the rewards the agent can expect in the future also depends on the actions the agent takes in that state, which is determined by a policy.

A *policy* $\pi$ is a mapping from states to probabilities of selecting each possible action. 
$$\pi(a|s) = Pr(A_t = a| S_t = s)$$

The state-value function of a state $s$ under a policy $\pi$ is $v_{\pi}(s)$. 

$$ v_{\pi}(s) = \mathbb{E}_{\pi}[G_t|S_t=s] = \mathbb{E}_{\pi}[\sum_{k=0}^\infty \gamma^k R_{t+k+1}|S_t=s]$$

The action-value function of a state,action pair (s,a) under a policy $\pi$ is $q_\pi(s,a)$. This is expected return of taking action $a$ in state $s$ and following policy $\pi$ thereafter.


$$ q_{\pi}(s,a) = \mathbb{E}_{\pi}[G_t|S_t=s,A_t=a] = \mathbb{E}_{\pi}[\sum_{k=0}^\infty \gamma^k R_{t+k+1}|S_t=s,A_t=a]$$

The value functions can be estimated from experience as we did in the bandit case.
1. Monte Carlo: If an agent follows $\pi$ and averages over many random samples of the returns that followed from each state, for each state, the average $\rightarrow$ $v_\pi(s)$
2. Approximation: If there are too many states, we can maintain the value functions as parameterized functions

## Bellman Equation for $v_\pi$

**Exercise 3.11**

If the current state is $S_t$, and actions are selected according to a stochastic
policy $\pi$, then what is the expectation of $R_{t+1}$ in terms of $\pi$ and the four-argument
function p?

This is just probability rules.

$\mathbb{E}[R_{t+1}] = \sum\limits_{a} \pi(a|s) \sum\limits_{s',r} p(s',r|s,a)$

**Exercise 3.12, 3.18** 
Give an equation for $v_{\pi}$ in terms of $q_{\pi}$ and $\pi$.

![](https://i.imgur.com/JBxPKiV.png)


The value of a state depends on the values of the actions possible in that state and on how likely each action is to be taken under the current policy. 

\begin{equation}
\begin{split}
v_{\pi}(s) &= \mathbb{E}_{\pi}[q_{\pi}(s,a)] \\
& =\sum\limits_{a} \pi(a|s) \space q_{\pi}(s,a) \\
\end{split}
\end{equation}

**Exercise 3.13, 3.19** 
Give an equation for $q_{\pi}$ in terms of $v_{\pi}$ and the four-argument p.

![](https://i.imgur.com/DxbNaZM.png)

$q_\pi$ is just the one-step reward $R_{t+1}$ for taking action $a$ in state $s$ plus the discounted expected value of landing in the successor states.

\begin{equation}
\begin{split}
q_{\pi}(s,a) & = \mathbb{E}[R_{t+1} + \gamma G_{t+1}|S_t = s, A_t = a]\\
& =\mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1})|S_t = s, A_t = a]\\
& = \sum\limits_{s',r}p(s',r|s,a) \space [r+\gamma v_\pi(s')] \\
& = r(s,a) +\sum\limits_{s',r}p(s',r|s,a) \space \gamma v_\pi(s')
\end{split}
\end{equation}

From line 1, the reward after $R_{t+1}$ is just the expected reward from the next state $S_{t+1}$ which is why $v_\pi(S_{t+1})$ is substituted in line 2.

Notice how the expectation is not taken with respect to $\pi$

**Bellman Expectation for $v_\pi$**
Using the above, notice it can be viewed as a recurrence relation or one-step lookahead.

![](https://i.imgur.com/0aF2EUe.png)

\begin{equation}
\begin{split}
v_{\pi}(s) & = \sum\limits_{a} \pi(a|s) \space q_{\pi}(s,a) & \text{by 3.12}\\
 & = \sum\limits_{a} \pi(a|s) \sum\limits_{s',r}p(s',r|s,a) \space [r+\gamma v_\pi(s')] &\text{by 3.13}\\
 & = \sum\limits_{a} \pi(a|s)  \left[r(s,a) + \sum\limits_{s',r} p(s',r|s,a) \space \gamma v_\pi(s')\right] 
\end{split}
\end{equation}

Note how the final expression can be read as an expected value where $\pi(a|s)p(s',r|s,a)$ is the weight on $(r+\gamma v_\pi(s'))$ summed over all possibilities. Breaking it down
1. how likely are we to take each action in state $s$, thats $\pi$
2. then how likely are we to end up in $s'$ and recieve reward $r$, thats the dynamics $p$
3. how good is $v_\pi(s)$ is just the sample of what we got $r$ + the discounted expected value of future return in $s'$; thats in brackets

In the diagram, the open circles represent states and the solid circles represent actions. These diagrams are called *backup diagrams* because they diagram the update or 'backup' operation transferring value information *back* from the successor state to the starting state or action.

**Exercise 3.17** What is the Bellman equation for action values, that is, for $q_\pi$? It must give the action value $q_\pi(s, a)$ in terms of the action values, $q_\pi(s', a')$, of possible successors to the state–action pair (s, a).
![](https://i.imgur.com/40n4AOY.png)

\begin{equation}
\begin{split}
q_{\pi}(s,a) & = \sum\limits_{s',r}p(s',r|s,a)\left[r+\gamma \sum_{a'}\space\pi(a'|s')q_\pi(s', a')\right]\\
& = r(s,a) + \sum\limits_{s',r}p(s',r|s,a)\sum_{a'}\gamma \space\pi(a'|s')q_\pi(s', a')\\
\end{split}
\end{equation}

## Optimal Policy and Value Functions

Value functions define a partial ordering over policies. 
$$\pi \geq \pi' \textbf{ iff } v_{\pi}(s) \geq v_{\pi}(s') \space \forall \space s \in \mathcal{S} $$

There is always at least one optimal policy $\pi_*(s)$ which is better than all others. The optimal value function $v_*(s)$ and action-value function $q_*(s,a)$ is unique, but *not* the optimal policy. 

$$v_{*}(s) = \max\limits_{\pi} v_{\pi}(s) $$

$$q_{*}(s,a) = \max\limits_{\pi} q_{\pi}(s,a) $$

We can then write,

$$q_{*}(s,a) = \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1})|S_t = s, A_t = a]$$

And, the optimal policy is just

$$\pi_*(s) = \arg\max_{a\in\mathcal{A}} q_*(s,a)$$

### Bellman Optimality Equation for $v_*$
![](https://i.imgur.com/vTLY8XJ.png)

The value of a state under an optimal policy is the expected return after taking the *best* action from that state. Using the bellman equation for $v_\pi$

\begin{equation}
\begin{split}
v_{*}(s) & = \max_{a \in \mathcal{A(s)}} q_{\pi_*}(s,a) \\
& = \max_{a} \sum\limits_{s',r}p(s',r|s,a) \space [r+\gamma v_*(s')]) \\
& = \max_{a} \left[r(s,a) + \sum\limits_{s',r}p(s',r|s,a) \space \gamma v_*(s')\right]
\end{split}
\end{equation}

The $\pi$ disappeared because now we just take the best action.

### Bellman Optimality Equation for $q_*$

\begin{equation}
\begin{split}
q_{*}(s,a) & = \sum\limits_{s',r}p(s',r|s,a) \space [r+\gamma \max_{a'} q_*(s',a')]) \\
& = \sum\limits_{s',r}p(s',r|s,a) \space [r+\gamma v_*(s')] \\ 
\end{split}
\end{equation}

The second line is valid because the best q(s',a') is just the optimal value of being in s'

#### Solving

These Bellman optimality equations have no closed-form solution, non-linear, and have many iterative solution methods

#### Why do we need $q_*$

If you have the optimal value function $v_*$, then a one-step lookahead gives the optimal action. See how this is depicted in the bellman optimality equation.

Any policy which chooses the best action after a one-step lookahead is called greedy with respect to the optimal value function. However, because $v_*$ already takes into account the reward consequences of future behavior, **the greedy policy is optimal**. So the traditional concern of acting greedy and preventing access to better alternatives is alleviated.

Having $q_*$ means the agent does not need to do a one-step lookahead because it caches the results of one-step ahead searches. It simply needs to choose the action in state $s$ that maximizes $q_*$. **Notice how at the cost representing values for every state-action pair, rather than just the states, $q_*$ allows optimal actions to be selected without having to know about environment dynamics.**










