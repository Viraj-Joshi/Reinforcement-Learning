---
tags: RL
---
# Planning and Learning

Model-based RL methods require a model of the environment (e.g DP)

Model-free RL do not (e.g MC and TD learning)

Model-based RL methods rely on **planning** while model-free methods rely on **learning**.

They both compute value functions and involve backing-up value estimates of future states.

We can form algorithms that intermix model-based and model-free methods.

## Models and Planning

A **model** of the environment is anything an agent uses to predict how the environment responds to an action in a given state. They **simulate** the environment, producing **simulated experience**.

What we have been using, $p(s',r|s,a)$, is a *distributional model*, that is, one that produces a description of all possibilities and their probabilities.

Other models use *sample models*, that is, they produce just one of the possibilities according to its probability. These are often easier to obtain in practice.

**Planning** is the computational process of using a given model to produce an improved policy. This is a rich field in AI, but here we focus on *state-space planning*, which is a search through the state space for an optimal policy/path to a goal. 

State-space planning methods share a common structure that highlights the similarity to learning methods.

![](https://i.imgur.com/mSkesbc.png)

Both learning and planning methods estimate value functions through backing-up operations. However, planning uses simulated experience while learning uses real experience generated by the environment. As a result, learning methods can be substituted into planning because learning methods take experience as input.

Here is a simple planning method.
![](https://i.imgur.com/zXYzcUb.png)

Planning often most benefits the problem when it is intermixed with learning - we will see this next. 

## Dyna

Examine the following:![](https://i.imgur.com/aQhpglM.png)

Experience can be either used to improve the model - called **model learning**- or to directly improve the value function - called **direct RL**- using the RL methods discussed before. Model learning indirectly improves the value function and is called **indirect RL**. 

Dyna is an framework that integrates planning, acting, model-learning, and direct-RL. In other words, integrating model-free and model-based methods.

![](https://i.imgur.com/0v8CxRc.png)

On the model-free left side, direct RL directly improves the value function using real experience. 

On the model-based right side, the model learned gives rise to simulated experience. Selecting the starting state/action pair, from already experienced state-action pairs, is termed *search control*. Finally, planning is just applying RL methods to the simulated experience, usually using the same method as in the direct RL update. This is also improves the value function.

Thus, learning and planning are alike in their goal and share the same machinery, differing only in the source of experience.

In Dyna, planning takes the most computation relative to direct RL, acting, and model learning.

---

![](https://i.imgur.com/c0iIXzi.png)
DynaQ is an algorithm that uses one-step tabular Q-learning for direct RL (d) and one-step tabular Q-planning for planning (f). In (f), the uniformly sampled transitions are only those that have experienced by the agent in real-time. The model-learning method (e) is table-based for *tabular problems* and just records the most recent transition from $S_t$ taking action $A_t$ as $S_{t+1}$ and $R_{t+1}$.

DynaQ intermixes the planning and acting, while also model-learning as real expereince is gained. As the model changes, the planning process computes a different way of behaving.

**Example**
In a deterministic, episodic grid-world where some movements are blocked and reward is zero everywhere except for going into the goal G. The graph below shows the number of steps taken by the agent to reach goal in each episode for each level of planning $n$.
![](https://i.imgur.com/AmtEoce.png)
We notice that the planning agents ($n>0$) find the solution much faster (*sample efficient)* than the non-planning agent ($n=0$).

Examining the policy halfway through the second episode explains why this is the case. An arrow in a position indicates the greedy policy, otherwise all action-values were equal.

![](https://i.imgur.com/MlyT2w6.png)
The first episode was 1700 steps, so the model has plenty of experience to draw from.

The non-planning agent adds only one additional step per episode. With the planning agents, one step was also learned during the first episode, but during the second episode, a policy that almost reaches the starting state has been developed.

## When The Model Is Wrong

What happens when the model is incorrect, unlike in the previous example? This can happen when the environment is stochastic and all actions from each state have not been experienced, the environment has changed, or a model using function approximation has not generalized perfectly.

Our first example shows how suboptimal policy computed by planning is corrected when the suboptimal policy is too optimistic. 

**Example**
Say after 1000 timesteps, a gridworld's blocked spots changes as follows:

![](https://i.imgur.com/sFw4y6P.png)

At first, the policy computed by planning is to use the short path on the right, but when it is blocked, the planned policy attempts to exploit the short path, but finds it cannot, and eventually switches to the longer path.

What happens when the environment changes to be 'better' than it was before?

**Example**
![](https://i.imgur.com/fERubE5.png)

After 3000 timesteps, a shorter path opens up. Unfortunately, Dyna-Q never realizes the shortcut because the more it planned using its model, the less likely it would realize a shortcut was there. It takes too many exploratory actions to reveal the shortcut.

This is exactly the exploration-exploitation dilemna stated in previous chapters. We want the agent to find changes in the environment that can lead to greater reward, but not so much that it degreades performance.

Dyna-Q+ is an agent which attempts to mitigate this issue by giving a bonus reward to state-action pairs that have not been tried for a while in real-expereince. If the modeled reward is $r$, and the transition has not been tried in $\tau$ time steps, then planning updates are done as if that transition produced  reward of $r + \kappa\sqrt\tau$ , for some small $\kappa$. This is reminscient of the UCB bandit algorithm.

**Note**
Just like the bandit algorithms who continue to explore, we notice their performance in the long-run deteriorates.

If the environment never changes, it is most likely that the performance of DynaQ+ compared to Dyna should first improve and then deteriorate. This is because exploring helps at first but can perform sub-optimally when the environment is stable.


## Prioritized Sweeping
In Dyna, simulated transitions were drawn randomly. However, planning can be more efficient if the updates are focused on certain state-action pairs.

This is analogous to asynchronous DP, where only certain states were updated before making the policy greedy. In fact, there is a prioritized sweeping update based on the bellman error of states.

Intuitively, in the last example at the start of the second episode, only the state-action pair leading into the goal state has a positive value, so only updates of states leading into the goal have an update that change their value. As a result, there is a high probability a randomly drawn simulated transition used in a planning update will be produce no change in the value function. As the state space gets larger, this problem magnifies.

This suggests focusing our search 'backwards' from states whose values have changed results in efficient planning. Say, if the value function is correct for the environment but then it changes, and the value function is changed for some state $s$. Then, the predecessor states are the states where taking action $a$ leads to $s$ and should have their values updated. If these predecessors need to have their values updated, then their predecessor states need to have their values updated and so on.

As this frontier of updates propagates, it produces state-action pairs that are meaningful to update. However, not all are equal; some value estimates will change a lot and some a little. For example, the value of predecessor state-action pairs of state-action pairs that have changed a lot are more likely to also change a lot. 

The idea of prioritized sweeping is maintain a priority queue of state-action pairs, prioritized by how much their value estimate would change after their successors were updated. When the top pair is updated, the predecessor pairs of state-action pairs are inserted if their values are estimated to change by more than some threshold. 

![](https://i.imgur.com/DSGfl46.png)
**Example**
In a large state-space, we can see the advantage in reducing computation through prioritized sweeping.
![](https://i.imgur.com/GUYoFTN.png)

#### Extension to stochastic environments

The model maintains the count of the number of times each state-action pair has been experienced and what the next staets were. Then, the update is not a sample update, but an expected update.

---
This is just one way to distribute computation.

## Expected v Sample Updates

## Trajectory Sampling

One way to distribute updates are exhaustively across the entire state/action space as we did in DP (each state is updated once per sweep). Not only can this be impossible in large state spaces but exhaustive sweeps give equal importance to each state, rather than where it is needed. 

Another way is to distribute updates according to the on-policy distribution. This distribution is easily generated, the agent interacts with the model following the current policy from the start state $S_t$.

Following the on-policy distribution of updates seems at least better than uniform, because the agent should updates the states that would arise in real situations and ignores the vast, uninteresting parts of the state space (e.g don't want to update values of unlikely states in chess).

## Planning at Decision Time

All of the planning we have seen has improved the table entries or function approximation parameters using simulated experience such that an action can be selected for the agent at the current state. This is *background planning*.

Another way to use planning is a computation upon encountering state $S_t$ to output selected action $A_t$. On the next step $S_{t+1}$, planning begins again to produce $A_{t+1}$. This is **decision-time planning**

Decision-time planning is still a computation that uses simulated experience. Practically, because in large state spaces, the agent is unlikely to return to its current state, value estimates are often discarded after selecting the action. 

This type of planning is most useful when a fast response is not required (e.g chess). If low-latency is required, then planning in the background computes a policy that is immediately applicable to each newly encounted state.

## Heuristic Search

Given a starting state $S_t$, an expectimax tree of possible continuations can be considered using the model. The value of a leaf node is the approximate value function and backed up towards the root using expected updates with maxes. The action selected at $S_t$ is the max action-value reacheable from it.

![](https://i.imgur.com/l9J6YTR.png)

Heuristic search can be implemented as a sequence of one-step updates (shown here outlined in blue) backing up values from the leaf nodes toward the root by taking a max over actions and expectation over states. The ordering shown here is for a selective depth-first search.

In DP, we selected the greedy action looking ahead one-step.

Here, we search deeper than one step because if one has a perfect model and an imperfect action-value function, then in fact deeper search will usually yield better policies. 

Obviously, if the search is all the way to the end of the episode, then the effect of the imperfect value function is eliminated, and the action must be optimal. If the search is at depth $k$ such that $\gamma^k$ is very small, then the actions will be correspondingly near optimal.

The effectiveness of heuristic search is due to the focus on the states and actions most likely to follow your current state (e.g if you are playing chess, you focus on the your current position, likely next moves and successor states). Most problems have state spaces far too large to store values estimates for each, so heurstic search considering only the relevant ones makes it effective.

The tree size scales as $(|S||A|)^H$ where H is the horizon. This is infeasible for most real problems to calculate in timely fashion.

## Rollout Algorithms

These are decision-time planning algorithms based on MC control applied to simulated trajectories that begin at the current environment state $S_t$.

In other words, the action-values following a state $S_t$ are estimated by averaging the returns of many simulated trajectories that start with that state-action pair and follow a rollout policy. This algorithm is termed '*rollout*' because we *rollout* many trajectories from a state-action pair. When the estimate are accurate enough, the action with the highest estimated value is executed, and the process repeats with the next state.  

This sampling approach saves us computation of computing the value of the full expectimax tree.

![](https://i.imgur.com/rI3kSIM.png)

In some problems, a rollout algorithm with random rollout policy can produce good performance.

Notice that these algorithms do not produce a complete optimal action-value function for the policy, instead they produced MC estimates of action-values for each state following a *rollout policy*. The action-value estimates are discarded after choosing the best action-value.

### Why can we do this?

The Policy Improvement Theorem tells us given $\pi$ and $\pi'$ such that they are identical policies except that $\pi'(s)=a\neq \pi(s)$ for some state $s$. Then, if $q_\pi(s,a) \geq v_\pi(s)$, then $\pi'$ is as good as $\pi$.

In the rollout algorithm, the rollout policy is $\pi$. The MC estimates of each action $a' \in \mathcal{A}(s)$ are produced. Then, the policy $\pi'$ which at state $s$, takes the action $\arg\max_{a'} q_\pi(s,a')$, is at least as good as policy $\pi$.

This result is like one step of PI.

So then a rollout algorithm improves upon a rollout policy, rather than finding an optimal policy.

### Performance

Intuitively, one would expect that the better a rollout policy and the better the action-value estimates, the better the policy produced? 

As a decision-time planning method, the agent has a time constraint. The computational complexity depends on the branching number from each state, the length of simulated trajectories, the time to execute the rollout policy, and the number of rollouts. To speed up the computation, we can run MC trials in parallel on seperate processors and truncate the simulated trajectories with evaluation functions (bootstrapping from previous chapters)

### Similarities with learning algorithms

- estimate action values by averaging returns of a collection of sample trajectories, just like MC control (except with simulated trajectories)
- using simulated trajectories avoids exhaustive sweeps of DP by trajectory sampling
- using sample update avoids the need for distributional models
- take advantage of the policy improvement theorem by acting greedily w.r.t action values ()

## Monte Carlo Tree Search (MCTS)

This is a decision-time planning algorithm responsible for expert-level performance of AlphaGo(2016) in the game Go.

It is very similar to rollout algorithms.

Like rollout algorithms, MCTS is an iterative process where its executed after encountering each new state to select an action for that state.

The algorithm sketch is as follows for some model:

Repeat until time or some computational resource is exhausted:
1. Selection - Starting at the root $S_t$, the **tree policy** picks a leaf node accordong to $Q(S,A)$ in an $\epsilon$-soft manner
2. Expansion - Optionally, the tree is expanded from the selected node by adding one or more child nodes reached from the selected node via unexplored actions
3. Simulation - From the selected node, or from one of its newly-added child nodes (if any), simulation of a complete episode is run with actions selected by the **rollout policy**. **This rollout can very long, so often, action-value estimates are used to truncate the trajectory.** The result is a Monte Carlo trial with actions selected first by the tree policy and beyond the tree by the rollout policy
    - only requirement is that rollout policy be cheap to compute, not necessarily random.
5. Backup - The return generated by the simulated episode is backed up to update the action values attached to the edges of the tree traversed by the tree policy in this iteration of MCTS. No values are saved for the states and actions visited by the rollout policy beyond the tree
    - this step improves the tree policy (because it is $\epsilon$-soft)
7. Select the best action
    - Often, after selecting the action, MCTS is run, sometimes just with a tree of solely $S_{t+1}$, but most often the tree containing any descendants of $S_{t+1}$ left over from the tree constructed in the previous execution of MCTS. In addition, it can retain and update approximate value functions at the tree edgesor policies from one iteration to another.

![](https://i.imgur.com/TZIP85d.png)
This converges to the optimal search tree $q_*(s,a)$

MCTS can be interleaved with model learning (like Dyna).

### Why is this effective?

MCTS has the effect of focusing on MC trials on the trajectories whose initial segments are common to high-return trajectories previously simulated.

In addition, because the tree is grown incrementally, the action-value estimates that are stored are those relevant to the initial segments common to high-yield simulated trajectories. As a result, we avoid maintaining the impossibly large global action-value function and focus on the updates that matter (current decision) using trajectory sampling.

Like rollout algorithms, it is parallelizable and only relies on the model to provide samples.















