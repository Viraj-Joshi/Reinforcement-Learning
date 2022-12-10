---
tags: RL
---
# Dynamic Programming (DP)

Given a perfect model of the environment as an MDP, we can compute optimal policies using DP. These algorithms are limited in use due to computational requirements and the model assumption.

let $<\mathcal{S},\mathcal{A},\mathcal{R}>$ be finite and the model dynamics $p(s',r|s,a)$ be given. DP can be applied to continuous state and action spaces by quantizing these spaces and obtaining an approximate solution using finite-state DP methods.

The heart of DP algorithms is turning Bellman equations into iterative updates.

## Policy Evaluation

Policy Evaluation - What is $v_\pi$ for an arbitrary $\pi$?

Remember that 
$$v_\pi(s) = \sum\limits_{a} \pi(a|s) \sum\limits_{s',r}p(s',r|s,a) \space [r+\gamma v_\pi(s')] $$

As long as $\gamma < 1$ (to prevent infinite reward in a continuing environment) or termination guranteed by policy, $v_\pi(s)$ exists and is unique

This is a system of $|\mathcal{S}|$ equations in $|\mathcal{S}|$ unknowns that can be solved. However, it can be solved iteratively as well.

Let $\{v_k\}_{k=0}^{\infty}$ be a sequence of approximate value functions. The DP approach is applying the Bellman equation as an update to produce successive approximations where $v_0$ is arbitrary except the terminal state must have a value of 0.

The update is:

$$v_{k+1}(s) = \sum\limits_{a} \pi(a|s) \sum\limits_{s',r}p(s',r|s,a) \space [r+\gamma v_k(s')]$$
$v_k = v_\pi$ is a fixed point because the Bellman equation assures of equality

Note, $v_k(s)$ is the expected reward from state $s$ after acting $k$ steps from that state. So we see that the values propagate 'backwards' from the best and worst states.

Ex: TODO

This type of update is called an ***expected update*** because they are based on the expectation of next possible states rather than a sample transistion to the next state. We saw this in the backup diagrams.


In an implementation, one can use two arrays - one for $v_k$ and one for $v_{k+1}$ - or one array that is updated in place. Using the second approach, sometimes new value estimates are used in the bracketed term of the bellman equation. 

A *sweep* is going through the state space and updating corresponding $v_\pi(s)$

In the inplace approach, the order of which states are updated during the sweep have a significant influence on convergence rate, but in general, this converges faster than the two-array approach (intutively, because you get new data faster).

![](https://i.imgur.com/rFPHTev.png)

In practice, the algorithm terminates when the max element of successive updates is less than some $\theta$. When updates result in no changes in value, convergence has occured such that it satisfies the Bellman equations.

It doesn't matter what $v_\pi$ is initialized to because in the update it doesn't take into account what the previous value was. Values will eventually propagate backwards from the states whose values have converged.

**Notice how the updated estimate of values of states relies on the estimates of the values of successor states. This is bootstrapping and is a concept used throughout DP methods.**

#### Is it possible to do a multistep lookahead?
Yes, this will come back later when we are not given a model.

**Exercise 4.3**
What are the update equations analogous for q-values?
$$q_\pi(s,a) = \sum\limits_{s',r} p(s',r|s,a) \space [r+\sum\limits_{a'} \pi(a'|s') \gamma q_\pi(s',a')] $$
So the update is 
$$q_{k+1}(s,a) = \sum\limits_{s',r} p(s',r|s,a) \space [r+\sum\limits_{a'} \pi(a'|s') \gamma q_k(s',a')] $$

## Policy Improvement

Now, we know what $v_\pi$ is for an arbitrary policy $\pi$. What happens if for some state $s$, we want to know whether to change the policy to choose an action $a \neq \pi(s)$ such that we can obtain higher returns? Is this a better policy $\pi'$ ?

Consider selecting $a$ in $s$ and thereafter following $\pi$. This is exactly
$$q_\pi(s,a) = \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1})|S_t =s,A_t=a]=\sum_{s',r} p(s',r|s,a)[r+\gamma v_\pi(s')]$$

If this is greater than $v_\pi$ ,that is, to deviate from the policy at state $s$ and then follow $\pi$, *rather* than to follow $\pi$ all the time, then one should take $a$ everytime $s$ is encountered from now on. Then $\pi'$ is at least as good of a policy.

The above is true is due to the Policy Improvement Theorem

### Policy Improvement Theorem

Let $\pi$ and $\pi'$ be deterministic policies be as we described earlier, such that
\begin{equation}
\begin{split}
& q_\pi(s',\pi'(s')) = v_\pi(s') \space \forall s' \neq s & \space \textbf{(1)} \\
& q_\pi(s,\pi'(s)) \geq v_\pi(s) & \space \textbf{(2)} \\
\end{split}
\end{equation}



This implies,

$$ v_{\pi'}(s) \geq v_\pi(s) \space \forall s \in \mathcal{S}$$

We see $\pi'$ must be at least as good as $\pi$.

Proof:

\begin{equation}
\begin{split}
v_\pi(s) & \leq q_\pi(s,\pi'(s)) \\
& = \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1})|S_t =s,A_t=\pi'(s)]\\
& = \mathbb{E}_{\pi'}[R_{t+1} + \gamma v_\pi(S_{t+1})|S_t =s]\\
& \leq \mathbb{E}_{\pi'}[R_{t+1} + \gamma q_\pi(S_{t+1},\pi'(S_{t+1}))|S_t =s] & \textbf{by (2) }\\
& = \mathbb{E}_{\pi'}[R_{t+1} + \gamma \mathbb{E}[R_{t+2} +\gamma v_{\pi}(S_{t+2})|S_{t+1},A_{t+1}=\pi'(S_{t+1})] \space | \space S_t =s] \\
& = \mathbb{E}_{\pi'}[R_{t+1} + \gamma R_{t+1} + \gamma^2 v_{\pi}(S_{t+2})|S_t = s] \\
& ... \leq \mathbb{E}_{\pi'}[R_{t+1} + \gamma R_{t+1} + \gamma^2 R_{t+1} + \gamma^3v_{\pi}(S_{t+3})|S_t = s] \\
&.\\
&.\\
& \leq \mathbb{E}_{\pi'}[R_{t+1} + \gamma R_{t+1} + \gamma^2 R_{t+1} + \gamma^3 R_{t+3} + \cdots|S_t = s]\\
& = v_{\pi'}(s)\\
\end{split}
\end{equation}

The first three dots indicate the substitution of $q_\pi(S_{t+2},\pi'(S_{t+2})|S_{t+1})$ and applying (2)

Using this theorem, then knowing if choosing an action, in state $s$, different than that of the current policy is a new policy at least as good as the current one, we should consider changing policy at *all* states.

The action at each state should be selected greedily according to $q_{\pi}(s,a)$.
Specifically,

\begin{equation}
\begin{split}
\pi'(s) & = \arg \max_{a} q_\pi (s,a)\\
& = \arg \max_{a} \mathbb{E}[R_{t+1} + \gamma v_\pi(S_{t+1})|S_t =s,A_t=a]\\
& = \arg \max_{a} \sum_{s',r} p(s',r|s,a) [r+ \gamma v_\pi(s')]
\end{split}
\end{equation}

If we select in this manner,then $q_\pi(s,\pi'(s)) \geq v_\pi(s)$ is by construction because we lookahead and choose the best action according to $v_\pi$. So we meet the prerequisite of the policy improvement theorem and our new policy is at least as good as our old policy.

The process of making a new, better policy $\pi'$ by being greedy with respect to the value function of the old policy $\pi$ is **policy improvement**.

If the greedy policy $\pi'$ is as good as $\pi$ but not better, then $v_{\pi'}=v_\pi$ and

$$v_{\pi'} = \max_a \sum_{s',r} p(s',r|s,a)[r+\gamma v_{\pi'}(s')]$$

by the fact $\pi'$ is greedy

But this is just the Bellman optimality equation, and so, $v_{\pi'} = v_*$ and $\pi'=\pi$ is optimal. PI gives an improved policy except when the original policy is already optimal.



## Policy Iteration
Policy Iteration is applying alternate procedures of *Policy Evaluation (E)* and *Policy Improvement (I)* being repeated to achieve montonically improving policies and value functions.
![](https://i.imgur.com/iuu89TF.png)

![](https://i.imgur.com/ed0j1fk.png)
Notice the **bootstrapping** in PE and PI.
Notice the PE step does not have a summation over $\pi$ because $\pi$ is deterministic

This has a subtle bug in the termination in that it will be infinite if the policy keeps switching between optimal policies.
Any of the below fix this:
1. Make the $\arg\max$  deterministic (i.e. always selecting the action with the smallest index).
2. Terminate whenever the same policy appears twice during the policy iteration (when the policy obtained at the end of step 3 is the same as one obtained in some previous iteration of step 3 ).

Ex: In a deterministic grid-world, where the reward is -1 on all transitions and the terminal state is shaded.
![](https://i.imgur.com/9hIIURT.png)

So in one iteration of policy iteration, we have an optimal policy. This is not normally the case.

## Value Iteration
Notice how in policy iteration that each iteration involves policy evaluation, which itself is a iterative computation requiring multiple state space sweeps.

It turns out we can truncate the policy evaluation step to just **one** sweep *as well as* combine the policy improvement step without losing the convergence gurantee of policy iteration. 

Following the update rule below is known as **value iteration**,
\begin{equation}
\begin{split}
v_{k+1}(s) & = \max_a \mathbb{E}[R_{t+1} + \gamma v_k(S_{t+1})|S_t =s, A_t = a]\\
& =\max_a  \sum_{s',r} p(s',r|s,a) [r+ \gamma v_k(s')]
\end{split}
\end{equation}

And, is simply just the Bellman optimality equation turned into an update. This only has one sweep of the state space per iteration. Technically, one can achieve faster convergence by doing k sweeps of policy evaluation instead of one, then policy improvement, at the cost of more computation.

![](https://i.imgur.com/jBMabV7.png)
Notice the **bootstrapping** in the value iteration update

The stopping condition is just when the value function changes less than some $\theta$.

Notice how we don't explictly update the policy in this algorithm, we just maintain a values functions $v_0,\cdots,v_k$ and have an implicit $\pi$ doing a one step lookahead.

E.g
![](https://i.imgur.com/pmybcJO.jpg)

Here, we see that VI looks like a lot more iterations, but PI has hid the computation of Policy Evaluation.

The difference between policy iteration and value iteration is best described by how much you improve your estimate of the value function before trying to improve the policy.

On that note, what if one doesn't sweep through the entire state space and update the values for every state, instead, what if the updates were *focused* on the state values of frequently visited/valuable states? 

This is Asynchronous Dynamic Programming.
## Asynchronous DP

These are in-place iterative DP algorithms that update the state values in any order, using whatever other state values are available. As a result, we avoid the sweep of the entire state space every iteration, which can be *very* expensive, like in backgammon with $10^{20}$ states ( at a rate of 1e6 states/s, this takes 1000 years for a sweep).

Some ordering strategies could include:
1. At iteration $k$, randomly select a state to update such that every state has some probability of being selected.
3. At iteration$k$ , update the value of state $s_{k\%N}$  (so we iteratively update all states), where  is the number of states.
3. At iteration $k$, starting from a random initial state (where every state has some probability of being selected as this start state), the agent follows a stochastic policy $\pi(a|s)$ for one episode, and then update only the states visited in that episode. Repeat at each iteration.
4. Prioritized Sweeping - Use the magnitude of Bellman error to guide state selection.
    - Backup the state with the largest $$\left|\max_a \left(\sum_{s',r}p(s',r|s,a)[r + \gamma v(s')\right) - v(s)\right|$$
    - Update Bellman error of affected states (reverse dynamics) after each backup. 
    - Store these <state,error> pairs efficiently in a Priority Queue

These are guaranteed asymptotic convergence to $v_*$ if all states $\{s_k\}$ occur an infinite number of times. However, to converge correctly, we must continue to update the values of all the states (cannot just ignore some states after some point)

As a result of updating select states, the agent can improve its policy without getting locked into a hopeless sweep. We can take advantage of being able to order the updates such that the value information propagates more efficiently. Efficiency comes from the notion that some states do not need their values updated as often. 

This is even more useful when the model is unknown, and the agent has to learn through real-time interaction. That is, intermixing of computation and real-time interaction. The DP updates(computation) are applied based on experience(e.g states visited in its trajectory), and the value/policy estimates are used in decision-making(interaction). As a result, we *focus* the updates on the states relevant to the agent.

## Generalized Policy Iteration (GPI)

One can view these DP methods in the following way.
![](https://i.imgur.com/az7k05S.png)
1. Policy iteration consists of two simultaneous, interacting processes, one updating the current value function by evaluating the policy, and the other updating the policy greedily with respect to the current value function (PI).
2. Value iteration has only a single iteration of policy evaluation performed in between each policy improvement.
3. Async DP blurs this line further with only updating select states before improving the policy.

These are all a generalized idea of interaction between improvement and evaluation (GPI) with the policy always being improved with respect to the value function and the value function always being driven toward the value function for the policy

The diagram portrays this because each process (PI and PE) pull in opposite directions, because making the policy greedy w.r.t value function pushes the value function away from the evaluation of the old policy and making the value function consistent with the policy causes the policy not to be greedy. However, they interact overall to find a joint solution of $v_*$ and $\pi_*$. 

- The joint solution is the convergence of both of these quantities. This is when the improvements stop and Bellman optimality holds.
$$v_\pi = \max_{a} q_\pi(s,a) \space \forall s \in \mathcal{S}$$


GPI is a key RL idea. All value-based RL methods follow this framework. However, not all of RL *today* is characterized by this framework (see PG methods).



## Efficiency of DP

DP methods are polynomial in the number of states(k) and actions(n) to find an optimal policy even though the number of deterministic policies is $k^n$.

This is exponentially faster than direct search and better than LP solutions which falter at smaller number of states than DP. So in the cases where DP is feasible, it handles the large state space better than competing methods.

However, when the state and action space become very large, nearing continuous, it's unlikely the DP method will be feasible. In practice, DP methods can solve MDPs with millions of states. **With larger state spaces, async DP methods are preferred because relatively few states occur along optimal solution trajectories**. This makes the problem tractable using DP.

## A remark on the update equations

1. Just as there are four primary value functions $(v_\pi, v_*, q_\pi,$ and $q_*)$, there are four corresponding Bellman equations and four corresponding expected updates.
2. Bootstrapping will come back in later methods in the full RL case (without a model).









