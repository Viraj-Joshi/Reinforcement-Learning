---
tags: RL
---
# N-Step Bootstrapping

Simply put, this is the unification of TD(0) and MC. 

### N-Step Prediction
Consider estimating $v_\pi$

Let 1-step TD be TD(0). This is backing-up the value of sucessor state plus the immediate reward.

Then, 2-step TD is just backing-up the value of state two steps later plus the immediate reward of the first two rewards. And so on, until we have n-step TD.

![](https://i.imgur.com/H2XLKwW.png)

This is still TD because we change the value of an earlier estimate based on how it differs from the value of an estimate *n*-steps later.

The benefit of n-step learning is that it trades off the biased estimates of TD(0) and the high-variance estimates of MC to be a middle-ground.

---

Mathematically, the target of each update is

MC: $G_t = R_{t+1}+\gamma R_{t+2} + \gamma^2R_{t+3} \cdots + \gamma^{T-t-1} R_T$, which is the return

TD(0): $G_{t:t+1} = R_{t+1} + \gamma V_t(S_{t+1})$, where $V_t(S_{t+1})$  estimates the remaining sum of reward terms of the full return

Two-Step Return: $G_{t:t+1} = R_{t+1} + \gamma R_{t+2} +\gamma^2 V_{t+1}(S_{t+2})$

N-Step Return: $G_{t:t+n} = R_{t+1} + \gamma R_{t+2} +\gamma^{n-1}R_{t+n} + V_{t+n-1}(S_{t+n})$

If $t+n \geq T$, that is the n-step return extends beyond termination of the episode, then $G_{t:t+n} = G_t$

Notice that any algorithm cannot use the n-step return until it has seen $R_{t+n}$ and computed $V_{t+n-1}$, which is first available at timestep $t+n$. This implies the first n-1 steps of episode have no updates.
The update rule for $0 \leq t < T$:

\begin{equation}
V_{t+n}(S)=
\left\{\begin{array}{lr}
        V_{t+n-1}(S_t) + \alpha[G_{t:t+n} - V_{t+n-1}(S_t)] & \text{for } S=S_t\\
        V_{t+n-1}(S), & \text{for } s \neq S_t\\
        \end{array}\right\}
\end{equation}

The full prediction algorithm is:

![](https://i.imgur.com/E7m0A67.png)

$\tau$ is the time where the return $G_{t:t+n} starts from, so $S_\tau$ is the state which needs updating.

#### Why is $\tau=t-n+1$?

Assume our trajectories are stored as $[\tau_0,\tau_1,...]$ where $\tau_i= [(S_0,A_0,R_1),(S_1,A_1,R_2),(S_2,A_2,R_3),\cdots]$

We can calculate the first return $G_{0:n}$, at timestep $n-1$ when we receive $R_n$. So if the current timestep $t$ satisfies $t-(n-1) \geq 0$, the return can be calculated. From here on out, because t increases, $t \geq n-1$ and $G_{t:t+n}$ can be calculated.

The plus 1 in the starting index of $i$ serves to index properly into the reward. Examine at $\tau=0$, the first reward is $R_1$, so we start at $i=\tau+1$ in the summation. Then, to not discount the first reward $R_1$,we subtract that 1 from $\tau$ in the gamma term.

```
def on_policy_n_step_td(
    env_spec:EnvSpec,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    n:int,
    alpha:float,
    initV:np.array
) -> Tuple[np.array]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        n: how many steps?
        alpha: learning rate
        initV: initial V values; np array shape of [nS]
    ret:
        V: $v_pi$ function; numpy array shape of [nS]
    """

    nS,nA,gamma = env_spec.nS,env_spec.nA, env_spec.gamma
    V = initV

    for traj in trajs:
        R = []
        for t in range(len(traj)):
            s,a,r,s_prime = traj[t]
            R.append(r)
            tau = t-n+1
            if tau >= 0:
                G = sum([(gamma**(j-tau))*R[j] for j in range(tau,min(tau+n,len(traj)))])
                if tau + n < len(traj):
                    G = G + gamma**n * V[traj[tau+n][0]]
                V[traj[tau][0]] += alpha * (G - V[traj[tau][0]])
    return V
```
See how the code removes the +1 in $i$ and -1 in $\gamma$ the summation of $G$ because the rewards are 0-indexed instead of 1-indexed. (e.g $R_1$ is the first element of the array, so $\tau=0$ is actually the correct way to index into it.)


---
### Tradeoffs
Given the following MRP:
![](https://i.imgur.com/epcKxIa.png)

A way to view bias-variance tradeoffs of this method is to examine the value of state A under each learning method.

In order for state A to be influenced
- under MC learning: must end at +1 goal state *and* visit A
- under TD(0): must visit +1 goal state at least once anytime and then wait for the value to propagate back to A through bootstrapping. So for very long chains, you don't have to wait for the very low probability event of visiting A and making it all the way over to the right.
- under N-step: the propagation speed of A being updated speeds up as n states get updated towards 1 in a single trajectory.

Examine:
![](https://i.imgur.com/7h75ELZ.png)

## N-Step SARSA

For control, we adapt N-step prediction to Q values **for on-policy learning**. The backup diagram would start and end with actions.
![](https://i.imgur.com/MRpXYIZ.png)



The n-step return(target) is 

$$G_{t:t+n} = R_{t+1} + \gamma R_{t+2} +\gamma^{n-1}R_{t+n} + \gamma^n Q_{t+n-1}(S_{t+n},A_{t+n})$$ with $G_{t:t+n} = G_t$ if $t+n \geq T$ as before

Then, the update rule is
\begin{equation}
Q_{t+n}(S,A)=
\left\{\begin{array}{lr}
        Q_{t+n-1}(S_t,A_t) + \alpha[G_{t:t+n} - Q_{t+n-1}(S_t,A_t)] & \text{for } S=S_t,A=A_t\\
        Q_{t+n-1}(S,A), & \text{for } s \neq S_t \text{ or } a \neq A_t\\
        \end{array}\right\}
\end{equation}

The control algorithm is familar:
![](https://i.imgur.com/RRmk866.png)



Expected SARSA is very similar except the bootstrap estimate is the expectation of the values of the actions from the last state under the target policy $\pi$.

If $t+n < T$ as before
\begin{equation}
\begin{split}
G_{t:t+n} & = R_{t+1} + \gamma R_{t+2} +\gamma^{n-1}R_{t+n} + \gamma^n \sum_a \pi(a|s) Q_{t+n-1}(s,a)  \\
& = R_{t+1} + \gamma R_{t+2} +\gamma^{n-1}R_{t+n} + \gamma^n \overline{V}_{t+n-1}(S_{t+n-1}) & \mathbf{(1)}
\end{split}
\end{equation}

Otherwise, $G_{t:t+n} = G_t$ for $t+n \geq T$ and if $S_{t+n-1}$ is terminal, $\overline{V}_{t+n-1}(S_{t+n-1}) = 0$

## N-Step Off-Policy

Let $\pi$ be the greedy policy while $b$ is the behavior policy, often exploratory($\epsilon$-greedy)

To use data from $b$ to learn $\pi$, we weight the n-step returns $\rho$,the importance sampling ratio or the relative probability under the two policies of taking the n actions from $A_t$ to $A_{t+n-1}$, like MC off-policy methods.

\begin{equation}
\rho_{t:h} = \prod_{k=t}^{\min(h,T-1)} \frac{\pi(A_k|S_k)}{b(A_k|S_k)}
\end{equation}

Notice if $\pi=b$, then the following method is on-policy and generalizes n-step SARSA.

Then, the update for off-policy n-step TD (prediction problem) for the state at time,$S_t$, is

$$V_{t+n}(S_t) = V_{t+n-1}(S_t) + \alpha \rho_{t:t+n-1}[G_{t:t+n} - V_{t+n-1}(S_t)]$$

The weight $\rho$ ends at $t+n-1$ because the last state in the return is at $t+n$, so the last action that led to the last state is $t+n-1$. Similarly, the weight $\rho$ starts at $t$ because the first state in the return is at $t$, so the action following is at $t$. Look at the backup diagram.


---
The off-policy n-step SARSA update for Q is
\begin{equation}
Q_{t+n}(S,A)=
\left\{\begin{array}{lr}
        Q_{t+n-1}(S_t,A_t) + \alpha \rho_{t+1:t+n}[G_{t:t+n} - Q_{t+n-1}(S_t,A_t)] & \text{for } S=S_t,A=A_t\\
        Q_{t+n-1}(S), & \text{for } s \neq S_t \text{ or } a \neq A_t\\
        \end{array}\right\}
\end{equation}

The $\rho$ is different than that of the update for V (**Look at the backup diagram for Q-values!**)
- starts at $t+1$ because the first action is prescribed by the Q function, so we only weight the actions that are chosen by the behavior policy. 
- ends at $t+n$ because the last action in the return is at $t+n$(because the last reward is at $t+n$, followed by the last action $t+n$). 

![](https://i.imgur.com/q51rL2y.png)

#### N-Step Expected SARSA off-policy

The algorithm is the same as off-policy n-step SARSA, but the return $G_{t:h}$ is weighted by a $\rho$ that ends one step earlier because the last term of the return is the expectation over all action-values and uses the expected SARSA return equation $\mathbf {(1)}$

\begin{equation}
Q_{t+n}(S,A)=
\left\{\begin{array}{lr}
        Q_{t+n-1}(S_t,A_t) + \alpha \rho_{t+1:t+n-1}[G_{t:t+n} - Q_{t+n-1}(S_t,A_t)] & \text{for } S=S_t,A=A_t\\
        Q_{t+n-1}(S,A), & \text{for } S \neq S_t \text{ or } A \neq A_t\\
        \end{array}\right\}
\end{equation}

N-Step expected SARSA generalizes n-step Q-learning because the expectation of the policy $\pi$ at the end can be the greedy policy.

## Control Variate

## N-Step Tree Backup Algorithm

It turns out it possible to do off-policy learning without importance sampling. 

The N-step Tree Backup generalizes N-Step Expected SARSA in a way. The importance weights in N-Step Expected SARSA ends one step earlier because we take an expectation of the action-values under a policy $\pi$. Tree backups generalize this to the n-step case.

Examine the following backup digram.
![](https://i.imgur.com/f2xFzBH.png)
At each state node, the actions dangling off to the side are not selected while the spine off the tree is the trajectory taken. 

In a typical backup diagram, the node at the top diagram is updated towards a target consisting of the rewards along the tree plus the estimated value of nodes at the bottom. In the tree-backup, the target is exactly that *plus* the estimated values (**bootstrapping**) of the dangling leaf action-nodes at all levels.

Lets build up an recursive equation for the target.

The one-step return for $t < T-1$ is exactly Expected SARSA.
$$G_{t:t+1} = R_{t+1} + \gamma \sum\limits_a \pi(a|S_{t+1})Q_t(S_{t+1},a)$$

The two-step return for $t<T-2$ is
$$ G_{t:t+2} = R_{t+1} + \gamma\sum\limits_{a \neq A_{t+1}} \pi(a|S_{t+1}) Q_{t+1}(S_{t+1},a)) + \\ \gamma \pi(A_{t+1},S_{t+1})\left(R_{t+2} + \gamma \sum\limits_a \pi(a|S_{t+2})Q_{t+1}(S_{t+2},a)\right) 
\\ = R_{t+1} + \gamma\sum\limits_{a \neq A_{t+1}} \pi(a|S_{t+1}) Q_{t+1}(S_{t+1},a)) +\pi(A_{t+1},S_{t+1}) G_{t+1:t+2}$$

Notice how the bootstrapped values of the actions not taken are weighted by the probability of that action occuring under the policy and the probability of the action actually taken in the current state weights all the values further down the tree.

We can easily see the general recursive formulation is
$$G_{t:t+n} = R_{t+1} + \gamma\sum\limits_{a \neq A_{t+1}} \pi(a|S_{t+1}) Q_{t+n-1}(S_{t+1},a)) +\pi(A_{t+1},S_{t+1}) G_{t+1:t+n}$$

One implementation detail is that $G_{T-1:t+n}=R_T$.

Using this return, we use the usual update rule

\begin{equation}
Q_{t+n}(S,A)=
\left\{\begin{array}{lr}
        Q_{t+n-1}(S_t,A_t) + \alpha [G_{t:t+n} - Q_{t+n-1}(S_t,A_t)] & \text{for } S=S_t,A=A_t\\
        Q_{t+n-1}(S,A), & \text{for } S \neq S_t \text{ or } A \neq A_t\\
        \end{array}\right\}
\end{equation}

---
![](https://i.imgur.com/uW3rGbL.png)

## $n$-step $Q(\sigma)$

Just as we unified TD(0) and MC using n-step algorithms. There is a unification of the three control n-step algorithms - n-step SARSA,n-step expected SARSA , and tree backup.

Consider the first three backup diagrams below with $n=4$ :
![](https://i.imgur.com/j4Jagy2.png)

n-step SARSA uses only sample transitions; n-step expected SARSA has all sample transitions except for the last state-to-action one which is fully branched to calculate expected value; Tree-backup has all transitions fully branched

The last diagram decides on each transition to another state  whether to take the sample reward from the action taken from that state or consider the expectation over all actions. If one always sampled, this is just n-step SARSA; if one never sampled, this is just Tree-backup; if one always sampled except for the last state, this is n-step Expected SARSA.

We propose $n$-step $\mathbf Q(\sigma)$ where $\sigma_t \in [0,1]$ denotes the degree of sampling on step $t$.
\begin{equation}
\sigma_t=
\left\{\begin{array}{lr}
         1& \text{for } \text{full sampling}\\
         0& \text{for pure expectation}\\
         \text{otherwise} & \text{continuous blend}
        \end{array}\right\}
\end{equation}

Let us develop the recursive formulation for the target of $n$-step $\mathbf Q(\sigma) update$

Beginning with the n-step return for tree-backup
\begin{equation}
\begin{split}
G_{t:h} & = R_{t+1} + \gamma\sum\limits_{a \neq A_{t+1}} \pi(a|S_{t+1}) Q_{h-1}(S_{t+1},a)) +\pi(A_{t+1},S_{t+1}) G_{t+1:h}\\
& = R_{t+1} + \gamma \overline{V}_{h-1}(S_{t+1}) -\gamma \pi(A_{t+1}|S_{t+1})Q_{h-1}(S_{t+1},A_{t+1})  +\pi(A_{t+1},S_{t+1}) G_{t+1:h} \\
& = R_{t+1} - \gamma \pi(A_{t+1}|S_{t+1})\left[G_{t+1:h}-Q_{h-1}(S_{t+1},A_{t+1})\right] + \gamma \overline{V}_{h-1}(S_{t+1}) & (\mathbf *)
\end{split}
\end{equation}

Now, the term outside the brackets can be replaced by how we slide linearly between full sampling and pure expectation.

$$G_{t:h}= R_{t+1} - \gamma \left(\sigma_{t+1}\rho_{t+1} +(1-\sigma_{t+1})\pi(A_{t+1}|S_{t+1})\right)\left[G_{t+1:h}-Q_{h-1}(S_{t+1},A_{t+1})\right] + \gamma \overline{V}_{h-1}(S_{t+1})$$

Note if $\sigma_t = 0$, we have the tree-backup n-step return for that step (eq.$\mathbf *$), while if we have $\sigma_t = 1$, we have the off-policy n-step SARSA(full sampling) return.

---

![](https://i.imgur.com/tE9eneM.png)













