# Eligibility Traces

As we have seen before, we start with the prediction problem of estimating $v_\pi(s)$ as $\hat{v}(s,\mathbf w)$
## $\lambda$-Return
We defined the $n$-step return before as 
$$G_{t+n} = R_t + \gamma R_{t+1} + \cdots + \gamma^{n-1}R_{t+n} + \gamma^n \hat v(S_{t+n},\mathbf w_{t+n-1})$$

We used this return as a target. Now, we introduce the concept of averaging many different $n$-step returns. In fact, one can even average an infinite number of n-step returns so long as the weight placed on each return sums to 1.

If one averages the one-step return and the infinite-step return, this is another way of interrelating TD and MC methods.

![](https://i.imgur.com/aYAbosk.png)

So far, we have seen $TD(\lambda)$ where $TD(0)$ updates toward the one-step return as a target and $TD(1)$ updates toward the full return as a target. Hence, TD(0) is the one-step TD method and TD(1) is a MC algorithm.

However, one can update towards a target in between the two.

Define the $\lambda$-return as 
$$G_t^{\lambda} = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_{t:t+n}$$

![](https://i.imgur.com/bBPGIL9.png)

In the above, we see in general that $TD(\lambda)$ weights each of the n-step returns as follows: the one-step return is given the largest weight $(1-\lambda)$, then the two-step return $(1-\lambda)\lambda$, and so on. All $n$-step returns past the terminal state are just the full return $G_t$ and given weight $\lambda^{T-t-1}$. We seperate the final return from the $\lambda$-return

$$G_t^{\lambda} = (1-\lambda) \sum_{n=1}^{T-t-1} \lambda^{n-1} G_{t:t+n} + \lambda^{T-t-1}G_t$$

**Exercise**
Exercise 12.1 Just as the return can be written recursively in terms of the first reward and itself one-step later, so can the $\lambda$-return. Derive the analogous recursive relationship.

Using the $\lambda$-return, we define the offline $\lambda$-return algorithm. It makes no changes to the weight vector during the episode and at the episode, makes a sequence of offline updates using the semi-gradient rule.

$$\mathbf w_{t+1} = \mathbf w + \alpha(G_t^\lambda - \hat v(S_t,\mathbf w_t)) \nabla \hat v(S_t,\mathbf w), \space t=0,\dots,T-1$$

### Forward View
![](https://i.imgur.com/fB0DTmU.png)

The offline return looks at the next steps to determine the future reward. **Note that this return can only be calculated in episodic cases.**

## TD($\lambda$)

TD($\lambda$) is a fully incremental, online method, updating the weight vector every transition of the episode and can be applied in the continuing case. **TD($\lambda$) approximates the offline $G_t^{\lambda}$ return algorithm and requires less memory than the forward-view.**

The eligibility trace $z_t$ is a short-term memory vector decaying over time, of same dimension as the long-term memory weight vector.

$z_t$ accumulates $\nabla \hat{v}(s,\mathbf w_t)$ and fades away by $\gamma\lambda$.  Specifically, $z_{t} = \gamma\lambda z_{t-1} + \nabla \hat{v}(s,\mathbf w)$ where $\lambda$ is the trace-decay parameter. In the tabular case, this is $z_{t} = \gamma\lambda E_{t-1} + E_t(s)$ where $E_t(s)$ is the one-hot vector $\mathbb{1}_{\{S_t = s\}} \text{ for each } s$.

Intuitively, the eligibility trace indicates the eligibility of each component of the weight vector for undergoing change should a 'reinforcing event' occur. **I.e how do we assign credit to the states we have seen for the error we just saw?** The reinforcing events are the one-step TD errors  $\delta_t$. Then, $\mathbf w_t$ is updated in the semi-gradient fashion as usual: 

$$\mathbf w_{t+1} = \mathbf w_t - \alpha\delta_tz_t$$

![](https://i.imgur.com/cLgza0M.png)

### Backward View
TD($\lambda$) updates past states based on the one-step TD error. The figure below illustrates this concept.

![](https://i.imgur.com/p1Ag0gZ.png)

It is simpler to understand first in the tabular case: Examine $V_{t+1}(s) \leftarrow V_t(s) + \alpha\delta_tE_t(s)$. $E_t(s)$ is 'assigning credit' to the states most reponsible for the error just encountered. For example, the value function will nudged very slightly toward a TD error that has low eligibility trace because this means the state has received little credit attributed to the error.

Now, in the case of function approximation:
At each point in time $t$, we change the weights corresponding to all prior states by the TD error, weighted by how much these prior states contributed to the eligibility trace.

We can see if $\lambda=0$, then the trace at $t$ is exactly gradient of the value estimate at that point. Then, the TD(0) update is the one-step TD semi-gradient TD update. 

For $\lambda < 1$, the gradient estimates of temporally distant states are contributing much less to the eligibility trace, and so the components of the weight corresponding to these states change less. They are given *less credit* for the TD error.

#### TD(1)
If $\lambda=1$, then credit given to earlier states falls by $\gamma$ per step. In passing back the undiscounted reward $R_{t+1}$, in the TD error, $k$ steps, it needs to be discounted, like any reward in a return, by $\gamma^k$, which is what the falling eligibility trace achieves. This is also just Monte Carlo method. Furthermore, if $\gamma=1$, this is just like undiscounted, episodic Monte Carlo method.

However, TD(1) is more general than earlier Monte Carlo methods, which were 1. limited by episodic tasks and 2. can only learn after the episode was over.

TD(1) can be applied to discounted continuing tasks. TD(1) is online and learns in an n-step TD way from the incomplete ongoing episode, *where the n steps are all the way up to the current step.

---
Examine the following experiment.

![](https://i.imgur.com/tVuGilm.png)

If $\alpha$ is chosen too large, TD($\lambda$) can be much more unstable than the off-line return algorithm but otherwise is approximating the offline algorithm.

---
## N-Step Truncated TD $\lambda$-Return (TTD) Method

In the episodic case,The $\lambda$-return is not known until the end of the episode. In the continuing case, the $\lambda$-return is never known because it depends on some $n$-step return for some arbitrarily large $n$ to complete. However, because the reward becomes decayed exponentially for long delayed rewards, we can approximate the $\lambda$-return by truncating the sequence.

Given some horizon $h$, the truncated $\lambda$-return at time $t$ is

$$G_{t:h}^\lambda = (1-\lambda) \sum_{n=1}^{h-t-1} \lambda^{n-1} G_{t:t+n} + \lambda^{h-t-1}G_{t:h}$$

Whereas n-step return methods only used the n-step return every nth step to update its value function, here we sum over all of the geometrically weighted k-step returns ($1\leq k \leq n$).

In other words, examine the compound backup diagram below
![](https://i.imgur.com/bOTBdJ6.png)
The only difference from the full $\lambda$ return is the longest compound update is at most $n$ steps rather than the length of the episode.

The update is $$\mathbf w_{t+n} = \mathbf w_{t+n-1} + \alpha[G_{t:t+n}^\lambda - \hat{v}(S_t,\mathbf w_{t+n-1})]\nabla\hat{v}(S_t,\mathbf w_{t+n-1}), \space 0 \leq t \leq T$$

## Redoing Updates: Online $\lambda$-return Method

TTD still requires the choice of the horizon h - short enough so updates can be made sooner and long enough to approximate the offline $\lambda$-return.

In exchange for more computation, we can remove this question.

At each timestep, one redoes all of the updates, since the beginning of the current episode, using the latest data from the slightly longer horizon. These redone updates will be better because they use more accurate targets, which results in better $\mathbf w$. I.e at each horizon, we form a sequence of weight vector updates. 

Examine ![](https://i.imgur.com/igzRVJ9.png)

Before taking an action at $t=2$, we have $R_2$, $S_1$, and $\mathbf w_1$. So we can form the better target $G_{0:2}^\lambda$ for $\hat{v}(s_0)$ and a better target $G_{1:2}^\lambda$ for $\hat{v}(s_1)$. This pattern of redoing updates continues as the horizon advances.

We denote the weight vectors computed at different horizons as $\mathbf w_t^h$. 

1. The first weight vector in each sequence $\mathbf w_0^h$ is inherited from the previous episode (so each sequence starts with the same initial weight vector). 
2. At the final horizon $h=T$, we obtain the final weights $\mathbf w_T^T$, which are passed on to form the initial weights of the next episode.

Each sequence of updates at horizon $h$ can be written as:
$$\mathbf w_{t+1} = \mathbf w_t - \alpha[G_{t:h}^\lambda - v(s_t,\mathbf w_t^h)]\nabla\hat{v}(s_t,\mathbf w_t^h),\space 0 \leq h \leq t\leq T$$

**Note this is a forward-view algorithm** because the targets are the n-step truncated returns. **This is known as the online $\lambda$-return algorithm**

It is strictly more complex than the offline $\lambda$-return algorithm. In return, we expect it to perform better 
1. during the episode because we benefit from the updates as we go along (unlike the offline algorithm which makes no updates until the end of the episode)
2. at the end of the episode because the weight vector has had more updates.

See below.

![](https://i.imgur.com/VBwdzma.png)



## True Online $TD(\lambda)$

Is there backward view of the previous algorithm? Yes. This is known as the ***True* Online $TD(\lambda)$** algorithm.

It is the 'true' online algorithm because it is truer to the Offline Lambda Return algorithm (remember the online $TTD(\lambda)$, $TD(\lambda)$, and the online $\lambda$-return algorithm only approximate).

However, it really is only the closest to the offline algorithm when we use a linear function approximator, if one is using a non-linear function approximator, $TD(\lambda)$ is better

Previously, the online $\lambda$-return algorithm is computing each row of weight vectors as horizon advances. In this algorithm, the desired weight vector $\mathbf w_T^T$ at each timestep is only built up from weight vectors on the diagonal $\mathbf w_t^t$, which themselves are the only vectors calculated.

![](https://i.imgur.com/cvd5b0r.png)
Skipping the derivation, let $\hat{v}(s,\mathbf w) := \mathbf w^T \mathbf x(s)$ and $\mathbf x_t = \mathbf x(s_t)$, it has been shown, the update rule that produces the same sequence of weight vectors $\mathbf w_t$ on the diagonal as the previous method, is as follows:
$$w_{t+1} = \mathbf w_t\delta_t\mathbf z_t + \alpha (\mathbf w^T_t \mathbf x_t - \mathbf w_{t-1}^T \mathbf x_t)(\mathbf z_t - \mathbf x_t)$$

The eligibility trace here is called a dutch trace.
$$\mathbf z_t = \gamma\lambda z_{t-1} + (1-\alpha\gamma\lambda \mathbf z_{t-1}^T \mathbf x_t)\mathbf x_t$$

This algorithm is much less expensive and simpler.

![](https://i.imgur.com/91YU9Bv.png)

## SARSA $(\lambda)$

We can extend eligibility traces to deal with action-value pairs with some slight modifications.

As seen before, the $n$-step return is 
$$G_{t:t+n} = R_{t+1} + \cdots + \gamma^{n-1}R_{t+n} + \gamma^n \hat{q}(s_{t+n},a_{t+n},\mathbf w_{t+n-1}), \space t+n < T$$

Similarly then, the offline $\lambda$-return algorithm **(forward view)** uses the following weight update
$$\mathbf w_{t+1} = \mathbf w_t - \alpha[G_{t}^\lambda - q(s_t,a_t,\mathbf w_{t})]\nabla\hat{v}(s_t,\mathbf w_t),\space 0 \leq t\leq T$$

---

SARSA $(\lambda)$ approximates this foward view algorithm. Notice we make subtle changes to accomodate action-value pairs.

$$\mathbf w_{t+1} = \mathbf w_t + \alpha\delta_tz_t$$

where $\delta_t$ is the action-value TD error $R_{t+1} + \hat{q}(s_{t+1},a_{t+1},\mathbf w_t) - \hat{q}(s_{t},a_{t},\mathbf w_t)$ and $z_t = \gamma\lambda z_{t-1} + \nabla \hat{q}(s_t,a_t,\mathbf w_t),\space 0 \leq t \leq T$

![](https://i.imgur.com/TCUlPAG.png)

```
def SarsaLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:StateActionFeatureVectorWithTile,
    num_episode:int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    w = np.zeros((X.feature_vector_len()))
    for _ in range(num_episode):
        s = env.reset()
        done = False
        a = epsilon_greedy_policy(s,done,w,epsilon=.01)
        x = X(s,done,a)
        z = np.zeros(x.shape)
        Q_old = 0
        while not done:
            s_prime,r,done,info = env.step(a)
            a_prime = epsilon_greedy_policy(s_prime,done,w,epsilon=.01)
            x_prime = X(s_prime,done,a_prime)
            Q = w.T@x
            Q_prime = w.T@x_prime
            delta = r + gamma*Q_prime - Q
            z = gamma*lam*z + (1-alpha*gamma*lam*z.T@x)*x
            w+= alpha*(delta+Q-Q_old)*z - alpha*(Q-Q_old)*x

            Q_old = Q_prime
            x = x_prime
            a = a_prime
            s = s_prime
    return w
```


### Example
![](https://i.imgur.com/vhWbT9g.png)
In this environment, the reward is 0 until you reach the goal state and $\gamma=1$. 10-step SARSA equally increments the last 10 action-values.  SARSA using eligibilty traces would increment all action-values up to the beginning of the episode depending on how temporally distant it is.

## Schedule $\gamma$, $\lambda$

We have only used constant $\gamma$ and $\lambda$. Now, we introduce dependencies on them such that: $\lambda: \mathcal{S} \times \mathcal{A} \rightarrow [0,1]$ and $\gamma: \mathcal{S} \rightarrow [0,1]$ are functions of states and actions.

## Off Policy Traces














