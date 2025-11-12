# POLICY ITERATION ALGORITHM

## AIM
To develop a Python program to find the optimal policy for the given MDP using the policy iteration algorithm.

## PROBLEM STATEMENT
The aim of this experiment is to find optimal policy for the mdp using policy iteration. Policy iteration includes policy evaluation and policy improvement where evaluation function is used to find optimal value function of each state and then improvement function is used to find best policy by comparing all the action value function as well as policy.

## POLICY ITERATION ALGORITHM
-> Step1 :
We are going to do policy evaluation of each state to get the state value function where the initial policy is defined randomly to the mdp.

-> Step2:
Once we obtain convergence in the policy evaluation then implement policy improvement where we are going to find best optimal policy until the previous and current policy are same.
</br>
</br>

## POLICY IMPROVEMENT FUNCTION
#### Name : ROHIT G
#### Register Number : 212222240083
```python
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    # Write your code here to improve the given policy
    for s in range(len(P)):
      for a in range(len(P[s])):
        for prob,next_state,reward,done in P[s][a]:
          Q[s][a]+=prob*(reward+gamma*V[next_state]*(not done))
          new_pi=lambda s:{s:a for s, a in enumerate(np.argmax(Q,axis=1))}[s]
    return new_pi
```
## POLICY ITERATION FUNCTION
#### Name : ROHIT G
#### Register Number : 212222240083
```python
def policy_iteration(P, gamma=1.0, theta=1e-10):
   random_actions=np.random.choice(tuple(P[0].keys()),len(P))
   pi = lambda s: {s:a for s, a in enumerate(random_actions)}[s]
   while True:
    old_pi = {s:pi(s) for s in range(len(P))}
    V = policy_evaluation(pi, P,gamma,theta)
    pi = policy_improvement(V,P,gamma)
    if old_pi == {s:pi(s) for s in range(len(P))}:
      break
   return V, pi
```

## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy
<img width="472" height="152" alt="image" src="https://github.com/user-attachments/assets/7afb763e-5ecd-4163-a9dc-5681e178ebd0" />
<img width="622" height="20" alt="image" src="https://github.com/user-attachments/assets/afa31a47-8635-4aca-84c8-f02ef63c1757" />
<img width="467" height="149" alt="image" src="https://github.com/user-attachments/assets/c65cdf60-5b89-4622-ac84-bed8796bd5bd" />








### 2. Policy, Value function and success rate for the Improved Policy
<img width="466" height="141" alt="image" src="https://github.com/user-attachments/assets/7a0e2f93-c955-4b42-97a3-d06545916e21" />
<img width="610" height="25" alt="image" src="https://github.com/user-attachments/assets/9033a762-9f17-4bc8-8839-e8a4e72508eb" />
<img width="482" height="144" alt="image" src="https://github.com/user-attachments/assets/57741c43-c804-4107-a37a-59e6a97858bc" />







### 3. Policy, Value function and success rate after policy iteration
<img width="444" height="166" alt="image" src="https://github.com/user-attachments/assets/991d54db-ec2c-46d9-b002-d986189b8f92" />
<img width="617" height="28" alt="image" src="https://github.com/user-attachments/assets/b44eb74c-49ad-4501-ac16-4187404b4d42" />
<img width="811" height="116" alt="image" src="https://github.com/user-attachments/assets/69526d7f-b676-447a-acef-ee36e669be60" />




## RESULT:
Thus, The Python program to find the optimal policy for the given MDP using the policy iteration algorithm is successfully executed.
