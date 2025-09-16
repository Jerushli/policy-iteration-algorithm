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
#### Name : JERUSHLIN JOSE JB
#### Register Number : 212222240039
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
#### Name : JERUSHLIN JOSE JB
#### Register Number : 212222240039
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
<img width="743" height="176" alt="image" src="https://github.com/user-attachments/assets/488c19bc-c43c-462d-8a5e-5b275c46f3c4" />
<img width="770" height="45" alt="image" src="https://github.com/user-attachments/assets/52c48577-3152-4223-9897-e3f839b33934" />
<img width="691" height="178" alt="image" src="https://github.com/user-attachments/assets/7222c4ed-d81b-40ed-a654-f4f4e1674b58" />




### 2. Policy, Value function and success rate for the Improved Policy
<img width="804" height="170" alt="image" src="https://github.com/user-attachments/assets/b906eb8e-cfa6-4312-ae6c-910da1a6f735" />
<img width="769" height="54" alt="image" src="https://github.com/user-attachments/assets/98eff415-eb1d-43ee-83a6-b1ee368cb618" />
<img width="696" height="175" alt="image" src="https://github.com/user-attachments/assets/5efecba1-7ff8-4cee-ace5-7a0beffc0071" />




### 3. Policy, Value function and success rate after policy iteration
<img width="710" height="189" alt="image" src="https://github.com/user-attachments/assets/d4289d09-c95e-4632-b44c-e0dd931c3a64" />
<img width="808" height="54" alt="image" src="https://github.com/user-attachments/assets/069135e3-c7a4-408a-bb09-04897ed35082" />
<img width="827" height="184" alt="image" src="https://github.com/user-attachments/assets/77dfdaa0-b9d2-439d-acff-2e16728a6063" />


## RESULT:
Thus, The Python program to find the optimal policy for the given MDP using the policy iteration algorithm is successfully executed.
