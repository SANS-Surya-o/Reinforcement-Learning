'''
Monte carlo Control with Exploring Starts for Selling an Asset
'''



import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt

# Problem setup
N = 100                # Max offer value
alpha = 0.9            # Discount factor
C = 10                 # Maintenance cost
P = np.ones(N + 1)     # Uniform probability distribution
P /= P.sum()

states = list(range(N + 1))
actions = [0, 1]       # 0 = reject, 1 = accept


Q = defaultdict(lambda: np.zeros(len(actions)))
Returns = defaultdict(int)
policy = {s : 1 for s in states}


def generate_episode(policy, P):
    episode = []
    state = np.random.choice(states)
    action = np.random.choice(actions)
    first = True
    while True:
        if first:
            first = False
        else:
            action = policy[state]
        
        if action == 1:  # Accept
            reward = state
            episode.append((state, action, reward))
            break
        else:  # Reject
            reward = -C
            episode.append((state, action, reward))
            state = np.random.choice(states, p=P)
    return episode


def first_visit_mc_control(num_episodes=100000):
    for ep in range(num_episodes):
        episode = generate_episode(policy, P)

        # Record first-visit time for each (s,a)
        first_visit_time = {}
        visited_states = set()
        for t, (s, a, _) in enumerate(episode):
            if (s, a) not in first_visit_time:
                first_visit_time[(s, a)] = t
            visited_states.add(s)

        # Traverse backward to compute G
        G = 0
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = alpha * G + r
            if first_visit_time.get((s, a)) == t:
                Returns[(s, a)] += 1
                Q[s][a] = Q[s][a] + (G - Q[s][a]) / Returns[(s, a)]


        # Improve policy
        for s in visited_states:
            policy[s] = np.argmax(Q[s])

    return Q, policy


# Train
Q, learned_policy = first_visit_mc_control()
for s in states:
    print(f"State {s}: values = {Q[s][learned_policy[s]]}, Policy = {learned_policy[s]}")

# Convert policy to a more readable format

# Determine learned threshold
mc_threshold = None
for s in states:
    if policy[s] == 1:
        mc_threshold = s
        break

print(f"Monte Carlo learned threshold i* ≈ {mc_threshold}")


plt.figure(figsize=(10, 5))
actions_plot = [policy[s] for s in states]
plt.step(states, actions_plot, where='mid')
plt.xlabel('Offer Value (State)')
plt.ylabel('Action (0 = Reject, 1 = Accept)')
plt.title('Monte Carlo Learned Policy')
plt.grid(True)
plt.savefig("exploring_starts.png", dpi=300)
