# from problem_models.selling_an_asset import SellingAssetProblem
from Problems.base_class import RLProblem
from Problems.selling_an_asset import SellingAssetProblem

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import defaultdict

from plot import animate_progress, plot_q_values


def epsilon_soft_policy(Q, state, actions, epsilon=0.1):
    policy_probs = np.ones(len(actions)) * epsilon / len(actions)
    best_action = np.argmax(Q[state])
    policy_probs[best_action] += 1.0 - epsilon
    return np.random.choice(actions, p=policy_probs)


def generate_episode(policy_fn, Q, actions, problem, epsilon):
    episode = []
    state = problem.reset()
    while True:
        action = policy_fn(Q, state, actions, epsilon)
        next_state, reward, done = problem.step(action)
        episode.append((state, action, reward))
        if done:
            break
        state = next_state
    return episode


def mc_control(problem: RLProblem, num_episodes=5000, gamma=0.9, epsilon=0.1):
    actions = problem.get_possible_actions(0)
    Q = defaultdict(lambda: np.zeros(len(actions)))
    Returns = defaultdict(list)
    policy_progress = []
    value_progress = []

    for ep in range(num_episodes):
        episode = generate_episode(epsilon_soft_policy, Q, actions, problem, epsilon)

        G = 0
        visited = set()
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r
            if (s, a) not in visited:
                visited.add((s, a))
                Returns[(s, a)].append(G)
                Q[s][a] = np.mean(Returns[(s, a)])

        # Track policy/value progress
        value_progress.append([max(Q[s]) for s in problem.get_all_states()])
        policy_progress.append([np.argmax(Q[s]) if s in Q else 0 for s in problem.get_all_states()])

    return Q, value_progress, policy_progress


if __name__ == "__main__":
    problem = SellingAssetProblem(N=100, alpha=0.9, C=10)
    Q, value_progress, policy_progress = mc_control(problem, num_episodes=100000, gamma=0.9, epsilon=0.1)

     # Animate progression
    animate_progress(
        value_progress=value_progress,
        policy_progress=policy_progress,
        get_all_states=problem.get_all_states,
        interval=5000,
        save_gif=True,
        value_title="MC: Value Function Progression",
        policy_title="MC: Policy Evolution",
        x_label="Offer Value",
        value_y_label="Estimated Value",
        policy_y_label="Action (0=Reject, 1=Accept)"
    )

    # Plot final Q values
    plot_q_values(
        Q=Q,
        get_all_states=problem.get_all_states,
        action_names=["Reject", "Accept"],
        x_label="Offer Value",
        y_label="Q-Value",
        title="Monte Carlo: Final Q(s,a) Values",
        name_of_plot="mc_q_values.png"
    )

    # Print learned threshold
    learned_policy = {s: np.argmax(Q[s]) for s in problem.get_all_states()}
    threshold = next((s for s in sorted(learned_policy.keys()) if learned_policy[s] == 1), None)
    print(f"Monte Carlo learned threshold i* ≈ {threshold}")