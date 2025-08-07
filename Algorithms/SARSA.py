# from problem_models.selling_an_asset import SellingAssetProblem
from Problems.base_class import RLProblem
from Problems.selling_an_asset import SellingAssetProblem

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from plot import animate_progress, plot_q_values, plot_regret

def epsilon_greedy_policy(Q, state, actions, epsilon):
    """Choose an action using epsilon-greedy policy."""
    if np.random.rand() < epsilon:
        return np.random.choice(actions)
    else:
        return np.argmax(Q[state])

def sarsa(problem: RLProblem, num_episodes=1000, alpha0=0.1, alpha_decay=1e-4,  gamma=0.9, epsilon0=0.1, epsilon_decay=1e-5):

    Q = defaultdict(lambda: np.zeros(len(problem.get_possible_actions(0))))
    # Initialise Q values 
    policy_progress = []
    value_progress = []
    regrets = []


    for ep in range(num_episodes):
        alpha = max(alpha0 / (1 + alpha_decay * ep), 0.01)
        epsilon = max(epsilon0 / (1 + epsilon_decay * ep), 0.01)
        # Initialize S
        S = problem.reset()
        # Select the first action using epsilon greedy policy
        A = epsilon_greedy_policy(Q, S, problem.get_possible_actions(S), epsilon)
        done = False
        episode_reward = 0
        theoretical_i = problem.theoretical_threshold()
        theoretical_best_reward = 0
        t_done = False
        while not done:
            S_prime, R, done = problem.step(A)
            episode_reward += R


            if not t_done:
                theoretical_best_reward += R
            
            if S_prime != None and S_prime >= theoretical_i and not t_done:
                t_done = True
                theoretical_best_reward += S_prime

            if not done:
                A_prime = epsilon_greedy_policy(Q, S_prime, problem.get_possible_actions(S_prime), epsilon)
                td_target = R + gamma * Q[S_prime][A_prime]
            else:
                A_prime = None
                td_target = R
            td_error = td_target - Q[S][A]
            Q[S][A] += alpha * td_error
            # print(f"Episode {ep}, State {S}, Action {A}, Reward {R}, Next State {S_prime}, Next Action {A_prime}, TD Target {td_target}, TD Error {td_error}")
            S, A = S_prime, A_prime

        # Store value function and policy progression for plotting
        value_progress.append([max(Q[s]) for s in problem.get_all_states()])
        policy_progress.append([np.argmax(Q[s]) if s in Q else 0 for s in problem.get_all_states()])
        regrets.append(theoretical_best_reward - episode_reward)

    return Q, value_progress, policy_progress, regrets



if __name__ == "__main__":
    problem = SellingAssetProblem(N=100, alpha=0.9, C=10)
    Q, value_progress, policy_progress, regrets = sarsa(problem, num_episodes=1000000, gamma=0.9)

    # Animate Progression
    # animate_progress(value_progress, policy_progress, problem.get_all_states, interval=5000,
    #                  name_of_plot="SARSA_PROGRESS_CHECK",
    #                  file_location="./RESULTS"
    #                  )

    # Save final plots
    plot_q_values(Q, problem.get_all_states, action_names=problem.get_possible_actions(0),
                  x_label="Offer Value",
                  y_label="Q-Value",
                  name_of_plot="SARSA_FINAL_VALUES_CHECK",
                  file_location="./SARSA_RESULTS")

    plot_regret(regrets, file_path="./SARSA_RESULTS/regret_plot_SARSA.png")