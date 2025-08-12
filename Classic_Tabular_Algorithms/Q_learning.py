# from problem_models.selling_an_asset import SellingAssetProblem
from Problems.base_class import RLProblem
from Problems.selling_an_asset import SellingAssetProblem

import numpy as np
from collections import defaultdict
from plot import animate_progress, plot_q_values
from plot import plot_regret


def epsilon_greedy_policy(Q, state, actions, epsilon):
    """Choose an action using epsilon-greedy policy."""
    if np.random.rand() < epsilon:
        return np.random.choice(actions)
    else:
        return np.argmax(Q[state])


def q_learning(problem: RLProblem, num_episodes=1000, alpha0=0.1, alpha_decay=1e-5, gamma=0.9, epsilon0=0.1, epsilon_decay=1e-5):

    Q = defaultdict(lambda: np.zeros(len(problem.get_possible_actions(0))))
    policy_progress = []
    value_progress = []
    regrets = []

    for ep in range(num_episodes):
        # Initialize state
        S = problem.reset()
        done = False
        alpha = alpha0 / (1 + alpha_decay * ep)
        epsilon = epsilon0 / (1 + epsilon_decay * ep)
        episode_reward = 0
        theoretical_i = problem.theoretical_threshold()
        theoretical_best_reward = 0
        t_done = False        
        while not done:
            # Choose action A from S using ε-greedy
            A = epsilon_greedy_policy(Q, S, problem.get_possible_actions(S), epsilon)

            # Take action A, observe reward R and next state S'
            S_prime, R, done = problem.step(A)
            episode_reward += R
            if not t_done:
                theoretical_best_reward += R
            
            if S_prime != None and S_prime >= theoretical_i and not t_done:
                t_done = True
                theoretical_best_reward += S_prime

            if not done:
                # Q-learning: off-policy max over next state's actions
                td_target = R + gamma * np.max(Q[S_prime])
            else:
                td_target = R

            td_error = td_target - Q[S][A]
            Q[S][A] += alpha * td_error

            S = S_prime  # Move to next state
        # Track value and policy for animation
        regrets.append(theoretical_best_reward - episode_reward)
        value_progress.append([max(Q[s]) for s in problem.get_all_states()])
        policy_progress.append([np.argmax(Q[s]) if s in Q else 0 for s in problem.get_all_states()])

    return Q, value_progress, policy_progress, regrets






if __name__ == "__main__":
    problem = SellingAssetProblem(N=100, alpha=0.9, C=10)
    Q, value_progress, policy_progress, regrets = q_learning(problem, num_episodes=10000, gamma=0.9, epsilon=0.1)

    # Plot Q-values
    # animate_progress(value_progress, policy_progress, 
    #                 get_all_states=problem.get_all_states,
    #                 interval=1,
    #                 x_label="Offer Value",
    #                 value_y_label="Estimated Value",
    #                 policy_y_label="Action (0=Reject, 1=Accept)",
    #                 value_title="Value Function Progress",
    #                 policy_title="Policy Evolution",
    #                 name_of_plot="Q_Learning_Progress",
    #                 file_location="./dummy",)

    plot_q_values(Q, problem.get_all_states, action_names=problem.get_possible_actions(0),
                       x_label="Offer Value",
                       y_label="Q-Value",
                       name_of_plot="Q_Values_Final",
                       file_location="./dummy")

    plot_regret(regrets, file_path="./dummy/regret_plot.png")
