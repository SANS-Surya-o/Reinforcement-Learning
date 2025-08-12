# from problem_models.selling_an_asset import SellingAssetProblem
from Problems.base_class import RLProblem
from Problems.selling_an_asset import SellingAssetProblem

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from plot import animate_progress, plot_q_values

def modified_policy_iteration(problem: RLProblem, max_iter=100, eval_iter=1, tol=1e-4):
    N = max(problem.get_all_states())
    P = np.ones(N + 1) / (N + 1)  # Uniform offer distribution
    alpha = problem.alpha
    C = problem.C

    V = np.zeros(N + 1)
    policy = np.zeros(N + 1, dtype=int)  # 0 = reject, 1 = accept

    value_progress = []
    policy_progress = []
    for it in range(max_iter):
        # --- Policy Evaluation (Partial) ---
        for _ in range(eval_iter):
            V_new = np.zeros_like(V)
            for i in range(N + 1):
                if policy[i] == 1:
                    V_new[i] = i  # Accept gives immediate reward i
                else:
                    V_new[i] = -C + alpha * np.dot(P, V)  # Reject incurs cost, continue
            V[:] = V_new

        # --- Policy Improvement ---
        policy_stable = True
        for i in range(N + 1):
            accept_reward = i
            reject_reward = -C + alpha * np.dot(P, V)
            new_action = 1 if accept_reward > reject_reward else 0
            if new_action != policy[i]:
                policy_stable = False
            policy[i] = new_action

        # Store for plotting
        value_progress.append(V.copy())
        policy_progress.append(policy.copy())

        print(f"Iteration {it + 1}: Threshold = {next((s for s in range(N+1) if policy[s] == 1), 'None')}")
        if policy_stable:
            break

    return V, policy, value_progress, policy_progress

def calculate_theoretical_threshold(N, alpha, C, P):
    max_value = float('-inf')
    best_i = 1

    for i in range(1, N + 1):
        sum_P = sum(P[j] for j in range(i))
        sum_jP = sum(j * P[j] for j in range(i, N + 1))
        
        if sum_P < 1 / alpha:
            fn_value = (sum_jP - C * sum_P) / (1 - alpha * sum_P)
        else:
            fn_value = float('-inf')
        
        if fn_value > max_value:
            max_value = fn_value
            best_i = i

    return best_i




if __name__ == "__main__":
    # Initialize the selling asset problem
    problem = SellingAssetProblem(N=100, alpha=0.9, C=10)

    # Run Modified Policy Iteration (reward maximization version)
    V, policy, value_progress, policy_progress = modified_policy_iteration(
        problem, max_iter=100, eval_iter=3
    )

    # Animate the progression of value function and policy
    animate_progress(
        value_progress=value_progress,
        policy_progress=policy_progress,
        get_all_states=problem.get_all_states,
        interval=1,
        save_gif=True,
        value_title="Modified Policy Iteration: Value Function Progression",
        policy_title="Modified Policy Iteration: Policy Evolution",
        x_label="Offer Value",
        value_y_label="Estimated Value",
        policy_y_label="Action (0=Reject, 1=Accept)",
        name_of_plot="Modified_Policy_Iteration_Progress",
        file_location="./RESULTS"
    )
    
    ### THE BELOW FUNCTION NEEDS SOME FIXING - LITE WILL NEVER DO : ) LOL

    # plot_q_values(
    #     Q=None,  # No Q-values in this case
    #     get_all_states=problem.get_all_states,
    #     action_names=["Reject", "Accept"],
    #     x_label="Offer Value",
    #     y_label="Value",
    #     title="Modified Policy Iteration: Final Value Function",
    #     name_of_plot="Modified_Policy_Iteration_Final_Values",
    # )

    # Compute and print learned acceptance threshold i*
    threshold = next((s for s in range(len(policy)) if policy[s] == 1), None)
    print(f"Learned acceptance threshold i* ≈ {threshold}")
