import numpy as np
import matplotlib.pyplot as plt


N = 100            # Max offer value
alpha = 0.9         # Discount factor
C = 10             # Daily maintenance cost


P = np.ones(N + 1)
P /= P.sum()


V = np.zeros(N + 1)
policy = np.zeros(N + 1, dtype=int)  # 0 = reject, 1 = accept


def calculate_theoretical_threshold(N, alpha, C, P):
    min_value = float('inf')
    best_i = 1
    
    for i in range(1, N + 1):
        # Calculate sum of P[j] for j=0 to i-1
        sum_P = sum(P[j] for j in range(i))
        
        # Calculate sum of j*P[j] for j=0 to i-1  
        sum_jP = sum(j * P[j] for j in range(i,N+1))
        
        # Calculate the function value
        if sum_P < 1/alpha:  # Ensure denominator is positive
            fn_value = (C * sum_P - sum_jP) / (1 - alpha * sum_P)
        else:
            fn_value = float('inf')
            
        if fn_value < min_value:
            min_value = fn_value
            best_i = i
    
    return best_i, min_value

def expected_reject_value(V, P, alpha, C):
    return C + alpha * np.dot(P, V)

def modified_policy_iteration(P, alpha, C, V, policy, max_iter=100, eval_iter=1, tol=1e-4):
    N = len(V) - 1
    history = []

    for it in range(max_iter):
        # Partial Policy Evaluation
        for _ in range(eval_iter):
            V_new = np.zeros_like(V) # Initilaise V to values from which the current policy was improved upon in the last iteration
            for i in range(N + 1):
                if policy[i] == 1:
                    V_new[i] = -i
                else:
                    V_new[i] = expected_reject_value(V, P, alpha, C)
            V[:] = V_new

        # Policy Improvement
        policy_stable = True
        for i in range(N + 1):
            accept_val = -i
            reject_val = expected_reject_value(V, P, alpha, C)
            new_action = 1 if accept_val < reject_val else 0
            if new_action != policy[i]:
                policy_stable = False
            policy[i] = new_action

        print(f"Iteration {it + 1}: Policy = {policy} Values = {V}")
        history.append(V.copy())

        if policy_stable:
            break

    return V, policy, history



############################## Finite Horizon Value Iteration (Backward Induction)
def finite_horizon_value_iteration(P, alpha, C, N, T):
    V = np.zeros((T + 1, N + 1))  # Value function: time x state
    policy = np.zeros((T + 1, N + 1), dtype=int)

    for t in range(T - 1, -1, -1):  # Backward induction from T-1 to 0
        for i in range(N + 1):
            accept_val = -i
            reject_val = C + np.dot(P, V[t + 1])
            if accept_val <= reject_val: # equality coz we dont like risk ..
                policy[t, i] = 1  # Accept
                V[t, i] = accept_val
            else:
                policy[t, i] = 0  # Reject
                V[t, i] = reject_val
        print(f"Time {t}: Policy = {policy[t]} ; Value = {V[t]}")
    return V, policy


T = 20 
V_finite, policy_finite = finite_horizon_value_iteration(P, alpha, C, N, T)

# Plot finite horizon policy at selected timesteps
plt.figure(figsize=(10, 6))
for t_plot in [0, T//4, T//2, 3*T//4, T-1]:
    plt.step(range(N + 1), policy_finite[t_plot], where='mid', label=f'Time t={t_plot}')
plt.title('Finite-Horizon Optimal Policy at Different Times')
plt.xlabel('Offer Value (State)')
plt.ylabel('Action (0=Reject, 1=Accept)')
plt.legend()
plt.grid(True)
plt.savefig('finite_horizon_policy.png', dpi=300)






# ############################## Resuls
# # Calculate theoretical optimal threshold
# i_star_theoretical, min_cost = calculate_theoretical_threshold(N, alpha, C, P)
# print(f"Theoretical optimal threshold i* = {i_star_theoretical}")
# print(f"Minimum function value = {min_cost:.6f}")

# # Run MPI
# V_final, policy_final, V_history = modified_policy_iteration(P, alpha, C, V, policy)

# # print("Final ")

# # Find empirical threshold
# empirical_threshold = None
# for i in range(N + 1):
#     if policy_final[i] == 1:
#         empirical_threshold = i
#         break

# print(f"Empirical threshold from MPI = {empirical_threshold}")

# # Plot results
# fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# # Value function over iterations
# for i, V_iter in enumerate(V_history):
#     axs[0].plot(range(N + 1), V_iter, label=f'Iter {i+1}')
# axs[0].set_title('Value Function over Iterations')
# axs[0].set_xlabel('Offer Value (State)')
# axs[0].set_ylabel('Value')
# axs[0].legend()
# axs[0].grid(True)

# # Final policy plot
# axs[1].step(range(N + 1), policy_final, where='mid', color='black')
# axs[1].axvline(x=i_star_theoretical, color='red', linestyle='--', linewidth=2, 
#                label=f'Theoretical i* = {i_star_theoretical}')
# axs[1].set_title('Optimal Policy')
# axs[1].set_xlabel('Offer Value (State)')
# axs[1].set_ylabel('Action (0=Reject, 1=Accept)')
# axs[1].set_yticks([0, 1])
# axs[1].legend()
# axs[1].grid(True)

# # Plot the function being minimized
# i_values = range(1, N + 1)
# fn_values = []
# for i in i_values:
#     sum_P = sum(P[j] for j in range(i))
#     sum_jP = sum(j * P[j] for j in range(i,N+1))
#     if sum_P < 1/alpha:
#         fn_value = (C * sum_P - sum_jP) / (1 - alpha * sum_P)
#     else:
#         fn_value = float('inf')
#     fn_values.append(fn_value)

    # """
    # Solves the finite-horizon selling problem using backward induction.
    # Returns: V_t[state], policy_t[state] for t=0 to T.
    # """

# axs[2].plot(i_values, fn_values, 'b-', linewidth=2)
# axs[2].axvline(x=i_star_theoretical, color='red', linestyle='--', linewidth=2,
#                label=f'Theoretical i* = {i_star_theoretical}')
# if empirical_threshold:
#     axs[2].axvline(x=empirical_threshold, color='green', linestyle=':', linewidth=2,
#                    label=f'Empirical = {empirical_threshold}')
# axs[2].set_title('Function to Minimize')
# axs[2].set_xlabel('Threshold i')
# axs[2].set_ylabel('Function Value')
# axs[2].legend()
# axs[2].grid(True)



# plt.savefig('selling_an_asset_results.png', dpi=300)