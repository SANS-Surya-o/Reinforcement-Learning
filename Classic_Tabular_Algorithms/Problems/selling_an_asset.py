import numpy as np
from Problems.base_class import RLProblem

class SellingAssetProblem(RLProblem):
    def __init__(self, N=100, alpha=0.9, C=10, P=None):
        self.N = N
        self.alpha = alpha
        self.C = C
        self.P = P if P is not None else np.ones(N + 1) / (N + 1)
        self.states = list(range(N + 1))
        self.actions = [0, 1]  # 0 = reject, 1 = accept
        self.state = None

    def reset(self):
        self.state = np.random.choice(self.states, p=self.P)
        return self.state

    def step(self, action):
        if action == 1:  # Accept
            reward = self.state
            done = True
            next_state = None
        else:  # Reject
            reward = -self.C
            done = False
            next_state = np.random.choice(self.states, p=self.P)
        self.state = next_state
        return next_state, reward, done

    def get_possible_actions(self, state):
        return self.actions

    def get_all_states(self):
        return self.states

    def is_terminal(self, state):
        # Accepting ends the episode, rejecting continues
        return state is None
    
    def theoretical_threshold(self):
        N = self.N
        alpha = self.alpha
        C = self.C
        P = self.P
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
