import abc

class RLProblem(abc.ABC):
    @abc.abstractmethod
    def reset(self):
        """Resets the environment and returns the initial state."""
        pass

    @abc.abstractmethod
    def step(self, action):
        """
        Executes the given action.
        Returns:
            next_state, reward, done
        """
        pass

    @abc.abstractmethod
    def get_possible_actions(self, state):
        """Returns the list of possible actions in a given state."""
        pass

    @abc.abstractmethod
    def get_all_states(self):
        """Returns the list of all possible states."""
        pass

    @abc.abstractmethod
    def is_terminal(self, state):
        """Returns True if the state is terminal, False otherwise."""
        pass

    def sample_initial_state(self):
        """Optionally define a method for sampling an initial state."""
        return self.reset()
    
    def theoretical_threshold(self):
        """Optionally define a method to compute the theoretical threshold."""
        return None
