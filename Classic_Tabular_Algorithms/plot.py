import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Dict, Callable
import numpy as np
import os

def animate_progress(value_progress: List[List[float]],
                     policy_progress: List[List[int]],
                     get_all_states: Callable[[], List],
                     interval: int = 100,
                     save_gif: bool = True,
                     save_mp4: bool = False,
                     value_title: str = "Value Function Progression",
                     policy_title: str = "Policy Progression",
                     x_label: str = "State",
                     value_y_label: str = "Value",
                     policy_y_label: str = "Action",
                     name_of_plot: str = "rl_progress",
                     file_location: str = "./RESULTS",
                     title: str = "RL Progress") -> None:
    
    os.makedirs(file_location, exist_ok=True)

    states = get_all_states()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    line1, = ax1.plot([], [], lw=2)
    ax1.set_xlim(min(states), max(states))
    ax1.set_ylim(min(min(v) for v in value_progress), max(max(v) for v in value_progress))
    ax1.set_title(value_title)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(value_y_label)
    ax1.grid(True)

    ax2 = axes[1]
    line2, = ax2.step([], [], where='mid')
    ax2.set_xlim(min(states), max(states))
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0, 1])
    ax2.set_title(policy_title)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(policy_y_label)
    ax2.grid(True)

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        return line1, line2

    def animate(i):
        idx = i * interval
        if idx >= len(value_progress):
            idx = len(value_progress) - 1
        line1.set_data(states, value_progress[idx])
        line2.set_data(states, policy_progress[idx])
        ax1.set_title(f"{value_title} - Episode {idx}")
        ax2.set_title(f"{policy_title} - Episode {idx}")
        return line1, line2

    num_frames = len(value_progress) // interval
    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=num_frames, interval=200, blit=True)

    if save_gif:
        ani.save(os.path.join(file_location, name_of_plot + ".gif"), writer="imagemagick", fps=5)
    if save_mp4:
        ani.save(os.path.join(file_location, name_of_plot + ".mp4"), writer="ffmpeg", fps=5)
    plt.close()


def plot_q_values(Q: Dict,
                  get_all_states: Callable[[], List],
                  action_names: List[str] = None,
                  x_label: str = "State",
                  y_label: str = "Q-Value",
                  name_of_plot: str = "q_values_plot.png",
                  file_location: str = "./RESULTS",
                  title: str = "Q-values and Policy") -> None:
    
    os.makedirs(file_location, exist_ok=True)

    states = get_all_states()
    num_actions = len(Q[next(iter(Q))])
    q_values_by_action = [[Q[s][a] for s in states] for a in range(num_actions)]
    optimal_policy = [int(max(range(num_actions), key=lambda a: Q[s][a])) for s in states]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for a in range(num_actions):
        label = action_names[a] if action_names else f"Q(s, a={a})"
        linestyle = '--' if a == 0 else '-'
        axes[0].plot(states, q_values_by_action[a], label=label, linestyle=linestyle)
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(y_label)
    axes[0].set_title("Q(s, a) Values")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].step(states, optimal_policy, where='mid', label="Optimal Action")
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("Optimal Action")
    axes[1].set_yticks(range(num_actions))
    if action_names:
        axes[1].set_yticklabels(action_names)
    axes[1].set_title("Learned Policy")
    axes[1].grid(True)

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(file_location, name_of_plot), dpi=300)
    plt.close()


def plot_regret(regrets, file_path="regret_plot.png"):
    folder = os.path.dirname(file_path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    cumulative = np.cumsum(regrets)

    plt.figure(figsize=(10, 6))
    plt.plot(cumulative, label="Cumulative Regret", color="red")
    plt.title("Cumulative Regret Over Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Regret")
    plt.grid(True)
    plt.legend()
    plt.savefig(file_path, dpi=300)
    plt.close()
