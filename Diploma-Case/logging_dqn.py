import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

from stable_baselines.bench.monitor import Monitor, load_results
from stable_baselines.results_plotter import ts2xy, plot_results
from stable_baselines import results_plotter

from set_environment_dqn import Environment
from dqn_agent import train_dqn

# smoothing
def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")

# environment
env = Environment(
    final_time=5,
    n_elem=50,
    num_steps_per_update=14,
    number_of_control_points=2,
    alpha=75,
    beta=75,
    target_position=[-0.8, 0.5, 0.35],
    num_obstacles=12,
    GENERATE_NEW_OBSTACLES=True
)

# log dir (same as PPO/TRPO)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
identifer = "dqn_id-" + timestamp
log_dir = "./log_" + identifer + "/"
os.makedirs(log_dir, exist_ok=True)

# wrap env with Monitor → writes monitor.csv
env = Monitor(env, log_dir)

# train
rewards, lengths, agent = train_dqn(env, num_episodes=200)

# save Q-network
torch.save(agent.q_net.state_dict(), os.path.join(log_dir, "dqn_policy.pth"))

# plot like Case4
x, y = ts2xy(load_results(log_dir), "timesteps")
if len(y) > 0:
    y = moving_average(y, window=20)
    x = x[len(x) - len(y):]
    fig = plt.figure("DQN Learning Curve")
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title("DQN Smoothed")
    plt.savefig(os.path.join(log_dir, "convergence_plot_dqn.png"))
    plt.close()
    
env.post_processing(os.path.join(log_dir, "dqn.mp4"), SAVE_DATA=True)
