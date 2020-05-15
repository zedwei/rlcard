''' An example of solve Leduc Hold'em with CFR
'''
import numpy as np

import rlcard
from rlcard.agents import CFRAgent, RandomAgent
from rlcard import models
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger
from rlcard.envs.wrapper import OpenSpiel2RLCard as env_converter
import pyspiel

# Make environment and enable human mode
#env = rlcard.make('leduc-holdem', config={'seed': 0, 'allow_step_back':True})
#eval_env = rlcard.make('leduc-holdem', config={'seed': 0})
game = pyspiel.load_game("leduc_poker")
env = env_converter({'allow_step_back': True, 'allow_raw_data': True, 'seed':0, 'record_action': True}, game)
env2 = env_converter({'allow_step_back': True, 'allow_raw_data': True, 'seed':3, 'record_action': True}, game)
eval_env = env_converter({'allow_step_back': True, 'allow_raw_data': True, 'seed':1, 'record_action': True}, game)

# Set the iterations numbers and how frequently we evaluate the performance and save model
evaluate_every = 100
save_plot_every = 1000
evaluate_num = 10000
episode_num = 10000

# The paths for saving the logs and learning curves
log_dir = './experiments/leduc_holdem_cfr_result/'

# Set a global seed
set_global_seed(0)

# Initilize CFR Agent
agent = CFRAgent(env)
#agent.load()  # If we have saved model, we first load the model

agent2 = RandomAgent(env.action_num) 

# Evaluate CFR against pre-trained NFSP
#eval_env.set_agents([agent, models.load('leduc-holdem-nfsp').agents[0]])
eval_env.set_agents([agent, agent2])

# Init a Logger to plot the learning curve
logger = Logger(log_dir)

for episode in range(episode_num):
    agent.train()
    print('\rIteration {}'.format(episode), end='')
    # Evaluate the performance. Play with NFSP agents.
    if episode % evaluate_every == 0:
        agent.save() # Save model
        logger.log_performance(env.timestep, tournament(eval_env, evaluate_num)[0])

# Close files in the logger
logger.close_files()

# Plot the learning curve
logger.plot('CFR')
