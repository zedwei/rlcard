''' An example of learning a NFSP Agent on Tractor
'''

import tensorflow as tf
import os

from tqdm import tqdm

import rlcard
from rlcard.agents import NFSPAgent
from rlcard.agents import RandomAgent, TractorRuleAgent
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger
from rlcard.games.tractor.utils import tournament_tractor

# Make environment
env = rlcard.make('tractor', config={'seed': 0})
eval_env = rlcard.make('tractor', config={'seed': 0})

# Set the iterations numbers and how frequently we evaluate the performance
evaluate_every = 1000
evaluate_num = 1000
episode_num = 100000
# episode_num = 10000

# The intial memory size
memory_init_size = 1000

# Train the agent every X steps
train_every = 256

# The paths for saving the logs and learning curves
log_dir = './experiments/tractor_nfsp_result/'

# Set a global seed
set_global_seed(0)

# Mitigation for gpu memory issue
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
with tf.Session(config=config) as sess:
# with tf.Session() as sess:
    
    # Initialize a global step
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Set up the agents
    agents = []
    for i in range(2):
        nfsp_agent = NFSPAgent(sess,
                               scope='nfsp' + str(i),
                               action_num=env.action_num,
                               state_shape=env.state_shape,
                              #  hidden_layers_sizes=[512,1024,2048,1024,512],
                               hidden_layers_sizes=[512,1024,512],
                               reservoir_buffer_capacity=int(1e4),
                              #  anticipatory_param=0.5,
                               anticipatory_param=0.1,
                               batch_size=256,
                               train_every = train_every,
                              #  rl_learning_rate=0.00005,
                              #  sl_learning_rate=0.00001,
                               rl_learning_rate=0.0001,
                               sl_learning_rate=0.00005,
                               min_buffer_size_to_learn=memory_init_size,
                               q_replay_memory_size=int(1e5),
                               q_replay_memory_init_size=memory_init_size,
                               q_update_target_estimator_every=1000,
                               q_discount_factor=0.99,
                               q_epsilon_start=0.06,
                               q_epsilon_end=0,
                               q_epsilon_decay_steps=int(1e6),
                               q_batch_size=256,
                               q_train_every=train_every,
                               q_mlp_layers=[512,1024,512],
                               evaluate_with='average_policy')
        agents.append(nfsp_agent)

    random_agent = RandomAgent(action_num=eval_env.action_num)
    rule_agent = TractorRuleAgent(action_num=eval_env.action_num)

    # env.set_agents([agents[0], random_agent, agents[1], random_agent])
    # eval_env.set_agents([agents[0], random_agent, agents[1], random_agent])

    env.set_agents([agents[0], rule_agent, agents[1], rule_agent])
    eval_env.set_agents([agents[0], rule_agent, agents[1], rule_agent])

    # Initialize global variables
    sess.run(tf.global_variables_initializer())

    # Init a Logger to plot the learning curvefrom rlcard.agents.random_agent import RandomAgent

    logger = Logger(log_dir)

    latest_rl_loss = None
    latest_sl_loss = None

    for episode in tqdm(range(episode_num)):
        # First sample a policy for the episode
        for agent_id in [0, 2]:
            env.agents[agent_id].sample_episode_policy()

        # Generate data from the environment
        trajectories, _ = env.run(is_training=True)

        # Feed transitions into agent memory, and train the agent
        for agent_id in [0, 2]:
            for ts in trajectories[agent_id]:
                rl_loss, sl_loss = env.agents[agent_id].feed(ts)
                if rl_loss != None:
                  latest_rl_loss = rl_loss
                if sl_loss != None:
                  latest_sl_loss = sl_loss

        # Evaluate the performance. Play with random agents.
        if episode % evaluate_every == 0:
            logger.log_performance(env.timestep, tournament_tractor(eval_env, evaluate_num)[0])
            tqdm.write('INFO - Agent 0, episode {}, rl-loss: {}, sl-loss: {}'.format(episode, latest_rl_loss, latest_sl_loss))

    # Close files in the logger
    logger.close_files()

    # Plot the learning curve
    logger.plot('NFSP')
    
    # Save model
    save_dir = 'models/tractor_nfsp'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(save_dir, 'model'))
    
