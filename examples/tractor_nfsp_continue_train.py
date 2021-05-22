''' An example of learning a NFSP Agent on Tractor
'''

import tensorflow as tf
import os

from tqdm import tqdm, trange

import rlcard
from rlcard.agents import NFSPAgent
from rlcard.agents import RandomAgent, TractorRuleAgent
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger
from rlcard.games.tractor.utils import tournament_tractor, MovingAvg

TRACTOR_PATH = os.path.join(rlcard.__path__[0], 'models\\tractorV2')

# Make environment
env = rlcard.make('tractor', config={'seed': 0})
eval_env = rlcard.make('tractor', config={'seed': 0})

# Set the iterations numbers and how frequently we evaluate the performance
evaluate_every = 5000
evaluate_num = 1000
episode_num = 100000
# episode_num = 5000

# The intial memory size
memory_init_size = 10000

# Train the agent every X steps
train_every = 64
# train_every = 256

# The paths for saving the logs and learning curves
log_dir = './experiments/tractor_nfsp_result/'

# Set a global seed
set_global_seed(0)

rl_loss_avg = MovingAvg(100)
sl_loss_avg = MovingAvg(100)
payoff_avg = MovingAvg(100)

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
    for i in range(1):
        nfsp_agent = NFSPAgent(sess,
                               scope='nfsp' + str(i),
                               action_num=env.action_num,
                               state_shape=env.state_shape,
                               hidden_layers_sizes=[2048, 2048],
                            #    anticipatory_param=0.1,
                               anticipatory_param=0.9,
                               batch_size=256,
                               train_every = train_every,
                               rl_learning_rate=0.00002,
                               sl_learning_rate=0.0002,
                               min_buffer_size_to_learn=memory_init_size,
                               q_replay_memory_init_size=memory_init_size,
                               q_update_target_estimator_every=500,
                               q_discount_factor=0.99,
                               q_epsilon_start=1,
                               q_epsilon_end=0.1,
                               q_epsilon_decay_steps=100000,
                               q_batch_size=256,
                               q_train_every=train_every,
                               q_mlp_layers=[2048, 2048],
                               reservoir_buffer_capacity=500000,
                               q_replay_memory_size=100000,
                            #    evaluate_with='average_policy')
                               evaluate_with='best_response')
        agents.append(nfsp_agent)

    random_agent = RandomAgent(action_num=eval_env.action_num)
    rule_agent = TractorRuleAgent(action_num=eval_env.action_num)

    # env.set_agents([agents[0], random_agent, agents[1], random_agent])
    # eval_env.set_agents([agents[0], random_agent, agents[1], random_agent])

    # env.set_agents([agents[0], rule_agent, agents[1], rule_agent])
    # eval_env.set_agents([agents[0], rule_agent, agents[1], rule_agent])

    env.set_agents([agents[0], agents[0], agents[0], agents[0]])
    eval_env.set_agents([agents[0], rule_agent, agents[0], rule_agent])


    # Initialize global variables
    sess.run(tf.global_variables_initializer())

    # Init a Logger to plot the learning curvefrom rlcard.agents.random_agent import RandomAgent
    logger = Logger(log_dir)

    latest_rl_loss = None
    latest_sl_loss = None

    # load the pre-trained model
    check_point_path = os.path.join(TRACTOR_PATH, 'nfsp_continue_350k_0.99')
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(check_point_path))

    graph = tf.get_default_graph()

    t = trange(episode_num, desc='rl-loss:', leave=True)
 
    for episode in t:
        # First sample a policy for the episode
        for agent_id in [0]:
            env.agents[agent_id].sample_episode_policy(use_rule_policy=False)

        # Generate data from the environment
        trajectories, payoffs = env.run(is_training=True)

        payoff_avg.append(payoffs[0])

        # Feed transitions into agent memory, and train the agent
        for agent_id in [0, 1, 2, 3]:
            for ts in trajectories[agent_id]:
                rl_loss, sl_loss = env.agents[agent_id].feed(ts)
                if rl_loss != None: # and agent_id == 0:
                  rl_loss_avg.append(rl_loss)
                if sl_loss != None: # and agent_id == 0:
                  sl_loss_avg.append(sl_loss)

        t.set_description("rl: {}, sl: {}, payoff: {}, e: {}, rsv: {}".format(
            round(rl_loss_avg.get(), 3), 
            round(sl_loss_avg.get(), 3), 
            round(payoff_avg.get(), 3), 
            round(env.agents[0].get_rl_epsilon(), 3),
            env.agents[0].get_reservoir_buffer_size()
            ), refresh=True)

        # Evaluate the performance. Play with random agents.
        if episode % evaluate_every == evaluate_every - 1:
            logger.log_performance(env.timestep, tournament_tractor(eval_env, evaluate_num)[0])

    # Close files in the logger
    logger.close_files()

    # Plot the learning curve
    logger.plot('NFSP')
    
    # Save model
    save_dir = 'models/tractor_nfsp_continue_train'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(save_dir, 'model'))
    
