''' An example of learning a Deep-Q Agent on Tractor
'''

import tensorflow as tf
import os

from tqdm import tqdm, trange

import rlcard
from rlcard.agents import DQNAgent
from rlcard.agents import RandomAgent, TractorRuleAgent
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger
from rlcard.games.tractor.utils import tournament_tractor, MovingAvg, ACTION_LIST

# Set a global seed
set_global_seed(0)

# Make environment
env = rlcard.make('tractor', config={'seed': 0})
eval_env = rlcard.make('tractor', config={'seed': 0})

# Set the iterations numbers and how frequently we evaluate the performance
evaluate_every = 2000
evaluate_num = 1000
# episode_num = 500000
episode_num = 100000

# The intial memory size
memory_init_size = 1000

# Train the agent every X steps
train_every = 64

# The paths for saving the logs and learning curves
log_dir = './experiments/tractor_dqn_result/'

# Mitigation for gpu memory issue
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
with tf.Session(config=config) as sess:
# with tf.Session() as sess:

    # Initialize a global step
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Set up the agents
    agent = DQNAgent(sess,
                     scope='dqn',
                     action_num=env.action_num,
                     replay_memory_init_size=memory_init_size,
                     train_every=train_every,
                     state_shape=env.state_shape,
                    #  mlp_layers=[512,1024,512],
                    mlp_layers=[2048,2048],
                    # mlp_layers=[4096,4096],
                     replay_memory_size=100000,
                     update_target_estimator_every=500,
                    # update_target_estimator_every=30,
                     discount_factor=0.5,
                    #  discount_factor=0,
                     epsilon_start=1,
                     epsilon_end=0.1,
                     epsilon_decay_steps=100000,
                     batch_size=256,
                    # batch_size=512,
                     learning_rate=0.00002,
                    # learning_rate=0.0005,
                     use_rule_policy=False
                 )
    random_agent = RandomAgent(action_num=eval_env.action_num)
    rule_agent = TractorRuleAgent(action_num=eval_env.action_num)

    # env.set_agents([agent, rule_agent, rule_agent, rule_agent])
    # eval_env.set_agents([agent, rule_agent, rule_agent, rule_agent])

    env.set_agents([agent, rule_agent, agent, rule_agent])
    eval_env.set_agents([agent, rule_agent, agent, rule_agent])

    # env.set_agents([agent, random_agent, agent, random_agent])
    # eval_env.set_agents([agent, random_agent, agent, random_agent])

    # Initialize global variables
    sess.run(tf.global_variables_initializer())

    # Init a Logger to plot the learning curve
    logger = Logger(log_dir)

    # Init moving average calculator
    m_avg = MovingAvg(100)
    payoff_avg = MovingAvg(100)

    # Store a test state to track Q value
    # state, player_id = env.reset()
    # predefined_hands = [['TD', 'AD'],
    #                     ['KS', 'AS'],
    #                     ['TD', '3D'],
    #                     ['3D', '4D']]

    # predefined_hands = [['AH','6H'],
    #                     ['KH','QH'],
    #                     ['4H','3H'],
    #                     ['JH','TH']]

    # predefined_hands = [['AH'],
    #                     ['KH'],
    #                     ['4H'],
    #                     ['JH']]
    # state = env.reset_predefine_state(predefined_hands)
    # for i in range(4):
    #     print(env.game.players[i].current_hand)

    
    t = trange(episode_num, desc='rl-loss:', leave=True)
    for episode in t:
        # Generate data from the environment
        trajectories, payoffs = env.run(is_training=True)
        payoff_avg.append(payoffs[0])

        # Feed transitions into agent memory, and train the agent
        for player in [0, 2]:
        # for player in [0]:
            for ts in trajectories[player]:
                rl_loss = agent.feed(ts)
                if rl_loss != None:
                    m_avg.append(rl_loss)
        
        t.set_description("rl loss: {}, payoff: {}, epsilon: {}".format(
            round(m_avg.get(), 2), 
            round(payoff_avg.get(), 2), 
            round(agent.epsilons[min(agent.total_t, agent.epsilon_decay_steps-1)], 2)
            ), refresh=True)

        # q = env.agents[0].eval_step(state)[1]
        # probs = {ACTION_LIST[i]:round(q[i],3) for i in range(len(q)) if q[i] != -100}
        # probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        # tqdm.write(str(probs))

        # Evaluate the performance. Play with random agents.
        if episode % evaluate_every == 0:
            logger.log_performance(env.timestep, tournament_tractor(eval_env, evaluate_num)[0])

    # Close files in the logger
    logger.close_files()

    # Plot the learning curve
    logger.plot('DQN')
    
    # Save model
    save_dir = 'models/tractor_dqn'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(save_dir, 'model'))
    
