''' An example of learning a Deep-Q Agent on Tractor
'''

import tensorflow as tf
import os

from tqdm import tqdm, trange

import rlcard
from rlcard.agents import DQNAgent
from rlcard.agents import RandomAgent, TractorRuleAgent
from rlcard.utils import set_global_seed
from rlcard.utils import Logger
from rlcard.games.tractor.utils import tournament_tractor, MovingAvg, ACTION_LIST

TRACTOR_PATH = os.path.join(rlcard.__path__[0], 'models\\tractorV6')

# Set a global seed
set_global_seed(0)

# Make environment
env = rlcard.make('tractor', config={'seed': 0})
eval_env = rlcard.make('tractor', config={'seed': 0})

# Set the iterations numbers and how frequently we evaluate the performance
evaluate_every = 5000
evaluate_num = 1000
episode_num = 1000000

# The intial memory size
memory_init_size = 100000

# Train the agent every X steps
train_every = 64

# The paths for saving the logs and learning curves
save_dir = 'models/tractor_dqn_continue_train'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Mitigation for gpu memory issue
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

agents = []

with tf.Session(config=config) as sess:
    # Initialize a global step
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Set up the agents
    for i in range(2):
        agent = DQNAgent(sess,
                        scope='dqn' if i==0 else 'dqn' + str(i),
                        action_num=env.action_num,
                        replay_memory_init_size=memory_init_size,
                        train_every=train_every,
                        state_shape=env.state_shape,
                        mlp_layers=[2048,2048],
                        replay_memory_size=400000,
                        update_target_estimator_every=500,
                        discount_factor=0.99,
                        epsilon_start=0.5,
                        epsilon_end=0.1,
                        epsilon_decay_steps=100000,
                        batch_size=256,
                        learning_rate=0.00002,
                        use_rule_policy=False
                    )
        agents.append(agent)
    
    random_agent = RandomAgent(action_num=eval_env.action_num)
    rule_agent = TractorRuleAgent(action_num=eval_env.action_num)


    # 2 dqn agent vs 2 rule agent
    # env.set_agents([agents[0], rule_agent, agents[0], rule_agent])
    # eval_env.set_agents([agents[0], rule_agent, agents[0], rule_agent])

    # env.set_agents([rule_agent, agents[1], rule_agent, agents[1]])
    # eval_env.set_agents([rule_agent, agents[1], rule_agent, agents[1]])


    # 4 dqn agent with single brain
    # env.set_agents([agents[0], agents[0], agents[0], agents[0]])
    # eval_env.set_agents([agents[0], rule_agent, agents[0], rule_agent])

    # 4 dqn agent with two brains
    env.set_agents([agents[0], agents[1], agents[0], agents[1]])
    eval_env.set_agents([agents[0], rule_agent, agents[0], rule_agent])

    # Initialize global variables
    sess.run(tf.global_variables_initializer())

    # Init a Logger to plot the learning curve
    logger = Logger(save_dir)

    # Init moving average calculator
    rl_avg = [MovingAvg(100), MovingAvg(100)]
    payoff_avg = [MovingAvg(100), MovingAvg(100)]

    # load the pre-trained model
    check_point_path = os.path.join(TRACTOR_PATH, 'tractor_dqn_2505k_2230k')
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint(check_point_path))
    graph = tf.get_default_graph()
    print('INFO: Loaded model from {}'.format(check_point_path))

    t = trange(episode_num, desc='rl-loss:', leave=True)
    for episode in t:
        if episode % 10 < 6:
            env.set_agents([agents[0], agents[1], agents[0], agents[1]])
            feed_range = [0, 1, 2, 3]
        elif episode % 10 < 8:
            env.set_agents([agents[0], rule_agent, agents[0], rule_agent])
            feed_range = [0, 2]
        else:
            env.set_agents([rule_agent, agents[1], rule_agent, agents[1]])
            feed_range = [1, 3]


        # Generate data from the environment
        trajectories, payoffs = env.run(is_training=True)
        for i in [0, 1]:
            payoff_avg[i].append(payoffs[i])

        # Feed transitions into agent memory, and train the agent
        # for agent_id in [0, 1, 2, 3]:
        # for agent_id in [1, 3]:
        for agent_id in feed_range:
            for ts in trajectories[agent_id]:
                rl_loss = env.agents[agent_id].feed(ts)
                if rl_loss != None:
                    rl_avg[agent_id % 2].append(rl_loss)

        t.set_description("rl_0: {}, rl_1: {}, po_0: {}, po_1: {}, epsilon: {}".format(
            round(rl_avg[0].get(), 2), 
            round(rl_avg[1].get(), 2), 
            round(payoff_avg[0].get(), 2), 
            round(payoff_avg[1].get(), 2), 
            round(env.agents[feed_range[0]].epsilons[min(agent.total_t, env.agents[feed_range[0]].epsilon_decay_steps-1)], 2)
            ), refresh=True)

        # q = env.agents[0].eval_step(state)[1]
        # probs = {ACTION_LIST[i]:round(q[i],3) for i in range(len(q)) if q[i] != -100}
        # probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        # tqdm.write(str(probs))

        # Evaluate the performance. Play with random agents.
        if episode % evaluate_every == evaluate_every - 1:
            logger.log_performance(env.timestep, tournament_tractor(eval_env, evaluate_num)[0])
            logger.log("rl_0: {}, rl_1: {}, po_0: {}, po_1: {}, epsilon: {}".format(
            round(rl_avg[0].get(), 2), 
            round(rl_avg[1].get(), 2), 
            round(payoff_avg[0].get(), 2), 
            round(payoff_avg[1].get(), 2), 
            round(env.agents[feed_range[0]].epsilons[min(agent.total_t, env.agents[feed_range[0]].epsilon_decay_steps-1)], 2)
            ))
            saver.save(sess, os.path.join(save_dir, 'model'))

    # Close files in the logger
    logger.close_files()

    # Plot the learning curve
    logger.plot('DQN')

    
