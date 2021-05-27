''' Wrrapers of pretrained models.
'''

import os

import rlcard
from rlcard.models.model import Model
from rlcard.agents import DQNAgent


TRACTOR_PATH = os.path.join(rlcard.__path__[0], 'models\\tractorV4')

class TractorNFSPModel(Model):
    ''' A pretrained model on Tractor with NFSP
    '''

    def __init__(self):
        ''' Load pretrained model
        '''
        import tensorflow as tf
        from rlcard.agents import NFSPAgent, RandomAgent
        self.graph = tf.Graph()

        # Mitigation for gpu memory issue
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(graph=self.graph, config=config)

        env = rlcard.make('tractor')
        with self.graph.as_default():
            self.nfsp_agents = []
            # for i in range(env.player_num):
            #     agent = NFSPAgent(self.sess,
            #                       scope='nfsp' + str(i),
            #                       action_num=env.action_num,
            #                       state_shape=env.state_shape,
            #                       hidden_layers_sizes=[512,1024,2048,1024,512],
            #                       q_mlp_layers=[512,1024,2048,1024,512])
            #     self.nfsp_agents.append(agent)

            for i in range(1):
                agent = NFSPAgent(self.sess,
                                scope='nfsp' + str(i),
                                action_num=env.action_num,
                                state_shape=env.state_shape,
                                hidden_layers_sizes=[2048,2048],
                                q_mlp_layers=[2048,2048],
                                # evaluate_with='average_policy')
                                evaluate_with='best_response')

                self.nfsp_agents.append(agent)

        check_point_path = os.path.join(TRACTOR_PATH, 'nfsp_continue_350k_0.99')

        with self.sess.as_default():
            with self.graph.as_default():
                saver = tf.train.Saver()
                saver.restore(self.sess, tf.train.latest_checkpoint(check_point_path))
    @property
    def agents(self):
        ''' Get a list of agents for each position in a the game

        Returns:
            agents (list): A list of agents

        Note: Each agent should be just like RL agent with step and eval_step
              functioning well.
        '''
        return self.nfsp_agents


class TractorDQNModel(Model):
    ''' A pretrained model on Tractor with NFSP
    '''

    def __init__(self):
        ''' Load pretrained model
        '''
        import tensorflow as tf
        self.graph = tf.Graph()

        # Mitigation for gpu memory issue
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(graph=self.graph, config=config)

        env = rlcard.make('tractor')
        with self.graph.as_default():
            self.dqn_agents = []
            for i in range(1):
                agent = DQNAgent(self.sess,
                     scope='dqn',
                     action_num=env.action_num,
                     state_shape=env.state_shape,
                     mlp_layers=[2048,2048],
                     replay_memory_size=100000,
                     update_target_estimator_every=100,
                     discount_factor=0.5,
                     epsilon_start=1,
                     epsilon_end=0.1,
                     epsilon_decay_steps=100000,
                     batch_size=256,
                     learning_rate=0.00002,
                     use_rule_policy=False
                )
                self.dqn_agents.append(agent)

        check_point_path = os.path.join(TRACTOR_PATH, 'tractor_dqn_100k')

        with self.sess.as_default():
            with self.graph.as_default():
                saver = tf.train.Saver()
                saver.restore(self.sess, tf.train.latest_checkpoint(check_point_path))
    @property
    def agents(self):
        ''' Get a list of agents for each position in a the game

        Returns:
            agents (list): A list of agents

        Note: Each agent should be just like RL agent with step and eval_step
              functioning well.
        '''
        return self.dqn_agents