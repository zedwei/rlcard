''' Wrrapers of pretrained models.
'''

import os

import rlcard
from rlcard.models.model import Model

TRACTOR_PATH = os.path.join(rlcard.__path__[0], 'models\\tractor')

class TractorNFSPModel(Model):
    ''' A pretrained model on Tractor with NFSP
    '''

    def __init__(self):
        ''' Load pretrained model
        '''
        import tensorflow as tf
        from rlcard.agents import NFSPAgent
        self.graph = tf.Graph()

        # Mitigation for gpu memory issue
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(graph=self.graph, config=config)

        env = rlcard.make('tractor')
        with self.graph.as_default():
            self.nfsp_agents = []
            for i in range(env.player_num):
                agent = NFSPAgent(self.sess,
                                  scope='nfsp' + str(i),
                                  action_num=env.action_num,
                                  state_shape=env.state_shape,
                                  hidden_layers_sizes=[512,1024,2048,1024,512],
                                  q_mlp_layers=[512,1024,2048,1024,512])
                self.nfsp_agents.append(agent)

        check_point_path = os.path.join(TRACTOR_PATH, '20210504')

        with self.sess.as_default():
            with self.graph.as_default():
                saver = tf.train.Saver()
                print(check_point_path)
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