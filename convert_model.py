''' An example of learning a Deep-Q Agent on Tractor
'''
from tensorflow.python.tools import freeze_graph
import tensorflow as tf
import os
import rlcard

path = os.path.join(rlcard.__path__[0], 'models\\tractorV2\\dqn_40k-70k_selftrain_1agent_0.5df_e400k_v2cr')
check_point_path = os.path.join(rlcard.__path__[0], 'models\\tractorV2\\dqn_70k-100k_selftrain_1agent_0.5df_e400k_v2cr')
output_path = os.path.join(rlcard.__path__[0], 'models\\pb\\dqn_40k-70k_selftrain_1agent_0.5df_e400k_v2cr')


def save_as_pb(self, directory, filename):

    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save check point for graph frozen later
    ckpt_filepath = check_point_path
    pbtxt_filename = filename + '.pbtxt'
    pbtxt_filepath = os.path.join(directory, pbtxt_filename)
    pb_filepath = os.path.join(directory, filename + '.pb')
    # This will only save the graph but the variables will not be saved.
    # You have to freeze your model first.
    tf.train.write_graph(graph_or_graph_def=self.sess.graph_def, logdir=directory, name=pbtxt_filename, as_text=True)

    # Freeze graph
    # Method 1
    freeze_graph.freeze_graph(input_graph=pbtxt_filepath, input_saver='', input_binary=False, input_checkpoint=ckpt_filepath, output_node_names='cnn/output', restore_op_name='save/restore_all', filename_tensor_name='save/Const:0', output_graph=pb_filepath, clear_devices=True, initializer_nodes='')
    
    # Method 2
    '''
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    output_node_names = ['cnn/output']

    output_graph_def = graph_util.convert_variables_to_constants(self.sess, input_graph_def, output_node_names)
    # For some models, we would like to remove training nodes
    # output_graph_def = graph_util.remove_training_nodes(output_graph_def, protected_nodes=None)

    with tf.gfile.GFile(pb_filepath, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    '''
    
    return pb_filepath


from rlcard.agents import DQNAgent, RandomAgent
graph = tf.Graph()

# Mitigation for gpu memory issue
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

exporter = tf.saved_model.builder.SavedModelBuilder(output_path)
# latest_ckpt = tf.train.latest_checkpoint(check_point_path)

sess = tf.Session(graph=graph, config=config)

env = rlcard.make('tractor')
with graph.as_default():
    dqn_agents = []
    for i in range(1):
        agent = DQNAgent(sess,
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
        dqn_agents.append(agent)

# check_point_path = os.path.join(TRACTOR_PATH, 'dqn_10k_blindcard')

with sess.as_default():
    with graph.as_default():
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(check_point_path))

        exporter.add_meta_graph_and_variables(
            sess, 
            tags=[tf.saved_model.tag_constants.SERVING])
        exporter.save(as_text=True)