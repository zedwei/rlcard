import subprocess
import sys
from distutils.version import LooseVersion

reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
installed_packages = [r.decode().split('==')[0] for r in reqs.split()]

if 'tensorflow-gpu' in installed_packages or 'tensorflow' in installed_packages:
    import tensorflow as tf
    if LooseVersion(tf.__version__) < LooseVersion('1.14.0') \
            or LooseVersion(tf.__version__) >= LooseVersion('2.0.0'):
        print('WAINING - RLCard supports Tensorflow >=1.14 and <2.0\nThe detected version is {} \nIf the models can not be loaded, please install Tensorflow via\n$ pip install rlcard[tensorflow]\n'.format(tf.__version__))
    from rlcard.agents.deep_cfr_agent import DeepCFR
    from rlcard.agents.dqn_agent import DQNAgent
    from rlcard.agents.nfsp_agent import NFSPAgent
if 'torch' in installed_packages:
    from rlcard.agents.dqn_agent_pytorch import DQNAgent as DQNAgentPytorch
    from rlcard.agents.nfsp_agent_pytorch import NFSPAgent as NFSPAgentPytorch

from rlcard.agents.cfr_agent import CFRAgent
from rlcard.agents.limit_holdem_human_agent import HumanAgent as LimitholdemHumanAgent
from rlcard.agents.nolimit_holdem_human_agent import HumanAgent as NolimitholdemHumanAgent
from rlcard.agents.leduc_holdem_human_agent import HumanAgent as LeducholdemHumanAgent
from rlcard.agents.blackjack_human_agent import HumanAgent as BlackjackHumanAgent
from rlcard.agents.random_agent import RandomAgent
from rlcard.agents.uno_human_agent import HumanAgent as UnoHumanAgent
from rlcard.agents.tractor_rule_agent import TractorRuleAgent
