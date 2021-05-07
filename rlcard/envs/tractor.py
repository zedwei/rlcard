import numpy as np
import functools

from rlcard.envs import Env
from rlcard.utils import reorganize
from rlcard.games.tractor.utils import reorganize_with_payoff_trace
from rlcard.games.tractor import Game
from rlcard.games.tractor.utils import encode_cards, ACTION_LIST, ACTION_SPACE


class TractorEnv(Env):
    ''' Tractor Environment
    '''

    def __init__(self, config):

        self.name = 'tractor'
        self.game = Game()
        super().__init__(config)
        # self.state_shape = [3, 3, 54]
        self.state_shape = [5, 3, 54]

    def run(self, is_training=False):
        '''
        Run a complete game, either for evaluation or training RL agent.

        Args:
            is_training (boolean): True if for training purpose.

        Returns:
            (tuple) Tuple containing:

                (list): A list of trajectories generated from the environment.
                (list): A list payoffs. Each entry corresponds to one player.

        Note: The trajectories are 3-dimension list. The first dimension is for different players.
              The second dimension is for different transitions. The third dimension is for the contents of each transiton
        '''
        if self.single_agent_mode:
            raise ValueError('Run in single agent not allowed.')

        trajectories = [[] for _ in range(self.player_num)]
        state, player_id = self.reset()

        # Loop to play the game
        trajectories[player_id].append(state)
        while not self.is_over():
            # Agent plays
            if not is_training:
                action, _ = self.agents[player_id].eval_step(state)
            else:
                action = self.agents[player_id].step(state)

            # action, _ = self.agents[player_id].eval_step(state)

            # Environment steps
            next_state, next_player_id = self.step(action, self.agents[player_id].use_raw)
            # Save action
            trajectories[player_id].append(action)

            # Set the state and player
            state = next_state
            player_id = next_player_id

            # Save state.
            if not self.game.is_over():
                trajectories[player_id].append(state)

        # Add a final state to all the players
        for player_id in range(self.player_num):
            state = self.get_state(player_id)
            trajectories[player_id].append(state)

        # Payoffs
        # Option 1: get final payoffs after each game
        payoffs = self.get_payoffs()
        trajectories = reorganize(trajectories, payoffs)

        # Option 2: get payoffs after each round
        # payoffs_with_trace = self.get_payoffs_trace()
        # trajectories = reorganize_with_payoff_trace(trajectories, payoffs_with_trace, payoffs)

        return trajectories, payoffs
        
    def _extract_state(self, state):
        ''' Extract useful information from state for RL. Must be implemented in the child class.

        Args:
            state (dict): The raw state

        Returns:
            (numpy.array): The extracted state
            x*3*54 array
                x:  current hand
                    union of others' hand
                    current cards in round
                    # team-mate's possible hand
                    # up-player possible hand
                    # down-player possible hand
        '''
        # obs = np.zeros((3, 3, 54), dtype=int)
        obs = np.zeros((5, 3, 54), dtype=int)
        # for index in range(3):
        for index in range(5):
            obs[index][0] = np.ones(54, dtype=int)
        encode_cards(obs[0], state['current_hand'])
        encode_cards(obs[1], state['others_hand'][0])
        encode_cards(obs[2], state['others_hand'][1])
        encode_cards(obs[3], state['others_hand'][2])
        
        current_round = [x for x in state['current_round'] if x != None]
        if (len(current_round) > 0):
            current_round = functools.reduce(lambda z,y : z + y, current_round)
            # encode_cards(obs[2], current_round)
            encode_cards(obs[4], current_round)

        extracted_state = {'obs': obs, 'legal_actions': self._get_legal_actions()}
        if self.allow_raw_data:
            extracted_state['raw_obs'] = state
            # TODO: state['actions'] can be None, may have bugs
            if state['actions'] == None:
                extracted_state['raw_legal_actions'] = []
            else:
                extracted_state['raw_legal_actions'] = [a for a in state['actions']]
        if self.record_action:
            extracted_state['action_record'] = self.action_recorder
        return extracted_state


    def get_payoffs(self):
        ''' Get the payoffs of players. Must be implemented in the child class.

        Returns:
            (list): A list of payoffs for each player.

        Note: Must be implemented in the child class.
        '''
        return self.game.judger.judge_payoffs(self.game.winner_id, self.game.round.score)

    def get_payoffs_trace(self):
        return self.game.round.score_trace
        
    def _decode_action(self, action_id):
        ''' Decode Action id to the action in the game.

        Args:
            action_id (int): The id of the action

        Returns:
            (string): The action that will be passed to the game engine.

        Note: Must be implemented in the child class.
        '''
        return ACTION_LIST[action_id].split(',')

    def _get_legal_actions(self):
        ''' Get all legal actions for current state.

        Returns:
            (list): A list of legal actions' id.

        Note: Must be implemented in the child class.
        '''
        legal_action_id = []
        legal_actions = self.game.state['actions']
        if legal_actions:
            for action in legal_actions:
                action_str = ','.join(action)
                legal_action_id.append(ACTION_SPACE[action_str])
        return legal_action_id

    def get_perfect_information(self):
        ''' Get the perfect information of the current state

        Returns:
            (dict): A dictionary of all the perfect information of the current state

        Note: Must be implemented in the child class.
        '''
        state = {}
        state['hand_cards'] = [player.current_hand for player in self.game.players]
        state['banker_id'] = self.game.state['banker_id']
        state['trace'] = self.game.state['trace']
        state['current_player'] = self.game.round.current_player
        state['legal_actions'] = self.game.state['actions']
        return state