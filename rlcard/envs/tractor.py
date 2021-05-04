import numpy as np
import functools

from rlcard.envs import Env
from rlcard.games.tractor import Game
from rlcard.games.tractor.utils import encode_cards, ACTION_LIST, ACTION_SPACE, cards2str


class TractorEnv(Env):
    ''' Tractor Environment
    '''

    def __init__(self, config):

        self.name = 'tractor'
        self.game = Game()
        super().__init__(config)
        self.state_shape = [3, 3, 54]

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
        obs = np.zeros((3, 3, 54), dtype=int)
        for index in range(3):
            obs[index][0] = np.ones(54, dtype=int)
        encode_cards(obs[0], state['current_hand'])
        encode_cards(obs[1], state['others_hand'][0])
        
        # self._encode_cards(obs[2], state['others_hand'][1])
        # self._encode_cards(obs[3], state['others_hand'][2])
        
        current_round = [x.split(',') for x in state['current_round'] if x != None]
        if (len(current_round) > 0):
            current_round = functools.reduce(lambda z,y : z + y, current_round)
            current_round = ','.join(current_round)
            encode_cards(obs[2], current_round)

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
        return self.game.judger.judge_payoffs(self.game.winner_id)
        
    def _decode_action(self, action_id):
        ''' Decode Action id to the action in the game.

        Args:
            action_id (int): The id of the action

        Returns:
            (string): The action that will be passed to the game engine.

        Note: Must be implemented in the child class.
        '''
        return ACTION_LIST[action_id]

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
                legal_action_id.append(ACTION_SPACE[action])
        return legal_action_id

    def get_perfect_information(self):
        ''' Get the perfect information of the current state

        Returns:
            (dict): A dictionary of all the perfect information of the current state

        Note: Must be implemented in the child class.
        '''
        state = {}
        state['hand_cards'] = [cards2str(player.current_hand) for player in self.game.players]
        state['banker_id'] = self.game.state['banker_id']
        state['trace'] = self.game.state['trace']
        state['current_player'] = self.game.round.current_player
        state['legal_actions'] = self.game.state['actions']
        return state