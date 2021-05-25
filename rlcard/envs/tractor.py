import numpy as np
import functools

from rlcard.envs import Env
from rlcard.utils import reorganize
from rlcard.games.tractor.utils import reorganize_with_payoff_trace
from rlcard.games.tractor import Game
from rlcard.games.tractor.utils import encode_cards, ACTION_LIST, ACTION_SPACE, NUM_DICT


class TractorEnv(Env):
    ''' Tractor Environment
    '''

    def __init__(self, config):

        self.name = 'tractor'
        self.game = Game()
        super().__init__(config)
        self.state_shape = [9, 3, 72]

    def run(self, is_training=False, debug=False):
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

        if debug:
            for i in range(4):
                print(self.game.players[i].current_hand)

        # Loop to play the game
        trajectories[player_id].append(state)
        while not self.is_over():
            # Agent plays
            if not is_training:
                action, prob = self.agents[player_id].eval_step(state)
                if debug and player_id == 0:
                    print(','.join(self.game.players[player_id].current_hand))
                    print(ACTION_LIST[action])
                    probs = {ACTION_LIST[i]:prob[i] for i in range(len(prob)) if prob[i] != -100 and prob[i] != 0}
                    probs = {k: round(v, 4) for k, v in probs.items()}
                    probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                    print(probs)
                    print()
            else:
                action = self.agents[player_id].step(state)

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
        # Get payoffs after each round
        payoffs = self.get_payoffs()
        payoffs_with_trace = self.get_payoffs_trace()
        trajectories = reorganize_with_payoff_trace(trajectories, payoffs_with_trace, payoffs)

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
        obs = np.zeros((9, 3, 72), dtype=int)
        
        # for index in range(5):
        for index in range(8):
            obs[index][0] = np.ones(72, dtype=int)
        encode_cards(obs[0], state['current_hand'])
        
        # "guess" play - real scenario
        encode_cards(obs[1], state['guessed_others_hand'][0])
        encode_cards(obs[2], state['guessed_others_hand'][1])
        encode_cards(obs[3], state['guessed_others_hand'][2])

        # separatedly provide current round cards from each player
        for i in range(3):
            if state['offseted_current_round'][i] != None:
                encode_cards(obs[i+4], state['offseted_current_round'][i])

        # remaining cards, possible banker cards
        encode_cards(obs[7], state['remaining_cards'])

        # other features
        obs[8][0][NUM_DICT[self.game.round.trump[0]]] = 1

        extracted_state = {'obs': obs, 'legal_actions': self._get_legal_actions(), 'trump': self.game.round.trump}

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

    def reset_predefine_state(self, predefined_hands):
        '''
        Reset environment in with pre-defined player hands
        '''
        state, player_id = self.game.init_game(predefined_hands)
        return self._extract_state(state)