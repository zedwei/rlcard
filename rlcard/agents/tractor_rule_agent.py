import numpy as np
from rlcard.games.tractor.utils import ACTION_LIST, CARD_RANK_DICT, CARD_RANK_STR, calc_score

class TractorRuleAgent(object):
    ''' A random agent. Random agents is for running toy examples on the card games
    '''

    def __init__(self, action_num):
        ''' Initilize the random agent

        Args:
            action_num (int): The size of the ouput action space
        '''
        self.use_raw = False
        self.action_num = action_num

    @staticmethod
    def step(state):
        ''' Predict the action given the curent state in gerenerating training data.

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
        '''
        # print(state)
        first_player_in_round = True
        has_score = False
        for k in range(4, 7):
            for i in range(len(state['obs'][k][0])):
                if state['obs'][k][0][i] == 0:
                    first_player_in_round = False
                    if calc_score([CARD_RANK_STR[i]], state['trump']) > 0:
                        has_score = True
                        break

        if first_player_in_round or not has_score:
            return np.random.choice(state['legal_actions'])
            # return state['legal_actions'][0]

        else:
            actions = [ACTION_LIST[x].split(',') for x in state['legal_actions']]

            max_score = -3
            max_index = 0
            for i in range(len(actions)):
                if actions[i][0] == 'pass':
                    score = -1
                elif actions[i][0] == 'pass_score':
                    score = -2
                else:
                    score = CARD_RANK_DICT[actions[i][0]]

                score += (len(actions[i]) * 100)
                
                if score > max_score:
                    max_score = score
                    max_index = i

            return state['legal_actions'][max_index]

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation.
            Since the random agents are not trained. This function is equivalent to step function

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
            probs (list): The list of action probabilities
        '''

        probs = [0 for _ in range(self.action_num)]
        action = self.step(state)

        # if state['current_player_id'] == state['first_player_id']:
        #     for i in state['legal_actions']:
        #         probs[i] = 1/len(state['legal_actions'])
        # else:
        #     probs[action] = 1

        probs[action] = 1

        return action, probs
