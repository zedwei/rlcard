# -*- coding: utf-8 -*-
''' Implement Tractor Game class
'''

import numpy as np
import functools
from heapq import merge

from rlcard.games.tractor import Player, Round, Judger
from rlcard.games.tractor.utils import tractor_sort_card, CARD_SCORE, ACTION_SPACE

class TractorGame(object):
    ''' Game class. This class will interact with outer environment.
    '''

    def __init__(self, allow_step_back=False):
        self.allow_step_back = allow_step_back
        self.np_random = np.random.RandomState()
        self.num_players = 4

    def init_game(self, predefined_hands=None):
        ''' Initialize all characters in the game and start round 1
        '''
        # initialize public variables
        self.winner_id = None
        self.history = []

        # initialize players
        self.players = [Player(num, self.np_random)
                        for num in range(self.num_players)]

        # initialize round to deal cards
        self.round = Round(self.np_random)
        self.round.initiate(self.players, predefined_hands)

        # initialize judger
        self.judger = Judger(self.players, self.np_random)

        for player in self.players:
            player.judger = self.judger

        # get state of first player
        player_id = self.round.current_player.player_id
        player = self.players[player_id]
        others_hands = self._get_others_current_hand(player)
        actions = self.judger.playable_cards[player_id]
        state = player.get_state(self.round.public, others_hands, actions)
        self.state = state

        return state, player_id

    def step(self, action):
        # perform action
        player = self.players[self.round.current_player.player_id]
        next_id, end_of_game = self.round.proceed_round(player, action, self.judger)
        self.round.current_player = self.players[next_id]
        self.judger.calc_playable_cards(player)

        if end_of_game:
            self.winner_id = []

            # TODO: hack to compute total score ignoring banker cards temporarily
            available_cards = self.round.dealer.deck[0:100]
            total_score = sum([CARD_SCORE[x] for x in available_cards if x in CARD_SCORE.keys()])

            if self.round.score[0] > total_score // 2:
                self.winner_id.extend([0, 2])
            if self.round.score[1] > total_score // 2:
                self.winner_id.extend([1, 3])

        # get next state
        state = self.get_state(next_id)
        self.state = state

        return state, next_id
        
    
    def get_state(self, player_id):
        player = self.players[player_id]
        others_hands = self._get_others_current_hand(player)
        if self.is_over():
            actions = None
        else:
            actions = player.available_actions(self.round.first_player, self.judger, self.round)
        state = player.get_state(self.round.public, others_hands, actions)

        return state

    def step_back(self):
        ''' Takes one step backward and restore to the last state
        '''
        raise NotImplementedError

    def get_player_num(self):
        ''' Retrun the number of players in the game
        '''
        return self.num_players

    def get_action_num(self):
        ''' Return the number of possible actions in the game
        '''
        return len(ACTION_SPACE.keys())

    def get_player_id(self):
        ''' Return the current player that will take actions soon
        '''
        return self.round.current_player.player_id

    def is_over(self):
        ''' Return whether the current game is over
        '''
        if self.winner_id is None:
            return False
        return True

    def _get_others_current_hand(self, player):
        player_down = self.players[(player.player_id + 1) % len(self.players)]
        player_front = self.players[(player.player_id + 2) % len(self.players)]
        player_up = self.players[(player.player_id - 1) % len(self.players)]

        # others_hand = []
        # others_hand.extend(player_up.current_hand)
        # others_hand.extend(player_front.current_hand)
        # others_hand.extend(player_down.current_hand)
        # # others_hand.extend(self.round.dealer.deck[100:108])
        
        # TODO: update logic to more restrictive card guess
        # player_down_hand = others_hand
        # player_front_hand = others_hand
        # player_up_hand = others_hand

        # start with all others' hands known
        player_down_hand = player_down.current_hand
        player_front_hand = player_front.current_hand
        player_up_hand = player_up.current_hand

        return [player_down_hand, player_front_hand, player_up_hand]
