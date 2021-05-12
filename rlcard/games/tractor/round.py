# -*- coding: utf-8 -*-
''' Implement Tractor Round class
'''

import numpy as np
import functools

from rlcard.games.tractor import Dealer
from rlcard.games.tractor.utils import CARD_RANK_STR, CARD_SCORE

class TractorRound(object):
    ''' Round stores the id the ongoing round and can call other Classes' functions to keep the game running
    '''

    def __init__(self, np_random):
        ''' When the game starts, round id should be 1
        '''
        self.np_random = np_random
        self.trace = []

        self.greater_player = None
        self.first_player = None
        self.current_player = None
        self.dealer = Dealer(self.np_random)
        # self.deck_str = cards2str(self.dealer.deck)
        self.trump = '2S' # either '2S' or 'BJ' for simplicity
        self.score = [0, 0]
        self.score_trace = []
        self.current_round = [None, None, None, None]
        self.played_player_in_round = 0

    def initiate(self, players, predefined_hands=None):
        ''' Call dealer to deal cards and determine banker.

        Args:
            players (list): list of DoudizhuPlayer objects
        '''
        banker_id = self.dealer.deal_cards_and_determine_role(players, predefined_hands)
        self.banker_id = banker_id
        self.current_player = players[banker_id]
        self.first_player = self.current_player
        self.greater_player = self.current_player

        self.public = {'deck': self.dealer.deck[0:108],
                    #    'banker_cards': self.dealer.deck[100:108],
                       'banker_id': self.banker_id, 
                       'trump': self.trump,
                       'score': self.score, 
                       'score_trace' : self.score_trace,
                       'trace': self.trace, 
                       'current_round': self.current_round,
                       'current_player_id': self.current_player.player_id,
                       'first_player_id': self.first_player.player_id,
                       'greater_player_id': self.greater_player.player_id}

    def proceed_round(self, player, action, judger):
        ''' Call other Classes's functions to keep one round running

        Args:
            player (object): object of TractorPlayer
            action (str): string of legal specific action

        Returns:
            int: player id who plays next
            bool: if game is over
        '''
        # play cards
        (self.greater_player, played_cards) = player.play(action, self.first_player, self.greater_player, judger, self.trump)

        self.trace.append((player.player_id, played_cards))
        self.current_round[player.player_id] = played_cards
        self.played_player_in_round += 1

        end_of_game = False
        if self.played_player_in_round < 4:
            # current round isn't ended
            next_id = (player.player_id + 1) % 4
        else:
            # current round ends
            # calculate score in current round
            # score = 1.0 * self.calc_score_in_round() / 10
            score = self.calc_score_in_round()
            self.score[self.greater_player.player_id % 2] += score

            # add delta score to score_trace
            score_trace_element = [0, 0]
            score_trace_element[self.greater_player.player_id % 2] += score
            # score_trace_element[(self.greater_player.player_id + 1) % 2] -= score
            self.score_trace.append(score_trace_element)            

            # reset round status with next player
            next_id = self.greater_player.player_id
            self.reset(self.greater_player)

            if len(player.current_hand) == 0:
                end_of_game = True

        self.public['current_player_id'] = next_id
        self.public['first_player_id'] = self.first_player.player_id
        self.public['greater_player_id'] = self.greater_player.player_id
        
        return next_id, end_of_game

    def calc_score_in_round(self):
        cards = functools.reduce(lambda z,y : z + y, self.current_round)
        scores = [CARD_SCORE[x] for x in cards if x in CARD_SCORE.keys()]
        score_in_round = sum(scores)
        return score_in_round

    def reset(self, player):
        for i in range(4):
            self.current_round[i] = None
        # self.current_round = [None, None, None, None]
        self.played_player_in_round = 0
        self.first_player = self.greater_player
        self.current_player = self.greater_player