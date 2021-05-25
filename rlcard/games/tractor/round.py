# -*- coding: utf-8 -*-
''' Implement Tractor Round class
'''

import numpy as np
import functools
import random

from rlcard.games.tractor import Dealer
from rlcard.games.tractor.utils import TRUMP_CANDIDATE_STR
from rlcard.games.tractor.utils import is_same_suit, calc_score, get_suit

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
 
        self.score = [0, 0]
        self.score_trace = []
        self.current_round = [None, None, None, None]
        self.played_player_in_round = 0

    def initiate(self, players, predefined_hands=None, predefined_trump=None):
        ''' Call dealer to deal cards and determine banker.

        Args:
            players (list): list of DoudizhuPlayer objects
        '''

       # self.trump = '2S'
        if predefined_trump != None:
            self.trump = predefined_trump
        else:
            self.trump = random.choice(TRUMP_CANDIDATE_STR)
        self.dealer = Dealer(self.trump, self.np_random)

        banker_id = self.dealer.deal_cards_and_determine_role(players, predefined_hands)
        self.banker_id = banker_id
        self.current_player = players[banker_id]
        self.first_player = self.current_player
        self.greater_player = self.current_player

        self.suit_avail = [[True, True, True, True, True], [True, True, True, True, True], [True, True, True, True, True], [True, True, True, True, True]]
        self.remaining_cards = []
        self.remaining_cards.extend(self.dealer.deck)

        self.public = {'trump': self.trump,
                       'banker_id': self.banker_id,
                       'banker_cards': self.dealer.deck[100:108],
                       'score': self.score, 
                       'score_trace' : self.score_trace,
                       'trace': self.trace, 
                       'current_round': self.current_round,
                       'current_player_id': self.current_player.player_id,
                       'first_player_id': self.first_player.player_id,
                       'greater_player_id': self.greater_player.player_id,
                       'suit_avail': self.suit_avail,
                       'remaining_cards': self.remaining_cards}

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

        # update the overall remaining cards list
        for card in played_cards:
            self.remaining_cards.remove(
                card)

        # update missing suit info of the current player
        if (self.played_player_in_round != 0):
            if not is_same_suit(self.current_round[self.first_player.player_id][0], played_cards[-1]):
                missing_suit = get_suit(self.current_round[self.first_player.player_id][0])
                self.suit_avail[player.player_id][missing_suit] = False

        self.played_player_in_round += 1

        end_of_game = False
        if self.played_player_in_round < 4:
            # current round isn't ended
            next_id = (player.player_id + 1) % 4
        else:
            # end of current round
            
            # calculate score in current round
            score = self.calc_score_in_round()

            # decide if it's end of the game
            if len(player.current_hand) == 0:
                end_of_game = True
                # calculate score from banker
                banker_score = calc_score(self.dealer.deck[100:108], self.trump)
                banker_score = banker_score * (2 ** len(self.greater_player.played_cards))
                score = score + banker_score

            self.score[self.greater_player.player_id % 2] += score

            # add delta score to score_trace
            score_trace_element = [0, 0]
            score_trace_element[self.greater_player.player_id % 2] += score
            # score_trace_element[(self.greater_player.player_id + 1) % 2] -= score
            self.score_trace.append(score_trace_element)            

            # reset round status with next player
            next_id = self.greater_player.player_id
            self.reset(self.greater_player)

        self.public['current_player_id'] = next_id
        self.public['first_player_id'] = self.first_player.player_id
        self.public['greater_player_id'] = self.greater_player.player_id
        
        return next_id, end_of_game

    def calc_score_in_round(self):
        cards = functools.reduce(lambda z,y : z + y, self.current_round)
        return calc_score(cards, self.trump)

    def reset(self, player):
        for i in range(4):
            self.current_round[i] = None
        self.played_player_in_round = 0
        self.first_player = self.greater_player
        self.current_player = self.greater_player