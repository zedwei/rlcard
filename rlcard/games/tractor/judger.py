# -*- coding: utf-8 -*-
''' Implement Doudizhu Judger class
'''
import collections
import numpy as np

from rlcard.games.tractor.utils import CARD_RANK_STR

class TractorJudger(object):
    ''' Judger decides whether the round/game ends and return the winner of the round/game
    '''

    def __init__(self, players, trump, np_random):
        ''' Initialize the Judger class for Tractor
        '''
        self.np_random = np_random
        self.trump = trump
        self.playable_cards = [[] for _ in range(4)]
        self._recorded_removed_playable_cards = [[] for _ in range(4)]
        for player in players:
            player_id = player.player_id
            current_hand = player.current_hand
            self.playable_cards[player_id] = self.playable_cards_from_hand(current_hand, self.trump)

    def calc_playable_cards(self, player):
        ''' Recalculate all legal cards the player can play according to his
        current hand.

        Args:
            player (TractorPlayer object): object of TractorPlayer

        Returns:
            list: list of string of playable cards
        '''
        player_id = player.player_id
        current_hand = player.current_hand

        self.playable_cards[player_id] = self.playable_cards_from_hand(current_hand, self.trump)
        return self.playable_cards[player_id]

    def get_playable_cards(self, player):
        ''' Provide all legal cards the player can play according to his
        current hand.

        Args:
            player (DoudizhuPlayer object): object of DoudizhuPlayer
            init_flag (boolean): For the first time, set it True to accelerate
              the preocess.

        Returns:
            list: list of string of playable cards
        '''
        return self.playable_cards[player.player_id]
        
    @staticmethod
    def playable_cards_from_hand(current_hand, trump):
        ''' Get playable cards from hand
            current_hand: list of st ring
        Returns:
            set: set of string of playable cards
        '''

        playable_cards = []

        cards_dict = collections.defaultdict(int)
        for card in current_hand:
            cards_dict[card] += 1

        non_zero_indexes = []
        more_than_1_indexes = []
        for i in range(len(CARD_RANK_STR)):
            card_count = cards_dict[CARD_RANK_STR[i]]
            if card_count >= 1:
                non_zero_indexes.append(i)
            if card_count >= 2:
                more_than_1_indexes.append(i)

        # solo
        for i in non_zero_indexes:
            playable_cards.append([CARD_RANK_STR[i]])
        
        # pair
        for i in more_than_1_indexes:
            playable_cards.append([CARD_RANK_STR[i], CARD_RANK_STR[i]])
        
        # Tractor type  1 - normal tractors with the same type
        for i in range(len(more_than_1_indexes)):
            if i==0:
                continue
            if ((more_than_1_indexes[i] == more_than_1_indexes[i-1]+1) and
               (CARD_RANK_STR[more_than_1_indexes[i-1]][1] == CARD_RANK_STR[more_than_1_indexes[i]][1])):
               playable_cards.append([
                   CARD_RANK_STR[more_than_1_indexes[i-1]],
                   CARD_RANK_STR[more_than_1_indexes[i-1]],
                   CARD_RANK_STR[more_than_1_indexes[i]],
                   CARD_RANK_STR[more_than_1_indexes[i]]
               ])
            
            if ((more_than_1_indexes[i] == more_than_1_indexes[i-1]+2) and
               (CARD_RANK_STR[more_than_1_indexes[i-1]][1] == CARD_RANK_STR[more_than_1_indexes[i]][1]) and
               (CARD_RANK_STR[more_than_1_indexes[i]-1][0] == trump[0])):
                playable_cards.append([
                   CARD_RANK_STR[more_than_1_indexes[i-1]],
                   CARD_RANK_STR[more_than_1_indexes[i-1]],
                   CARD_RANK_STR[more_than_1_indexes[i]],
                   CARD_RANK_STR[more_than_1_indexes[i]]
               ])


        # Tractor type 2 - NSNSNJNJ, NHNHNJNJ, NCNCNJNJ, NDNDNJNJ
        if cards_dict['NJ'] > 1:
            for second_card in ['NS', 'NH', 'NC', 'ND']:
                if cards_dict[second_card] > 1:
                    playable_cards.append([second_card, second_card, 'NJ', 'NJ'])

        # Tractor type 3 - AJAJNSNS, AJAJNHNH, AJAJNCNC, AJAJNDND
        if cards_dict['AJ'] > 1:
            for second_card in ['NS', 'NH', 'NC', 'ND']:
                if cards_dict[second_card] > 1:
                    playable_cards.append(['AJ', 'AJ', second_card, second_card])

        # Tractor type 4 - NJNJBJBJ - covered by type 1
        # if (cards_dict['NJ'] > 1 and cards_dict['BJ'] > 1):
        #     playable_cards.append(['NJ', 'NJ', 'BJ', 'BJ'])

        # Tractor type 5 - NSNSBJBJ, NHNHBJBJ, NCNCBJBJ, NDNDBJBJ when no trump suit
        if (trump[1] == 'J' and cards_dict['BJ'] > 1):
            for first_card in ['NS', 'NH', 'NC', 'ND']:
                if cards_dict[first_card] > 1:
                    playable_cards.append([first_card, first_card, 'BJ', 'BJ'])

        # # AAK
        # for suit in ['S', 'H', 'C', 'D']:
        #     if (cards_dict['A'+suit] > 1 and cards_dict['K'+suit] > 0):
        #         playable_cards.append(['A'+suit, 'A'+suit, 'K'+suit])

        # # AKK
        # for suit in ['S', 'H', 'C', 'D']:
        #     if (cards_dict['A'+suit] > 0 and cards_dict['K'+suit] > 1):
        #         playable_cards.append(['A'+suit, 'K'+suit, 'K'+suit])

        # # AAQ & AQQ
        # if (trump[0] == 'K'):
        #     # AAQ
        #     for suit in ['S', 'H', 'C', 'D']:
        #         if (cards_dict['A'+suit] > 1 and cards_dict['Q'+suit] > 0):
        #             playable_cards.append(['A'+suit, 'A'+suit, 'Q'+suit])

        #     # AQQ
        #     for suit in ['S', 'H', 'C', 'D']:
        #         if (cards_dict['A'+suit] > 0 and cards_dict['Q'+suit] > 1):
        #             playable_cards.append(['A'+suit, 'Q'+suit, 'Q'+suit])


        return playable_cards

    @staticmethod
    def judge_payoffs(winner_id, scores):
        # use final score as payoff
        payoffs = np.array([0, 0, 0, 0])
        for player_id in range(4):
            payoffs[player_id] = scores[player_id % 2]
        return payoffs