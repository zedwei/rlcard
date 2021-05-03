# -*- coding: utf-8 -*-
''' Implement Doudizhu Judger class
'''
import collections
import numpy as np

from rlcard.games.tractor.utils import cards2str, CARD_RANK_STR

class TractorJudger(object):
    ''' Judger decides whether the round/game ends and return the winner of the round/game
    '''

    def __init__(self, players, np_random):
        ''' Initialize the Judger class for Tractor
        '''
        self.np_random = np_random
        self.playable_cards = [set() for _ in range(4)]
        self._recorded_removed_playable_cards = [[] for _ in range(4)]
        for player in players:
            player_id = player.player_id
            current_hand = cards2str(player.current_hand)
            self.playable_cards[player_id] = self.playable_cards_from_hand(current_hand)

    def calc_playable_cards(self, player):
        ''' Recalculate all legal cards the player can play according to his
        current hand.

        Args:
            player (TractorPlayer object): object of TractorPlayer

        Returns:
            list: list of string of playable cards
        '''
        player_id = player.player_id
        current_hand = cards2str(player.current_hand)

        self.playable_cards[player_id] = self.playable_cards_from_hand(current_hand)
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
    def playable_cards_from_hand(current_hand):
        ''' Get playable cards from hand
            current_hand: string joined with ','
        Returns:
            set: set of string of playable cards
        '''

        # first convert string represnetation to list
        current_hand = current_hand.split(',')

        cards_dict = collections.defaultdict(int)
        for card in current_hand:
            cards_dict[card] += 1
        cards_count = np.array([cards_dict[k] for k in CARD_RANK_STR])
        playable_cards = set()

        non_zero_indexes = np.argwhere(cards_count > 0)
        more_than_1_indexes = np.argwhere(cards_count > 1)

        # solo
        for i in non_zero_indexes:
            playable_cards.add(CARD_RANK_STR[i[0]])
        
        # pair
        for i in more_than_1_indexes:
            playable_cards.add(','.join([CARD_RANK_STR[i[0]], CARD_RANK_STR[i[0]]]))
        
        # Tractor type  1 - normal tractors with the same type
        for i in range(len(more_than_1_indexes)):
            if i==0:
                continue
            if ((more_than_1_indexes[i] == more_than_1_indexes[i-1]+1) and
               (CARD_RANK_STR[more_than_1_indexes[i-1][0]][1] == CARD_RANK_STR[more_than_1_indexes[i][0]][1])):
               playable_cards.add(','.join([
                   CARD_RANK_STR[more_than_1_indexes[i-1][0]],
                   CARD_RANK_STR[more_than_1_indexes[i-1][0]],
                   CARD_RANK_STR[more_than_1_indexes[i][0]],
                   CARD_RANK_STR[more_than_1_indexes[i][0]]
               ]))

        # Tractor type 2 - 2H2H2S2S, 2C2C2S2S, 2D2D2S2S
        if cards_dict['2S'] > 1:
            for second_card in ['2H', '2C', '2D']:
                if cards_dict[second_card] > 1:
                    playable_cards.add(','.join([second_card, second_card, '2S', '2S']))

        # Tractor type 3 - ASAS2H2H, ASAS2C2C, ASAS2D2D
        if cards_dict['AS'] > 1:
            for second_card in ['2H', '2C', '2D']:
                if cards_dict[second_card] > 1:
                    playable_cards.add(','.join(['AS', 'AS', second_card, second_card]))

        # Tractor type 4 - 2S2SBJBJ
        if (cards_dict['2S'] > 1 and cards_dict['BJ'] > 1):
            playable_cards.add(','.join(['2S', '2S', 'BJ', 'BJ']))

        return playable_cards

    @staticmethod
    def judge_payoffs(winner_id):
        payoffs = np.array([0, 0, 0, 0])
        for player_id in winner_id:
            payoffs[player_id] = 1
        return payoffs