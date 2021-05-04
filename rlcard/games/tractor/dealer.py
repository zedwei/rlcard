# -*- coding: utf-8 -*-
''' Implement Tractor Dealer class
'''
import functools

from rlcard.games.tractor.utils import tractor_sort_card, CARD_RANK_STR


class TractorDealer(object):
    ''' Dealer stores a deck of playing cards, remained cards
    holded by dealer, and can deal cards to players

    Note: deck variable means all the cards in a single game, and should be a list of Card objects.
    '''

    deck = []

    def __init__(self, np_random):
        ''' The dealer should have all the cards at the beginning of a game
        '''
        self.np_random = np_random
        # self.deck = init_108_deck()
        # self.deck.sort(key=functools.cmp_to_key(tractor_sort_card))

        self.deck = []
        self.deck.extend(CARD_RANK_STR)
        self.deck.extend(CARD_RANK_STR)

        self.banker = None

    def shuffle(self):
        ''' Shuffle the cards holded by dealer
        '''
        self.np_random.shuffle(self.deck)

    def deal_cards(self, players):
        ''' Deal specific number of cards to a specific player

        Args:
            player_id: the id of the player to be dealt cards
            num: number of cards to be dealt
        '''
        hand_num = 25
        for index, player in enumerate(players):
            current_hand = self.deck[index * hand_num : (index+1) * hand_num]
            current_hand.sort(key=functools.cmp_to_key(tractor_sort_card))
            player.current_hand = current_hand
            player.initial_hand = current_hand
    
    def deal_cards_and_determine_role(self, players):
        ''' Deal cards and determine banker according to players' hand

        Args:
            players (list): list of TractorPlayer objects

        Returns:
            int: banker's player_id
        '''
        # deal cards
        self.shuffle()
        self.deal_cards(players)
        players[0].role = 'banker'
        players[1].role = 'opponent'
        players[2].role = 'declarer'
        players[3].role = 'opponent'

        self.banker = players[0]
        return self.banker.player_id
