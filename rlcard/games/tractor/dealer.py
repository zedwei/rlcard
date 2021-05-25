# -*- coding: utf-8 -*-
''' Implement Tractor Dealer class
'''
import functools

from rlcard.games.tractor.utils import tractor_sort_card, CARD_STR


class TractorDealer(object):
    ''' Dealer stores a deck of playing cards, remained cards
    holded by dealer, and can deal cards to players

    Note: deck variable means all the cards in a single game, and should be a list of Card objects.
    '''

    deck = []

    def __init__(self, trump, np_random):
        ''' The dealer should have all the cards at the beginning of a game
        '''
        self.np_random = np_random

        # two decks of cards
        self.deck = []
        self.deck.extend(CARD_STR)
        self.deck.extend(CARD_STR)

        for i in range(len(self.deck)):
            if self.deck[i][0] == trump[0]:
                self.deck[i] = 'N' + self.deck[i][1]
            if self.deck[i][1] == trump[1]:
                self.deck[i] = self.deck[i][0] + 'J'

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
        # exclude 8 banker cards
        hand_num = (len(self.deck) - 8) // 4

        for index, player in enumerate(players):
            current_hand = self.deck[index * hand_num : (index+1) * hand_num]
            current_hand.sort(key=functools.cmp_to_key(tractor_sort_card))
            player.current_hand = current_hand
            player.initial_hand = current_hand.copy()
    
    def deal_cards_and_determine_role(self, players, predefined_hands=None):
        ''' Deal cards and determine banker according to players' hand

        Args:
            players (list): list of TractorPlayer objects

        Returns:
            int: banker's player_id
        '''
        # deal cards
        if predefined_hands == None:
            self.shuffle()
            self.deal_cards(players)
        else:
            for player_id in range(4):
                current_hand = predefined_hands[player_id]
                current_hand.sort(key=functools.cmp_to_key(tractor_sort_card))
                players[player_id].current_hand = current_hand

        # Assume player[0] is always the banker
        # TODO: if later on multiple agents are trained, randomization needs to be added here
        players[0].role = 'banker'
        players[1].role = 'opponent'
        players[2].role = 'declarer'
        players[3].role = 'opponent'
        self.banker = players[0]

        return self.banker.player_id
