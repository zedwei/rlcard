# -*- coding: utf-8 -*-
''' Implement Tractor Player class
'''
import functools
import random
from rlcard.games.tractor import Dealer, Judger
from rlcard.games.tractor.utils import get_valid_cards, tractor_sort_card, get_suit, get_pass_cards_sequence
from rlcard.games.tractor.utils import SUIT_RANK

class TractorPlayer(object):
    ''' Player stores cards in the player's hand, and can determine the actions can be made according to the rules
    '''

    def __init__(self, player_id, np_random):
        ''' Every player should have a unique player id
        '''
        self.np_random = np_random
        self.player_id = player_id
        self.initial_hand = None
        self.current_hand = []
        self.role = ''
        self.played_cards = None

        # Record cards removed from self._current_hand for each play()
        # and restore cards back to self._current_hand when play_back()
        self._recorded_played_cards = []

    def get_state(self, public, others_hands, actions):
        state = {}
        state['current_hand'] = self.current_hand
        state['others_hand'] = others_hands
        state['current_round'] = public['current_round'].copy()
        state['offseted_current_round'] = self.get_offseted_current_round(public['current_round'])
        state['current_player_id'] = public['current_player_id']
        state['first_player_id'] = public['first_player_id']
        state['greater_player_id'] = public['greater_player_id']
        state['guessed_others_hand'] = self.guess_othershand(public['suit_avail'], public['remaining_cards'], public['banker_id'], public['banker_cards'])
        state['remaining_cards'] = self.guess_banker(public['remaining_cards'], public['banker_id'], public['banker_cards'])
        state['score'] = public['score']
        state['actions'] = actions

        return state

    def available_actions(self, first_player=None, judger=None):
        ''' Get the actions can be made based on the rules

        Returns:
            list: a list of available orders
        '''
        actions = []
        playable_cards = judger.get_playable_cards(self)
        if first_player == self:
            actions = playable_cards
        else:
            actions = get_valid_cards(first_player, playable_cards)
        return actions

    def play(self, action, first_player=None, greater_player=None, judger=None, trump=None):
        ''' Perfrom action

        Args:
            action (list of card string): specific action
            greater_player (Tractor object): The player who played current biggest cards.

        Returns:
            object of TractorPlayer: If there is a new greater_player, return it, if not, return None
            string: cards played
        '''
        removed_cards = []
        # pass or scor
        if (action[0][0] == 'p' or action[0][0] == 's'):
            is_get_score = False if action[0][0] == 'p' else True
            second_suit = SUIT_RANK[action[0][5]]
            suit_candidate = [0,1,2,3,4]

            target_hand = first_player.played_cards

            target_suit = get_suit(target_hand[0])
            suit_sequence = [target_suit]
            suit_candidate.remove(target_suit)

            if second_suit in suit_candidate:
                suit_sequence.append(second_suit)
                suit_candidate.remove(second_suit)
            
            random.shuffle(suit_candidate)
            suit_sequence.extend(suit_candidate)

            if len(target_hand) == 1 or len(target_hand) == 2 or len(target_hand) == 4:
                # Single
                # Current player MUST NOT have any card with the same suit according to how actions are picked
                # Pair
                # Current player MUST NOT have any pairs with the same suit
                # Tractor
                # Current player MUST NOT have any tractors with the same suit
                playable_cards = judger.get_playable_cards(self)
                sorted_card_list = get_pass_cards_sequence(self.current_hand, playable_cards, is_get_score, len(target_hand) >= 2, suit_sequence, trump)
                for i in range(len(target_hand)):
                    removed_cards.append(sorted_card_list[i])
                    self.current_hand.remove(sorted_card_list[i])
            else:
                raise NotImplementedError

            self._recorded_played_cards.append(removed_cards)
            self.played_cards = removed_cards
            return (greater_player, self.played_cards)
        else:
            # action matches greater_player card type
            for play_card in action:
                for _, remain_card in enumerate(self.current_hand):
                    if play_card == remain_card:
                        removed_cards.append(self.current_hand[_])
                        self.current_hand.remove(self.current_hand[_])
                        break
            
            if len(removed_cards) == 0:
                print(self.initial_hand)
                print(self.current_hand)
                print(action)
                print(judger.get_playable_cards(self))
                raise Exception("Can't find action cards {} in current_hand {}".format(action, self.current_hand))
                
            self._recorded_played_cards.append(removed_cards)
            self.played_cards = removed_cards

            greater_player_cards = greater_player.played_cards
            if (greater_player == None or tractor_sort_card(action[0], greater_player_cards[0])) > 0:
                return (self, self.played_cards)
            else:
                return (greater_player, self.played_cards)
    
    def get_offseted_current_round(self, current_round):
        offseted_current_round = [None, None, None]
        player_down = (self.player_id + 1) % 4
        player_front = (self.player_id + 2) % 4
        player_up = (self.player_id + 3) % 4
        offseted_current_round[0] = current_round[player_down]
        offseted_current_round[1] = current_round[player_front]
        offseted_current_round[2] = current_round[player_up]
        return offseted_current_round

    def guess_othershand(self, suit_avail, remaining_cards, banker_id, banker_cards):
        othershand = []

        cards = remaining_cards.copy()
        for card in self.current_hand:
            cards.remove(card)

        if self.player_id == banker_id:
            for card in banker_cards:
                cards.remove(card)

        player_ids =  [(self.player_id + 1) % 4, (self.player_id + 2) % 4, (self.player_id + 3) % 4]
        for player_id in player_ids:
            available_cards = []
            for card in cards:
                if suit_avail[player_id][get_suit(card)]:
                    available_cards.append(card)
            # available_cards.sort(key=functools.cmp_to_key(tractor_sort_card))
            othershand.append(available_cards)

        return othershand

    def guess_banker(self, remaining_cards, banker_id, banker_cards):
        if self.player_id == banker_id:
            return banker_cards
        else:
            cards = remaining_cards.copy()
            for card in self.current_hand:
                cards.remove(card)
            return cards