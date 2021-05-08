# -*- coding: utf-8 -*-
''' Implement Tractor Player class
'''
from rlcard.games.tractor import Dealer, Judger
from rlcard.games.tractor.utils import get_valid_cards, is_same_suit, tractor_sort_card

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
        # state['deck'] = public['deck']
        # state['banker_cards'] = public['banker_cards']
        # state['banker_id'] = public['banker_id']
        # state['trace'] = public['trace'].copy()
        # state['self'] = self.player_id
        # state['initial_hand'] = self.initial_hand
        state['current_hand'] = self.current_hand
        state['others_hand'] = others_hands
        state['current_round'] = public['current_round'].copy()
        state['current_player_id'] = public['current_player_id']
        state['first_player_id'] = public['first_player_id']
        state['greater_player_id'] = public['greater_player_id']
        state['score'] = public['score']
        state['actions'] = actions
        return state

    def available_actions(self, first_player=None, judger=None, game_round=None):
        ''' Get the actions can be made based on the rules

        Returns:
            list: a list of available orders
        '''
        actions = []
        playable_cards = judger.get_playable_cards(self)
        if first_player == self:
            actions = playable_cards
        else:
            actions = get_valid_cards(first_player, playable_cards, game_round.trump)
        return actions

    def play(self, action, first_player=None, greater_player=None, judger=None, trump=None):
        ''' Perfrom action

        Args:
            action (string): specific action
            greater_player (Tractor object): The player who played current biggest cards.

        Returns:
            object of TractorPlayer: If there is a new greater_player, return it, if not, return None
            string: cards played
        '''
        removed_cards = []
        #if (action == 'pass' or action == 'pass_score'):
        if (action[0] == 'pass'):
            target_hand = first_player.played_cards
            if len(target_hand) == 1:
                # Single
                # Current player MUST NOT have any card with the same suit according to how actions are picked
                # Strategy: pick the 1st card in hand, and that card MUST NOT be the trump suit

                # if (len(self.current_hand) == 0):
                #     print(target_hand)
                removed_cards.append(self.current_hand[0])
                self.current_hand.remove(self.current_hand[0])

            elif len(target_hand) == 2:
                # Pair
                # Current player MUST NOT have any pairs with the same suit
                # Strategy: first exhaust same suit from smallest ranked card, then pick the global smallest
                
                cards_to_remove = 2
                for target_card in target_hand:
                    for _, remain_card in enumerate(self.current_hand):
                        if is_same_suit(target_card, remain_card, trump):
                            removed_cards.append(self.current_hand[_])
                            self.current_hand.remove(self.current_hand[_])
                            cards_to_remove -= 1
                            break
                for _ in range(cards_to_remove):
                    removed_cards.append(self.current_hand[0])
                    self.current_hand.remove(self.current_hand[0])

            elif len(target_hand) == 4:
                # Tractor
                # Current player MUST NOT have any tractors with the same suit
                # Strategy: first exhaust pairs with the same suit (right now is random), 
                # then singles from same suit, then global smallest
                # TODO: perf optimization

                # play pairs first
                playable_cards = judger.get_playable_cards(self)
                cards_to_remove = 4
                for cards in playable_cards:
                    if cards_to_remove == 0:
                        break
                    if len(cards) == 2:  # a pair
                        if is_same_suit(target_hand[0], cards[0], trump): # same suit
                            for cardstr in cards:
                                for _, remain_card in enumerate(self.current_hand):
                                    if cardstr == remain_card:
                                        removed_cards.append(self.current_hand[_])
                                        self.current_hand.remove(self.current_hand[_])
                                        cards_to_remove -= 1
                                        break

                # try to remove single with the same suit
                for _ in range(cards_to_remove):
                    for _, remain_card in enumerate(self.current_hand):
                        if is_same_suit(target_hand[0], remain_card, trump):
                            removed_cards.append(self.current_hand[_])
                            self.current_hand.remove(self.current_hand[_])
                            cards_to_remove -= 1
                            break

                # pick the global smallest
                for _ in range(cards_to_remove):
                    removed_cards.append(self.current_hand[0])
                    self.current_hand.remove(self.current_hand[0])

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
            self._recorded_played_cards.append(removed_cards)
            self.played_cards = removed_cards

            greater_player_cards = greater_player.played_cards
            if (greater_player == None or tractor_sort_card(action[0], greater_player_cards[0])) > 0:
                return (self, self.played_cards)
            else:
                return (greater_player, self.played_cards)
        