''' Tractor utils
'''
import numpy as np

from rlcard.core import Card

CARD_RANK_STR = ['3H', '4H', '5H', '6H', '7H', '8H', '9H', 'TH', 'JH', 'QH', 'KH', 'AH',
            '3C', '4C', '5C', '6C', '7C', '8C', '9C', 'TC', 'JC', 'QC', 'KC', 'AC',
            '3D', '4D', '5D', '6D', '7D', '8D', '9D', 'TD', 'JD', 'QD', 'KD', 'AD',
            '3S', '4S', '5S', '6S', '7S', '8S', '9S', 'TS', 'JS', 'QS', 'KS', 'AS',
            '2H', '2C', '2D', '2S', 'BJ', 'RJ']

CARD_RANK_DICT = {'3H': 0, '4H': 1, '5H': 2, '6H': 3, '7H': 4, '8H': 5, '9H': 6, 'TH': 7, 'JH': 8, 'QH': 9, 'KH': 10, 'AH': 11, '3C': 12, '4C': 13, '5C': 14, '6C': 15, '7C': 16, '8C': 17, '9C': 18, 'TC': 19, 'JC': 20, 'QC': 21, 'KC': 22, 'AC': 23, '3D': 24, '4D': 25, '5D': 26, '6D': 27, '7D': 28, '8D': 29, '9D': 30, 'TD': 31, 'JD': 32, 'QD': 33, 'KD': 34, 'AD': 35, '3S': 36, '4S': 37, '5S': 38, '6S': 39, '7S': 40, '8S': 41, '9S': 42, 'TS': 43, 'JS': 44, 'QS': 45, 'KS': 46, 'AS': 47, '2H': 48, '2C': 49, '2D': 50, '2S': 51, 'BJ': 52, 'RJ': 53}

CARD_VALUE = [103,104,105,106,107,108,109,110,111,112,113,114,
              203,204,205,206,207,208,209,210,211,212,213,214,
              303,304,305,306,307,308,309,310,311,312,313,314,
              403,404,405,406,407,408,409,410,411,412,413,414,
              501,501,501,502,503,504
             ]

CARD_SCORE = {
                '5S':  5, '5H':  5, '5C':  5, '5D' : 5,
                'TS': 10, 'TH': 10, 'TC': 10, 'TD': 10,
                'KS': 10, 'KH': 10, 'KC': 10, 'KD': 10
            }

SUIT_RANK = {
                'S' : 0,
                'H' : 1,
                'C' : 2,
                'D' : 3,
                'J' : 4
            }

ACTION_LIST = ['2C', '2C,2C', '2C,2C,2S,2S', '2D', '2D,2D', '2D,2D,2S,2S', '2H', '2H,2H', '2H,2H,2S,2S', '2S', '2S,2S', '2S,2S,BJ,BJ', '3C', '3C,3C', '3C,3C,4C,4C', '3D', '3D,3D', '3D,3D,4D,4D', '3H', '3H,3H', '3H,3H,4H,4H', '3S', '3S,3S', '3S,3S,4S,4S', '4C', '4C,4C', '4C,4C,5C,5C', '4D', '4D,4D', '4D,4D,5D,5D', '4H', '4H,4H', '4H,4H,5H,5H', '4S', '4S,4S', '4S,4S,5S,5S', '5C', '5C,5C', '5C,5C,6C,6C', '5D', '5D,5D', '5D,5D,6D,6D', '5H', '5H,5H', '5H,5H,6H,6H', '5S', '5S,5S', '5S,5S,6S,6S', '6C', '6C,6C', '6C,6C,7C,7C', '6D', '6D,6D', '6D,6D,7D,7D', '6H', '6H,6H', '6H,6H,7H,7H', '6S', '6S,6S', '6S,6S,7S,7S', '7C', '7C,7C', '7C,7C,8C,8C', '7D', '7D,7D', '7D,7D,8D,8D', '7H', '7H,7H', '7H,7H,8H,8H', '7S', '7S,7S', '7S,7S,8S,8S', '8C', '8C,8C', '8C,8C,9C,9C', '8D', '8D,8D', '8D,8D,9D,9D', '8H', '8H,8H', '8H,8H,9H,9H', '8S', '8S,8S', '8S,8S,9S,9S', '9C', '9C,9C', '9C,9C,TC,TC', '9D', '9D,9D', '9D,9D,TD,TD', '9H', '9H,9H', '9H,9H,TH,TH', '9S', '9S,9S', '9S,9S,TS,TS', 'AC', 'AC,AC', 'AD', 'AD,AD', 'AH', 'AH,AH', 'AS', 'AS,AS', 'AS,AS,2C,2C', 'AS,AS,2D,2D', 'AS,AS,2H,2H', 'BJ', 'BJ,BJ', 'BJ,BJ,RJ,RJ', 'JC', 'JC,JC', 'JC,JC,QC,QC', 'JD', 'JD,JD', 'JD,JD,QD,QD', 'JH', 'JH,JH', 'JH,JH,QH,QH', 'JS', 'JS,JS', 'JS,JS,QS,QS', 'KC', 'KC,KC', 'KC,KC,AC,AC', 'KD', 'KD,KD', 'KD,KD,AD,AD', 'KH', 'KH,KH', 'KH,KH,AH,AH', 'KS', 'KS,KS', 'KS,KS,AS,AS', 'QC', 'QC,QC', 'QC,QC,KC,KC', 'QD', 'QD,QD', 'QD,QD,KD,KD', 'QH', 'QH,QH', 'QH,QH,KH,KH', 'QS', 'QS,QS', 'QS,QS,KS,KS', 'RJ', 'RJ,RJ', 'TC', 'TC,TC', 'TC,TC,JC,JC', 'TD', 'TD,TD', 'TD,TD,JD,JD', 'TH', 'TH,TH', 'TH,TH,JH,JH', 'TS', 'TS,TS', 'TS,TS,JS,JS', 'pass', 'pass_score']

ACTION_SPACE = {'2C': 0, '2C,2C': 1, '2C,2C,2S,2S': 2, '2D': 3, '2D,2D': 4, '2D,2D,2S,2S': 5, '2H': 6, '2H,2H': 7, '2H,2H,2S,2S': 8, '2S': 9, '2S,2S': 10, '2S,2S,BJ,BJ': 11, '3C': 12, '3C,3C': 13, '3C,3C,4C,4C': 14, '3D': 15, '3D,3D': 16, '3D,3D,4D,4D': 17, '3H': 18, '3H,3H': 19, '3H,3H,4H,4H': 20, '3S': 21, '3S,3S': 22, '3S,3S,4S,4S': 23, '4C': 24, '4C,4C': 25, '4C,4C,5C,5C': 26, '4D': 27, '4D,4D': 28, '4D,4D,5D,5D': 29, '4H': 30, '4H,4H': 31, '4H,4H,5H,5H': 32, '4S': 33, '4S,4S': 34, '4S,4S,5S,5S': 35, '5C': 36, '5C,5C': 37, '5C,5C,6C,6C': 38, '5D': 39, '5D,5D': 40, '5D,5D,6D,6D': 41, '5H': 42, '5H,5H': 43, '5H,5H,6H,6H': 44, '5S': 45, '5S,5S': 46, '5S,5S,6S,6S': 47, '6C': 48, '6C,6C': 49, '6C,6C,7C,7C': 50, '6D': 51, '6D,6D': 52, '6D,6D,7D,7D': 53, '6H': 54, '6H,6H': 55, '6H,6H,7H,7H': 56, '6S': 57, '6S,6S': 58, '6S,6S,7S,7S': 59, '7C': 60, '7C,7C': 61, '7C,7C,8C,8C': 62, '7D': 63, '7D,7D': 64, '7D,7D,8D,8D': 65, '7H': 66, '7H,7H': 67, '7H,7H,8H,8H': 68, '7S': 69, '7S,7S': 70, '7S,7S,8S,8S': 71, '8C': 72, '8C,8C': 73, '8C,8C,9C,9C': 74, '8D': 75, '8D,8D': 76, '8D,8D,9D,9D': 77, '8H': 78, '8H,8H': 79, '8H,8H,9H,9H': 80, '8S': 81, '8S,8S': 82, '8S,8S,9S,9S': 83, '9C': 84, '9C,9C': 85, '9C,9C,TC,TC': 86, '9D': 87, '9D,9D': 88, '9D,9D,TD,TD': 89, '9H': 90, '9H,9H': 91, '9H,9H,TH,TH': 92, '9S': 93, '9S,9S': 94, '9S,9S,TS,TS': 95, 'AC': 96, 'AC,AC': 97, 'AD': 98, 'AD,AD': 99, 'AH': 100, 'AH,AH': 101, 'AS': 102, 'AS,AS': 103, 'AS,AS,2C,2C': 104, 'AS,AS,2D,2D': 105, 'AS,AS,2H,2H': 106, 'BJ': 107, 'BJ,BJ': 108, 'BJ,BJ,RJ,RJ': 109, 'JC': 110, 'JC,JC': 111, 'JC,JC,QC,QC': 112, 'JD': 113, 'JD,JD': 114, 'JD,JD,QD,QD': 115, 'JH': 116, 'JH,JH': 117, 'JH,JH,QH,QH': 118, 'JS': 119, 'JS,JS': 120, 'JS,JS,QS,QS': 121, 'KC': 122, 'KC,KC': 123, 'KC,KC,AC,AC': 124, 'KD': 125, 'KD,KD': 126, 'KD,KD,AD,AD': 127, 'KH': 128, 'KH,KH': 129, 'KH,KH,AH,AH': 130, 'KS': 131, 'KS,KS': 132, 'KS,KS,AS,AS': 133, 'QC': 134, 'QC,QC': 135, 'QC,QC,KC,KC': 136, 'QD': 137, 'QD,QD': 138, 'QD,QD,KD,KD': 139, 'QH': 140, 'QH,QH': 141, 'QH,QH,KH,KH': 142, 'QS': 143, 'QS,QS': 144, 'QS,QS,KS,KS': 145, 'RJ': 146, 'RJ,RJ': 147, 'TC': 148, 'TC,TC': 149, 'TC,TC,JC,JC': 150, 'TD': 151, 'TD,TD': 152, 'TD,TD,JD,JD': 153, 'TH': 154, 'TH,TH': 155, 'TH,TH,JH,JH': 156, 'TS': 157, 'TS,TS': 158, 'TS,TS,JS,JS': 159, 'pass': 160, 'pass_score': 161}

# def init_108_deck():
#     ''' Initialize two standard decks of cards, 54 * 2 = 108 cards

#     Returns:
#         (list): A list of Card object
#     '''
#     suit_list = ['S', 'H', 'D', 'C']
#     rank_list = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
#     res = [Card(suit, rank) for suit in suit_list for rank in rank_list]
#     res.append(Card('BJ', ''))
#     res.append(Card('RJ', ''))
#     res = np.repeat(res, 2)
#     return res.tolist()

# def cards2str(cards):
#     ''' Get the corresponding string representation of cards

#     Args:
#         cards (list): list of Card objects

#     Returns:
#         string: string representation of cards
#     '''
#     return ','.join(str(card) for card in cards)

def hand2type(cards, trump):
    ''' Get type of the hand
    Args:
        cards: list of string
        trump: string of current trump i.e. either '2S' or 'BJ'

    Returns:
        int: hand type id

    Note:
        10: Spade single, 11: Heart single, 12: Club single, 13: Diamond single, 14: Trump suit single
        20: Spade pair, 21: Heart pair, 22: Club pair, 23: Diamond pair, 24: Trump suit pair
        40: Spade tractor, 41: ..., 42: ..., 43: ..., 44: ...
    '''
    # check single or pair or tractor
    ret_type = len(cards) * 10
    
    # check suit
    if (cards[0][0] == trump[0] or cards[0][1] == trump[1] or cards[0][1] == 'J'):
        ret_type += 4
    else:
        ret_type += SUIT_RANK[cards[0][1]]
    
    return ret_type

def trump_type_to_win(target_type):
    return target_type // 10 * 10 + 4

def is_same_suit(card_1, card_2, trump):
    suit = []
    for card in [card_1, card_2]:
        if (card[0] == trump[0] or card[1] == trump[1] or card[1] == 'J'):
            suit.append(4)
        else:
            suit.append(SUIT_RANK[card[1]])
    return suit[0] == suit[1]

def tractor_sort_card(card_1, card_2):
    ''' Compare the rank of two cards of Card object

    Args:
        card_1 (object): object of Card
        card_2 (object): object of card
    '''
    index_1 = CARD_RANK_DICT[card_1]
    index_2 = CARD_RANK_DICT[card_2]
    if index_1 > index_2:
        return 1
    if index_1 < index_2:
        return -1
    return 0

# def compare_card_str(card_1, card_2):
#     index_1 = CARD_RANK_STR.index(card_1)
#     index_2 = CARD_RANK_STR.index(card_2)
#     if CARD_VALUE[index_1] > CARD_VALUE[index_2]:
#         return 1
#     elif CARD_VALUE[index_1] < CARD_VALUE[index_2]:
#         return -1
#     else:
#         return 0

def get_valid_cards(first_player, playable_cards, trump):
    '''
        Returns:
        list: list of string of valid cards based on first player's card type

        Note:
        'pass' means pass without scores
        'pass_score' means pass with scores
    '''
    valid_cards = []
    target_hand = first_player.played_cards # list of string
    target_type = hand2type(target_hand, trump)
    playable_cards_type = [(cards, hand2type(cards, trump)) for cards in playable_cards]

    matched_playable_cards = [x[0] for x in playable_cards_type if x[1] == target_type]
    if len(matched_playable_cards) > 0:
        # matched type exist, have to pick from one of them
        valid_cards.extend(matched_playable_cards)
    else:
        # matched type doesn't exist, have 3 choices:
        # - use trump cards to beat ONLY WHEN there is no matched type
        same_type = [x[0] for x in playable_cards_type if x[1] % 10 == target_type % 10]
        if len(same_type) == 0:
            trump_type = trump_type_to_win(target_type)
            if (trump_type != target_type):
                matched_trump_cards = [x[0] for x in playable_cards_type if x[1] == trump_type]
                valid_cards.extend(matched_trump_cards)

        # - 'pass' without feeding any score
        valid_cards.append(['pass'])

        # - 'pass_score'
        # valid_cards.append(['pass_score'])

    return valid_cards

def encode_cards(plane, cards):
    ''' Encode cards and represerve it into plane.

    Args:
        cards (string): list of string of cards
    '''
    if not cards:
        return None
    
    for card in cards:
        rank = CARD_RANK_DICT[card]
        if plane[0][rank] == 1:
            plane[0][rank] = 0
            plane[1][rank] = 1
        elif plane[1][rank] == 1:
            plane[1][rank] = 0
            plane[2][rank] = 1
    