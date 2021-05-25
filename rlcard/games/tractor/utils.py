''' Tractor utils
'''
import numpy as np
from tqdm import tqdm

from rlcard.core import Card

CARD_STR = [
            '2S', '3S', '4S', '5S', '6S', '7S', '8S', '9S', 'TS', 'JS', 'QS', 'KS', 'AS',
            '2H', '3H', '4H', '5H', '6H', '7H', '8H', '9H', 'TH', 'JH', 'QH', 'KH', 'AH',
            '2C', '3C', '4C', '5C', '6C', '7C', '8C', '9C', 'TC', 'JC', 'QC', 'KC', 'AC',
            '2D', '3D', '4D', '5D', '6D', '7D', '8D', '9D', 'TD', 'JD', 'QD', 'KD', 'AD',
            'BJ', 'RJ']

CARD_RANK_STR = [
            '2S', '3S', '4S', '5S', '6S', '7S', '8S', '9S', 'TS', 'JS', 'QS', 'KS', 'AS',
            '2H', '3H', '4H', '5H', '6H', '7H', '8H', '9H', 'TH', 'JH', 'QH', 'KH', 'AH',
            '2C', '3C', '4C', '5C', '6C', '7C', '8C', '9C', 'TC', 'JC', 'QC', 'KC', 'AC',
            '2D', '3D', '4D', '5D', '6D', '7D', '8D', '9D', 'TD', 'JD', 'QD', 'KD', 'AD',
            '2J', '3J', '4J', '5J', '6J', '7J', '8J', '9J', 'TJ', 'JJ', 'QJ', 'KJ', 'AJ',
            'NS', 'NH', 'NC', 'ND', 'NJ', 'BJ', 'RJ']

TRUMP_CANDIDATE_STR = [
            '2S', '3S', '4S', '5S', '6S', '7S', '8S', '9S', 'TS', 'JS', 'QS', 'KS', 'AS',
            '2H', '3H', '4H', '5H', '6H', '7H', '8H', '9H', 'TH', 'JH', 'QH', 'KH', 'AH',
            '2C', '3C', '4C', '5C', '6C', '7C', '8C', '9C', 'TC', 'JC', 'QC', 'KC', 'AC',
            '2D', '3D', '4D', '5D', '6D', '7D', '8D', '9D', 'TD', 'JD', 'QD', 'KD', 'AD',
            '2J', '3J', '4J', '5J', '6J', '7J', '8J', '9J', 'TJ', 'JJ', 'QJ', 'KJ', 'AJ'
            ]

CARD_RANK_DICT = {'2S': 0, '3S': 1, '4S': 2, '5S': 3, '6S': 4, '7S': 5, '8S': 6, '9S': 7, 'TS': 8, 'JS': 9, 'QS': 10, 'KS': 11, 'AS': 12, '2H': 13, '3H': 14, '4H': 15, '5H': 16, '6H': 17, '7H': 18, '8H': 19, '9H': 20, 'TH': 21, 'JH': 22, 'QH': 23, 'KH': 24, 'AH': 25, '2C': 26, '3C': 27, '4C': 28, '5C': 29, '6C': 30, '7C': 31, '8C': 32, '9C': 33, 'TC': 34, 'JC': 35, 'QC': 36, 'KC': 37, 'AC': 38, '2D': 39, '3D': 40, '4D': 41, '5D': 42, '6D': 43, '7D': 44, '8D': 45, '9D': 46, 'TD': 47, 'JD': 48, 'QD': 49, 'KD': 50, 'AD': 51, '2J': 52, '3J': 53, '4J': 54, '5J': 55, '6J': 56, '7J': 57, '8J': 58, '9J': 59, 'TJ': 60, 'JJ': 61, 'QJ': 62, 'KJ': 63, 'AJ': 64, 'NS': 65, 'NH': 66, 'NC': 67, 'ND': 68, 'NJ': 69, 'BJ': 70, 'RJ': 71}

CARD_SCORE = {
                '5S':  5, '5H':  5, '5C':  5, '5D' : 5, '5J' : 5,
                'TS': 10, 'TH': 10, 'TC': 10, 'TD': 10, 'TJ' : 10,
                'KS': 10, 'KH': 10, 'KC': 10, 'KD': 10, 'KJ' : 10,
            }

CARD_SCORE_5 = {
                '5S':  5, '5H':  5, '5C':  5, '5D' : 5, '5J' : 5,
                'TS': 10, 'TH': 10, 'TC': 10, 'TD': 10, 'TJ' : 10,
                'KS': 10, 'KH': 10, 'KC': 10, 'KD': 10, 'KJ' : 10,
                'NS':  5, 'NH':  5, 'NC':  5, 'ND' : 5, 'NJ' : 5,
            }

CARD_SCORE_TK = {
                '5S':  5, '5H':  5, '5C':  5, '5D' : 5, '5J' : 5,
                'TS': 10, 'TH': 10, 'TC': 10, 'TD': 10, 'TJ' : 10,
                'KS': 10, 'KH': 10, 'KC': 10, 'KD': 10, 'KJ' : 10,
                'NS': 10, 'NH': 10, 'NC': 10, 'ND': 10, 'NJ' : 10,
            }


SUIT_RANK = {
                'S' : 0,
                'H' : 1,
                'C' : 2,
                'D' : 3,
                'J' : 4,
            }

NUM_DICT = {
    '2': 0,
    '3': 1,
    '4': 2,
    '5': 3,
    '6': 4,
    '7': 5,
    '8': 6,
    '9': 7,
    'T': 8,
    'J': 9,
    'Q': 10,
    'K': 11,
    'A': 12,
}

ACTION_LIST = ['3H', '4H', '5H', '6H', '7H', '8H', '9H', 'TH', 'JH', 'QH', 'KH', 'AH', '3C', '4C', '5C', '6C', '7C', '8C', '9C', 'TC', 'JC', 'QC', 'KC', 'AC', '3D', '4D', '5D', '6D', '7D', '8D', '9D', 'TD', 'JD', 'QD', 'KD', 'AD', '3J', '4J', '5J', '6J', '7J', '8J', '9J', 'TJ', 'JJ', 'QJ', 'KJ', 'AJ', 'NH', 'NC', 'ND', 'NJ', 'BJ', 'RJ', '3H,3H', '4H,4H', '5H,5H', '6H,6H', '7H,7H', '8H,8H', '9H,9H', 'TH,TH', 'JH,JH', 'QH,QH', 'KH,KH', 'AH,AH', '3C,3C', '4C,4C', '5C,5C', '6C,6C', '7C,7C', '8C,8C', '9C,9C', 'TC,TC', 'JC,JC', 'QC,QC', 'KC,KC', 'AC,AC', '3D,3D', '4D,4D', '5D,5D', '6D,6D', '7D,7D', '8D,8D', '9D,9D', 'TD,TD', 'JD,JD', 'QD,QD', 'KD,KD', 'AD,AD', '3J,3J', '4J,4J', '5J,5J', '6J,6J', '7J,7J', '8J,8J', '9J,9J', 'TJ,TJ', 'JJ,JJ', 'QJ,QJ', 'KJ,KJ', 'AJ,AJ', 'NH,NH', 'NC,NC', 'ND,ND', 'NJ,NJ', 'BJ,BJ', 'RJ,RJ', '3H,3H,4H,4H', '4H,4H,5H,5H', '5H,5H,6H,6H', '6H,6H,7H,7H', '7H,7H,8H,8H', '8H,8H,9H,9H', '9H,9H,TH,TH', 'TH,TH,JH,JH', 'JH,JH,QH,QH', 'QH,QH,KH,KH', 'KH,KH,AH,AH', '3C,3C,4C,4C', '4C,4C,5C,5C', '5C,5C,6C,6C', '6C,6C,7C,7C', '7C,7C,8C,8C', '8C,8C,9C,9C', '9C,9C,TC,TC', 'TC,TC,JC,JC', 'JC,JC,QC,QC', 'QC,QC,KC,KC', 'KC,KC,AC,AC', '3D,3D,4D,4D', '4D,4D,5D,5D', '5D,5D,6D,6D', '6D,6D,7D,7D', '7D,7D,8D,8D', '8D,8D,9D,9D', '9D,9D,TD,TD', 'TD,TD,JD,JD', 'JD,JD,QD,QD', 'QD,QD,KD,KD', 'KD,KD,AD,AD', '3J,3J,4J,4J', '4J,4J,5J,5J', '5J,5J,6J,6J', '6J,6J,7J,7J', '7J,7J,8J,8J', '8J,8J,9J,9J', '9J,9J,TJ,TJ', 'TJ,TJ,JJ,JJ', 'JJ,JJ,QJ,QJ', 'QJ,QJ,KJ,KJ', 'KJ,KJ,AJ,AJ', 'NJ,NJ,BJ,BJ', 'BJ,BJ,RJ,RJ', 'NH,NH,NJ,NJ', 'NC,NC,NJ,NJ', 'ND,ND,NJ,NJ', 'AJ,AJ,NH,NH', 'AJ,AJ,NC,NC', 'AJ,AJ,ND,ND', '2H', '2C', '2D', '2J', '2H,2H', '2C,2C', '2D,2D', '2J,2J', '2H,2H,4H,4H', '2C,2C,4C,4C', '2D,2D,4D,4D', '2J,2J,4J,4J', '2H,2H,3H,3H', '3H,3H,5H,5H', '2C,2C,3C,3C', '3C,3C,5C,5C', '2D,2D,3D,3D', '3D,3D,5D,5D', '2J,2J,3J,3J', '3J,3J,5J,5J', '4H,4H,6H,6H', '4C,4C,6C,6C', '4D,4D,6D,6D', '4J,4J,6J,6J', '5H,5H,7H,7H', '5C,5C,7C,7C', '5D,5D,7D,7D', '5J,5J,7J,7J', '6H,6H,8H,8H', '6C,6C,8C,8C', '6D,6D,8D,8D', '6J,6J,8J,8J', '7H,7H,9H,9H', '7C,7C,9C,9C', '7D,7D,9D,9D', '7J,7J,9J,9J', '8H,8H,TH,TH', '8C,8C,TC,TC', '8D,8D,TD,TD', '8J,8J,TJ,TJ', '9H,9H,JH,JH', '9C,9C,JC,JC', '9D,9D,JD,JD', '9J,9J,JJ,JJ', 'TH,TH,QH,QH', 'TC,TC,QC,QC', 'TD,TD,QD,QD', 'TJ,TJ,QJ,QJ', 'JH,JH,KH,KH', 'JC,JC,KC,KC', 'JD,JD,KD,KD', 'JJ,JJ,KJ,KJ', 'QH,QH,AH,AH', 'QC,QC,AC,AC', 'QD,QD,AD,AD', 'QJ,QJ,AJ,AJ', '3S', '4S', '5S', '6S', '7S', '8S', '9S', 'TS', 'JS', 'QS', 'KS', 'AS', 'NS', '3S,3S', '4S,4S', '5S,5S', '6S,6S', '7S,7S', '8S,8S', '9S,9S', 'TS,TS', 'JS,JS', 'QS,QS', 'KS,KS', 'AS,AS', 'NS,NS', '3S,3S,4S,4S', '4S,4S,5S,5S', '5S,5S,6S,6S', '6S,6S,7S,7S', '7S,7S,8S,8S', '8S,8S,9S,9S', '9S,9S,TS,TS', 'TS,TS,JS,JS', 'JS,JS,QS,QS', 'QS,QS,KS,KS', 'KS,KS,AS,AS', 'NS,NS,NJ,NJ', 'AJ,AJ,NS,NS', '2S', '2S,2S', '2S,2S,4S,4S', '2S,2S,3S,3S', '3S,3S,5S,5S', '4S,4S,6S,6S', '5S,5S,7S,7S', '6S,6S,8S,8S', '7S,7S,9S,9S', '8S,8S,TS,TS', '9S,9S,JS,JS', 'TS,TS,QS,QS', 'JS,JS,KS,KS', 'QS,QS,AS,AS', 'NS,NS,BJ,BJ', 'NH,NH,BJ,BJ', 'NC,NC,BJ,BJ', 'ND,ND,BJ,BJ', 'pass', 'pass_score']

ACTION_SPACE = {'3H': 0, '4H': 1, '5H': 2, '6H': 3, '7H': 4, '8H': 5, '9H': 6, 'TH': 7, 'JH': 8, 'QH': 9, 'KH': 10, 'AH': 11, '3C': 12, '4C': 13, '5C': 14, '6C': 15, '7C': 16, '8C': 17, '9C': 18, 'TC': 19, 'JC': 20, 'QC': 21, 'KC': 22, 'AC': 23, '3D': 24, '4D': 25, '5D': 26, '6D': 27, '7D': 28, '8D': 29, '9D': 30, 'TD': 31, 'JD': 32, 'QD': 33, 'KD': 34, 'AD': 35, '3J': 36, '4J': 37, '5J': 38, '6J': 39, '7J': 40, '8J': 41, '9J': 42, 'TJ': 43, 'JJ': 44, 'QJ': 45, 'KJ': 46, 'AJ': 47, 'NH': 48, 'NC': 49, 'ND': 50, 'NJ': 51, 'BJ': 52, 'RJ': 53, '3H,3H': 54, '4H,4H': 55, '5H,5H': 56, '6H,6H': 57, '7H,7H': 58, '8H,8H': 59, '9H,9H': 60, 'TH,TH': 61, 'JH,JH': 62, 'QH,QH': 63, 'KH,KH': 64, 'AH,AH': 65, '3C,3C': 66, '4C,4C': 67, '5C,5C': 68, '6C,6C': 69, '7C,7C': 70, '8C,8C': 71, '9C,9C': 72, 'TC,TC': 73, 'JC,JC': 74, 'QC,QC': 75, 'KC,KC': 76, 'AC,AC': 77, '3D,3D': 78, '4D,4D': 79, '5D,5D': 80, '6D,6D': 81, '7D,7D': 82, '8D,8D': 83, '9D,9D': 84, 'TD,TD': 85, 'JD,JD': 86, 'QD,QD': 87, 'KD,KD': 88, 'AD,AD': 89, '3J,3J': 90, '4J,4J': 91, '5J,5J': 92, '6J,6J': 93, '7J,7J': 94, '8J,8J': 95, '9J,9J': 96, 'TJ,TJ': 97, 'JJ,JJ': 98, 'QJ,QJ': 99, 'KJ,KJ': 100, 'AJ,AJ': 101, 'NH,NH': 102, 'NC,NC': 103, 'ND,ND': 104, 'NJ,NJ': 105, 'BJ,BJ': 106, 'RJ,RJ': 107, '3H,3H,4H,4H': 108, '4H,4H,5H,5H': 109, '5H,5H,6H,6H': 110, '6H,6H,7H,7H': 111, '7H,7H,8H,8H': 112, '8H,8H,9H,9H': 113, '9H,9H,TH,TH': 114, 'TH,TH,JH,JH': 115, 'JH,JH,QH,QH': 116, 'QH,QH,KH,KH': 117, 'KH,KH,AH,AH': 118, '3C,3C,4C,4C': 119, '4C,4C,5C,5C': 120, '5C,5C,6C,6C': 121, '6C,6C,7C,7C': 122, '7C,7C,8C,8C': 123, '8C,8C,9C,9C': 124, '9C,9C,TC,TC': 125, 'TC,TC,JC,JC': 126, 'JC,JC,QC,QC': 127, 'QC,QC,KC,KC': 128, 'KC,KC,AC,AC': 129, '3D,3D,4D,4D': 130, '4D,4D,5D,5D': 131, '5D,5D,6D,6D': 132, '6D,6D,7D,7D': 133, '7D,7D,8D,8D': 134, '8D,8D,9D,9D': 135, '9D,9D,TD,TD': 136, 'TD,TD,JD,JD': 137, 'JD,JD,QD,QD': 138, 'QD,QD,KD,KD': 139, 'KD,KD,AD,AD': 140, '3J,3J,4J,4J': 141, '4J,4J,5J,5J': 142, '5J,5J,6J,6J': 143, '6J,6J,7J,7J': 144, '7J,7J,8J,8J': 145, '8J,8J,9J,9J': 146, '9J,9J,TJ,TJ': 147, 'TJ,TJ,JJ,JJ': 148, 'JJ,JJ,QJ,QJ': 149, 'QJ,QJ,KJ,KJ': 150, 'KJ,KJ,AJ,AJ': 151, 'NJ,NJ,BJ,BJ': 152, 'BJ,BJ,RJ,RJ': 153, 'NH,NH,NJ,NJ': 154, 'NC,NC,NJ,NJ': 155, 'ND,ND,NJ,NJ': 156, 'AJ,AJ,NH,NH': 157, 'AJ,AJ,NC,NC': 158, 'AJ,AJ,ND,ND': 159, '2H': 160, '2C': 161, '2D': 162, '2J': 163, '2H,2H': 164, '2C,2C': 165, '2D,2D': 166, '2J,2J': 167, '2H,2H,4H,4H': 168, '2C,2C,4C,4C': 169, '2D,2D,4D,4D': 170, '2J,2J,4J,4J': 171, '2H,2H,3H,3H': 172, '3H,3H,5H,5H': 173, '2C,2C,3C,3C': 174, '3C,3C,5C,5C': 175, '2D,2D,3D,3D': 176, '3D,3D,5D,5D': 177, '2J,2J,3J,3J': 178, '3J,3J,5J,5J': 179, '4H,4H,6H,6H': 180, '4C,4C,6C,6C': 181, '4D,4D,6D,6D': 182, '4J,4J,6J,6J': 183, '5H,5H,7H,7H': 184, '5C,5C,7C,7C': 185, '5D,5D,7D,7D': 186, '5J,5J,7J,7J': 187, '6H,6H,8H,8H': 188, '6C,6C,8C,8C': 189, '6D,6D,8D,8D': 190, '6J,6J,8J,8J': 191, '7H,7H,9H,9H': 192, '7C,7C,9C,9C': 193, '7D,7D,9D,9D': 194, '7J,7J,9J,9J': 195, '8H,8H,TH,TH': 196, '8C,8C,TC,TC': 197, '8D,8D,TD,TD': 198, '8J,8J,TJ,TJ': 199, '9H,9H,JH,JH': 200, '9C,9C,JC,JC': 201, '9D,9D,JD,JD': 202, '9J,9J,JJ,JJ': 203, 'TH,TH,QH,QH': 204, 'TC,TC,QC,QC': 205, 'TD,TD,QD,QD': 206, 'TJ,TJ,QJ,QJ': 207, 'JH,JH,KH,KH': 208, 'JC,JC,KC,KC': 209, 'JD,JD,KD,KD': 210, 'JJ,JJ,KJ,KJ': 211, 'QH,QH,AH,AH': 212, 'QC,QC,AC,AC': 213, 'QD,QD,AD,AD': 214, 'QJ,QJ,AJ,AJ': 215, '3S': 216, '4S': 217, '5S': 218, '6S': 219, '7S': 220, '8S': 221, '9S': 222, 'TS': 223, 'JS': 224, 'QS': 225, 'KS': 226, 'AS': 227, 'NS': 228, '3S,3S': 229, '4S,4S': 230, '5S,5S': 231, '6S,6S': 232, '7S,7S': 233, '8S,8S': 234, '9S,9S': 235, 'TS,TS': 236, 'JS,JS': 237, 'QS,QS': 238, 'KS,KS': 239, 'AS,AS': 240, 'NS,NS': 241, '3S,3S,4S,4S': 242, '4S,4S,5S,5S': 243, '5S,5S,6S,6S': 244, '6S,6S,7S,7S': 245, '7S,7S,8S,8S': 246, '8S,8S,9S,9S': 247, '9S,9S,TS,TS': 248, 'TS,TS,JS,JS': 249, 'JS,JS,QS,QS': 250, 'QS,QS,KS,KS': 251, 'KS,KS,AS,AS': 252, 'NS,NS,NJ,NJ': 253, 'AJ,AJ,NS,NS': 254, '2S': 255, '2S,2S': 256, '2S,2S,4S,4S': 257, '2S,2S,3S,3S': 258, '3S,3S,5S,5S': 259, '4S,4S,6S,6S': 260, '5S,5S,7S,7S': 261, '6S,6S,8S,8S': 262, '7S,7S,9S,9S': 263, '8S,8S,TS,TS': 264, '9S,9S,JS,JS': 265, 'TS,TS,QS,QS': 266, 'JS,JS,KS,KS': 267, 'QS,QS,AS,AS': 268, 'NS,NS,BJ,BJ': 269, 'NH,NH,BJ,BJ': 270, 'NC,NC,BJ,BJ': 271, 'ND,ND,BJ,BJ': 272, 'pass': 273, 'pass_score': 274}

def hand2type(cards):
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
    ret_type += get_suit(cards[0])
    
    return ret_type

def get_suit(card):
    if card[0] == 'N':
        return 4
    else:
        return SUIT_RANK[card[1]]

def trump_type_to_win(target_type):
    return target_type // 10 * 10 + SUIT_RANK['J']

def is_same_suit(card_1, card_2):
    return get_suit(card_1) == get_suit(card_2)

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

def calc_score(cards, trump):
    if trump[0] == '5':
        score_dict = CARD_SCORE_5
    elif trump[0] == 'T' or trump[0] == 'K':
        score_dict = CARD_SCORE_TK
    else:
        score_dict = CARD_SCORE

    scores = [score_dict[x] for x in cards if x in score_dict.keys()]
    score_in_round = sum(scores)
    return score_in_round

def get_valid_cards(first_player, playable_cards):
    '''
        Returns:
        list: list of string of valid cards based on first player's card type

        Note:
        'pass' means pass without scores
        'pass_score' means pass with scores
    '''
    valid_cards = []
    target_hand = first_player.played_cards # list of string
    target_type = hand2type(target_hand,)
    playable_cards_type = [(cards, hand2type(cards)) for cards in playable_cards]

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


def reorganize_with_payoff_trace(trajectories, payoffs_with_trace, payoffs):
    ''' Reorganize the trajectory to make it RL friendly

    Args:
        trajectory (list): A list of trajectories
        payoffs (list): A list of payoffs for the players. Each entry corresponds to one player

    Returns:
        (list): A new trajectories that can be fed into RL algorithms.

    '''
    player_num = len(trajectories)
    new_trajectories = [[] for _ in range(player_num)]

    for player in range(player_num):
        for i in range(0, len(trajectories[player])-2, 2):
            reward = payoffs_with_trace[i // 2][player % 2]
            if i ==len(trajectories[player])-3:
                #reward += payoffs[player] # add a final payoff for each game
                done = True
            else:
                done = False
            transition = trajectories[player][i:i+3].copy()
            transition.insert(2, reward)
            transition.append(done)

            new_trajectories[player].append(transition)
    return new_trajectories

def tournament_tractor(env, num):
    ''' Evaluate he performance of the agents in the environment

    Args:
        env (Env class): The environment to be evaluated.
        num (int): The number of games to play.

    Returns:
        A list of avrage payoffs for each player
    '''
    payoffs = [0 for _ in range(env.player_num)]

    print()
    for iter in range(num):
        print('\rEvaluating {}/{} episodes...'.format(iter, num), end='')
        _, _payoffs = env.run(is_training=False)
        if isinstance(_payoffs, list):
            for _p in _payoffs:
                for i, _ in enumerate(payoffs):
                    if _p[i] > _p[(i+1)%2]:
                        payoffs[i] += 1
        else:
            for i, _ in enumerate(payoffs):
                if _payoffs[i] > _payoffs[(i+1)%2]:
                    payoffs[i] += 1
    for i, _ in enumerate(payoffs):
        payoffs[i] /= num
    return payoffs

class MovingAvg():
    def __init__(self, m_len):
        self.arr = []
        self.m_len = m_len
    
    def append(self, element):
        if (len(self.arr) == self.m_len):
            self.arr.pop(0)
        self.arr.append(element)
    
    def get(self):
        return float(sum(self.arr)) / max(len(self.arr), 1)

    def get_latest(self):
        return self.arr[-1] if len(self.arr)>0 else 0

    def get_first(self):
        return self.arr[0] if len(self.arr)>0 else 0