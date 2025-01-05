from itertools import combinations_with_replacement, product

# player 1 has plays on the rows
# player 2 has plays on the columns

# strategies are k (number of possible dice throws) dictionaries
# i.e. if 3 dice per player strat[0]=strat used if dice throws is (1,1,1)=
# {'first move to play': {'possible enemy move': ['response',{'possible enemy move after response':['',{}]}]}, 'second possible enemy move': ['',{}]}

def available_bids(current_bid, num_dice, num_faces):

    all_combinations = product(range(1, 2*(num_dice) + 1), range(1, num_faces + 1)) # Get all possible bids

    if current_bid is not None: # Check that this is not the first round

        curr_quant, curr_num = current_bid

        valid_bids = [  # Filter bids to only keep the legal ones
            (new_quant, new_num)
            for new_quant, new_num in all_combinations
            if new_quant > curr_quant or (new_quant == curr_quant and new_num > curr_num)
        ]
        
        valid_bids.append('Bluff')  # Add Bluff

    else:   # If its the first round all combinations are legal, except Bluff
        
        valid_bids = list(all_combinations)

    return valid_bids

def build_strat_ds(current_bid, dice_throw, num_dice, num_faces):

    bids=available_bids(current_bid=current_bid, num_dice=num_dice, num_faces=num_faces)

    # Filter for only sensible bids

    if bids == ['Bluff']:

        return None

    a={}

    for bid in available_bids:

        a[bid]=build_strat_ds(current_bid=bid, dice_throw=dice_throw, num_dice=num_dice, num_faces=num_faces)

    return a




def calc_possible_strats(player_num, num_dice, num_faces):

    dice_throws=(list(combinations_with_replacement(iterable=range(1, num_faces + 1), r=num_dice)))

    if player_num==1:

        for dice in dice_throws:

            build_strat_ds()

    elif player_num==2:
