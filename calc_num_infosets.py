import itertools as iter
import math
import time

p1dice=2
p2dice=1
die_faces=3

def num_outcomes(dice, faces):
    return math.comb(faces + dice - 1, dice)

outcomes_p1 = num_outcomes(p1dice, die_faces)
outcomes_p2 = num_outcomes(p2dice, die_faces)

possible_bids=(p1dice+p2dice)*die_faces

possible_bid_sequences=len(list(iter.product((0, 1), repeat=possible_bids)))

count_num_infosets=int(possible_bid_sequences*((outcomes_p1/2)+(outcomes_p2/2)))
count_space_needed=2*count_num_infosets

for bidseq in list(iter.product((0, 1), repeat=possible_bids)):
    
    last_bid=-1
    
    for i,bid in enumerate(bidseq):
        
        if bid==1:
            
            last_bid=i
            
    available_bids=possible_bids-last_bid
    
    if available_bids==possible_bids+1:
        
        available_bids=possible_bids
        
    count_space_needed+=2*available_bids*(outcomes_p1 if (sum(bidseq)%2==0) else outcomes_p2)

print('There are ',possible_bids, ' possible bids')
print('There are ',len(list(iter.product((0, 1), repeat=possible_bids))), ' possible bid sequences')
print('There are ',count_num_infosets, ' infosets')
print('Needed space is ', count_space_needed)

