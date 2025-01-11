import random
import numpy as np

class InfoSet():
    
    def __init__():
        
        
    
class InfosetStore():
    
    def __init__():
        
        
    
    def compute_bounds(b1,b2):
        
        
    
class GameState():
    
    

def cfr(gamestate, current_player, t, reach1, reach2, flag, update_player):
    
    # Check if node (gamestate) is terminal, if it is return the payoff for update_player
    
    # Check if node (gamestate) is a chance node (in our case at the top of the tree), its enough to check if the gamestate has attributes p1roll and p2roll set.
    # Do an if gs.p1roll==0 and if so define an expected value variable and iterate through all possible chance outcomes for player1. For all possible chance
    # outcomes create a gamestate which has that outcome as p1roll, define a new chancereachprobability by multiplying the chanceprobability of current outcome 
    # with current chancereach and recursively call cfr with cfr(new_gs, current_player, bidseq, reach1, reach2, newChanceReach, phase, updatePlayer) nothing 
    # besides chancereach and gs have changed because no actual play has taken place 
    
    # Do an elseif with the same approach but for player2, both this and the previous return EV which is the sum of the recursive cfrs weighted by the chance
    # probabilities of obtaining them
    
    # Check for possible pruning optimization, if flag is 1 and (current_player==update_player==1 and reach2<=0)||(current_player==update_player==2 and reach1<=0)
    # then we set flag to 2. Then if flag is 2 and (current_player==update_player==1 and reach1<=0)||(current_player==update_player==2 and reach2<=0) we 
    # automatically return 0
    
    # Create an Infoset object
    
if __name__=='main':
    
    # Create the InfoSetStore iss as it is done in line 564 of bluff.cpp
    
    # Create two variables, one for the number of possible chance outcomes for player 1 and one for player 2
    
    # Define iscWidth which is the number of bits needed to encode the chance outcomes, ceiling(log2 of the max of the two numChanceOutcomes)
    
    # Create a vector of size numbids where we assign in order the bids in the format (quantity*10)+face,
    # don't forget that wildcard bids are favoured (more probable), follow function initBids() in bluff.cpp.
    
    # Define a number of iterations to run
    
    # Set the bidsequence as a long long 0 or find an equivalent way with strings
    
    # Create a for, that runs for the defined number of iterations and at each step for each (of the two) player(s) creates an instance of the GameState class and 
    # runs the cfr algorithm with cfr(gamestate_i, i, 0, bidseq, 1.0, 1.0, 1, 1), gets the expected value this way
    
    # If we are at the last iteration (or every few iterations) compute the bounds with the iss.compute_bounds function and best reponses with function
    # compute_best_responses, to write it take after computeBestResponse in file br.cpp