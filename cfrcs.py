import random
import numpy as np
import itertools as iter
from collections import Counter
import math
import struct
import time

class InfoSet():
    
    def __init__(self, actions):

        self.cfr = [0.0]*(numbids+1)                # Stores counterfactual regrets of the information set with respect to each action
        self.total_move_probs = [0.0]*(numbids+1)   # Stores move probabilities for the information set compounded over all iterations
        self.curr_move_probs = [0.0]*(numbids+1)    # Stores move probabilities for the information set over the current iteration

        self.actions_here = actions                 # Stores the amount of actions available at this information set
        self.last_update = 0                        # 

        for i in range(actions):                    # Set the first action elements of cfr and total_move_probs to zero while curr_move_probs as uniform 
            self.cfr[i] = 0.0
            self.total_move_probs[i] = 0.0
            self.curr_move_probs[i] = 1.0 / actions
    
def get_sizes():
    
    # Get the parameters
    
    p1dice=params['p1dice']
    p2dice=params['p2dice']
    die_faces=params['diefaces']
    
    # Define a function that returns the number of outcomes of a player
    
    def num_outcomes(dice, faces): return math.comb(faces + dice - 1, dice)

    outcomes_p1 = num_outcomes(p1dice, die_faces)
    outcomes_p2 = num_outcomes(p2dice, die_faces)

    # Calculate number of possible bids (bluff included since we start from 0)

    possible_bids=(p1dice+p2dice)*die_faces

    # Create all possible bidsequences

    possible_bid_sequences=len(list(iter.product((0, 1), repeat=possible_bids)))

    # Half of the bidsequences are assigned to one player, half to the other, the two halves should be multiplied by the roll relative outcomes and counted, 
    # the space needed includes 2 slots per infoset

    count_num_infosets=int(possible_bid_sequences*((outcomes_p1/2)+(outcomes_p2/2)))
    count_space_needed=2*count_num_infosets

    # The space needed includes 2*num_available_actions slots per infoset, run over all of them and count

    for bidseq in list(iter.product((0, 1), repeat=possible_bids)):
        
        last_bid=-1
        
        for i,bid in enumerate(bidseq):
            
            if bid==1:
                
                last_bid=i
                
        available_bids=possible_bids-last_bid
        
        if available_bids==possible_bids+1:
            
            available_bids=possible_bids
            
        count_space_needed+=2*available_bids*(outcomes_p1 if (sum(bidseq)%2==0) else outcomes_p2)
        
    return count_space_needed,count_num_infosets
    
class InfosetStore():
    
    def __init__(self):
        
        ROWS=100   # This is arbitrary and indicates the ideal number of rows that our table is gonna have
        
        # Get the dimensions
        
        size,indexsize=get_sizes()    
        
        self.size,self.indexsize=size,indexsize
            
        # Calculate the sizes of the table
        
        self.rowsize = self.size // (ROWS-1) 
        self.lastrowsize = self.size - self.rowsize*(ROWS-1)
        self.rows = ROWS
        
        # Sometimes there are problems when size is small, this fixes it
        
        i = 0
        while (self.lastrowsize > self.rowsize):
            
            i+=1
            self.rows = ROWS-i
            self.rowsize = self.size // (self.rows-1)
            self.lastrowsize = self.size - self.rowsize*(self.rows-1)
        
        # Define an array for the keys of the information sets and a parallel array for the positions of the information of the relative is in the table
        
        self.index_keys=np.repeat(self.size,self.indexsize)
        self.index_values=np.repeat(self.size,self.indexsize)
        
        # Create the table
        
        self.table=[[] for _ in range(self.rows)]

        for i in range(self.rows): 
            if (i != (self.rows-1)):
                self.table[i] = [0.0]*self.rowsize
            else:
                self.table[i] = [0.0]*self.lastrowsize
        
        # Set InfosetStore in adding mode, initialize position for next Infoset to be saved and initialize a counter of how many Infosets there are
                
        self.adding_infosets = True
        self.next_infoset_pos = 0
        self.added = 0
    
    def compute_bounds(self,b1,b2,iter):
        
        # Run over the whole indexes
        
        for i in range(self.indexsize):
            
            # If we find a place which is set
            
            if (self.index_values[i] < self.size):
                
                # Get the corresponding key and calculate the positions in the table
                
                key = self.index_keys[i]
                
                pos = self.index_values[i]
                row = pos // self.rowsize
                col = pos % self.rowsize
                curr_rowsize = self.rowsize if (row < (self.rows-1)) else self.lastrowsize
                
                # Extract the data from the table
                
                actions_here=self.table[row][col]
                row, col, pos, curr_rowsize = self.next(row, col, pos, curr_rowsize)
                
                last_update = self.table[row][col]
                row, col, pos, curr_rowsize = self.next(row, col, pos, curr_rowsize)

                # Find the max of the regrets for this infoset

                running_max = -math.inf
                
                for a in range(int(actions_here)): 
                
                    cfr = self.table[row][col] 
                    
                    if cfr > running_max:
                        
                        running_max = cfr; 
                        
                    # Call next two times to skip the curr_move_prob    
                    
                    row, col, pos, curr_rowsize = self.next(row, col, pos, curr_rowsize)
                    row, col, pos, curr_rowsize = self.next(row, col, pos, curr_rowsize)
                    
                # Get the positive maximum regret
                
                delta = running_max 
                delta = max(0.0, delta)
                
                # Use the positive maximum regret to update the bound for 1 or for 2 based on whose is this infoset
                
                if key%2==0:
                    
                    b1+=delta
                    
                else:
                    
                    b2+=delta
                
        b1 /= iter
        b2 /= iter
        
        return b1,b2
        
    def next(self, row, col, pos, curr_rowsize):
        
        pos+=1
        col+=1 
        
        if col >= curr_rowsize:
            col = 0 
            row+=1
            curr_rowsize = self.rowsize if (row < (self.rows-1)) else self.lastrowsize
            
        return row, col, pos, curr_rowsize
        
    def get(self, infosetkey, infoset, actions_here, starting_move):
        
        # Retrieve the position of the infoset
        
        pos, hash_index = self.get_pos_from_index(infosetkey)
        
        # If the infoset doesn't exist, return False
        
        if (pos >= self.size): return False

        # Otherwise calculate the position in the table

        row = pos // self.rowsize
        col = pos % self.rowsize
        curr_rowsize = self.rowsize if row < (self.rows-1) else self.lastrowsize

        # Extract the data and put it into the infoset

        x = int(self.table[row][col])
        infoset.actions_here = x
        row, col, pos, curr_rowsize = self.next(row, col, pos, curr_rowsize)
        
        x = int(self.table[row][col])
        infoset.last_update= x; 
        row, col, pos, curr_rowsize = self.next(row, col, pos, curr_rowsize)

        for i in range(actions_here):
            
            infoset.cfr[starting_move+i] = self.table[row][col]

            row, col, pos, curr_rowsize = self.next(row, col, pos, curr_rowsize)

            infoset.total_move_probs[starting_move+i] = self.table[row][col]

            row, col, pos, curr_rowsize = self.next(row, col, pos, curr_rowsize)

        # Do Regret matching to update the cfr and the curr_move_probs

        tot_pos_regret = 0.0
        all_negative = True

        # Run over all possible actions

        for i in range(actions_here): 
            
            # Get the actual bid number and retrieve the current regret for that bid in this infoset
            
            movenum = starting_move+i
            cfr = infoset.cfr[movenum]
            
            # If it is > 0 then update the total positive regret and set all_negative to False
            
            if cfr > 0.0:
                
                tot_pos_regret = tot_pos_regret + cfr
                all_negative = False

        # Run over all possible actions

        for i in range(actions_here):
        
            # Ge the move number
        
            movenum = starting_move+i
            
            # If not all regrets were negative, then update each regret with the formula 2.18 of the thesis of Lanctot
            
            if not all_negative:
                
                if infoset.cfr[movenum] <= 0.0:
                    
                    infoset.curr_move_probs[movenum] = 0.0
                    
                elif tot_pos_regret > 0.0:
                    
                    infoset.curr_move_probs[movenum] = infoset.cfr[movenum] / tot_pos_regret

            # If they were all negative, play at random

            else:
                
                infoset.curr_move_probs[movenum] = 1.0/actions_here

        return infoset
        
    def put(self, infosetkey, infoset, actions_here, starting_move):
        
        # Keep track of the infoset being new or not
        
        newinfoset = False; 

        # Extract the position in the table using the two parallel arrays

        thepos, hash_index = self.get_pos_from_index(infosetkey)
        
        # If we haven't found it and we are in adding mode
        
        if (self.adding_infosets and thepos >= self.size):
            
            # Use the internal variable to know where to add the next infoset and calculate row and col 
            
            newinfoset = True
            
            pos = self.next_infoset_pos
            row = self.next_infoset_pos // self.rowsize
            col = self.next_infoset_pos % self.rowsize
            
            curr_rowsize = self.rowsize if (row < (self.rows-1)) else self.lastrowsize
            
            # Save the data into the arrays as well for later retrieval
            
            self.index_keys[hash_index] = infosetkey
            self.index_values[hash_index] = pos
            
        else:
            
            # If it already exists, just setup the variable to overwrite it
            
            newinfoset = False; 
            
            pos = thepos 
            row = pos // self.rowsize
            col = pos % self.rowsize
            curr_rowsize = self.rowsize if (row < (self.rows-1)) else self.lastrowsize
            
        # Write the number of available actions in the position, call next to update the variables to the next slot (loop if necessary)
            
        x = actions_here
        self.table[row][col] = x
        row, col, pos, curr_rowsize = self.next(row, col, pos, curr_rowsize)

        # Write the last_update in the following position and push the indexes again

        x = infoset.last_update
        self.table[row][col] = x 
        row, col, pos, curr_rowsize = self.next(row, col, pos, curr_rowsize)

        # For as many actions as are available we need to write 2 slots, one for the regret, one for the total_move_prob

        for i in range(actions_here):
        
            self.table[row][col] = infoset.cfr[starting_move+i]

            row, col, pos, curr_rowsize = self.next(row, col, pos, curr_rowsize)

            self.table[row][col] = infoset.total_move_probs[starting_move+i]

            row, col, pos, curr_rowsize = self.next(row, col, pos, curr_rowsize)

        # If we added something, update the next_infoset position and the added variable

        if (newinfoset and self.adding_infosets):
            
            self.next_infoset_pos = pos
            self.added+=1
            
        return newinfoset
            
    def get_pos_from_index(self, infosetkey):
    
        # Using linear probing we start from i=infosetkey%self.indexsize, we keep track of misses and hash_index
    
        i = infosetkey % self.indexsize
        misses = 0 
        hash_index=0

        # If we haven't checked all the array...

        while misses < self.indexsize:
            
            # If we found the key we were looking for, return position
            
            if (self.index_keys[i] == infosetkey and self.index_values[i] < self.size): 
            
                hash_index = i 
                return self.index_values[i],hash_index 
            
            # If we find a place where the value is still at size (the default) then we can stop because of linear probing
            
            elif self.index_values[i] >= self.size:
                
                hash_index = i
                return self.size, hash_index 
            
            # Update and loop when needed
            
            i += 1 
            if i >= self.indexsize:
                i = 0 
                
        # We shouldn't arrive here ever
        assert(False)
        
    def dump_to_disk(self, filename):
        
        # Open the file
        
        with open(filename, 'w') as file:

            # Write the metadata, one for each line for better readability
            
            file.write(str(self.indexsize))
            file.write('\n')
            file.write(str(self.size))
            file.write('\n')
            file.write(str(self.rowsize))
            file.write('\n')
            file.write(str(self.rows))
            file.write('\n')
            file.write(str(self.lastrowsize))
            file.write('\n')

            # Write the indexes, one array per line
            
            for i in range(self.indexsize):
                
                file.write(str(self.index_keys[i]))
                file.write(',')
                
            file.write('\n')
                
            for i in range(self.indexsize):
                
                file.write(str(self.index_values[i]))
                file.write(',')

            # Write the table, one element per line
            
            pos, row, col = 0, 0, 0
            curr_row_size = self.rowsize
            
            while pos < self.size:
                
                file.write('\n')
                file.write(str(self.table[row][col]))
                row, col, pos, curr_row_size = self.next(row, col, pos, curr_row_size)

    def read_from_disk(self, filename):
        
        # Handle possible errors
        
        try:
            
            # Open the file
            
            with open(filename, 'r') as file:
                
                # Read the metadata (they're all ints) 
                
                self.indexsize = int(file.readline().replace('\n',''))
                self.size = int(file.readline().replace('\n',''))
                self.rowsize = int(file.readline().replace('\n',''))
                self.rows = int(file.readline().replace('\n',''))
                self.lastrowsize = int(file.readline().replace('\n',''))
                
                # Read the indexes and turn them back into ints
                
                self.index_keys = file.readline().replace(',\n','').split(',')
                self.index_values = file.readline().replace(',\n','').split(',')
                
                self.index_keys = [int(index) for index in self.index_keys]
                self.index_values = [int(val) for val in self.index_values]
                
                # Allocate table rows
                
                self.table = []
                
                for i in range(self.rows):
                    
                    if i != self.rows - 1:
                        
                        row = [0.0] * self.rowsize
                        
                    else:
                        
                        row = [0.0] * self.lastrowsize
                        
                    self.table.append(row)
                    
                # Read table rows from the file, transform the data back into float
                
                pos, row, col = 0, 0, 0
                curr_row_size = self.rowsize
                
                while pos < self.size:
                    
                    self.table[row][col] = float(file.readline())
                    row, col, pos, curr_row_size = self.next(row, col, pos, curr_row_size)
                
            return True
        
        except (IOError, struct.error, AssertionError) as e:
            
            print(f"Error reading file: {e}")
            return False
        
class GameState():
    
    def __init__(self):
        
        self.p1roll = -1
        self.p2roll = -1
        self.curbid = -1
        self.prevbid = -1
        self.calling_player = -1
        
    def __string__(self):
        
        return f'P1roll:\t{self.p1roll}\nP2roll:\t{self.p2roll}\nCurbid:\t{self.curbid}\nPrevbid:\t{self.prevbid}\nCallingPlayer:\t{self.calling_player}'
        
    def copy(self):

        ngs = GameState()

        ngs.p1roll = self.p1roll
        ngs.p2roll = self.p2roll
        ngs.curbid = self.curbid
        ngs.prevbid = self.prevbid
        ngs.calling_player = self.calling_player

        return ngs
        
def get_move_prob(infoset, action, actions_here):
    
    # Sums the total probabilities and gives back the probability of the wanted action normalized over all others
    
    den = 0.0
    
    for i in range(actions_here):
        
        if infoset.total_move_probs[i]>0.0:
        
            den += infoset.total_move_probs[i]

    if den > 0.0: 
        
        return (infoset.total_move_probs[action] / den)
    
    else:
        
        return (1.0 / actions_here)
        
def get_chance_probability(player, outcome):
    
    # Define the chance probability based on who's the player
    
    chance_outcome =  num_chance_outcomes_1 if (player == 1) else num_chance_outcomes_2
    chance_probability = chance_prob_1 if (player == 1) else chance_prob_2
    return chance_probability[outcome]

def compute_action_dist(bidseq, player, fixed_player, action, opp_action_dist, opp_reach, actions_here, opp_chance_outcomes):
    
    """
    print('########################')
    print('Bidseq: ',bin(bidseq))
    print('Player: ',player)
    print('Fixed player (-i): ',fixed_player)
    print('Action taken: ',action)
    print('Opponent\'s reach probability: ',opp_reach)
    print('Available actions: ',actions_here)
    print('Opponent\'s chance outcomes: ', opp_chance_outcomes)
    print('########################')
    """
    
    weight = 0.0 
    
    # Run over the opponent's chance outcomes
    
    for i in range(len(opp_chance_outcomes)): 

        # Get the chance outcome, create an infoset and populate it with the current one

        chance_outcome = opp_chance_outcomes[i]
    
        infoset = InfoSet(actions_here)
        
        infosetkey = bidseq << isc_width 
        infosetkey = infosetkey | chance_outcome 
        infosetkey = infosetkey << 1
        if (player == 2): infosetkey = infosetkey | 1 

        infoset = iss.get(infosetkey, infoset, actions_here, 0) 

        #// apply out-of-date mccfr patch if needed. note: we know it's always the fixed player here
        #if (mccfrAvgFix)
        #    fixAvStrat(infosetkey, is, actionshere, newOppReach[i]); 
    
        # Get the opponent's probability of playing action action over the actions_here actions available and update the opp_reach  
    
        opp_prob = get_move_prob(infoset, action, actions_here)
        opp_reach[i] = opp_reach[i] * opp_prob 

        # Set the probability of action action as a sum of chance probability outcomes and opp_reach probabilities of arriving to this gamestate 

        weight += get_chance_probability(fixed_player, chance_outcome)*opp_reach[i] 

        opp_action_dist[action]=weight 
        
        """
        print(f'Opponenet\'s probability of playing action {action}: ',opp_prob)
        print('Probability of being here: ', opp_reach[i])
        print('Weight: ',weight)
        print('Chance outcome is number: ',chance_outcome)
        print('Infosetkey: ',infosetkey)
        """
            
    return opp_action_dist
        
def expectimax_br (gamestate, bidseq, player, fixed_player, opp_reach, opp_chance_outcomes):
    
    # Define the update player (the opposite of the fixed one), and an all_equal_zero flag
    
    update_player = 3-fixed_player
    all_equal_zero = True
    
    """
    print('Gamestate:\n',gamestate.__string__())
    print('Number of bids: ',numbids+1)
    print('Bidsequence: ', bin(bidseq))
    print('Current player: ',player)
    print('Fixed player: ', fixed_player)
    print('Opponent\'s reach probabilities: ',opp_reach)
    print('Opponent\'s chance outcomes: ',opp_chance_outcomes)
    print('Update player: ', update_player)
    print('#####################################')
    """
    
    # Check if all opp_reach probabilities are 0
    
    for i in opp_reach:
        
        if i!=0:
            
            all_equal_zero=False

    # If we are at the update player's turn and all opp_reach probabilities are zero, return -inf

    if (player == update_player & all_equal_zero):
        
        return -math.inf
    
    # If we are at a terminal node
    
    if gamestate.curbid==numbids:
        
        # Check if all opp_reach prob are zero, if so return -inf
        
        if all_equal_zero:
            
            return -math.inf

        # Define an opp_dist vector containing, for each opponent chance outcome, the probability that he arrives here with that chance outcome

        opp_dist=[0.0]*len(opp_chance_outcomes) 
    
        for i in range(len(opp_chance_outcomes)):
            
            opp_dist[i] = get_chance_probability(fixed_player, opp_chance_outcomes[i])*opp_reach[i] 

        # Normalize the probabilities

        total=sum(opp_dist)
        opp_dist=[a/total for a in opp_dist]

        # Define an expected payoff

        exp_payoff = 0.0 

        # Get average payoff over opponent's outcome 

        for i in range(len(opp_chance_outcomes)): 
            
            if update_player==1:
                
                gamestate.p2roll=opp_chance_outcomes[i]
                
            else: 
                
                gamestate.p1roll=opp_chance_outcomes[i]

            payoff = get_payoff(gamestate, update_player)

            exp_payoff += (opp_dist[i] * payoff)

        return exp_payoff
    
    # Check if we're at a chance node
    
    if (gamestate.p1roll == -1):
        
        # On fixed player chance nodes we can just use any die roll
        
        if (fixed_player == 1):
            
            new_gamestate = gamestate.copy(); 
            new_gamestate.p1roll = 1;   # Assigned a roll but it's never used

            return expectimax_br(new_gamestate, bidseq, player, fixed_player, opp_reach, opp_chance_outcomes)
        
        else:
            
            # Otherwise we compute an expected value over player 1's chance outcomes
            
            ev = 0.0 

            for i in range(num_chance_outcomes_1): 
            
                new_gamestate = gamestate.copy() 
                new_gamestate.p1roll = i; 
                
                ev += get_chance_probability(1,i) * expectimax_br(new_gamestate, bidseq, player, fixed_player, opp_reach, opp_chance_outcomes)

            return ev
        
    # Same for player 2
        
    elif (gamestate.p2roll == -1):
        
        if (fixed_player == 2):
            
            new_gamestate = gamestate.copy() 
            new_gamestate.p2roll = 1
            
            return expectimax_br(new_gamestate, bidseq, player, fixed_player, opp_reach,opp_chance_outcomes)
        
        else:
            
            ev = 0.0
            
            for i in range(num_chance_outcomes_2):
                
                new_gamestate = gamestate.copy() 
                new_gamestate.p2roll = i
                
                ev += get_chance_probability(2,i) * expectimax_br(new_gamestate, bidseq, player, fixed_player, opp_reach,opp_chance_outcomes)
                
            return ev
        
    # Start an expected value variable and calculate the number of available bids
        
    ev = 0.0 
    
    maxbid = numbids if (gamestate.curbid == -1) else numbids+1
    actions_here = maxbid - gamestate.curbid - 1
    
    max_ev = -math.inf
    child_evs=[0.0]*actions_here
    action = -1
    opp_action_dist={}
    
    # Run over possible bids
    
    for i in range(gamestate.curbid+1, maxbid):
        
        action+=1
        
        # If the current player is the fixed player let's compute the opponent's action's distribution
        
        if (player == fixed_player):
            
            opp_action_dist = compute_action_dist(bidseq, player, fixed_player, action, opp_action_dist, opp_reach, actions_here, opp_chance_outcomes) 

        # Create the child game and run it
        
        new_gamestate = gamestate.copy() 
        new_gamestate.prevbid = gamestate.curbid
        new_gamestate.curbid = i
        new_gamestate.calling_player = player 
        new_bidseq = bidseq | (1 << (numbids-i))

        child_evs[action] = expectimax_br(new_gamestate, new_bidseq, 3-player, fixed_player, opp_reach, opp_chance_outcomes)
        
        # After recursion keep track of max_ev seen for the update_player
        
        if player == update_player: 
            
            if child_evs[action] >= max_ev:
                
                max_ev = child_evs[action]
            
    # If we are currently the update_player, set the max_ev found to ex
            
    if player == update_player:
        
        ev = max_ev
        
    # If we are the fixed player normalize the values in opp_action_dist and calculate the expected value by weighting 
    # the children evs (varying action) with the opp_action_dist
        
    elif player == fixed_player:
        
        tot=0
        
        #print(opp_action_dist)
        
        for key in opp_action_dist:
            
            tot+=opp_action_dist[key]
        
        for key in opp_action_dist:
            
            opp_action_dist[key]=opp_action_dist[key]/tot
        
        for i in range(actions_here):
            
            ev += (opp_action_dist[i] * child_evs[i]) 
            
    return ev

            
def compute_best_responses(avgFix=False):
    
    # Follows theory in appendix B of Lanctot thesis 
    # Flag used for MC
    
    mccfrAvgFix = avgFix 
    
    # Get chance outcomes for player 1, fixing it 
    
    opp_chance_outcomes=[i for i in range(0, num_chance_outcomes_1)]

    # Calculate the expected value for player 2 fixing player 1

    gamestate1 = GameState()
    bidseq = 0
    reach1 = [1.0]*num_chance_outcomes_1 
    p2value = expectimax_br(gamestate1, bidseq, 1, 1, reach1, opp_chance_outcomes)

    # Get chance outcomes for player 2, fixing it

    opp_chance_outcomes=[i for i in range(0, num_chance_outcomes_2)]

    # Calculate the expected value for player 1 fixing 2

    gamestate2 = GameState() 
    bidseq = 0
    reach2 = [1.0]*num_chance_outcomes_2 
    p1value = expectimax_br(gamestate2, bidseq, 1, 2, reach2, opp_chance_outcomes)

    # Calculate convergence value

    conv = p1value + p2value

    return conv

        
def get_payoff(gamestate, update_player):
    
    # The actual bid to be tested is the previous (the last one was bluff), the player that made it is the other player, not the one calling bluff
    
    bid = bids[gamestate.prevbid]
    bidder = 3-gamestate.calling_player
    
    # Extract quantity and face of the bid from the code 
    
    quant = bid//10
    face = bid%10
    
    # Getting the actual roll from the roll number
    
    p1dice = list(iter.combinations_with_replacement(range(1,params['diefaces']+1),params['p1dice']))[gamestate.p1roll]
    p2dice = list(iter.combinations_with_replacement(range(1,params['diefaces']+1),params['p2dice']))[gamestate.p2roll]
    
    # Count number of matching dice
    
    matching=0
    
    for dice in p1dice:
        
        if dice == face or dice == params['diefaces']:
            
            matching+=1
            
    for dice in p2dice:
        
        if dice == face or dice == params['diefaces']:
            
            matching+=1
    
    # Set correct player as winner based on matching count
    
    if matching >= quant:
        
        winner = bidder
    
    else:
        
        winner = gamestate.calling_player
        
    # Return the payoff in the win or lose cases
        
    if winner == update_player:
        
        payoff=1.0
    
    else:
        
        payoff=-1.0
    
    """
    print('Update player: ',update_player)
    print('Player 1 roll : ', gamestate.p1roll)
    print('Which means: ',p1dice)
    print('Player 2 roll : ', gamestate.p2roll)
    print('Which means: ',p2dice)
    print('Bid: ', bid)
    print('Bidder: ',bidder)
    print('Quantity: ',quant)
    print('Face: ',face)
    print('Payoff: ',payoff)
    time.sleep(1)
    """
    
    return payoff

def init_infosets(gamestate, player, bidseq):
    
    # This looks like a cfr function but is used to traverse the tree and populate the iss
    
    # If we're at a terminal node, end the function
    
    if gamestate.curbid==numbids:     # numbids is the number for bluff since we start from 0 for the first bid
        
        print('Outcome: ', bin(bidseq))
        
        return
    
    # If p1roll has not been set yet, set one
    
    if gamestate.p1roll == -1:
        
        print('Player 1 has no roll, number of rolls: ',num_chance_outcomes_1)
        
        # Run over all possible chance outcomes for player 1
        
        for i in range(num_chance_outcomes_1):
            
            # Create a copy of the current gamestate with p1roll set
            
            new_gamestate = gamestate.copy()
            new_gamestate.p1roll = i

            print('Player 1 has rolled: ',new_gamestate.p1roll)
            print('Player 2\'s roll: ', new_gamestate.p2roll)
            time.sleep(1)
            
            # Recursively call the function to set infosets below, player and bidseq remain the same because this was a chance node
            
            init_infosets(new_gamestate, player, bidseq)
            
        return

    # Equivalent to before but for player 2

    elif gamestate.p2roll == -1:
        
        print('Player 2 has no roll, number of rolls: ',num_chance_outcomes_2)
        
        for i in range(num_chance_outcomes_2):
            
            new_gamestate = gamestate.copy()
            new_gamestate.p2roll = i

            print('Player 1 has rolled: ', new_gamestate.p1roll)
            print('Player 2 has rolled: ', new_gamestate.p2roll)
            time.sleep(1)
            
            init_infosets(new_gamestate, player, bidseq)
            
        return

    # If we are here this is neither a chance node nor a terminal node, calculate the maxbid which is bluff if the 
    # current bidseq is not 0, bluff-1 otherwise (if bidseq is 0, no bid has been done and bluff is off limits).
    # Also calculate the amount of available actions

    maxbid = numbids if (gamestate.curbid == -1) else numbids+1
    actions_here = maxbid - (gamestate.curbid+1)
    
    """
    print('Current bid: ', gamestate.curbid)
    print('Allowed maximum bid in this state: ',maxbid)
    print('Amount of available bids: ', actions_here)
    """
    
    # Initialize an empty infoset
    
    infoset = InfoSet(actions_here)
    
    # For each available bid in this infoset
    
    for i in range(gamestate.curbid+1, maxbid):

        # Create a new game which is a copy of the current one, with old curbid as new prevbid and the current bid in the for as new curbid, 
        # also calling_player is set and the bidsequence is updated by adding a 1 in the position of the bid in the for

        new_gamestate = gamestate.copy()
        new_gamestate.prevbid = gamestate.curbid
        new_gamestate.curbid = i
        new_gamestate.calling_player = player
        new_bidseq = bidseq | (1 << (numbids-i))
        
        """
        print('Creating state with bid: ', i)
        print('Previous bid of the new game: ', new_gamestate.prevbid)
        print('Current bid of the new game: ', new_gamestate.curbid)
        print('Calling player of the new game: ', new_gamestate.calling_player)
        print('Bid sequence of the new game: ', bin(new_bidseq))
        time.sleep(1)
        """
        
        # Recursively call the function with the new game, new_bidseq and the opposite player (the current one has made his move)
        
        init_infosets(new_gamestate, (3-player), new_bidseq)
        
    # Make room for the roll of the dice in the infosetkey by moving the bidseq to the left
        
    infosetkey = bidseq
    infosetkey = infosetkey << isc_width
    
    # Add the dice roll and a bit (0 for player 1, 1 for player 2) to build the infosetkey
    
    if (player == 1):
        
        infosetkey = infosetkey | gamestate.p1roll
        infosetkey = infosetkey << 1
        
        # Put the updated infoset back into the InfosetStore, the put function returns a boolean indicating wether the infoset was new or not 
        
        print('Current player: ',player)
        print('Bidseq:',bin(bidseq))
        print('Roll: ', gamestate.p1roll if player==1 else gamestate.p2roll)
        print('Number of available actions: ', actions_here)
        print('Infosetkey: ',bin(infosetkey))
        print('Maximum allowed bid: ',maxbid)
        
        new_flag=iss.put(infosetkey, infoset, actions_here, 0)
        
        print('Inserting infoset with key: ', bin(infosetkey), '. It was new' if new_flag else '. It already existed')
        
    # Do the same foor player 2
        
    elif (player == 2):
        
        infosetkey = infosetkey | gamestate.p2roll
        infosetkey = infosetkey << 1
        infosetkey = infosetkey | 1
        new_flag=iss.put(infosetkey, infoset, actions_here, 0)
        
        print('Inserting infoset with key: ', bin(infosetkey), '. It was new' if new_flag else '. It already existed')
        
    # Define a filename to save the InfosetStore
        
    filename = f"iss{params['p1dice']}{params['p2dice']}{params['diefaces']}.initial.txt"
    
    # Call the dump function that writes the iss to the file
    
    iss.dump_to_disk(filename)


def sample_chance_event(player):
    num_chance_outcomes = num_chance_outcomes_1 if player==1 else num_chance_outcomes_2
    chance_prob = chance_prob_1 if player==1 else chance_prob_2
    chance_outcomes = chance_outcomes_1 if player==1 else chance_outcomes_2
    
    roll = random.uniform(0, 1)
    sum = 0.0

    for i in range(num_chance_outcomes):
        chance_p = chance_prob[i]

        if roll >= sum and roll < sum + chance_p:
            outcome = i
            prob = chance_p
            return outcome, prob

        sum += chance_p

    # We shouldn't arrive here ever
    assert(False)
        

def cfrcs(gamestate, current_player, bidseq, reach1, reach2, chance_reach, flag, update_player):
    
    """
    print(gamestate.__string__())
    print('################################')
    time.sleep(1)
    """
    
    # Check if node (gamestate) is terminal, if it is return the payoff for update_player
    
    if gamestate.curbid==numbids:
        
        # Get the payoff of this gamestate
        
        return get_payoff(gamestate=gamestate,update_player=update_player)
    
    # Check if we are at a chance node for player 1, this is seen by the fact that p1roll has not been set yet
    
    if gamestate.p1roll == -1:
        
        # Define an expected value variable, since this is the first node of the tree, this will be the expected value of the game????
        
        ev = 0.0
        
        # Run over all possible rolls for player 1
        
        sampled_outcome, sampled_prob = sample_chance_event(1)
                    
        new_gamestate = gamestate.copy(); 
        new_gamestate.p1roll = sampled_outcome
        new_chance_reach = chance_reach*sampled_prob
        
        ev += cfrcs(new_gamestate, current_player, bidseq, reach1, reach2, new_chance_reach, flag, update_player)

        return ev

    # Do the same for player 2

    elif gamestate.p2roll == -1:
        
        # Define an expected value variable, since this is the first node of the tree, this will be the expected value of the game????
        
        ev = 0.0
        
        # Run over all possible rolls for player 1
        
        sampled_outcome, sampled_prob = sample_chance_event(2)

        new_gamestate = gamestate.copy(); 
        new_gamestate.p2roll = sampled_outcome; 
        new_chance_reach = chance_reach*sampled_prob

        ev += cfrcs(new_gamestate, current_player, bidseq, reach1, reach2, new_chance_reach, flag, update_player)

        return ev
    
    # Check for pruning optimization, if the reach_prob of the adversary is 0 we shouldn't calculate the regret later (it's useless),
    # if also our reach_prob is 0 we can stop altogether
    
    if (flag==1) and ((current_player==update_player==1 and reach2<=0) or (current_player==update_player==2 and reach1<=0)):
        
        flag=2
        
    if (flag==2) and ((current_player==update_player==1 and reach1<=0) or (current_player==update_player==2 and reach2<=0)):
        
        return 0.0
    
    # Define the number of possible moves, if gamestate.curbid==0 then bluff is not possible, otherwise it is
    
    maxbid = numbids if (gamestate.curbid == -1) else numbids+1
    actions_here = maxbid - gamestate.curbid - 1
    
    # Create an Infoset object
    
    infoset = InfoSet(actions_here)
    
    # Create an vector of zeros for the expected value of the moves (should be the same length as the number of possible bids)
    
    move_ev = [0.0]*actions_here 
    
    # Define the infosetkey and retrieve the infoset from the iss using get, this also updates the regrets and the curr move probs with regret matching
    
    infosetkey = bidseq
    infosetkey = infosetkey<<isc_width    
    
    if current_player == 1:
        infosetkey = infosetkey | gamestate.p1roll
        infosetkey = infosetkey << 1
        
    elif current_player == 2:
        infosetkey = infosetkey | gamestate.p2roll
        infosetkey = infosetkey << 1
        infosetkey = infosetkey | 1
        
    infoset = iss.get(infosetkey, infoset, actions_here, 0)
    
    # Define the expected value of the current strategy
    
    strat_ev = 0.0
    action = -1
    
    # Run over available bids
    
    for i in range(gamestate.curbid+1, maxbid):
        
        action+=1
        
        # Get the current move probability (updated with regret matching when using get) and use it to update the correct reach probability
        
        move_prob = infoset.curr_move_probs[action]; 
        newreach1 = move_prob*reach1 if (current_player == 1) else reach1 
        newreach2 = move_prob*reach2 if (current_player == 2) else reach2

        # Create the gamestate of the child game

        new_gamestate = gamestate.copy() 
        new_gamestate.prevbid = gamestate.curbid
        new_gamestate.curbid = i 
        new_gamestate.calling_player = current_player
        new_bidseq = bidseq
        newbidseq = new_bidseq | (1 << (numbids-i))
        
        # Run cfr of the child to get its payoff
        
        payoff = cfrcs(new_gamestate, 3-current_player, newbidseq, newreach1, newreach2, chance_reach, flag, update_player) 
        
        # Set the expected value of the action with the payoff and update the current strat with the payoff weighted by the current probability of the move being played
        
        move_ev[action] = payoff 
        strat_ev += move_prob*payoff 

    """
    print('Infosetkey: ',bin(infosetkey))
    print('Starting bid:',gamestate.curbid+1)
    print('Maxbid: ',maxbid)
    print('Num_actions: ',infoset.actions_here)
    print('Actions: ',action)
    print('MoveProb: ',move_prob)
    print(reach1, ' -> ', newreach1)
    print(reach2, ' -> ', newreach2)
    
    print('Moves ev: ',move_ev)
    print('Strat ev: ',strat_ev)
    """
    #time.sleep(1)
    
    # Define the myreach and oppreach probabilies based on which player is playing
    
    myreach = reach1 if (current_player == 1) else reach2 
    oppreach = reach2 if (current_player == 1) else reach1
    
    # If flag is still 1 and the current player is the updatePlayer then update the regrets with (move_ev[a]-strat_ev) weigthed by chance_reach*oppreach
    
    if (flag == 1 and current_player == update_player):
        
        for a in range(actions_here):
            
            infoset.cfr[a] += chance_reach*oppreach*(move_ev[a] - strat_ev) 
    
    # If flag is >=1 and the current player is the updatePlayer then update the strategies i.e. the totalmoveProb of the infoset by adding the 
    # current move_prob (updated with regret when calling getInfoset) weighted by myreach
    
    if (flag >= 1 and current_player == update_player):
    
        for a in range(actions_here):
            
            infoset.total_move_probs[a] += myreach*infoset.curr_move_probs[a]; 
    
    # Finally if the current player is the update_player call function put of iss to save the current (modified) infoset to the iss
    
    if (current_player == update_player):
        
        new_flag=iss.put(infosetkey, infoset, actions_here, 0)
    
    # Return strat_ev
    
    return strat_ev
    
def determine_chance_outcomes(player):
    
    # Extract parameters 
    
    num_dice=params[f'p{player}dice']
    num_faces=params['diefaces']
    
    # Calculate the possible chance outcomes using combinations with replacement
    
    chance_outcomes=list(iter.combinations_with_replacement(range(1,num_faces+1),num_dice))

    # Get the number of chance outcomes
        
    num_chance_outcomes=len(chance_outcomes)

    # Get the probability of each chance outcome

    chance_prob=[0.0]*num_chance_outcomes

    for i, roll in enumerate(chance_outcomes):

        # This basically creates a dictionary which counts how many times each element in roll appears
        count=Counter(roll)

        # Count the permutations
        perm=1
        for key in count:
            
            perm*=math.factorial(count[key])
            
        chance_prob[i]=math.factorial(num_dice)/((num_faces**num_dice)*perm)
    
    return num_chance_outcomes, chance_outcomes, chance_prob
    
def main():
    
    # Define global variables
    global params
    global numbids
    global iss
    global isc_width
    global bids
    
    params = {
        'p1dice':       1,
        'p2dice':       1,
        'diefaces':     6,
        'n_iter':       100000,
        'filename':     'iss116.initial.txt',
    }
    
    # Calculate number of bids (+1 for bluff)
    
    numbids = ((params['p1dice']+params['p2dice'])*params['diefaces'])
    
    # Create two variables, one for the number of possible chance outcomes for player 1 and one for player 2
    
    global num_chance_outcomes_1,chance_outcomes_1,chance_prob_1
    global num_chance_outcomes_2,chance_outcomes_2,chance_prob_2
    
    num_chance_outcomes_1, chance_outcomes_1, chance_prob_1 = determine_chance_outcomes(player=1)
    num_chance_outcomes_2, chance_outcomes_2, chance_prob_2 = determine_chance_outcomes(player=2)
    
    """
    print(num_chance_outcomes_1, ' ', chance_outcomes_1, ' ', chance_prob_1)
    print(num_chance_outcomes_2, ' ', chance_outcomes_2, ' ', chance_prob_2)
    """
    
    # Define isc_width which is the number of bits needed to encode the chance outcomes, ceiling(log2 of the max of the two num_chance_outcomes)
    
    isc_width = math.ceil(math.log2(max(num_chance_outcomes_1,num_chance_outcomes_2)))
    
    # Create a vector of size numbids where we assign in order the bids in the format (quantity*10)+face, that wildcard bids are more probable.
    
    bids = [0]*numbids
    next_wild = 1
    i = 0
    
    for dice in range(1,params['p1dice']+params['p2dice']+1):
        
        for face in range(1,params['diefaces']):
            
            bids[i] = dice*10 + face
            i+=1

        if (dice % 2 == 1):
            bids[i] = next_wild*10 + params['diefaces']
            i+=1
            next_wild+=1
            
    while next_wild <= (params['p1dice']+params['p2dice']):
        bids[i] = next_wild*10 + params['diefaces']
        i+=1
        next_wild+=1
    
    # Create the empty InfosetStore
    
    iss=InfosetStore()
    
    # Check if a file to import the InfosetStore has been specified, if not create it, if yes upload it
    
    if params['filename']==None:
        
        # Populate the InfosetStore with the initial strategies
        
        gamestate = GameState()
        bidseq = np.uint64(0)
        
        init_infosets(gamestate, 1, bidseq)
        
    else:
        
        # If a filename is present, just upload the iss from that file
        
        iss.read_from_disk(params['filename'])
        
        """
        print(iss.size)
        print(iss.table)
        """
        
    # Set the bidsequence at 0
    
    bidseq = 0
    
    # Run for the defined number of iterations
    mean1 = 0
    mean2 = 0
    
    for i in range(params['n_iter']):
        
        print(f'Running iteration {i+1}')
        
        # Create a game for player 1 and run the cfr function with update_player=1
        
        gs1 = GameState()
        bidseq = np.uint64(0)
        ev1 = cfrcs(gs1, 1, bidseq, 1.0, 1.0, 1.0, 1, 1)
        mean1 += (ev1/params['n_iter'])

        
        #print('Strat ev for player 1: ', ev1)
        
        # Create a game for player 2 and run the cfr function with update_player=2, player is still 1 because 1 always starts
        
        gs2 = GameState()
        bidseq = np.uint64(0)
        ev2 = cfrcs(gs2, 1, bidseq, 1.0, 1.0, 1.0, 1, 2)
        mean2 += (ev2/params['n_iter'])

        
        #print('Strat ev for player 2: ', ev2)
    
    # If we are at the last iteration (or every few iterations) compute the bounds with the iss.compute_bounds function and best reponses with function
    # compute_best_responses
    
        if (i%10==0 and i!=0) or i==params['n_iter']-1:
            
            print(f"ev1 = {mean1}, ev2 = {mean2}")
            b1,b2 = 0,0
            b1,b2 = iss.compute_bounds(b1, b2,i) 
            print(f" b1 = {b1}, b2 = {b2}, bound = {(2.0*max(b1,b2))}")
            
    # I haven't yet understood this very well
            
    conv = compute_best_responses()
    
    print(conv)
    
if __name__=='__main__':
    
    main()