import random
import numpy as np
import itertools as iter
from collections import Counter
import math
import struct

class InfoSet():
    
    def __init__(self, actions):

        self.cfr = [0.0]*(numbids+1)                # Stores counterfactual regrets of the information set with respect to each action
        self.total_move_probs = [0.0]*(numbids+1)   # Stores move probabilities for the information set compounded over all iterations
        self.curr_move_probs = [0.0]*(numbids+1)    # Stores move probabilities for the information set over the current iteration

        self.actions_here = actions
        self.last_update = 0

        for i in range(actions):
        
            self.cfr[i] = 0.0
            self.total_move_probs[i] = 0.0
            self.curr_move_probs[i] = 1.0 / actions
    
class InfosetStore():
    
    def __init__(self):
        
        ROWS=100   # This is arbitrary and indicates the ideal number of rows that our table is gonna have
        
        if (params['p1dice'] == 1 and params['p2dice'] == 1 and params['diefaces'] == 6):
            
            self.size=147432
            self.indexsize=100000
            
        # Calculate the sizes of the table
        
        self.rowsize = self.size / (ROWS-1) 
        self.lastrowsize = self.size - self.rowsize*(ROWS-1)
        rows = ROWS
        
        # Define an array for the keys of the information sets and a parallel array for the positions of the information of the relative is in the table
        
        self.index_keys=np.repeat(self.size,self.indexsize)
        self.index_values=np.repeat(self.size,self.indexsize)
        
        # Create the table
        
        self.table=[[] for _ in range(rows)]

        for i in range(rows): 
            if (i != (rows-1)):
                self.table[i] = [0.0]*self.rowsize
            else:
                self.table[i] = [0.0]*self.lastrowsize
        
        # Set InfosetStore in adding mode, initialize position for next Infoset to be saved and initialize a counter of how many Infosets there are
                
        self.adding_infosets = True
        self.next_infoset_pos = 0
        self.added = 0
    
    def compute_bounds(self,b1,b2):
        
        for i in range(self.indexsize):
            
            if (self.index_values[i] < self.size):
                
                key = self.index_keys[i] 
                b = b1 if (key % 2 == 0) else b2 
                
                pos = self.index_values[i]
                row = pos / self.rowsize
                col = pos % self.rowsize
                curr_rowsize = self.rowsize if (row < (self.rows-1)) else self.lastrowsize
                
                actions_here=self.table[row][col]
                row, col, pos, curr_rowsize = self.next(row, col, pos, curr_rowsize)
                
                lastUpdate = self.table[row][col]
                row, col, pos, curr_rowsize = self.next(row, col, pos, curr_rowsize)

                max = -math.inf
                
                for a in range(actions_here): 
                
                    cfr = self.table[row][col] 
                    
                    if cfr > max:
                        
                        max = cfr; 
                    row, col, pos, curr_rowsize = self.next(row, col, pos, curr_rowsize)
                    row, col, pos, curr_rowsize = self.next(row, col, pos, curr_rowsize)
                    
                delta = max 
                delta = max(0.0, delta); 
                
                b += delta
                
        b1 /= i 
        b2 /= i
        
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
        
        pos, hash_index = self.get_pos_from_index(infosetkey)
        
        if (pos >= self.size): return False

        row = pos / self.rowsize
        col = pos % self.rowsize
        curr_rowsize = self.rowsize if row < (self.rows-1) else self.lastrowsize

        x = self.table[row][col]
        infoset.actions_here = x
        row, col, pos, curr_rowsize = self.next(row, col, pos, curr_rowsize)
        
        x = self.table[row][col] 
        infoset.last_update= x; 
        row, col, pos, curr_rowsize = self.next(row, col, pos, curr_rowsize)

        for i in range(actions_here):
            
            infoset.cfr[starting_move+i] = self.table[row][col]

            row, col, pos, curr_rowsize = self.next(row, col, pos, curr_rowsize)

            infoset.total_move_probs[starting_move+i] = self.table[row][col]

            row, col, pos, curr_rowsize = self.next(row, col, pos, curr_rowsize)

        # Do Regret matching

        tot_pos_regret = 0.0
        all_negative = True

        for i in range(actions_here): 
            
            movenum = starting_move+i
            cfr = infoset.cfr[movenum]
            
            if cfr > 0.0:
                
                totPosReg = totPosReg + cfr
                all_negative = False

        prob_sum = 0.0

        for i in range(actions_here):
        
            movenum = starting_move+i
            
            if not all_negative:
                
                if infoset.cfr[movenum] <= 0.0:
                    
                    infoset.curr_move_probs[movenum] = 0.0
                    
                elif totPosReg > 0.0:
                    
                    infoset.curr_move_probs[movenum] = infoset.cfr[movenum] / tot_pos_regret

            else:
                
                infoset.curr_move_probs[movenum] = 1.0/actions_here
                
            prob_sum += infoset.curr_move_probs[movenum]

        return infoset
        
    def put(self, infosetkey, infoset, actions_here, starting_move):
        
        newinfoset = False; 

        thepos, hash_index = self.get_pos_from_index(infosetkey)
        
        if (self.adding_infosets and thepos >= self.size):      # If we haven't seen it the pos is still at size
            
            newinfoset = True
            
            pos = self.next_infoset_pos
            row = self.next_infoset_pos / self.rowsize
            col = self.next_infoset_pos % self.rowsize
            
            curr_rowsize = self.rowsize if (row < (self.rows-1)) else self.lastrowsize
            
            self.index_keys[hash_index] = infosetkey
            self.index_vals[hash_index] = pos
            
        else:
            
            newinfoset = False; 
            
            pos = thepos 
            row = pos / self.rowsize
            col = pos % self.rowsize
            curr_rowsize = self.rowsize if (row < (self.rows-1)) else self.lastrowsize
            
        x = actions_here
        self.table[row][col] = x
        row, col, pos, curr_rowsize = self.next(row, col, pos, curr_rowsize)

        x = infoset.last_update
        self.table[row][col] = x 
        row, col, pos, curr_rowsize = self.next(row, col, pos, curr_rowsize)

        for i in range(actions_here):
        
            self.table[row][col] = infoset.cfr[starting_move+i]

            row, col, pos, curr_rowsize = self.next(row, col, pos, curr_rowsize)

            self.table[row][col] = infoset.total_move_probs[starting_move+i]

            row, col, pos, curr_rowsize = self.next(row, col, pos, curr_rowsize)

        if (newinfoset and self.adding_infosets):
            
            self.next_infoset_pos = pos
            self.added+=1
            
    def get_pos_from_index(self, infosetkey):
    
        i = infosetkey % self.indexsize
        misses = 0 
        hash_index=0

        while misses < self.indexsize:
            
            if (self.index_keys[i] == infosetkey and self.index_values[i] < self.size): 
            
                hash_index = i 
                return self.index_values[i],hash_index 
            
            elif self.index_values[i] >= self.size:
                
                hash_index = i
                return self.size, hash_index 
            
            i += 1 
            if i >= self.indexsize:
                i = 0 
                
        assert(False)
        
    def dump_to_disk(self, filename):
        
        with open(filename, 'wb') as file:

            # Write metadata
            
            file.write(struct.pack('Q', self.indexsize))
            file.write(struct.pack('Q', self.size))
            file.write(struct.pack('Q', self.rowsize))
            file.write(struct.pack('Q', self.rows))
            file.write(struct.pack('Q', self.lastrowsize))

            # Write the indexes
            
            for i in range(self.indexsize):
                
                file.write(struct.pack('Q', self.index_keys[i]))
                file.write(struct.pack('Q', self.index_values[i]))

            # Write the table
            
            pos, row, col = 0, 0, 0
            curr_row_size = self.rowsize
            
            while pos < self.size:
                
                file.write(struct.pack('Q', self.table[row][col]))
                row, col, pos, curr_row_size = self.next_position(row, col, pos, curr_row_size)

    def read_from_disk(self, filename):
        
        try:
            
            with open(filename, 'rb') as file:
                
                # Read metadata (each 8 bytes)
                
                self.indexsize = struct.unpack('Q', self.read_bytes(file, 8))[0]
                self.size = struct.unpack('Q', self.read_bytes(file, 8))[0]
                self.rowsize = struct.unpack('Q', self.read_bytes(file, 8))[0]
                self.rows = struct.unpack('Q', self.read_bytes(file, 8))[0]
                self.lastrowsize = struct.unpack('Q', self.read_bytes(file, 8))[0]

                # Read the indexes
                
                self.index_keys = [struct.unpack('Q', self.read_bytes(file, 8))[0] for _ in range(self.indexsize)]
                self.index_values = [struct.unpack('Q', self.read_bytes(file, 8))[0] for _ in range(self.indexsize)]

                # Allocate table rows
                
                self.table = []
                
                for i in range(self.rows):
                    
                    if i != self.rows - 1:
                        
                        row = [0.0] * self.rowsize
                        
                    else:
                        
                        row = [0.0] * self.lastrowsize
                        
                    self.table.append(row)

                # Read table rows from the file
                
                pos, row, col = 0, 0, 0
                curr_row_size = self.rowsize
                
                while pos < self.size:
                    
                    self.table[row][col] = struct.unpack('d', self.read_bytes(file, 8))[0]
                    row, col, pos, curr_row_size = self.next(row, col, pos, curr_row_size)
                    
            return True
        
        except (IOError, struct.error, AssertionError) as e:
            
            print(f"Error reading file: {e}")
            return False
        
class GameState():
    
    def __init__(self):
        
        self.p1roll = 0
        self.p2roll = 0 
        self.curbid = 0
        self.prevbid = 0
        self.calling_player = 0
        
    def copy(self):

        ngs = GameState()

        ngs.p1roll = self.p1roll
        ngs.p2roll = self.p2roll
        ngs.curbid = self.curbid
        ngs.prevbid = self.prevbid
        ngs.calling_player = self.calling_player

        return ngs
        
def payoff(gamestate, update_player):
    
    # The actual bid to be tested is the previous (the last one was bluff), the player that made it is the other player, not the one calling bluff
    
    bid = gamestate.prevbid
    bidder = 3-gamestate.calling_player
    
    # Extract quantity and face of the bid from the code 
    
    quant = bid//10
    face = bid%10
    
    # Count number of matching dice
    
    matching=0
    
    for dice in gamestate.p1roll:
        
        if dice == face or dice == params['diefaces']:
            
            matching+=1
            
    for dice in gamestate.p2roll:
        
        if dice == face or dice == params['diefaces']:
            
            matching+=1
    
    # Set correct player as winner based on matching count
    
    if matching >= quant:
        
        winner = bidder
    
    else:
        
        winner = gamestate.calling_player
        
    # Return the payoff in the win or lose cases
        
    if winner == update_player:
        
        return 1.0
    
    else:
        
        return -1.0

def init_infosets(gamestate, player, bidseq):
    
    # This looks like a cfr function but is used to traverse the tree and populate the iss
    
    if gamestate.curbid==numbids+1:     # numbids+1 is the bid number for bluff
        
        return
    
    if gamestate.p1roll == 0:
        
        for i in range(num_chance_outcomes_1):
            
            new_gamestate = gamestate.copy()
            new_gamestate.p1roll = i

            init_infosets(new_gamestate, player, bidseq)
            
        return

    elif gamestate.p2roll == 0:
        
        for i in range(num_chance_outcomes_2):
            
            new_gamestate = gamestate.copy()
            new_gamestate.p2roll = i

            init_infosets(new_gamestate, player, bidseq)
            
        return

    maxbid = numbids if (gamestate.curbid == 0 ) else numbids+1
    actions_here = maxbid - gamestate.curbid
    
    infoset = InfoSet(actions_here)
    
    for i in range(gamestate.curbid+1, maxbid+1):

        new_gamestate = gamestate.copy()
        new_gamestate.prevbid = gamestate.curbid
        new_gamestate.curbid = i
        new_gamestate.calling_player = player
        new_bidseq = bidseq
        newbidseq = new_bidseq or (1 << (numbids+1-i))

        init_infosets(new_gamestate, (3-player), newbidseq)
        
    infosetkey = bidseq
    infosetkey = infosetkey << isc_width
    
    if (player == 1):
        
        infosetkey = infosetkey or gamestate.p1roll
        infosetkey = infosetkey << 1
        iss.put(infosetkey, infoset, actions_here, 0)
        
    elif (player == 2):
        
        infosetkey = infosetkey or gamestate.p2roll
        infosetkey = infosetkey << 1
        infosetkey = infosetkey or 1
        iss.put(infosetkey, infoset, actions_here, 0)
        
    filename = 'iss.initial.dat'
    
    iss.dump_to_disk(filename)

def cfr(gamestate, current_player, bidseq, reach1, reach2, chance_reach,flag, update_player):
    
    # Check if node (gamestate) is terminal, if it is return the payoff for update_player
    
    if gamestate.curbid==numbids+1:     # numbids+1 is the bid number for bluff
        
        return payoff(gamestate=gamestate,update_player=update_player)
    
    # Check if node (gamestate) is a chance node (in our case at the top of the tree), its enough to check if the gamestate has attributes p1roll and p2roll set.
    # Do an if gs.p1roll==0 and if so define an expected value variable and iterate through all possible chance outcomes for player1. For all possible chance
    # outcomes create a gamestate which has that outcome as p1roll, define a new chancereachprobability by multiplying the chanceprobability of current outcome 
    # with current chancereach and recursively call cfr with cfr(new_gs, current_player, bidseq, reach1, reach2, newChanceReach, flag, updatePlayer) nothing 
    # besides chancereach and gs have changed because no actual play has taken place 
    
    if gamestate.p1roll == 0:
        
        ev = 0.0
        
        for i in range(num_chance_outcomes_1):
            
            chance_prob = chance_prob_1[i]
            
            new_gamestate = gamestate.copy(); 
            new_gamestate.p1roll = i; 
            new_chance_reach = chance_prob*chance_reach

            ev += chance_prob*cfr(new_gamestate, current_player, bidseq, reach1, reach2, new_chance_reach, flag, update_player)

        return ev

    # Do an elseif with the same approach but for player2, both this and the previous return EV which is the sum of the recursive cfrs weighted by the chance
    # probabilities of obtaining them

    elif gamestate.p2roll == 0:
        
        ev = 0.0
        
        for i in range(num_chance_outcomes_2):
            
            chance_prob = chance_prob_2[i]
            
            new_gamestate = gamestate.copy()
            new_gamestate.p2roll = i
            new_chance_reach = chance_prob*chance_reach

            ev += chance_prob*cfr(new_gamestate, current_player, bidseq, reach1, reach2, new_chance_reach, flag, update_player)

        return ev
    
    # Check for possible pruning optimization, if flag is 1 and (current_player==update_player==1 and reach2<=0)||(current_player==update_player==2 and reach1<=0)
    # then we set flag to 2. Then if flag is 2 and (current_player==update_player==1 and reach1<=0)||(current_player==update_player==2 and reach2<=0) we 
    # automatically return 0
    
    if (flag==1) and ((current_player==update_player==1 and reach2<=0) or (current_player==update_player==2 and reach1<=0)):
        
        flag=2
        
    if (flag==2) and ((current_player==update_player==1 and reach1<=0) or (current_player==update_player==2 and reach2<=0)):
        
        return 0.0
    
    # Define the number of possible moves, if gamestate.curbid==0 then bluff is not possible, otherwise it is
    
    maxbid = numbids if (gamestate.curbid == 0 ) else numbids+1
    actions_here = maxbid - gamestate.curbid
    
    # Create an Infoset object as defined in line 43 of bluff.h
    
    infoset = InfoSet(actions_here)
    
    # Create an zero vector of length of the number of possible bids
    
    move_ev = [0.0]*actions_here 
    
    # Run getInfoset(gs, player, bidseq, is, 0, numactions) which in turn runs getInfosetKey, as described in 118-141 of bluff.cpp, this needs a function 
    # get to be defined in the class InfosetStore as in line 171 of infosetstore.cpp
    
    infosetkey = bidseq
    infosetkey = infosetkey<<isc_width
    if current_player == 1:
        
        infosetkey = infosetkey or gamestate.p1roll
        infosetkey = infosetkey << 1
        
    elif current_player == 2:
        
        infosetkey = infosetkey or gamestate.p2roll
        infosetkey = infosetkey << 1
        infosetkey = infosetkey or 1
        
    infoset = iss.get(infosetkey, infoset, actions_here, 0)
    
    # Write a for over all the possible moves where you get the current probabilities of the possible moves (which has been updated when calling getInfoset), 
    # create a new game from the previous one where the current move becomes the prev move, the move we're at in the for become the current move and the 
    # new bidsequence is updated by adding a 1 in the position of the bid we're at in the for. Then it calculates the payoff of the subtree where choice i is 
    # made by running cfr with 3-player as player, newreach probabilities updated using the current probabilities of moves (taken from is) and the new gs
    
    strat_ev = 0.0
    action = -1
    
    for i in range(gamestate.curbid+1, maxbid+1):
        
        action+=1
        
        newbidseq = bidseq
        move_prob = infoset.curr_move_probs[action]; 
        newreach1 = move_prob*reach1 if (current_player == 1) else reach1 
        newreach2 = move_prob*reach2 if (current_player == 2) else reach2

        new_gamestate = gamestate.copy() 
        new_gamestate.prevbid = gamestate.curbid
        new_gamestate.curbid = i 
        new_gamestate.calling_player = current_player
        new_bidseq = bidseq
        newbidseq = new_bidseq or (1 << (numbids+1-i))
        
        payoff = cfr(new_gamestate, 3-current_player, newbidseq, newreach1, newreach2, chance_reach, flag, update_player) 
    
        # Put the payoff in slot [action] of an array moveEV and sum to a variable started at 0 the payoff weighted by the action probability
    
        move_ev[action] = payoff 
        strat_ev += move_prob*payoff 
    
    # Define the myreach and oppreach probabilies based on which player is playing
    
    myreach = reach1 if (current_player == 1) else reach2 
    oppreach = reach2 if (current_player == 1) else reach1
    
    # If flag is still 1 and the current player is the updatePlayer then update each element in is.cfr by (moveEvs[a]-stratEv) weigthed by chancereach*oppreach
    
    if (flag == 1 and current_player == update_player):
        
        for a in range(actions_here):
            
            infoset.cfr[a] += (chance_reach*oppreach)*(move_ev[a] - strat_ev) 
    
    # If flag is >=1 and the current player is the updatePlayer then update the strategies i.e. the totalmoveProb of is by adding the current moveProb (updated 
    # with regret when calling getInfoset) weighted by myreach
    
    if (flag >= 1 and current_player == update_player):
    
        for a in range(actions_here):
            
            infoset.totalMoveProbs[a] += myreach*infoset.curMoveProbs[a]; 
    
    # Finally if the current player is the updatePlayer call function put of iss to save the current (modified) is to the iss
    
    if (current_player == update_player):
        
        iss.put(infosetkey, infoset, actions_here, 0)
    
    # Return stratEv
    
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
            
        chance_prob[i]=perm/num_chance_outcomes
    
    return num_chance_outcomes, chance_outcomes, chance_prob
    
if __name__=='main':
    
    # Define parameters
    global params
    global numbids
    global iss
    global isc_width
    
    params = {
        'p1dice':       1,
        'p2dice':       1,
        'diefaces':     6,
        'n_iter':       10000,
        'filename':     None,
    }
    
    numbids = ((params['p1dice']+params['p2dice'])*params['diefaces'])
    
    # Create two variables, one for the number of possible chance outcomes for player 1 and one for player 2
    
    num_chance_outcomes_1, chance_outcomes_1, chance_prob_1 = determine_chance_outcomes(player=1)
    num_chance_outcomes_2, chance_outcomes_2, chance_prob_2 = determine_chance_outcomes(player=2)
        
    # Define iscWidth which is the number of bits needed to encode the chance outcomes, ceiling(log2 of the max of the two numChanceOutcomes)
    
    isc_width = math.ceil(math.log2(max(num_chance_outcomes_1,num_chance_outcomes_2)))
    
    # Create a vector of size numbids where we assign in order the bids in the format (quantity*10)+face,
    # don't forget that wildcard bids are favoured (more probable), follow function initBids() in bluff.cpp.
    
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
        
        iss.read_from_disk(params['filename'])
    
    # Set the bidsequence as a long long 0 or find an equivalent way with strings
    
    bidseq = np.uint64(0)
    
    # Create a for, that runs for the defined number of iterations and at each step for each (of the two) player(s) creates an instance of the GameState class and 
    # runs the cfr algorithm with cfr(gamestate_i, 1, 0, bidseq, 1.0, 1.0, 1, i), gets the expected value this way
    
    for i in range(params['n_inter']):
        
        gs1 = GameState()
        bidseq = np.uint64(0)
        ev1 = cfr(gs1, 1, bidseq, 1.0, 1.0, 1.0, 1, 1)
        
        gs2 = GameState()
        bidseq = np.uint64(0)
        ev2 = cfr(gs2, 1, bidseq, 1.0, 1.0, 1.0, 1, 2)
    
    # If we are at the last iteration (or every few iterations) compute the bounds with the iss.compute_bounds function and best reponses with function
    # compute_best_responses, to write it take after computeBestResponse in file br.cpp
    
        if i==params['n_iter']-1:
            
            b1,b2 = 0.0
            b1,b2 = iss.compute_bounds(b1, b2) 
            print(f" b1 = {b1}, b2 = {b2}, bound = {(2.0*max(b1,b2))}")