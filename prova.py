from itertools import combinations_with_replacement, product
from graphviz import Digraph

# Function to perform backwards induction, it is a recursive function which
# has to be called by passing the root node. Still not tested

def backward_induction(node):
    if node.is_terminal():
        return node.payoffs  # Terminal node, return payoffs

    # For decision nodes, compute the optimal action
    optimal_action = None
    optimal_payoff = None

    for action, child in node.children.items():
        child_payoff = backward_induction(child)

        if optimal_payoff is None or is_better(child_payoff, optimal_payoff, node.player):
            optimal_action = action
            optimal_payoff = child_payoff

    node.payoffs = optimal_payoff  # Assign the computed optimal payoff to this node
    node.optimal_action = optimal_action  # Record the optimal action
    return optimal_payoff

# Function to choose the preferred path. Still not tested

def is_better(payoff1, payoff2, player):
    """
    Determines whether payoff1 is better than payoff2 for the given player.
    For simplicity, assume players are 1 (maximize first value) and 2 (maximize second value).
    """
    if player == 1:
        return payoff1[0] > payoff2[0]
    elif player == 2:
        return payoff1[1] > payoff2[1]
    return False

# Class that allows to model a Node of a tree.
# Each node has a player which is the player that takes the decision starting from that node,
# a payoff which is None everywhere except for terminal nodes, where it is a list of the payoffs of the players
# a move_path which is the sequence of moves that goes to that node, might be useful for backwards induction
# childrens, in the format of a dictionary where the 'key' is the move that goes to the children and the 'value' is
# another instance of the class, representing another node. 

class GameNode:
    def __init__(self, player=None, payoffs=None, move_path=None):
        if move_path is None:
            move_path = [] 
        self.player = player
        self.payoffs = payoffs
        self.move_path = move_path  
        self.children = {}  

    def is_terminal(self):
        return self.payoffs is not None

# From the current bid, following the rules of the game, calculate the legal bids.

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

# This is a recursive function that builds a tree, given the number of player,
# number of dice per player, number of faces on each die and the result of the initial throws.
# This function doesn't build the full tree with the Nature choices for the two players (dice throws)
# at the beginning, each of the possible dice throws is treated separately. 

def build_complete_tree(node, num_players, num_dice, num_faces, dice_throws, current_bid=None):

    # If the parent node is one reached by playing 'Bluff' or 'Check' there's no need to continue since they don't have children
    if current_bid == 'Bluff' or current_bid == 'Check':
        return
    
    # Calculate the legal bids

    possible_bids = available_bids(current_bid=current_bid, num_dice=num_dice, num_faces=num_faces)

    player = (node.player % num_players) + 1  # Calculate what is the player that needs to take the next choice 

    for bid in possible_bids: # Create a children for each possible bid

        # Update the current move path

        new_move_path = node.move_path + [bid]

        # Create a new GameNode for this child

        new_node = GameNode(player=player, move_path=new_move_path)

        # Add the new node as a child of the current node

        node.children[bid] = new_node

        # Create a list of the results of the thrown dice

        dice=[item for sublist in dice_throws for item in sublist]

        # Recursively build the tree for this new node (if not Bluff or Check)
        # This means that, after attaching the new node to the tree, we call this function on
        # the new node so that its children are created, and so on...

        if bid != "Bluff" and bid != "Check":
            build_complete_tree(new_node, num_players=num_players, num_dice=num_dice, num_faces=num_faces, dice_throws=dice_throws, current_bid=bid)
        elif bid=='Check':  # If 'Check' or 'Bluff' are played, we instead calculate the payoffs. Check is not supported yet
            print()
        elif bid=='Bluff':

            # Create the payoff vector in case the last bid was correct (all -1s except for the one that did the last bid, might be changed)

            payoffs= [-1]*num_players
            payoffs[node.children[bid].player-1]=1

            # Check whether the bid was correct or not and set the payoffs or their opposite

            if current_bid[0]<=sum(1 for die in dice if die == current_bid[1] or die == num_faces):
                print(payoffs)
                node.children[bid].payoffs=payoffs
            else:
                print(payoffs*(-1))
                node.children[bid].payoffs=[-1*x for x in payoffs]

# This function creates all the trees, one for each dice throw, given the number of player,
# the number of dice per player, and the number of faces per dice.

def build_trees(num_players=2, num_dice=1, num_faces=6):

    trees = {}
    dice = []

    # Create all possible combinations of dice throws per player

    for _ in range(num_players):
        dice.append(list(combinations_with_replacement(iterable=range(1, num_faces + 1), r=num_dice)))

    # Create all possible combination of dice throws in general

    all_combinations = list(product(*dice))

    # For each combination build a tree

    for dice_throws in all_combinations:

        # Initialize the root node for this tree

        root_node = GameNode(player=1, move_path=[])

        # Build the tree from the root node

        build_complete_tree(root_node, num_players=num_players, num_dice=num_dice, num_faces=num_faces, current_bid=None, dice_throws=dice_throws)

        # Add the tree to the dictionary
        trees[dice_throws] = root_node

    return trees

# This function takes a personalized Tree (one of those we built before) and transforms it in a Graphviz dot tree for visualization purposes

def build_dot_from_tree(node, dot=None):

    if dot is None:
        dot = Digraph(strict=True)  # Directed graph for tree structure

    # Put the number of the player playing on each node as a label

    label = f'{node.player}'

    # If the node is a terminal one, add the payoffs to the label

    if node.is_terminal(): 

        print(node.payoffs)
        label += f"\nPayoffs: {node.payoffs}"

    # Add the current node to the graph with the label
    dot.node(str(id(node)), label=label)

    # Recursively add the children to the graph
    for bid, child in node.children.items():
        dot.edge(str(id(node)), str(id(child)), label=str(bid))  # Label edges with the action

        # Call the function recursively to build all the tree
        build_dot_from_tree(child, dot)

    return dot

# Normal render function for Graphviz

def render_tree(dot):
    """ Render and view the tree using graphviz """
    dot.render('game_tree', format='png', view=True)

# Build trees for a simple version of the game

trees = build_trees(num_players=2, num_dice=1, num_faces=3)

# Take a specific tree of those created

root_node = trees[((2,),(1,))]

# Build the dot graph from the root node of the tree
dot = build_dot_from_tree(root_node)

# Render the tree to a PNG file and display it
render_tree(dot)


