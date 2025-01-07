import random
import numpy as np

class PerudoGame:
    def __init__(self, num_players=2, num_dice_per_player=5):
        self.num_players = num_players
        self.num_dice_per_player = num_dice_per_player
        self.dice = {player: [random.randint(1, 6) for _ in range(num_dice_per_player)] 
                     for player in range(num_players)}
        self.current_bid = None  # Format: (quantity, face)
        self.active_player = 0

    def roll_dice(self):
        for player in range(self.num_players):
            self.dice[player] = [random.randint(1, 6) for _ in range(len(self.dice[player]))]

    def is_valid_bid(self, quantity, face):
        if self.current_bid is None:
            return True
        return (quantity > self.current_bid[0]) or (quantity == self.current_bid[0] and face > self.current_bid[1])

    def make_bid(self, player, quantity, face):
        if not self.is_valid_bid(quantity, face):
            raise ValueError("Invalid bid!")
        self.current_bid = (quantity, face)
        self.active_player = (player + 1) % self.num_players

    def call_bluff(self):
        quantity, face = self.current_bid
        actual_count = sum(dice.count(face) for dice in self.dice.values())
        bluff = actual_count < quantity
        return bluff


class CounterfactualRegretMinimizer:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.regret_sum = np.zeros(num_actions)
        self.strategy = np.zeros(num_actions)
        self.strategy_sum = np.zeros(num_actions)

    def get_strategy(self, realization_weight):
        positive_regrets = np.maximum(self.regret_sum, 0)
        normalizing_sum = np.sum(positive_regrets)
        if normalizing_sum > 0:
            self.strategy = positive_regrets / normalizing_sum
        else:
            self.strategy = np.ones(self.num_actions) / self.num_actions
        self.strategy_sum += realization_weight * self.strategy
        return self.strategy

    def get_average_strategy(self):
        normalizing_sum = np.sum(self.strategy_sum)
        if normalizing_sum > 0:
            return self.strategy_sum / normalizing_sum
        return np.ones(self.num_actions) / self.num_actions

    def update_regret(self, action_utilities, realization_weight):
        for a in range(self.num_actions):
            print(a)
            self.regret_sum[a] += realization_weight * (action_utilities[a] - action_utilities.dot(self.strategy))


class PerudoTrainer:
    def __init__(self, num_players=2, num_dice=5):
        self.game = PerudoGame(num_players, num_dice)
        self.regret_minimizers = [CounterfactualRegretMinimizer(num_actions=2) for _ in range(num_players)]

    def train(self, iterations=10000):
        for _ in range(iterations):
            self.game.roll_dice()
            utilities = self.play_round()
            for p in range(self.game.num_players):
                self.regret_minimizers[p].update_regret(utilities[p], 1.0)

    def play_round(self):
        utilities = np.zeros(self.game.num_players)
        for p in range(self.game.num_players):
            strategy = self.regret_minimizers[p].get_strategy(1.0)
            action = np.random.choice(len(strategy), p=strategy)
            if action == 0:  # Bluff
                utilities[p] = -1  # Penalty for being called on a bluff
            elif action == 1:  # Truth
                utilities[p] = 1  # Reward for truthful play
        return utilities


# Example usage
if __name__ == "__main__":
    trainer = PerudoTrainer(num_players=2, num_dice=5)
    trainer.train(iterations=1000)

    # Print average strategies
    for player_id, rm in enumerate(trainer.regret_minimizers):
        print(f"Player {player_id} strategy: {rm.get_average_strategy()}")
