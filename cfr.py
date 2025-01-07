import random
import math
from collections import defaultdict

class CFR:
    def __init__(self, iss):
        self.iss = iss
        self.regret_sum = defaultdict(lambda: defaultdict(float))
        self.strategy_sum = defaultdict(lambda: defaultdict(float))
    
    def get_strategy(self, infoset, t):
        strategy = defaultdict(float)
        normalizing_sum = 0
        for action in self.iss.store[infoset][1]:
            strategy[action] = max(self.regret_sum[infoset][action], 0)
            normalizing_sum += strategy[action]
        for action in strategy:
            if normalizing_sum > 0:
                strategy[action] /= normalizing_sum
            else:
                strategy[action] = 1.0 / len(self.iss.store[infoset][1])
            self.strategy_sum[infoset][action] += strategy[action]
        return strategy
    
    def cfr(self, gs, player, t, p0, p1):
        if gs.is_terminal():
            return gs.get_payoff(player)
        infoset = gs.get_infoset(player)
        strategy = self.get_strategy(infoset, t)
        util = defaultdict(float)
        node_util = 0
        for action in strategy:
            next_gs = gs.copy()
            next_gs.apply_action(action)
            if player == 1:
                util[action] = self.cfr(next_gs, 2, t, p0 * strategy[action], p1)
            else:
                util[action] = self.cfr(next_gs, 1, t, p0, p1 * strategy[action])
            node_util += strategy[action] * util[action]
        for action in strategy:
            regret = util[action] - node_util
            if player == 1:
                self.regret_sum[infoset][action] += p1 * regret
            else:
                self.regret_sum[infoset][action] += p0 * regret
        return node_util
    
    def train(self, iterations):
        for t in range(1, iterations + 1):
            gs = GameState()
            self.cfr(gs, 1, t, 1, 1)
    
    def get_average_strategy(self):
        avg_strategy = defaultdict(lambda: defaultdict(float))
        for infoset in self.strategy_sum:
            normalizing_sum = sum(self.strategy_sum[infoset].values())
            for action in self.strategy_sum[infoset]:
                if normalizing_sum > 0:
                    avg_strategy[infoset][action] = self.strategy_sum[infoset][action] / normalizing_sum
                else:
                    avg_strategy[infoset][action] = 1.0 / len(self.strategy_sum[infoset])
        return avg_strategy

# Example usage
iss = InfoSetStore()
initializeInfosets(iss)
cfr = CFR(iss)
cfr.train(10000)
avg_strategy = cfr.get_average_strategy()
print(avg_strategy)