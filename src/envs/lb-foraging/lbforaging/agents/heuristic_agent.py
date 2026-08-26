import random
import numpy as np
from foraging import Agent
from foraging.environment import Action


class HeuristicAgent(Agent):
    name = "Heuristic Agent"

    def _center_of_players(self, players):
        coords = np.array([player.position for player in players])
        return np.rint(coords.mean(axis=0))

    def _move_towards(self, target, allowed):

        y, x = self.observed_position
        r, c = target

        if r < y and Action.NORTH in allowed:
            return Action.NORTH
        elif r > y and Action.SOUTH in allowed:
            return Action.SOUTH
        elif c > x and Action.EAST in allowed:
            return Action.EAST
        elif c < x and Action.WEST in allowed:
            return Action.WEST
        else:
            raise ValueError("No simple path found")

    def step(self, obs):
        raise NotImplemented("Heuristic agent is implemented by H1-H4")


class H1(HeuristicAgent):
    """
    H1 智能体始终前往最近的食物。
    """

    name = "H1"

    def step(self, obs):
        try:
            r, c = self._closest_food(obs)
        except TypeError:
            return random.choice(obs.actions)
        y, x = self.observed_position

        if (abs(r - y) + abs(c - x)) == 1:
            return Action.LOAD

        try:
            return self._move_towards((r, c), obs.actions)
        except ValueError:
            return random.choice(obs.actions)


class H2(HeuristicAgent):
    """
    H2 智能体前往最靠近可见玩家中心的可见食物。
    """

    name = "H2"

    def step(self, obs):

        players_center = self._center_of_players(obs.players)

        try:
            r, c = self._closest_food(obs, None, players_center)
        except TypeError:
            return random.choice(obs.actions)
        y, x = self.observed_position

        if (abs(r - y) + abs(c - x)) == 1:
            return Action.LOAD

        try:
            return self._move_towards((r, c), obs.actions)
        except ValueError:
            return random.choice(obs.actions)


class H3(HeuristicAgent):
    """
    H3 智能体始终前往等级相匹配的最近食物。
    """

    name = "H3"

    def step(self, obs):

        try:
            r, c = self._closest_food(obs, self.level)
        except TypeError:
            return random.choice(obs.actions)
        y, x = self.observed_position

        if (abs(r - y) + abs(c - x)) == 1:
            return Action.LOAD

        try:
            return self._move_towards((r, c), obs.actions)
        except ValueError:
            return random.choice(obs.actions)


class H4(HeuristicAgent):
    """
    H4 智能体前往离所有可见玩家最近的可见食物，且这些玩家与 H4 的
    等级总和必须足以装载该食物。
    """

    name = "H4"

    def step(self, obs):

        players_center = self._center_of_players(obs.players)
        players_sum_level = sum([a.level for a in obs.players])

        try:
            r, c = self._closest_food(obs, players_sum_level, players_center)
        except TypeError:
            return random.choice(obs.actions)
        y, x = self.observed_position

        if (abs(r - y) + abs(c - x)) == 1:
            return Action.LOAD

        try:
            return self._move_towards((r, c), obs.actions)
        except ValueError:
            return random.choice(obs.actions)
