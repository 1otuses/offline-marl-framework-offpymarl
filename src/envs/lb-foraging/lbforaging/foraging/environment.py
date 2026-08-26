import logging
from collections import namedtuple, defaultdict
from enum import Enum
from itertools import product
from gym import Env
import gym
from gym.utils import seeding
import numpy as np


class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4


class Player:
    def __init__(self):
        self.controller = None
        self.position = None
        self.level = None
        self.field_size = None
        self.score = None
        self.reward = 0
        self.history = None
        self.current_step = None

    def setup(self, position, level, field_size):
        self.history = []
        self.position = position
        self.level = level
        self.field_size = field_size
        self.score = 0

    def set_controller(self, controller):
        self.controller = controller

    def step(self, obs):
        return self.controller._step(obs)

    @property
    def name(self):
        if self.controller:
            return self.controller.name
        else:
            return "Player"


class ForagingEnv(Env):
    """
    包含基于等级的觅食游戏规则与动作的环境类。
    """

    metadata = {"render.modes": ["human"]}

    action_set = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST]
    Observation = namedtuple(
        "Observation",
        ["field", "actions", "players", "game_over", "sight", "current_step", "score"],
    )
    PlayerObservation = namedtuple(
        "PlayerObservation", ["position", "level", "history", "reward", "is_self"]
    )  # 仅当 is_self 时才可获得奖励

    def __init__(
        self,
        players,
        max_player_level,
        field_size,
        max_food,
        sight,
        max_episode_steps,
        force_coop,
        normalize_reward=True,
        food_pos="topright",
        _grid_observation=False,
    ):
        self.food_pos = food_pos
        self._grid_observation = _grid_observation

        self.logger = logging.getLogger(__name__)
        self.seed()
        self.players = [Player() for _ in range(players)]

        self.field = np.zeros(field_size, np.int32)

        self.max_food = max_food
        self._food_spawned = 0.0
        self.max_player_level = max_player_level
        self.sight = sight
        self.force_coop = force_coop
        self._game_over = None

        self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(5)] * len(self.players)))
        self.observation_space = gym.spaces.Tuple(tuple([self._get_observation_space()] * len(self.players)))
        # 非网格观测模式

        self._rendering_initialized = False
        self._valid_actions = None
        self._max_episode_steps = max_episode_steps

        self._normalize_reward = normalize_reward

        self.viewer = None

        self.n_agents = len(self.players)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_observation_space(self):
        if not self._grid_observation:
            """The Observation Space for each agent.
            - all of the board (board_size^2) with foods
            - player description (x, y, level)*player_count
            """
            field_x = self.field.shape[1]
            field_y = self.field.shape[0]
            # field_size = field_x * field_y

            max_food = self.max_food
            max_food_level = self.max_player_level * len(self.players)

            # 将 [x,y,r] 添加到观测中
            min_obs = [0] + [-1, -1, 0] * max_food + [0, 0, 1] * len(self.players)
            max_obs = [1] + \
                      [field_x, field_y, max_food_level] * max_food + \
                      [field_x, field_y, self.max_player_level] * len(self.players)
        else:
            # 网格观测空间
            grid_shape = (1 + 2 * self.sight, 1 + 2 * self.sight)

            # 智能体层：智能体等级
            agents_min = np.zeros(grid_shape, dtype=np.float32)
            agents_max = np.ones(grid_shape, dtype=np.float32) * self.max_player_level

            # 食物层：食物等级
            max_food_level = self.max_player_level * len(self.players)
            foods_min = np.zeros(grid_shape, dtype=np.float32)
            foods_max = np.ones(grid_shape, dtype=np.float32) * max_food_level

            # 可访问层：标记单元格是否可访问
            access_min = np.zeros(grid_shape, dtype=np.float32)
            access_max = np.ones(grid_shape, dtype=np.float32)

            # 合并各层
            min_obs = np.stack([agents_min, foods_min, access_min])
            max_obs = np.stack([agents_max, foods_max, access_max])

        return gym.spaces.Box(np.array(min_obs), np.array(max_obs), dtype=np.float32)

    @classmethod
    def from_obs(cls, obs):

        players = []
        for p in obs.players:
            player = Player()
            player.setup(p.position, p.level, obs.field.shape)
            player.score = p.score if p.score else 0
            players.append(player)

        env = cls(players, None, None, None, None)
        env.field = np.copy(obs.field)
        env.current_step = obs.current_step
        env.sight = obs.sight
        env._gen_valid_moves()

        return env

    @property
    def field_size(self):
        return self.field.shape

    @property
    def rows(self):
        return self.field_size[0]

    @property
    def cols(self):
        return self.field_size[1]

    @property
    def game_over(self):
        return self._game_over

    def _gen_valid_moves(self):
        self._valid_actions = {
            player: [
                action for action in Action if self._is_valid_action(player, action)
            ]
            for player in self.players
        }

    def neighborhood(self, row, col, distance=1, ignore_diag=False):
        if not ignore_diag:
            return self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows),
                max(col - distance, 0) : min(col + distance + 1, self.cols),
            ]

        return (
            self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows), col
            ].sum()
            + self.field[
                row, max(col - distance, 0) : min(col + distance + 1, self.cols)
            ].sum()
        )

    def adjacent_food(self, row, col):
        return (
            self.field[max(row - 1, 0), col]
            + self.field[min(row + 1, self.rows - 1), col]
            + self.field[row, max(col - 1, 0)]
            + self.field[row, min(col + 1, self.cols - 1)]
        )
    
    def adjacent_food_location(self, row, col):
        if row > 0 and self.field[row - 1, col] > 0:
            return row - 1, col
        elif row < self.rows - 1 and self.field[row + 1, col] > 0:
            return row + 1, col
        elif col > 0 and self.field[row, col - 1] > 0:
            return row, col - 1
        elif col < self.cols - 1 and self.field[row, col + 1] > 0:
            return row, col + 1

    def adjacent_players(self, row, col):
        return [
            player
            for player in self.players
            if abs(player.position[0] - row) == 1
            and player.position[1] == col
            or abs(player.position[1] - col) == 1
            and player.position[0] == row
        ]

    def spawn_players(self, max_player_level):
        for player in self.players:

            attempts = 0
            player.reward = 0

            while attempts < 1000:
                row = self.np_random.randint(0, 2)
                col = self.np_random.randint(0, 2)
                #row = self.np_random.integers(0, 2)
                #col = self.np_random.integers(0, 2)
                if self._is_empty_location(row, col):
                    player.setup(
                        (row, col),
                        self.np_random.randint(1, max_player_level),
                        self.field_size,
                    )
                    break
                attempts += 1

    def spawn_food(self, max_food, max_level):

        min_level = max_level if self.force_coop else 1
        level = min_level if min_level == max_level else self.np_random.randint(min_level, max_level)
        if self.food_pos == 'topright':
            self.field[0,     self.cols - 1]     = level # 右上角
        if self.food_pos == 'topright1':
            self.field[0,     self.cols - 1 - 1] = level # 右上角左侧
        if self.food_pos == 'topright2':
            self.field[0 + 1, self.cols - 1 - 1] = level # 右上角的左下方
        if self.food_pos == 'topright3':
            self.field[0 + 1, self.cols - 1]     = level # 右上角下方

        elif self.food_pos == 'right':
            self.field[self.rows // 2,     self.cols - 1]     = level # 右侧
        elif self.food_pos == 'right1':
            self.field[self.rows // 2 - 1, self.cols - 1]     = level # 右侧位置的上方
        elif self.food_pos == 'right2':
            self.field[self.rows // 2 - 1, self.cols - 1 - 1] = level # 右侧位置的左上方
        elif self.food_pos == 'right3':
            self.field[self.rows // 2,     self.cols - 1 - 1] = level # 右侧位置的左方
        elif self.food_pos == 'right4':
            self.field[self.rows // 2 + 1, self.cols - 1 - 1] = level # 右侧位置的左下方
        elif self.food_pos == 'right5':
            self.field[self.rows // 2 + 1, self.cols - 1]     = level # 右侧位置的下方

        elif self.food_pos == 'bottomright':
            self.field[self.rows - 1,     self.cols - 1]     = level # 右下角
        elif self.food_pos == 'bottomright1':
            self.field[self.rows - 1 - 1, self.cols - 1]     = level # 右下角上方
        elif self.food_pos == 'bottomright2':
            self.field[self.rows - 1 - 1, self.cols - 1 - 1] = level # 右下角的左上方
        elif self.food_pos == 'bottomright3':
            self.field[self.rows - 1,     self.cols - 1 - 1] = level # 右下角左侧

        elif self.food_pos == 'bottom':
            self.field[self.rows - 1,     self.cols // 2]     = level # 底部
        elif self.food_pos == 'bottom1':
            self.field[self.rows - 1,     self.cols // 2 + 1] = level # 底部位置的右方
        elif self.food_pos == 'bottom2': 
            self.field[self.rows - 1 - 1, self.cols // 2 + 1] = level # 底部位置的右上方
        elif self.food_pos == 'bottom3':
            self.field[self.rows - 1 - 1, self.cols // 2]     = level # 底部位置的上方
        elif self.food_pos == 'bottom4':
            self.field[self.rows - 1 - 1, self.cols // 2 - 1] = level # 底部位置的左上方
        elif self.food_pos == 'bottom5':
            self.field[self.rows - 1,     self.cols // 2 - 1] = level # 底部位置的左方

        elif self.food_pos == 'bottomleft':
            self.field[self.rows - 1,     0]     = level # 左下角
        elif self.food_pos == 'bottomleft1':
            self.field[self.rows - 1,     0 + 1] = level # 左下角右侧
        elif self.food_pos == 'bottomleft2':
            self.field[self.rows - 1 - 1, 0 + 1] = level # 左下角的右上方
        elif self.food_pos == 'bottomleft3':
            self.field[self.rows - 1 - 1, 0]     = level # 左下角的顶部

        self._food_spawned = self.field.sum()

    def _is_empty_location(self, row, col):

        if self.field[row, col] != 0:
            return False
        for a in self.players:
            if a.position and row == a.position[0] and col == a.position[1]:
                return False

        return True

    def _is_valid_action(self, player, action):
        if action == Action.NONE:
            return True
        elif action == Action.NORTH:
            return (
                player.position[0] > 0
                and self.field[player.position[0] - 1, player.position[1]] == 0
            )
        elif action == Action.SOUTH:
            return (
                player.position[0] < self.rows - 1
                and self.field[player.position[0] + 1, player.position[1]] == 0
            )
        elif action == Action.WEST:
            return (
                player.position[1] > 0
                and self.field[player.position[0], player.position[1] - 1] == 0
            )
        elif action == Action.EAST:
            return (
                player.position[1] < self.cols - 1
                and self.field[player.position[0], player.position[1] + 1] == 0
            )

        self.logger.error("Undefined action {} from {}".format(action, player.name))
        raise ValueError("Undefined action")

    def _transform_to_neighborhood(self, center, sight, position):
        return (
            position[0] - center[0] + min(sight, center[0]),
            position[1] - center[1] + min(sight, center[1]),
        )

    def get_valid_actions(self) -> list:
        return list(product(*[self._valid_actions[player] for player in self.players]))

    def _make_obs(self, player, sight):
        return self.Observation(
            actions=self._valid_actions[player],
            players=[
                self.PlayerObservation(
                    position=self._transform_to_neighborhood(
                        player.position, sight, a.position
                    ),
                    level=a.level,
                    is_self=a == player,
                    history=a.history,
                    reward=a.reward if a == player else None,
                )
                for a in self.players
                if (
                    min(
                        self._transform_to_neighborhood(
                            player.position, sight, a.position
                        )
                    )
                    >= 0
                )
                and max(
                    self._transform_to_neighborhood(
                        player.position, sight, a.position
                    )
                )
                <= 2 * sight
            ],
            # TODO：是否还需检查最大值？
            field=np.copy(self.neighborhood(*player.position, sight)),
            game_over=self.game_over,
            sight=sight,
            current_step=self.current_step,
            score = player.score,
        )

    def _make_gym_obs(self, sight):
        def make_obs_array(observation):
            obs = np.zeros(self.observation_space[0].shape, dtype=np.float32)
            # obs[: observation.field.size] = observation.field.flatten()
            # 当前玩家始终排在首位
            seen_players = [p for p in observation.players if p.is_self] + [
                p for p in observation.players if not p.is_self
            ]

            for i in range(self.max_food):
                obs[3 * i] = -1
                obs[3 * i + 1] = -1
                obs[3 * i + 2] = 0

            for i, (y, x) in enumerate(zip(*np.nonzero(observation.field))):
                obs[3 * i] = y
                obs[3 * i + 1] = x
                obs[3 * i + 2] = observation.field[y, x]

            for i in range(len(self.players)):
                obs[self.max_food * 3 + 3 * i] = -1
                obs[self.max_food * 3 + 3 * i + 1] = -1
                obs[self.max_food * 3 + 3 * i + 2] = 0

            for i, p in enumerate(seen_players):
                obs[self.max_food * 3 + 3 * i] = p.position[0]
                obs[self.max_food * 3 + 3 * i + 1] = p.position[1]
                obs[self.max_food * 3 + 3 * i + 2] = p.level

            obs[-1] = observation.score

            return obs

        def make_global_grid_arrays():
            """
            为网格观测空间创建全局数组。
            """
            grid_shape_x, grid_shape_y = self.field_size
            grid_shape_x += 2 * self.sight
            grid_shape_y += 2 * self.sight
            grid_shape = (grid_shape_x, grid_shape_y)

            agents_layer = np.zeros(grid_shape, dtype=np.float32)
            for player in self.players:
                player_x, player_y = player.position
                agents_layer[player_x + self.sight, player_y + self.sight] = player.level

            foods_layer = np.zeros(grid_shape, dtype=np.float32)
            foods_layer[self.sight:-self.sight, self.sight:-self.sight] = self.field.copy()

            access_layer = np.ones(grid_shape, dtype=np.float32)
            # 边界外区域不可访问
            access_layer[:self.sight, :] = 0.0
            access_layer[-self.sight:, :] = 0.0
            access_layer[:, :self.sight] = 0.0
            access_layer[:, -self.sight:] = 0.0
            # 智能体所在位置不可访问
            for player in self.players:
                player_x, player_y = player.position
                access_layer[player_x + self.sight, player_y + self.sight] = 0.0
            # 食物位置不可访问
            foods_x, foods_y = self.field.nonzero()
            for x, y in zip(foods_x, foods_y):
                access_layer[x + self.sight, y + self.sight] = 0.0

            return np.stack([agents_layer, foods_layer, access_layer])

        def get_agent_grid_bounds(agent_x, agent_y):
            return agent_x, agent_x + 2 * sight + 1, agent_y, agent_y + 2 * sight + 1

        def get_player_reward(observation):
            for p in observation.players:
                if p.is_self:
                    return p.reward

        observations = [self._make_obs(player, sight=sight) for player in self.players]
        if self._grid_observation:
            layers = make_global_grid_arrays()
            agents_bounds = [get_agent_grid_bounds(*player.position) for player in self.players]
            nobs = tuple([layers[:, start_x:end_x, start_y:end_y] for start_x, end_x, start_y, end_y in agents_bounds])
        else:
            nobs = tuple([make_obs_array(obs) for obs in observations])
        nreward = [get_player_reward(obs) for obs in observations]
        ndone = [obs.game_over for obs in observations]
        # ninfo = [{'observation': obs} for obs in observations]
        ninfo = {}

        return nobs, nreward, ndone, ninfo
    
    def reset(self):
        self.field = np.zeros(self.field_size, np.int32)
        self.spawn_players(self.max_player_level)
        player_levels = sorted([player.level for player in self.players])

        self.spawn_food(
            self.max_food, max_level=sum(player_levels[:3])
        )
        self.current_step = 0
        self._game_over = False
        self._gen_valid_moves()
        nobs, _, _, _ = self._make_gym_obs(sight=self.sight)
        return nobs
        #return nobs, _ # To cater gym==0.26.2

    def step(self, actions):
        self.current_step += 1

        for p in self.players:
            p.reward = 0

        actions = [
            Action(a) if Action(a) in self._valid_actions[p] else Action.NONE
            for p, a in zip(self.players, actions)
        ]

        # 检查动作是否有效
        for i, (player, action) in enumerate(zip(self.players, actions)):
            if action not in self._valid_actions[player]:
                self.logger.info(
                    "{}{} attempted invalid action {}.".format(
                        player.name, player.position, action
                    )
                )
                actions[i] = Action.NONE

        # 移动玩家
        # 如果两个或更多玩家尝试移动到同一位置，则全部移动失败
        collisions = defaultdict(list)

        # 所以检查是否有碰撞
        for player, action in zip(self.players, actions):
            if action == Action.NONE:
                collisions[player.position].append(player)
            elif action == Action.NORTH:
                collisions[(player.position[0] - 1, player.position[1])].append(player)
            elif action == Action.SOUTH:
                collisions[(player.position[0] + 1, player.position[1])].append(player)
            elif action == Action.WEST:
                collisions[(player.position[0], player.position[1] - 1)].append(player)
            elif action == Action.EAST:
                collisions[(player.position[0], player.position[1] + 1)].append(player)

        # 移动未发生碰撞的玩家

        for k, v in collisions.items():
            if len(v) > 1:  # 确保同一位置最多只有一个玩家到达
                continue
            v[0].position = k

        # 最后处理装载动作
        for player in self.players:
            # 寻找相邻的食物
            if self.adjacent_food(*player.position) == 0:
                # 没有相邻的食物
                continue
            frow, fcol = self.adjacent_food_location(*player.position)
            food = self.field[frow, fcol]

            adj_players = self.adjacent_players(frow, fcol)

            adj_player_level = sum([a.level for a in adj_players])

            if adj_player_level < food:
                # 装载失败
                continue

            # 否则食物装载成功，每个玩家获得分数
            for a in adj_players:
                a.reward = float(a.level * food)
                if self._normalize_reward:
                    a.reward = a.reward / float(
                        adj_player_level * self._food_spawned
                    )  # 奖励归一化
            # 食物被移除
            self.field[frow, fcol] = 0

        self._game_over = (
            self.field.sum() == 0 or self._max_episode_steps <= self.current_step
        )
        self._gen_valid_moves()

        for p in self.players:
            p.score += p.reward

        return self._make_gym_obs(sight=self.sight)

    def _init_render(self):
        from .rendering import Viewer

        self.viewer = Viewer((self.rows, self.cols))
        self._rendering_initialized = True

    def render(self, mode="human"):
        if not self._rendering_initialized:
            self._init_render()

        return self.viewer.render(self, return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()

    
    def get_state(self):
        if not self._grid_observation:
            # 新状态
            state = np.zeros(self.observation_space[0].shape, dtype=np.float32)

            # 食物位置和等级
            for i, (y, x) in enumerate(zip(*np.nonzero(self.field))):
                state[3 * i] = y
                state[3 * i + 1] = x
                state[3 * i + 2] = self.field[y, x]

            # 玩家位置和等级
            for i, p in enumerate(self.players):
                state[self.max_food * 3 + 3 * i] = p.position[0]
                state[self.max_food * 3 + 3 * i + 1] = p.position[1]
                state[self.max_food * 3 + 3 * i + 2] = p.level

            state[-1] = sum([player.score for player in self.players])
            return state
        else:
            agents_layer = np.zeros(self.field_size, dtype=np.float32)
            for player in self.players:
                player_x, player_y = player.position
                agents_layer[player_x, player_y] = player.level

            foods_layer = np.zeros(self.field_size, dtype=np.float32)
            foods_layer = self.field.copy()

            access_layer = np.ones(self.field_size, dtype=np.float32)
            # 智能体所在位置不可访问
            for player in self.players:
                player_x, player_y = player.position
                access_layer[player_x, player_y] = 0.0
            # 食物位置不可访问
            foods_x, foods_y = self.field.nonzero()
            for x, y in zip(foods_x, foods_y):
                access_layer[x, y] = 0.0

            return np.stack([agents_layer, foods_layer, access_layer]).flatten()

    def get_state_size(self):
        if not self._grid_observation:
            return self.observation_space[0].shape[0]
        else:
            return self.rows * self.cols * 3