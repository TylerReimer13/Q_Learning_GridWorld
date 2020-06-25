import numpy as np
import random
import matplotlib.pyplot as plt


class Environment:
    def __init__(self, rows, cols, start_pos, goal_pos):
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((rows, cols))
        self.start_pos = start_pos
        self.next_state = start_pos
        self.reward_tbl = np.ones((rows, cols)) * -1
        self.board[start_pos[0]][start_pos[1]] = 1.
        self.goal_pos = np.array(goal_pos)
        self.generate_reward_table()

        self.action_dict = {'U': np.array([-1, 0]),
                            'D': np.array([1, 0]),
                            'R': np.array([0, 1]),
                            'L': np.array([0, -1])
                            }

        self.current_pos = np.array(start_pos)
        self.reward_count = 0.
        self.goal_found = False

    @property
    def states(self):
        return [(x, y) for x in range(self.rows) for y in range(self.cols)]

    @property
    def num_states(self):
        return self.rows * self.cols

    @property
    def num_actions(self):
        return len(list(self.action_dict.keys()))

    @property
    def rewards(self):
        return self.reward_tbl

    @property
    def current_position(self):
        return self.current_pos

    @property
    def current_reward(self):
        return self.reward_count

    def check_in_bounds(self, possible_move):
        proceed = True
        future_pos = self.current_pos+possible_move
        if future_pos[0] < 0 or future_pos[0] > self.rows-1:
            proceed = False
        if future_pos[1] < 0 or future_pos[1] > self.cols-1:
            proceed = False
        return proceed

    def __call__(self, action):
        move = self.action_dict[action]
        if self.check_in_bounds(move):
            self.current_pos += move
            r = self.reward_tbl[self.current_pos[0]][self.current_pos[1]]
        else:
            self.current_pos += np.array([0, 0])
            r = -3

        self.reward_count += r
        self.next_state = self.current_pos
        comparison = self.current_pos == self.goal_pos
        if comparison.all():
            self.goal_found = True

        return self.goal_found, tuple(self.next_state), r

    def generate_reward_table(self):
        self.reward_tbl[self.goal_pos[0]][self.goal_pos[1]] = 10.

    def reset(self):
        self.goal_found = False
        self.reward_count = 0.
        self.current_pos = tuple(self.start_pos)
        return self.current_pos


class Agent:
    def __init__(self, action_table, states, alpha=.9, gamma=.9):
        self.action_table = action_table
        num_actions = len(self.action_table)
        self.enumerated_actions = dict(zip(range(num_actions), self.action_table))
        self.states = states
        self.q_table = dict([(s, np.zeros(num_actions)) for s in self.states])
        self.alpha = alpha  # Learning Rate
        self.gamma = gamma  # Discount Factor

    @property
    def q_values(self):
        return self.q_table

    def random_policy(self):
        # Only used for testing/debugging
        select_action = random.choice(self.action_table)
        return select_action

    def act(self, current_state, episode_num):
        rand_num = np.random.uniform(0., 1.)
        greedy = .99 - np.exp(-(1/8)*episode_num)
        if rand_num > greedy:
            action_idx = random.choice(list(self.enumerated_actions.keys()))
        else:
            action_idx = np.argmax(self.q_table[current_state])
        selected_action = self.enumerated_actions[action_idx]
        return selected_action, action_idx

    def update_q_table(self, current, nxt, actn, rwd):
        best_action = np.max(self.q_table[nxt])
        self.q_table[current][actn] += self.alpha*(rwd+(self.gamma*best_action)-self.q_table[current][actn])


start = np.array([0, 0])
goal = np.array([6, 5])
action_list = ['U', 'D', 'L', 'R']
my_env = Environment(8, 8, start, goal)
my_agent = Agent(action_list, my_env.states)

episode_count = []
reward_hist = []
moves_hist = []
episode = 0

for i in range(100):
    done = False
    episode += 1
    episode_count.append(episode)
    tot_reward = 0.
    num_moves = 0
    state = my_env.reset()
    while True:
        action, idx = my_agent.act(state, episode)
        done, next_state, reward = my_env(action)
        my_agent.update_q_table(state, next_state, idx, reward)
        state = next_state
        tot_reward += reward
        num_moves += 1
        if done:
            break
    reward_hist.append(tot_reward)
    moves_hist.append(num_moves)


print('FINAL Q VALUES:', my_agent.q_values)

plt.title('Total Number of moves per Episode')
plt.xlabel('Episode')
plt.ylabel('Num of Moves')
plt.grid()
plt.plot(episode_count, moves_hist)
plt.show()

plt.title('Total Reward per Episode')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.grid()
plt.plot(episode_count, reward_hist)
plt.show()
