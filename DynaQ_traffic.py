import numpy as np
import time

# wrapper for a game that is supported by our DynaQ implementation
# only exposes the information that is required by DynaQ
class DynaQGame:
    def __init__(self, game):
        self.game = game
    def actions(self):
        return self.game.actions
    def game_over(self):
        return self.game.game_over()
    def total_reward(self):
        return self.game.total_reward
    def buses(self):
        return self.game.get_num_buses()
    def state(self):
        return self.game.state_to_str()
    def play(self, action):
        return self.game.play(action)
    def reset(self):
        return self.game.reset()

class DynaQ:
    def __init__(self,
                 game,
                 alpha=0.1,
                 gamma=0.99,
                 epsilon=0.1,
                 epsilon_decay=0.999,
                 n_planning_steps=3,
                 num_episodes=500, #100, 250, 500
                 max_steps_per_episode=2000,
                 verbose=True):
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode

        self.alpha = alpha
        # discount
        self.gamma = gamma
        # greed algorithm factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        # planning steps
        self.n_planning_steps = n_planning_steps

        self.game = game

        self.actions = self.game.actions()

        # initial Q(s,a) and Model(s,a)
        s = self.game.state()
        self.Q = dict()
        self.Q[s] = dict(zip(self.actions, [0]*len(self.actions)))

        self.model = Model()

        self.verbose = verbose

    def run(self):
        rewards = []
        steps = []
        buses = []
        for episode in range(self.num_episodes):
            step = 0
            s = self.game.reset()
            while not self.game.game_over() and (self.max_steps_per_episode < 0 or step < self.max_steps_per_episode):
                # time1 = time.time()

                # get action from epslon-greedy
                a = np.random.choice(e_greedy(self.epsilon * self.epsilon_decay ** step, s, self.Q, self.actions), 1)[0]
                # tell game agent to execute action a
                s_next, r = self.game.play(a)

                # print("state", s_next, r)

                # State-value: allocate space for new state and action
                for state_q in [s, s_next]:
                    if state_q not in self.Q.keys():
                        self.Q[state_q] = dict(zip(self.actions, [0]*len(self.actions)))

                # update State-value
                self.Q[s][a] += self.alpha * (r + self.gamma * max(self.Q[s_next].values()) - self.Q[s][a])

                # update model
                self.model.insert(s, a, s_next, r)

                # update Q using simulated experience from model
                for _ in range(self.n_planning_steps):
                    ss, sa, ss_next, sr = self.model.get_simulate_experience()
                    self.Q[ss][sa] += self.alpha * (sr + self.gamma * max(self.Q[ss_next].values()) - self.Q[ss][sa])

                # current state is next state
                s = s_next
                step += 1
                # print("episode", time.time()-time1)
                if self.verbose and episode % 10 == 0:
                    print(s)
                    print(a)

            rewards.append(self.game.total_reward())
            buses.append(self.game.buses())
            steps.append(step)

            if self.verbose and episode % 10 == 0:
                print('Episode {} over, reward: {}, step: {}, buses {}, model {}'.format(
                    episode, self.game.total_reward(), step, self.game.buses(), 0))
                print('Size of Q: {}'.format(len(self.Q)))

        return self.num_episodes, rewards, steps, buses


class Model:
    def __init__(self):
        # double dict: Model(s,a) --> s', R
        self.model = dict()

        self.use_seed = False
        self.randomseed = 42
        # apply random seed
        if self.use_seed:
            np.random.seed(self.randomseed)

    def insert(self, s, a, s_next, r):
        # state must be tuple
        if s not in self.model.keys():
            self.model[s] = dict()
        # print(s, a, self.model)
        self.model[s][a] = [s_next, r]

    def get_simulate_experience(self):
        # random sample s and a
        s_index = np.random.choice(len(self.model.keys()), 1)[0]
        s = list(self.model.keys())[s_index]
        a = np.random.choice(list(self.model[s].keys()), 1)[0]
        s_next, r = self.model[s][a]
        return s, a, s_next, r


def e_greedy(epsilon, s, Q, actions):
    if np.random.binomial(1, epsilon) == 1:
        # explore
        return np.random.choice(list(actions), 1)
    else:
        # exploit
        max_value = max(Q[s].values())
        return [a for a in actions if Q[s][a] == max_value]

