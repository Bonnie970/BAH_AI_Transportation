import numpy as np

class DynaQ:
    def __init__(self, game, n_eps):
        self.n_eps = n_eps

        # step sizeimport numpy as np

        self.alpha = 0.7  # 0.1
        # discount
        self.gamma = 0.95
        # greed algorithm factor
        self.epslon = 0.1
        # planning steps
        self.n = 5

        self.mazeIndex = 0
        self.changeMaze = False

        self.game = game
        self.episode = 0

        # get initial state of game
        self.s = self.game.state

        # cumulative reward, steps
        self.steps = 0

        self.actions = self.game.actions

        # initial Q(s,a) and Model(s,a)
        self.Q = dict()
        print(self.s, type(self.s))
        self.Q[self.s] = dict(zip(self.actions, [0]*len(self.actions)))
        self.model = Model()

        self.verbose = False

    def run(self):
        rewards = []
        for _ in range(self.n_eps):

            while not self.game.game_over:
                # current state
                s = self.s
                # get action from epslon-greedy
                a = np.random.choice(e_greedy(self.epslon, s, self.Q, self.actions), 1)[0]
                # tell game agent to execute action a
                s_next, r = self.game.play(a)

                self.steps += 1

                if self.verbose:
                    if self.steps % 1000 == 0:
                        print(self.steps)
                        print(s, a, s_next, r)

                # State-value: allocate space for new state and action
                for involving_state in [s, s_next]:
                    if involving_state not in self.Q.keys():
                        self.Q[involving_state] = dict(zip(self.actions, [0]*len(self.actions)))

                # update State-value
                self.Q[s][a] += self.alpha * (\
                    r + self.gamma * max(self.Q[s_next].values()) \
                    - self.Q[s][a])

                # update model
                self.model.insert(s, a, s_next, r)

                # update Q using simulated experience from model
                for _ in range(self.n):
                    ss, sa, ss_next, sr = self.model.get_simulate_experience()
                    self.Q[ss][sa] += self.alpha * ( \
                        sr + self.gamma * max(self.Q[ss_next].values()) \
                        - self.Q[ss][sa])

                # current state is next state
                self.s = s_next

            if self.game.game_over:
                rewards.append(self.game.total_reward)
                self.episode += 1
                if self.verbose:
                    print('Episode {} over, reward: {}, step: {}'.format(self.episode, self.game.total_reward, self.steps))

                self.s = self.game.reset()
                self.steps = 0

        return rewards


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


def e_greedy(epslon, s, Q, actions):
    # explore
    if np.random.binomial(1, epslon) == 1:
        return np.random.choice(list(actions), 1)
    else:
        # exploit
        max_value = max(Q[s].values())
        return [a for a in actions if Q[s][a] == max_value]

