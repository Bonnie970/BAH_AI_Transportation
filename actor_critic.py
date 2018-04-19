import numpy as np
import tensorflow as tf
import itertools
from matplotlib import pyplot as plt
from env_transporation_simulator import TrafficSimulator

########################################################################################################################

precision = 6
epsilon = 10 ** (-precision)

########################################################################################################################


# for tracking test results for each episode
class EpisodeStats:
    def __init__(self, num_episodes):
        self.lengths = np.zeros(num_episodes)
        self.rewards = np.zeros(num_episodes)
        self.actions = [[] for _ in range(num_episodes)]

    def __str__(self):
        return "episode_lengths: " + str(self.lengths) + \
                "\nepisode_actions: " + str(self.actions)

    def add(self, i, reward, t, action):
        self.rewards[i] += reward
        self.lengths[i] = t
        self.actions[i].append(action)


# policy function approximator
class PolicyEstimator():
    def __init__(self,
                 env,
                 alpha_theta=0.9,
                 lambda_trace=0.5,
                 gamma=1.0,
                 useTrace=True,
                 tag=""
                 ):
        self.env = env
        self.scope = "policy_estimator-" + tag
        self.alpha_theta = alpha_theta
        self.lambda_trace = lambda_trace
        self.gamma = gamma
        self.useTrace = useTrace

        self._theta = np.zeros((self.env.observation_space_n, self.env.action_space_n), np.float32)
        self._trace = np.zeros((self.env.observation_space_n, self.env.action_space_n), np.float32)

        with tf.variable_scope(self.scope):
            # inputs
            self.state = tf.placeholder(tf.int32, [], "state")
            self.action = tf.placeholder(tf.int32, name="action")
            self.delta = tf.placeholder(tf.float32, name="delta")
            self.I = tf.placeholder(tf.float32, name="I")

            self.theta = tf.placeholder(tf.float32, name="theta",
                                        shape=(self.env.observation_space_n, self.env.action_space_n))
            self.trace = tf.placeholder(tf.float32, name="trace",
                                        shape=(self.env.observation_space_n, self.env.action_space_n))

            # prediction  computation
            state_one_hot = tf.one_hot(self.state, self.env.observation_space_n)
            self.action_probs = tf.squeeze(tf.nn.softmax(tf.tensordot(state_one_hot, self.theta, axes=1)))

            # updates
            action_prob = tf.log(tf.gather(self.action_probs, [self.action]))
            action_prob_grad = tf.gradients(action_prob, [self.theta])

            self.trace_new = tf.squeeze(self.gamma * self.lambda_trace * self.trace + self.I * action_prob_grad)

            if useTrace:
                self.theta_new = tf.squeeze(self.theta + self.alpha_theta * self.delta * self.trace_new)
            else:
                self.theta_new = tf.squeeze(self.theta + self.alpha_theta * self.I * self.delta * action_prob_grad)

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_probs, {self.state: state, self.theta: self._theta})

    def update(self, state, delta, action, I, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state,
                     self.delta: delta,
                     self.action: action,
                     self.I: I,
                     self.theta: self._theta,
                     self.trace: self._trace}
        if self.useTrace:
            self._theta, self._trace = sess.run([self.theta_new, self.trace_new], feed_dict)
        else:
            self._theta = sess.run(self.theta_new, feed_dict)


# value function approximator
# value function trace is computed here when enabled
class ValueEstimator():
    def __init__(self,
                 env,
                 alpha_w=0.9,
                 lambda_trace=0.5,
                 gamma=1.0,
                 useTrace=True,
                 tag=""
                 ):
        self.env = env
        self.scope = "value_estimator-" + tag
        self.alpha_w = alpha_w
        self.lambda_trace = lambda_trace
        self.gamma = gamma
        self.useTrace = useTrace

        self._w = np.zeros((self.env.observation_space_n), np.float32)
        self._trace = np.zeros((self.env.observation_space_n), np.float32)

        with tf.variable_scope(self.scope):
            # inputs
            self.state = tf.placeholder(tf.int32, [], "state")
            self.delta = tf.placeholder(tf.float32, name="delta")
            self.I = tf.placeholder(tf.float32, name="I")
            self.w = tf.placeholder(tf.float32, name="w", shape=(self.env.observation_space_n,))
            self.trace = tf.placeholder(tf.float32, name="trace", shape=(self.env.observation_space_n,))

            # prediction computation
            state_one_hot = tf.one_hot(self.state, self.env.observation_space_n)
            self.value = tf.squeeze(tf.reduce_sum(tf.multiply(state_one_hot, self.w)))

            # updates
            grad_v = tf.gradients(self.value, [self.w])

            if useTrace:
                self.trace_new = tf.squeeze(self.gamma * self.lambda_trace * self.trace + self.I * grad_v)
                self.w_new = tf.squeeze(self.w + self.alpha_w * self.delta * self.trace_new)
            else:
                self.w_new = tf.squeeze(self.w + self.I * self.alpha_w * self.delta * grad_v)

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state, self.w: self._w}
        return sess.run(self.value, feed_dict)

    def update(self, state, delta, I, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state,
                     self.delta: delta,
                     self.I: I,
                     self.w: self._w,
                     self.trace: self._trace}
        if self.useTrace:
            self._w, self._trace = sess.run([self.w_new, self.trace_new], feed_dict)
        else:
            self._w = sess.run(self.w_new, feed_dict)


# base actor critic algorithm
class ActorCritic:
    def __init__(self,
                 env,
                 policy_estimator,
                 value_estimator,
                 gamma=0.99,
                 num_episodes=100,
                 max_iters_per_ep=10000
                 ):
        # variables
        self.env = env
        self.policy_estimator = policy_estimator
        self.value_estimator = value_estimator
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.max_iters_per_ep = max_iters_per_ep

        self.stats = EpisodeStats(self.num_episodes)

    def run(self):
        for i_episode in range(self.num_episodes):
            state = self.env.reset()
            I = 1.0

            for t in itertools.count():

                action_probs = self.policy_estimator.predict(state)

                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, R, done = self.env.step(action)

                self.stats.add(i_episode, R, t, action)

                value_crt = self.value_estimator.predict(state)
                value_next = self.value_estimator.predict(next_state)

                td_delta = R + self.gamma * value_next - value_crt

                # dont update when its insignificant
                if np.abs(td_delta) > epsilon:
                    self.value_estimator.update(state, td_delta, I)
                    self.policy_estimator.update(state, td_delta, action, I)

                if t % 10 == 0:
                    print("\rStep {} @ Episode {}/{} ({})".format(
                        t, i_episode + 1, self.num_episodes, self.stats.rewards[i_episode - 1]), end="")

                if done or t > self.max_iters_per_ep:
                    break

                I *= self.gamma
                state = next_state


# holder for test parameters
class TestConfig:
    def __init__(self,
                 use_trace_policy,
                 use_trace_value,
                 gamma,
                 lambda_trace
                 ):
        self.use_trace_policy = use_trace_policy
        self.use_trace_value = use_trace_value
        self.gamma = gamma
        self.lambda_trace = lambda_trace

    def __str__(self):
        def bool2str(b):
            return "T" if b else "F"

        def float2str(f, precision=precision):
            return "0." + str(int((10 ** precision) * f))

        return "{}{}-{}-{}".format(
            bool2str(self.use_trace_policy),
            bool2str(self.use_trace_value),
            float2str(self.gamma),
            float2str(self.lambda_trace),
        )

env = TrafficSimulator()

# run cpu only -> this is actually faster on my machine
config = tf.ConfigProto(
    device_count={'GPU': 0}
)

tf.reset_default_graph()

stats_dict = {} # record results here
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    config_name = str(config)

    print("Running tests with configuration: {}".format(config_name))

    policy_estimator = PolicyEstimator(gamma=config.gamma,
                                       lambda_trace=config.lambda_trace,
                                       useTrace=config.use_trace_policy,
                                       tag=config_name)

    value_estimator = ValueEstimator(gamma=config.gamma,
                                     lambda_trace=config.lambda_trace,
                                     useTrace=config.use_trace_value,
                                     tag=config_name)

    actor_critic = ActorCritic(env,
                               policy_estimator,
                               value_estimator,
                               gamma=gamma,
                               num_episodes=num_episodes)

    actor_critic.run()

    stats = actor_critic.stats