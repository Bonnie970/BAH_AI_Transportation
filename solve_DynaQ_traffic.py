import DynaQ_traffic
import matplotlib.pyplot as plt
import traffic_env
import numpy as np
from scipy.signal import savgol_filter

########################################################################################################################

# plots the results of tests
def plotstats(test_name,
              stats_dict,
              num_episodes,
              max_savgol_winsize=151,
              min_savgol_winsize=15
              ):

    # iterator table of available colors
    class ColorIterator:
        def __init__(self):
            self.colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                      'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
                      'tab:olive', 'tab:cyan']
            self.index = 0
        def next(self):
            color = self.colors[self.index]
            self.index = (self.index + 1) % len(self.colors)
            return color

    colors = ColorIterator()

    fig, ax1 = plt.subplots()

    # setup visuals
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")

    # process data
    episodes = np.arange(num_episodes)

    # determine smoothing window size
    savgol_winsize = int(num_episodes / 2)
    savgol_winsize = min_savgol_winsize if savgol_winsize < min_savgol_winsize else savgol_winsize
    savgol_winsize = max_savgol_winsize if savgol_winsize > max_savgol_winsize else savgol_winsize
    savgol_winsize = savgol_winsize + 1 if savgol_winsize % 2 == 0 else savgol_winsize  # ensure odd

    print("Plotting results; smoothed with {}-wide savgol filter.".format(savgol_winsize))

    for stats_name in stats_dict:
        color = colors.next()

        stats = stats_dict[stats_name]
        stats_smooth = savgol_filter(stats, savgol_winsize, 4)

        ax1.plot(episodes, stats, 'o--', color=color, markersize=1, alpha=0.2)
        ax1.plot(episodes, stats_smooth, 'o--', color=color, markersize=1, alpha=0.8, label=stats_name)

    fig.tight_layout()
    ax1.legend()
    plt.title(test_name)
    plt.show()

########################################################################################################################

game = traffic_env.TrafficSimulator(bus_cost=500)

n_eps = 2000

dynaq = DynaQ_traffic.DynaQ(game, n_eps)

# set parameters
# this parameter defines how many steps dynaq learns from experiences
dynaq.n = 0
dynaq.alpha = 0.99
dynaq.epslon = 0.01
dynaq.verbose = True

rewards = dynaq.run()

plotstats("Test 1", {"Configuration 1": rewards}, n_eps)
