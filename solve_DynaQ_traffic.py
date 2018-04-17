import DynaQ_traffic
import matplotlib.pyplot as plt
import traffic_env
import numpy as np
from scipy.signal import savgol_filter

########################################################################################################################

# plots the results of tests
def plotstats(stats_dict,
              num_episodes,
              test_name="Test",
              x_axis_name="X axis",
              y_axis_name="Y axis",
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
    ax1.set_ylabel(y_axis_name)

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

game = traffic_env.TrafficSimulator()

n_eps = 5000

dynaq = DynaQ_traffic.DynaQ(game, num_episodes=n_eps)
rewards, steps, buses = dynaq.run()

plotstats({"Configuration 1": rewards}, n_eps, x_axis_name="Episodes", y_axis_name="Rewards")
plotstats({"Configuration 1": steps}, n_eps, x_axis_name="Episodes", y_axis_name="Steps")
plotstats({"Configuration 1": buses}, n_eps, x_axis_name="Episodes", y_axis_name="Buses sent")

