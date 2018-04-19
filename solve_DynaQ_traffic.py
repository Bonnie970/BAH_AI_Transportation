import numpy as np
import traffic_env

from DynaQ_traffic import DynaQ
from DynaQ_traffic import DynaQGame
from env_transporation_simulator import TrafficSimulator
from plotter import plotstats

########################################################################################################################

game = DynaQGame(game=TrafficSimulator())
dynaq = DynaQ(game)
n_eps, rewards, steps, buses = dynaq.run()

plotstats({"Configuration 1": rewards}, n_eps, x_axis_name="Episodes", y_axis_name="Rewards")
plotstats({"Configuration 1": steps}, n_eps, x_axis_name="Episodes", y_axis_name="Steps")
plotstats({"Configuration 1": buses}, n_eps, x_axis_name="Episodes", y_axis_name="Buses sent")

