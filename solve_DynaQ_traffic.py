from DynaQ_traffic import DynaQ
from traffic_env import TrafficSimulator
from plotter import plotstats

# [20, 150, 50, 35, 0] fails to converge
states=[20, 100, 30, 5, 0]
goal_state=[0, 0, 0, 0, sum(states)]
game = TrafficSimulator(bus_cost=30,
                        states=states,
                        goal_state=goal_state)

# !!! low alpha converges better
# !!! explore less in simple case --> explore more in complex case
dynaq = DynaQ(game, alpha=0.1, gamma=0.95, \
              epslon=0.1, planning_step=5, n_eps=5000, verbose=True)

dynaq.run()

rewards = [x[0] for x in dynaq.results]
n_eps = len(dynaq.results)

plotstats({"Configuration 1": rewards}, n_eps, \
          x_axis_name="Episodes", y_axis_name="Rewards")