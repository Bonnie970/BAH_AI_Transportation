import DynaQ_traffic
import matplotlib.pyplot as plot
import traffic_env

game = traffic_env.TrafficSimulator(bus_cost=50)

n_eps = 300

dynaq = DynaQ_traffic.DynaQ(game)

# set parameters
# this parameter defines how many steps dynaq learns from experiences
dynaq.n = 0
dynaq.alpha = 0.9
dynaq.epslon = 0.01
dynaq.verbose = True

for _ in range(n_eps):
  dynaq.run()

reward = [x[0] for x in dynaq.results]
eps = [x[1] for x in dynaq.results]

plot.plot(eps, reward)
plot.xlabel('Episode')
plot.ylabel('Cumulative reward')
plot.legend()
plot.show()