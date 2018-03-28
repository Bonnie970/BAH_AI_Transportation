import DynaQ_traffic
import matplotlib.pyplot as plot
import traffic_env

game = traffic_env.TrafficSimulator(bus_cost=50)

n_eps = 1000

dynaq = DynaQ_traffic.DynaQ(game)

# set parameters
dynaq.n = 3
dynaq.alpha = 0.7
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