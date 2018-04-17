class Bus:
    def __init__(self, states, capacity=50, init_station=0, terminal_station=4):
        self.capacity = capacity
        self.empty = self.capacity
        # count time between consecutive stations, reset at arrival of station
        self.time_count = 0
        self.station = init_station
        self.terminal_station = terminal_station
        self.states = states
        self.terminate_flag = self.station==self.terminal_station

    def move(self):
        self.time_count += 1

    def arrival(self):
        self.station += 1
        self.time_count = 0
        if self.station == self.terminal_station:
            self.terminate_flag = True
            # unload passenger at terminal state
            self.states[-1] += (self.capacity - self.empty)
            self.empty = self.capacity

    def take_passenger(self):
        if self.station != self.terminal_station:
            # check if bus has enough seats to take all ppl at current station
            num_passenger = self.states[self.station]
            if self.empty >= num_passenger:
                self.empty -= num_passenger
                self.states[self.station] = 0
            else:
                self.states[self.station] -= self.empty
                self.empty = 0


class TrafficSimulator:
    def __init__(self,
                 states=[20, 50, 0, 0, 0],  # initial conditions at each station
                 goal_state=[0, 0, 0, 0, 70],
                 actions_dict={'wait':0,'sendA':1}, #,'sendB':2,'sendC':3},  # wait, send new bus at A, B, or C, number corresponds to position A,B,C
                 traffic_condition=[10, 1, 1, 1],  # time required between each station
                 bus_cost=500,  # cost for starting a new bus
                 ):
        self.time = 0
        self.initial_states = states.copy()
        self.states = states.copy()
        self.goal_state = goal_state
        self.actions = actions_dict.keys()
        self.actions_dict = actions_dict
        self.traffic_condition = traffic_condition
        # initial buses
        self.buses = [Bus(self.states)]
        self.bus_states = [(bus.capacity - bus.empty) for bus in self.buses]
        self.twostates = tuple(self.states)#(tuple(self.states), tuple(self.bus_states))
        self.total_reward = 0
        self.bus_cost = bus_cost
        self.game_over = False
        self.pi = []

    def play(self, action):
        self.pi.append(action)
        if action not in self.actions:
            return -1
        # add extra bus if action is not wait
        extra_bus_fee = 0
        if action != 'wait':
            self.buses.append(Bus(self.states, init_station=self.actions_dict[action]))
            extra_bus_fee = -1 * self.bus_cost
        # loading bus, move bus
        for bus in self.buses:
            if bus.terminate_flag:
                continue
            bus.take_passenger()
            bus.move()
            if bus.time_count == self.traffic_condition[bus.station]:
                bus.arrival()

        self.bus_states = [(bus.capacity - bus.empty) for bus in self.buses]
        self.twostates = tuple(self.states)#(tuple(self.states), tuple(self.bus_states))
        self.time += 1
        reward = -1 * sum(self.states[:-1]) + extra_bus_fee #- 0.5 * sum(self.bus_states)
        self.total_reward += (reward + extra_bus_fee)

        if self.states == self.goal_state:
            print(self.pi)
            self.game_over = True

        # print('ACTION', action, 'STATE: ', self.twostates,'REWARD: ', reward, 'TOTAL REWARD: ', self.total_reward)
        return self.twostates, reward

    def reset(self):
        self.time = 0
        self.states = self.initial_states.copy()
        self.buses = [Bus(self.states)]
        self.total_reward = 0
        self.game_over = False
        self.pi = []
        # self.__init__() -> this doesnt work when using not default arguments in init
