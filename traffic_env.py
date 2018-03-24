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
                 states=[10, 10, 10, 10, 0],  # initial conditions at each station
                 goal_state=[0, 0, 0, 0, 40],
                 actions={'wait':0,'sendA':1,'sendB':2,'sendC':3},  # wait, send new bus at A, B, or C
                 traffic_condition=[10, 1, 1, 1]  # time required between each station
                 ):
        self.time = 0
        self.states = states
        self.goal_state = goal_state
        self.actions = actions
        self.traffic_condition = traffic_condition
        # initial buses
        self.buses = [Bus(self.states)]
        self.total_reward = 0

    def play(self, action):
        if action not in self.actions.keys():
            return -1
        # add extra bus if action is not wait
        pay_driver = 0
        if action != 'wait':
            self.buses.append(Bus(self.states, init_station=self.actions[action]))
            pay_driver = -25
        # loading bus, move bus
        for bus in self.buses:
            if bus.terminate_flag:
                continue
            bus.take_passenger()
            bus.move()
            if bus.time_count == self.traffic_condition[bus.station]:
                bus.arrival()

        self.time += 1
        reward = -1*sum(self.states[:-1])
        self.total_reward += reward + pay_driver

        return reward, self.states


