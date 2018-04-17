class BusA:
    def __init__(self,
                 current_passengers,
                 capacity=50,
                 ):
        self.capacity = capacity
        self.current_passengers = 0

    # returns number of passengers that could be loaded
    def load_passengers(self, num_passengers):
        loaded_passengers = 0

        available_capacity = self.capacity - self.current_passengers

        if available_capacity >= num_passengers:
            loaded_passengers = num_passengers
        else:
            loaded_passengers = available_capacity

        self.current_passengers += loaded_passengers

        return loaded_passengers

    # returns number of passengers that were unloaded
    def unload_passengers(self):
        current_passengers = self.current_passengers
        self.current_passengers = 0
        return current_passengers


# skeleton for a Dyna game
class DynaQGame:
    def __init__(self):
        self.state = tuple(None)
        self.actions = []
        self.game_over = False
        self.total_reward = 0
    # returns next state and the observed reward after taking action
    def play(self, action):
        pass
    # reset env and return initial state of a reset game
    def reset(self):
        pass

class TrafficSimulatorA:
    def __init__(self,
                 passengers_per_station=[20, 0, 0, 50, 0],  # initial conditions at each station
                 passengers_per_station_goal=[0, 0, 0, 0, 70],
                 actions_dict={'wait': 0,'sendA': 1}, #,'sendB':2,'sendC':3},  # wait, send new bus at A, B, or C, number corresponds to position A,B,C
                 time_between_stations=[10, 1, 1, 1],  # time required between each station
                 penalty_per_bus=10,  # cost for starting a new bus
                 ):
        self.time_step = 0


    def play(self, action):
        pass

    def reset(self):
        pass
