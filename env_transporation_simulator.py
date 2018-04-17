import numpy as np

class Passenger:
    def __init__(self,
                 start_time_step,
                 destination):
        self.start_time_step = start_time_step
        self.destination = destination

class BusA:
    def __init__(self,
                 time_to_next_station,
                 capacity=50,
                 ):
        self.time_to_destination = time_to_next_station

        self.capacity = capacity
        self.passengers = []

    # returns number of passengers that could be loaded
    def load_passengers(self, passengers):
        num_loaded_passengers = 0

        passengers_sorted_by_time = sorted(passengers, key=lambda passenger: passenger.start_time_step)

        available_capacity = self.capacity - len(self.passengers)
        while available_capacity < num_loaded_passengers:
            passenger = passengers_sorted_by_time.pop(0)
            if not passenger is None:
                self.passengers.append(passenger)
                num_loaded_passengers += 1
            else:
                break

        return num_loaded_passengers

    # returns passengers that were unloaded
    def unload_all_passengers(self):
        unloaded_passengers = self.passengers.copy()
        self.passengers = []
        return unloaded_passengers

    def unload_passengers(self):
        pass

    def step(self,):
        self.time_to_destination -= 1




# skeleton for a Dyna game
# define each of these
class DynaQGame:
    def __init__(self):
        self.state = ""
        self.actions = []
        self.game_over = False
        self.total_reward = 0
    # returns next state and the observed reward after taking action
    def play(self, action):
        pass
    # reset env and return initial state of a reset game
    def reset(self):
        pass



class Environment:
    def __init__(self,
                 actions_dict={'wait': -1, 'send0': 0},
                 time_between_stations=np.array([10, 1, 1, 1]),  # time required between each station
                 ):
        pass
        # self.initial_passengers = passengers_per_station_init.copy()

        # self.stations = passengers_per_station_init.copy()



    def add_passengers(self, num_station, num_passengers):
        pass

class TrafficSimulatorA:
    def __init__(self,
                 environment,
                 penalty_per_bus=10,  # cost for starting a new bus
                 ):
        self.time_step = 0

    def step(self):
        for bus in self.buses:
            bus.step()


    def play(self, action):
        pass

    def reset(self):
        pass
