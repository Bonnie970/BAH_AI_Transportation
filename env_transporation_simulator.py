import numpy as np
from enum import Enum

class Passenger:
    def __init__(self,
                 start_time_step,
                 destination):
        self.start_time_step = start_time_step
        self.destination = destination


class Station:
    def __init__(self):
        self.unique_passenger_id = 0
        self.passengers = {}

    def add_passenger(self, passenger):
        self.passengers[self.unique_passenger_id] = passenger
        self.unique_passenger_id += 1

    def add_passengers(self, passengers):
        for passenger in passengers:
            self.add_passenger(passenger)

    def get_passengers_sorted_by_time(self):
        pairs = self.passengers.items()
        passengers = sorted(pairs, key=lambda pair: pair[1].start_time_step)
        return passengers


class Bus:
    def __init__(self,
                 env,
                 current_station=0,
                 passengers_max_capacity=20,
                 ):
        self.env = env
        self.current_station = current_station
        self.time_to_destination = self.env.time_between_stations[current_station]

        self.passengers_max_capacity = passengers_max_capacity
        self.passengers = []

        self.load_passengers()

    # loads passengers (as capacity allows and ordered by arrival time)
    def load_passengers(self):
        station = self.env.stations[self.current_station]

        passengers_sorted = station.get_passengers_sorted_by_time()

        loaded_passengers = []
        available_capacity = self.passengers_max_capacity - len(self.passengers)
        while available_capacity > len(loaded_passengers) and len(passengers_sorted) > 0:
            id, passenger = passengers_sorted.pop(0)
            self.passengers.append(passenger)
            loaded_passengers.append(passenger)
            station.passengers.pop(id, None)

        return loaded_passengers

    def unload_all_passengers(self):
        unloaded_passengers = self.passengers.copy()
        self.passengers = []
        return unloaded_passengers

    def unload_passengers(self):
        unloaded_passengers = []
        for i in range(len(self.passengers)):
            i -= len(unloaded_passengers)
            passenger = self.passengers[i]
            if passenger.destination == self.current_station:
                unloaded_passengers.append(passenger)
                self.passengers.pop(i)

        return unloaded_passengers

    def next_time_step(self):
        # print("BUS:", self.time_to_destination, self.current_station, len(self.passengers))
        if self.time_to_destination <= 0:
            self.current_station = (self.current_station + 1) % self.env.num_stations
            self.time_to_destination = self.env.time_between_stations[self.current_station]

            unloaded_passengers = self.unload_passengers()
            loaded_passengers = self.load_passengers()

            return unloaded_passengers, loaded_passengers

        self.time_to_destination -= 1

        return [], []


class Environment:
    def __init__(self,
                 num_stations=10,
                 ):
        self.num_stations = num_stations

        self.stations = [Station() for _ in range(num_stations)]
        self.time_between_stations = [1 for _ in range(num_stations)]

        self.buses = []

        # initial configuration
        self.set_initial_configuration()

    def set_initial_configuration(self):
        # configure delays
        self.set_delay_between_stations(station=0, delay=2)
        self.set_delay_between_stations(station=1, delay=2)
        self.set_delay_between_stations(station=5, delay=5)

        # configure passengers
        self.add_passengers(time_step=0, station_source=0, station_destination=5, num_passengers=10)
        self.add_passengers(time_step=0, station_source=1, station_destination=5, num_passengers=5)
        self.add_passengers(time_step=0, station_source=2, station_destination=5, num_passengers=10)
        self.add_passengers(time_step=0, station_source=7, station_destination=2, num_passengers=5)
        self.add_passengers(time_step=0, station_source=9, station_destination=5, num_passengers=5)

        # configure buses
        # self.add_bus(Bus(env=self, current_station=0))

    def set_delay_between_stations(self, station, delay):
        self.time_between_stations[station] = delay

    def add_passengers(self, time_step, station_source, station_destination, num_passengers=1):
        for i in range(num_passengers):
            self.stations[station_source].add_passenger(Passenger(time_step, station_destination))

    def add_bus(self, bus=None):
        if bus is None:
            bus = Bus(env=self, current_station=0)
        self.buses.append(bus)

    def remove_bus(self):
        if len(self.buses) >= 1:
            # remove the first bus
            bus = self.buses.pop(0)
            # get passengers currently on the bus
            passengers = bus.unload_all_passengers()
            # add unloaded passengers to the station the bus was last at
            station = bus.current_station
            self.stations[station].add_passengers(passengers)

    def __str__(self):
        env_strs = ["" for station in self.stations]
        for index, station in enumerate(self.stations):
            env_strs[index] += '(' + str(len(station.passengers)) + ')' + '-'
        for bus in self.buses:
            env_strs[bus.current_station] += 'b-'
        env_str = ">"
        for s in env_strs:
            env_str += s
        env_str += '>'
        return env_str


# possible actions
class Actions(Enum):
    WAIT = 'wait'
    ADD_BUS = 'add bus'
    REMOVE_BUSS = 'remove buss'


class TrafficSimulator:
    def __init__(self,
                 penalty_per_bus=10,  # cost for starting a new bus
                 penalty_per_missed_passenger=10,
                 time_to_miss_passenger=10,
                 ):
        self.env = Environment()

        # initialize variables
        self.time_step = 0
        self.total_reward = 0

        self.actions = [Actions.WAIT, Actions.ADD_BUS, Actions.REMOVE_BUSS]

        self.penalty_per_bus = penalty_per_bus

    def reset(self):
        self.env = Environment()
        self.time_step = 0
        self.total_reward = 0
        self.actions = []
        return self.state_to_str()

    def get_num_buses(self):
        return len(self.env.buses)

    def step(self):
        reward = 0

        # reward for delivered passengers
        for bus in self.env.buses:
            unloaded_passengers, loaded_passengers = bus.next_time_step()

            for passenger in unloaded_passengers:
                reward += 1

        # penalty for each bus
        for bus in self.env.buses:
            reward -= 1 * self.penalty_per_bus

        # penalty for waiting passengers
        for station in self.env.stations:
            reward -= len(station.passengers)

        for station in self.env.stations:


        self.total_reward += reward

        return reward

    def play(self, action):

        if action == Actions.WAIT:
            pass
        elif action == Actions.ADD_BUS:
            self.env.add_bus()
        elif action == Actions.REMOVE_BUSS:
            self.env.remove_bus()

        # simulate environment
        reward = self.step()
        return self.state_to_str(), reward

    def game_over(self):
        passengers_left = 0
        for station in self.env.stations:
            passengers_left += len(station.passengers)
        return passengers_left == 0

    def state_to_str(self):
        s = str(self.env)
        return s


# sim = TrafficSimulator()
# import time
# for i in range(1000):
#     time.sleep(0.25)
#     sim.play("a")
#     print("\r[{}] {}".format('-' if i%2==0 else '+', sim.state_to_str()), end='')

