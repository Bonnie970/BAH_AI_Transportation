import numpy as np
from enum import Enum

class Passenger:
    def __init__(self,
                 start_time_in_minutes,
                 destination):
        self.start_time_in_minutes = start_time_in_minutes
        self.destination = destination


class Station:
    def __init__(self):
        self.unique_passenger_id = 0
        self.passengers = {}

    def add_passenger(self, passenger, num_passengers=1):
        self.passengers[self.unique_passenger_id] = passenger
        self.unique_passenger_id += 1

    def add_passengers(self, passengers):
        for passenger in passengers:
            self.add_passenger(passenger)

    def get_passengers_sorted_by_time(self):
        pairs = self.passengers.items()
        passengers = sorted(pairs, key=lambda pair: pair[1].start_time_in_minutes)
        return passengers


class Bus:
    def __init__(self,
                 env,
                 current_station=0,
                 passengers_max_capacity=50,
                 ):
        self.env = env
        self.current_station = current_station
        self.minutes_to_destination = self.env.minutes_between_stations[current_station]

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

    def next_time_step(self, minutes_per_time_step):
        # print("BUS:", self.time_to_destination, self.current_station, len(self.passengers))
        if self.minutes_to_destination <= 0:
            self.current_station = (self.current_station + 1) % self.env.num_stations
            self.minutes_to_destination = self.env.minutes_between_stations[self.current_station]

            unloaded_passengers = self.unload_passengers()
            loaded_passengers = self.load_passengers()

            return unloaded_passengers, loaded_passengers

        self.minutes_to_destination -= minutes_per_time_step

        return [], []


class DailyPassengerSchedule:
    def __init__(self,
                 ):
        self.minutes_in_a_day = 1440  # 60min*24h
        self.scheduled_passengers = {}

    def schedule_passenger(self,
                           time_in_minutes,
                           number_of_passengers,
                           start_station,
                           destination_station,
                           ):
        time_in_minutes %= self.minutes_in_a_day
        if time_in_minutes not in self.scheduled_passengers:
            self.scheduled_passengers[time_in_minutes] = []
        self.scheduled_passengers[time_in_minutes].append((start_station, destination_station, number_of_passengers))

    # assumes time_previous and time_current are on the same day
    def get_scheduled_passengers_between(self, time_previous, time_current):
        time_previous %= self.minutes_in_a_day
        time_current %= self.minutes_in_a_day
        scheduled_passengers = []
        for time in self.scheduled_passengers:
            if time >= time_previous and time < time_current:
                for scheduled_passenger in self.scheduled_passengers[time]:
                    scheduled_passengers.append(scheduled_passenger)
        return scheduled_passengers

class Environment:
    def __init__(self,
                 num_stations=10,
                 default_minutes_between_stations=10,
                 ):
        self.num_stations = num_stations

        self.current_time_in_minutes = 0

        self.stations = [Station() for _ in range(num_stations)]
        self.minutes_between_stations = [default_minutes_between_stations for _ in range(num_stations)]

        self.buses = []

        self.schedule = DailyPassengerSchedule()

        # initial configuration
        self.set_initial_configuration()

    def set_initial_configuration(self):
        # configure delays
        self.set_delay_between_stations(station=1, delay_in_minutes=15)
        self.set_delay_between_stations(station=3, delay_in_minutes=15)
        self.set_delay_between_stations(station=5, delay_in_minutes=20)
        self.set_delay_between_stations(station=8, delay_in_minutes=15)

        # configure schedule
        # midnight:
        self.schedule.schedule_passenger(time_in_minutes=0, number_of_passengers=2, start_station=0,
                                         destination_station=5)
        self.schedule.schedule_passenger(time_in_minutes=0, number_of_passengers=5, start_station=1,
                                         destination_station=4)
        self.schedule.schedule_passenger(time_in_minutes=0, number_of_passengers=2, start_station=1,
                                         destination_station=7)
        self.schedule.schedule_passenger(time_in_minutes=0, number_of_passengers=2, start_station=2,
                                         destination_station=8)
        self.schedule.schedule_passenger(time_in_minutes=0, number_of_passengers=2, start_station=5,
                                         destination_station=9)
        self.schedule.schedule_passenger(time_in_minutes=0, number_of_passengers=5, start_station=7,
                                         destination_station=5)
        self.schedule.schedule_passenger(time_in_minutes=0, number_of_passengers=2, start_station=9,
                                         destination_station=3)
        # 6AM: RUSH HOUR TO WORK
        self.schedule.schedule_passenger(time_in_minutes=360, number_of_passengers=10, start_station=0,
                                         destination_station=5)
        self.schedule.schedule_passenger(time_in_minutes=360, number_of_passengers=10, start_station=1,
                                         destination_station=4)
        self.schedule.schedule_passenger(time_in_minutes=360, number_of_passengers=15, start_station=1,
                                         destination_station=7)
        self.schedule.schedule_passenger(time_in_minutes=360, number_of_passengers=10, start_station=2,
                                         destination_station=8)
        self.schedule.schedule_passenger(time_in_minutes=360, number_of_passengers=15, start_station=5,
                                         destination_station=9)
        self.schedule.schedule_passenger(time_in_minutes=360, number_of_passengers=20, start_station=7,
                                         destination_station=5)
        self.schedule.schedule_passenger(time_in_minutes=360, number_of_passengers=15, start_station=9,
                                         destination_station=3)
        # midday:
        self.schedule.schedule_passenger(time_in_minutes=720, number_of_passengers=5, start_station=0,
                                         destination_station=5)
        self.schedule.schedule_passenger(time_in_minutes=720, number_of_passengers=5, start_station=1,
                                         destination_station=4)
        self.schedule.schedule_passenger(time_in_minutes=720, number_of_passengers=5, start_station=1,
                                         destination_station=7)
        self.schedule.schedule_passenger(time_in_minutes=720, number_of_passengers=5, start_station=2,
                                         destination_station=8)
        self.schedule.schedule_passenger(time_in_minutes=720, number_of_passengers=5, start_station=5,
                                         destination_station=9)
        self.schedule.schedule_passenger(time_in_minutes=720, number_of_passengers=5, start_station=7,
                                         destination_station=5)
        self.schedule.schedule_passenger(time_in_minutes=720, number_of_passengers=5, start_station=9,
                                         destination_station=3)
        # 6PM: RUSH HOUR FROM WORK
        self.schedule.schedule_passenger(time_in_minutes=1080, number_of_passengers=10, start_station=0,
                                         destination_station=5)
        self.schedule.schedule_passenger(time_in_minutes=1080, number_of_passengers=20, start_station=1,
                                         destination_station=4)
        self.schedule.schedule_passenger(time_in_minutes=1080, number_of_passengers=15, start_station=1,
                                         destination_station=7)
        self.schedule.schedule_passenger(time_in_minutes=1080, number_of_passengers=20, start_station=2,
                                         destination_station=8)
        self.schedule.schedule_passenger(time_in_minutes=1080, number_of_passengers=15, start_station=5,
                                         destination_station=9)
        self.schedule.schedule_passenger(time_in_minutes=1080, number_of_passengers=20, start_station=7,
                                         destination_station=5)
        self.schedule.schedule_passenger(time_in_minutes=1080, number_of_passengers=15, start_station=9,
                                         destination_station=3)

        # configure initial buses
        # self.add_bus(Bus(env=self, current_station=0))

    def set_delay_between_stations(self, station, delay_in_minutes):
        self.minutes_between_stations[station] = delay_in_minutes

    def add_passengers(self, start_time_in_minutes, station_source, station_destination, num_passengers=1):
        for i in range(num_passengers):
            self.stations[station_source].add_passenger(Passenger(start_time_in_minutes, station_destination))

    def add_bus(self, bus=None, station=0):
        if bus is None:
            bus = Bus(env=self, current_station=station)
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

    def next_time_step(self, minutes_per_time_step):
        time_previous = self.current_time_in_minutes
        self.current_time_in_minutes += minutes_per_time_step

        self.update_passengers(time_previous, self.current_time_in_minutes)

    def update_passengers(self, time_previous, time_current):
        # update passengers
        scheduled_passengers = self.schedule.get_scheduled_passengers_between(time_previous, time_current)
        for start_station, destination_station, number_of_passengers in scheduled_passengers:
            self.add_passengers(time_current, start_station, destination_station, number_of_passengers)

    def to_array(self):
        # column 0: passengers
        # column 1: buses at each station
        # column 2: traffic conditions
        state = np.zeros((self.num_stations, 3))
        for i in range(self.num_stations):
            state[i, 0] = len(self.stations[i].passengers)
            state[i, 2] = self.minutes_between_stations[i]
        for bus in self.buses:
            state[bus.current_station, 1] += 1
        return state

    def to_number(self):
        return np.sum(np.array([2**i*(len(station.passengers) > 0) for i, station in enumerate(self.stations)]))

    def __str__(self):
        env_strs = ["" for station in self.stations]
        for index, station in enumerate(self.stations):
            env_strs[index] += '({})'.format(str(len(station.passengers))) + \
                               '-[t{}]-'.format(self.minutes_between_stations[index])
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
    REMOVE_BUS = 'remove bus'


class TrafficSimulator:
    def __init__(self,
                 minutes_per_time_step=5,  # for simulation
                 penalty_per_bus=10,  # cost for starting a new bus
                 penalty_per_missed_passenger=10,
                 minutes_to_miss_passenger=60,
                 state_as_string=True,
                 goal_delivered_passengers=500
                 ):
        self.env = Environment()

        # initialize variables
        self.time_step = 0
        self.minutes_per_time_step = minutes_per_time_step

        self.total_reward = 0

        self.actions = [Actions.WAIT, Actions.ADD_BUS, Actions.REMOVE_BUS]

        self.penalty_per_bus = penalty_per_bus

        self.minutes_to_miss_passenger = minutes_to_miss_passenger
        self.penalty_per_missed_passenger = penalty_per_missed_passenger

        self.state_as_string = state_as_string

        # for actor-critic
        self.observation_space_n = 2 ** self.env.num_stations
        self.action_space_n = len(self.actions)

        self.delivered_passengers = 0
        self.goal_delivered_passengers = goal_delivered_passengers

    def reset(self):
        self.env = Environment()
        self.time_step = 0
        self.total_reward = 0
        self.actions = []
        self.delivered_passengers = 0
        return self.get_state()

    def get_num_buses(self):
        return len(self.env.buses)

    def play(self, action):
        # print("Playing action", action)

        if action == Actions.WAIT:
            pass
        elif action == Actions.ADD_BUS:
            self.env.add_bus()
        elif action == Actions.REMOVE_BUS:
            self.env.remove_bus()

        # simulate environment
        reward = self.step()

        return self.get_state(), reward

    def step(self):
        reward = 0

        # reward for delivered passengers
        for bus in self.env.buses:
            unloaded_passengers, loaded_passengers = bus.next_time_step(self.minutes_per_time_step)

            self.delivered_passengers += len(unloaded_passengers)
            reward += len(unloaded_passengers)

        # penalty for each bus
        for bus in self.env.buses:
            reward -= 1 * self.penalty_per_bus

        # penalty for waiting passengers
        for station in self.env.stations:
            # base waiting passenger penalty
            reward -= len(station.passengers)

            # # strong waiting passenger penalty if waiting for a prolonged amount of time
            # for passenger_id in station.passengers:
            #     passenger = station.passengers[passenger_id]
            #     waiting_time = self.time_step * self.minutes_per_time_step - passenger.start_time_in_minutes
            #     if waiting_time > self.minutes_to_miss_passenger:
            #         reward -= self.penalty_per_missed_passenger

        self.total_reward += reward

        self.env.next_time_step(self.minutes_per_time_step)

        return reward

    def game_over(self):
        return self.delivered_passengers > self.goal_delivered_passengers

    def state_to_num(self):
        return self.env.to_number()

    def state_to_str(self):
        return str(self.env)

    def get_state(self):
        return self.state_to_str() if self.state_as_string else self.state_to_num()


def main():
    sim = TrafficSimulator()

    import time
    for i in range(1000):
        print(sim.time_step, sim.env.current_time_in_minutes)
        print(sim.state_to_str())
        # print(sim.state_to_num())
        print("")
        # print("\r[{}] {}".format('-' if i%2==0 else '+', sim.state_to_str()), end='')

        if i % 100 == 0:
            sim.play(Actions.ADD_BUS)
        else:
            sim.play(Actions.WAIT)

        time.sleep(0.1)

if __name__=="__main__":
    main()

