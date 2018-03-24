import traffic_env

simulator1 = traffic_env.TrafficSimulator()

send_bus_time = 4

for _ in range(send_bus_time):
    print(simulator1.play('wait'))

print(simulator1.play('sendA'))

for _ in range(15-send_bus_time):
    print(simulator1.play('wait'))

print(simulator1.total_reward)

