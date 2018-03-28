import traffic_env

simulator1 = traffic_env.TrafficSimulator()

send_bus_time = 1
step = 0

while not simulator1.game_over:
    step += 1
    if step==send_bus_time:
        print(simulator1.play('sendA'))
    else:
        print(simulator1.play('wait'))


print(simulator1.total_reward)


simulator2 = traffic_env.TrafficSimulator()
simulator2.play('sendA')
simulator2.play('wait')
simulator2.play('wait')
simulator2.play('wait')
print(simulator2.states, simulator2.total_reward)
simulator2.reset()
print(simulator2.states, simulator2.total_reward)
