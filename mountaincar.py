import gym
import numpy as np
env= gym.make("MountainCar-v0")
env.reset()

print(env.observation_space.high) #Max value of Obs
print(env.observation_space.low) # Min value of Obs
print(env.action_space.n)

DISCRETE_OS_SIZE = [20]*len(env.observation_space.high)  # make Q table of 20 chunks
# distribute high to low into 20 chunks
discrete_os_win_size =(env.observation_space.high- env.observation_space.low)/DISCRETE_OS_SIZE

print(discrete_os_win_size)

q_table = np.random.uniform(low=-2,high=0,size=DISCRETE_OS_SIZE +[env.action_space.n])
# makes q tabel which has all combination from -2 to 0 in with 3 columns with 
# random Q  values which gets optimised over time 
print(q_table.shape)
print(q_table)
'''
done=False

while not done:
    action=2
    new_state , reward , done , _ =env.step(action) # Takes Action 
    # observation, reward, done and info
    print(new_state,reward,done,_)
    env.render()  # Display PopUp Window 


env.close()
'''