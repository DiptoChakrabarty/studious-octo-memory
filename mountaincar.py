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
#print(q_table)

# CONSTANTS
LR = 0.1
DIS = 0.95  # importance of future vs current reward
EPO = 20000
SHOW =2000
def new_discrete_state(state):
    # get discrete values of environment reset
    dis_state= (state - env.observation_space.low) / discrete_os_win_size
    return tuple(dis_state.astype(np.int))



for episodes in range(EPO):
    if episodes % SHOW == 0:
        print(episodes)
        render = True
    else:
        render = False
    discrete_state = new_discrete_state(env.reset())
    print(q_table[discrete_state])

    done=False

    while not done:
        action= np.argmax(q_table[discrete_state]) # check max q value from q table
        new_state , reward , done , _ =env.step(action) # Takes Action 
        # observation, reward, done and info
        new_discrete_state = new_discrete_state(new_state) # obtain discrete values for new state

        #print(new_state,reward,done,_)
        if render:
            env.render()  # Display PopUp Window 
        if not done:
            max_future_Q = np.max(q_table[new_discrete_state]) # get the max q value from q table
            current_q = q_table[discrete_state+ (action ,)]  # the current q value

            new_q = (1-LR)* current_q + LR *(reward + DIS * max_future_Q)  # the new  q value based on research paper formula

            q_table[discrete_state+(action,)]= new_q
            # update q table
        elif new_state[0] >= env.goal_position:
           # print("Wemade it on episode {}".format(episodes))
            q_table[discrete_state+(action,)]= 0
        discrete_state = new_discrete_state

    env.close()
