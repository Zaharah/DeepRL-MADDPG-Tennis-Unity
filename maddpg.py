from ddpg_agent import DDPGAgent, ReplayBuffer
class MADDPG:
    def __init__(self, random_seed, num_agents, state_size, action_size):
        self.agents = [DDPGAgent(state_size,action_size,random_seed) for x in range(num_agents)]

    def step(self, states, actions, rewards, next_states, dones):
        
