import ray
import tensorflow as tf
import tf_agents

#Create a custom agent from Ray that can use multiple multi agent reinforcement learning algorithms
#from rllib and MARLlib and that can be used with a gym environment
class RayAgent(ray.rllib.agents.agent.Agent):
    def __init__(self, config, env, logger_creator):
        #Call the parent class's constructor
        super().__init__(config, env, logger_creator)
        #Create a custom agent from Ray that can use multiple multi agent reinforcement learning algorithms
        #from rllib and MARLlib and that can be used with a gym environment
        self.rl_algorithm = MARLlib.PPO(config, env)
    #Step function using Ray for the agent using the RL algorithm selected to choose an action
    def step(self, observation):
        #Use the RL algorithm to choose an action
        action = self.rl_algorithm.choose_action(observation)
        #Return an action
        return {"action" : action}
    #Train function using Ray for the agent using the RL algorithm selected to train the agent
    def _train(self):
        #Train the RL algorithm
        self.rl_algorithm.train()

