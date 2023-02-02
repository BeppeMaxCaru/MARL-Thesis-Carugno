import environment
import graph
import agent

#print("Hello World")
    
#Generate graph
#Uncomment this line to test graph generation
#graphToPatrol = graph.Graph(10, 10)
#Done inside directly into _setup

#Generate environment
env = environment.myEnv(num_patrollers=1, num_attackers=0)

#Create agent and pass it the action and observation spaces
patroller = agent.PatrollerAgent("Patroller_0", env.action_space, env.observation_space, env.graph, env.target_nodes)

#Pass the agent to the environment
patroller = env.add_agent(patroller)

#Train the agent
patroller._train(env)
