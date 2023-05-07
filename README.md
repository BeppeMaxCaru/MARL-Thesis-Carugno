# MARL-Thesis-Carugno

Importante: il file di configurazione dell'env customizzato si trova in marllib/envs/base_env/config e si chiama patrolling.yaml
Il suo utilizzo come file di configurazione é necessario ed è dove la funzione make_env di marllib va a prendere i parametri di default per costruire l'ambiente -> vedi doc!


Qua sotto ci sono altre note importanti da ricordare e riordinare su Ray e MARLlib!
```
#def notes(self):
########################## IMPORTANT ##############################
        #This comment explains how to correctly create the observation space for a MARLlib agent!
        #READ IT CAREFULLY AND MAKE SURE YOU UNDERSTOOD IT BEFORE CREATING THE OBSERVATION_SPACE
        #OTHERWISE THE WHOLE FRAMEWORK WON'T WORK AND IT'S BEEN A PAIN FINDING OUT WHY
        # 1) The observation space has to be defined as a gym dictionary:
        """
        self.observation_space = gym.spaces.Dict({
            "obs": ..., # -> This is mandatory and has to contain all the observations of the agent
            "state": ..., # -> This is optional and can contain the global env state 
            "action_mask": ... # -> This is optional and can contain the action mask for the agent actions
        })
        """
        # 2) In MARLlib such dictionary requires from 1 to 3 inner spaces depending on the problem
        # and the 3 launch configuration parameters at the bottom of the yaml config file to use:
        
        #       a) mask_flag
        #       b) global_state_flag
        #       c) opp_action_in_cc
        
        # Each one of them is a boolean parameters!
        # By default it's strongly suggested to set them as False and then understand them before
        # chenging them to True depending on the problem -> the code for them are in the files 
        # marllib/marl/algos/utils
        
        #       a) mask_flag: it has to be set to True when you out of the possible actions you
        #       want to immediately discard some of them when selected since it will violate env
        #       env costraints so they cannot be taken
        #       Example: 
        #           Actions:            Up | Down | Left | Right | Stay 
        #           Actions mask:       0  | 1    | 1    | 0     | 1
        #       This mean that only actions Down, Left and Stay can be taken so we just pick
        #       pick one of them knowing that they're already valid rather than picking an invalid
        #       action and then having to check that the chosen action is valid
        #       (See for example the step function where I have to check that, chosen a move action,
        #       it's actually possible to move to the nearby node in that direction or there's no 
        #       edge to move there instead -> computationally more complex and inefficient)
        """
        self.observation_space = gym.spaces.Dict({
            "obs": ...,
            "state": ...,
            "action_mask": ...
        })
        """
        #       b) global_state_flag: it has to be set to True when also the environment can
        #       have different states, not just only the agents
        #       
        #       If True in the observation_space dict a second key has to be added called "state"
        #       with the corresponding gym space type
        """
        self.observation_space = gym.spaces.Dict({
            "obs": ...,
            "state": ...,
        })
        """
        
        ##################################################################
        
#def offline_graph_generation(self):
    #Passing an offline generated graph to the __init func instead of online generation
    
    #Configuration file settings + check
        #Define size of the squared grid        
        self._graph_size = env_config["size"]
        # Define the possible agents in the environment
        ...
        
        #Creating graph directly here instead of each time in reset!
        #This is the correct way to do so!
        #See RWARE example in MARLlin where Warehouse is initialized in __init__ and 
        #not in reset!!!!!!
        
        #Same action space for all agents so I just need to declare it once (for now)
        #5 possible actions: Stay, Up, Down, Left, Right
        self.action_space = gym.spaces.Discrete(5)
        #Assign to each agent the possibility to move: Stay, Up, Down, Left, Right
        #Removing action of Stay??? Would it make sense?
    
#def points_to_discuss_about_reset_func(self):
    #Reset observations for each agent once reset is called
        obs = {}

        ########### NEW ############
        #Reset obs spaces for each agent once reset method is called
        #Different strategies and possibilities on how to handle the reset!
        #Options:
        #1) Reset also all target nodes locations
        """
        If so I need to change the graph class
        """
        #2) Each reset also randomizes the agents starting locations for each episodes
        
        #Assign new random starting location to agents as a dictionary
        self._agents_locations = self._set_agents_random_starting_locations(self._graph)
        #print(self._agents_locations)
        
        #3) Use the same agents starting locations every reset -> risk of overfitting?
        """
        """
```

# How training works:

Relevant parameters to set:

- **batch_mode: ["complete_episodes" | "truncated_episodes"]**: 
completed episodes means that batches always contains full episodes -> train_batch_size is the equivalent of an epoch
truncated episodes means that the samples in a batch can be reached by also truncating episodes

- **train_batch_size**: It's the equivalent of an epoch. A number of timesteps (transition/samples) equivalent to train_batch_size is done and collected from a subsequent number of episodes. This train_batch_size is then used as a buffer to extract samples used to create minibatches of size sgd_minibatch_size. 

NB If the batch_mode selected is complete_episodes, train_batch_size has to be always bigger than the episode lenght

- **sgd_minibatch_size**: It's the size of the randomly sampled minibatches built to perform a nn weights update.

- **num_sgd_iter**: Defines how many network weights updates are done during an epoch using minibatches of size sgd_minibatch_size before moving to the next epoch.

Example:
- in stop config episode limit: 100
- Batch_mode: "complete_episodes"
- Episode_limit: 10 (this means each episode can have a maximum lenght of 10)
- train_batch_size: 20
- sgd_minibatches_size: 5
- num_sgd_iter: 10


# Configuration parameters to use during rware simulation to train IPPO according to the rware paper:

## Two options

**Configuration IPPO without parameters sharing and how to replicate it here**

In Rware paper:

- 


In Ray:

- In main_patrolling.py in ippo.fit share_policy='individual' to disable parameters sharing

**Configuration IPPO with parameters sharing and how to replicate it here**

In Ray:

- In main_patrolling.py in ippo.fit share_policy='group' to enable parameters sharing between agents of the same teams identified by teams prefix

