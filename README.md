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
-> First epoch is made out of episode 0 and episode 1 since they last 10 steps (transitions) and they fill a train_batch_size since 10 * 2 = 20
-> In this first epoch 10 network updates are done to the agents nn (num_sgd_iter = 10) using minibatches of size 5 with samples (transition) being sampled from the train batch which contains 20 timesteps
-> In this case one epoch contains 20 network updates for each nn of each agent
-> The total number of epochs is (100 * 10) / 20 = 50
-> The total number of network weights updates done for each agent is 50 * 10 = 500

# Training parameters as rware paper using IPPO as baseline

**In rware paper**
- Each episode has a maximum limit of 500 timesteps
- 40 million timesteps of training in total
- 41 metrics collections during training
- 40 mil / 40 = 1 checkpoint every 1 mil timesteps to monitor training
- scenario used for rware: tiny 4p -> map_size: "tiny", num_agents: 4, difficulty: "medium" (difficulty is the lenght of queue of tasks assigned to each agent, with medium is 1 so each agent has always only one task assigned -> double check this in the paper)

- NB tiny map is an 11*11 squared grid however that's not randomly initialized but always has
the same layout! Only the agents positions are initialized randomly!

- neural networks are used with the same size of 128 neurons in the hidden layer
- clipping value of surrogate PPO objective function is 0.2 -> ppo clipping value
- number of update epochs per training batch is 4
- size of experience replay is either 5K episodes or 1M samples (depending on which uses lower memory)

- Gamma is already set by default both in Ray and in EPYMarl to 0.99


- The architecture of the actor/policy and network and critic network are exactly the same
- They're built using the "encode-layer" value of the dictionary passed
- Their input size is the observation space dimension
- Their output layers are however different: for the actor/policy network a final layer is added with an output size equivalent to the number of action, instead for the critic network a differnt final layer is added with only 1 neuron and output size 1 to return the expected return value from a given state (which means that the input of the critic is the observation space too, exactly the same input as the actor/policy network)

- In EpyMARL config files for ippo non sharing e ippo sharing:
- Epochs is the number of updates performed on each training batch -> is the equivalent of num_sgd_iter and its value is 4!
- Buffer_size is the amount of episodes to collect to form a training batch -> is the equivalent of batch_episode and its value is 10 -> (batch_episode * episode_limit) is the amount of timesteps collected in the training buffer size called train_batch_size 
- Batch_size is the number of episodes to train on -> is the equivalent of sgd_minibatch_size and is value is 10
- Batch_size_run is 10 and is the number of environments run in parallel. Environments are run in parallel only to collect samples more efficiently, so the number of network updates has to be kept the same instead of being multiplied for the times of parallel environments run. 
- NB To replicate such conditions for RWARE from the EpyMARL paper we have to set num_sgd_iter = 4 and batch_episode = 10. Since episodes lasts 500 timesteps we build buffers containing 10 * 500 = 5000 timesteps. Since in the original configs files ("https://github.com/uoe-agents/epymarl/blob/main/src/config/algs/ippo_ns.yaml" & "https://github.com/uoe-agents/epymarl/blob/main/src/config/algs/ippo.yaml") of the paper of RWARE the number of episodes used to perform one epoch of training is the same as the number in the buffer it means that the full buffer is used to perform the training for a total of 4 neural networks weights updates since the num_sgd_iter is 4. After this 4 updates the training buffer is populated with the next ten episodes and this procedure is started again until the 40M timesteps are reached. This means that the total number of weights updates performed on each network is 40M / 5000 = 8000 buffers * 4 = 32000. 


# Configuration parameters to use during rware simulation to train IPPO according to the rware paper:

## Two options

In IPPO and MAPPO the number of update epochs per training batch is 4 and the clipping value of the surrogate objective is 0.2.

**Configuration IPPO without parameters sharing and how to replicate it here**

In Rware paper:

- Neural network architecture is FC (fully connected) -> MLP in our case
- 3 layers with the hidden one of size 128 -> 128
- For those two "core_arch":{"mlp": "128"}
- Learning rate lr = 0.0005
- Entropy coefficient = 0.001

In Ray:

- In main_patrolling.py in ippo.fit share_policy='individual' to disable parameters sharing

**Configuration IPPO with parameters sharing and how to replicate it here**

In Ray:

- In main_patrolling.py in ippo.fit share_policy='group' to enable parameters sharing between agents of the same teams identified by teams prefix

