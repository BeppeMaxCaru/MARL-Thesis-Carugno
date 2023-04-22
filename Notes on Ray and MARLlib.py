def notes(self):
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