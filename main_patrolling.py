import marllib as marllib
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY

import random
import numpy as np

from ray_rl_patrolling_environment import RayGraphEnv, RayGraphEnv_Coop

##################################### CUSTOM ENV ############################################

#A nice idea to simplify the environment would be to pass the graph to it externally
#so it doesn't have to be generated inside it
#This way a database of graphs can be built and an environment can be created from one of
#graph of this dataset that can be passed to the environment

#CUSTOM_ENV
ENV_REGISTRY["patrolling"] = RayGraphEnv
#CUSTOM COOP ENV as subclass of RayGraphEnv
COOP_ENV_REGISTRY["patrolling"] = RayGraphEnv_Coop

#marl.make_env is in marllib.marl in the __init__.py file
#It returns the environment and its configuration (config used to setup and initialize Ray) 
#as a tuple (env, env_config)
env = marl.make_env(
        ############# Default params
        environment_name="patrolling", 
        map_name="patrolling_graph",
        force_coop=True, #Change this value to True to register the environment as a coop one
        #By making force_coop True the total reward is split equally between agents
        ############# Additional optional params to change yaml file default values -> **env_params
        #NB Only parameters already specified in yaml file can be passed and changed otherwise error!

)
#Alternative: 
# env, env_config = marl.make_env(...)
# ...
# model = marl.build_model((env, env_config), mappo, ...)

print(env[0])
print(env[1])
#Fino qua tutto ok -> il codice della libreria è molto semplice

# pick algorithm
#Initialization done using the class _AlgoManager in marllib.marl in file __init__.py

#Algo parameters can be passed as additional optional parameters to override the
#default values of the algorithm present in its yaml config file
#Three options: $ENV, "test" or "common"
#Their respective parameters are in the marllib.marl.algos.hyperparameters folders
#Better not to touch them in the original yaml files and pass them here as additional parameters
ippo = marl.algos.ippo(
        ########### Default mandatory parameter
        hyperparam_source="test"
        ########### Additional parameters to override the default ones in test.yaml
        # ...
)

# customize model
#
# core_arch can be: mlp (multi-layer perceptron), gru (gated recurrent unit) or 
# lstm (long short-term memory)

#Successivamente nel dizionario presente come terzo parametro dopo core_arch
#si possono inserire come coppie di chiavi valori tutti i campi che sono
#presenti nel file yaml di config per modificarli, dove quest'ultimi vanno presi
#in base alla core_arch indicata e al modello che si è scelto come secondo parametro
#Vedi file __init__.py in marllib.marl nella funzione build_model per capire
#i parametri disponibili per il caso che si vuole studiare

#Esempio: in questo caso abbiamo mappo come algoritmo e "core_arch":"mlp" come architettura

# MLP yaml file parameters:
#model_arch_args:
#  hidden_state_size: 256
#  core_arch: "mlp"

#Observation space with shape (..., 1) so fc encoder otherwise if there are multiple channels
#the cnn encoder has to be used instead
#fc encoder params examples:

# model_arch_args:
  # fc_layer: 1 
  # out_dim_fc_0: 128

#model_arch_args:
#  fc_layer: 2
#  out_dim_fc_0: 128
#  out_dim_fc_1: 64

#model_arch_args:
# fc_layer: 3
# encode_layer: "xxx-xxx-xxx" 

#fc_layer has to contain the number of layers of the encoder

#If "encode_layer" is used the encoder is built using the corresponding value which
# has to be string with numbers with "-" between consecutive numbers where such numbers 
# are the number of neuron that are in each layer of the encoder
# NB if the encode-layer field is present only that will be used to build the encoder, 
# so the original fc_encoder.yaml file parameters will be ignored
# Instead if no encode-layer field is present a layer will be built for each output_dim_fc_{} value present
# using the index to order them based on the fc_encoder.yaml file

# Ex. {"encode_layer": "128-128"}

# then number of neurons equal to fc_layer can be added by putting
#a "-" between each number like in last example, otherwise the syntax to use is the one of the
#two first examples with out_dim_fc_.{i} for each fc_layer indicated 

#Also build_model returns a tuple, with a model and its configuration
#model, model_config = marl.build_model(...)

model = marl.build_model(env, ippo, {"core_arch": "mlp", "encode_layer": "256-128"})

# start learning
#mappo.fit(env, model, stop={'episode_reward_mean': 2000, 'timesteps_total': 10000}, local_mode=True, num_gpus=1,
#        num_workers=2, share_policy='group', checkpoint_freq=50, use_opponent_obs=False)

#possible stop conditions to use in the stop dictionary:
# 1) episode_reward_mean
# 2) timesteps_total
# 3) training_iteration

#Extremely important: the lenght of the training is the product of "episode limit" in
# get_env_info and "timesteps_total" in the stop conditions

ippo.fit(
        ########## Mandatory parameters
        env, # Tuple resulting from make_env
        model, # Tuple resulting from build_model
        stop={'timesteps_total': 40000000}, # Dictionary to define stop conditions
        local_mode=True,
        num_gpus=1,
        num_workers=4,
        share_policy='group', #Può essere "all", "group" oppure "individual"
        checkpoint_freq=5000,
        #seed=0
        #checkpoint_end=False,
)

print("finished training successfully")