import numpy as np

target_nodes_locations = np.random.randint(1, 1000, 10)
print(target_nodes_locations)

def gen_target_attacks_start_and_length_values(target_nodes_location, episode_duration, num_attacks, max_attack_duration):
            
        attacks_tracker_dict = {}
                
        #Range of values that can be sampled is equal to 1/scale in exp distribution
        importance_values = np.round(np.random.exponential(scale=100, size=len(target_nodes_location))).astype(int)
        #Scale them to upper limit of 1
        scaled_importance_vales = importance_values / max(importance_values)
                
        for i in range(len(target_nodes_location)):
            #sort to order attacks in time
            attack_start_times = np.sort(np.random.choice(episode_duration, num_attacks, replace=False))
            #Extract random attack lenght
            attack_durations = np.random.randint(1, max_attack_duration, num_attacks)
            
            final_attacks_duration = np.ceil(attack_durations * scaled_importance_vales[i]).astype(int)
            
            attacks_tracker_dict[str(target_nodes_location[i])] = list(zip([attack_start_times, final_attacks_duration]))
        
        return attacks_tracker_dict 


# Example usage
num_attacks = 10
episode_duration = 500
max_attack_duration = 50  # Adjust this value as needed

att_dict = gen_target_attacks_start_and_length_values(target_nodes_locations, num_attacks=10, episode_duration=500, max_attack_duration=50)
for key, values in att_dict.items():
    print(values)