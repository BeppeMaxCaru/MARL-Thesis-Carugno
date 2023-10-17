import numpy as np

target_nodes_locations = np.random.randint(1, 1000, 10)
print(target_nodes_locations)

def gen_target_attacks_start_and_length_values(target_nodes_location, episode_duration, num_attacks, max_attack_duration):
            
        attacks_tracker_dict = {}
                
        #Range of values that can be sampled is equal to 1/scale in exp distribution
        #importance_values = np.round(np.random.exponential(scale=100, size=len(target_nodes_location))).astype(int)
        #Scale them to upper limit of 1
        #scaled_importance_vales = importance_values / max(importance_values)
                
        for i in range(len(target_nodes_location)):
            #sort to order attacks in time
            attack_start_times = np.sort(np.random.choice(episode_duration, num_attacks, replace=False))
            #Extract random attack lenght
            attack_durations = np.random.randint(1, max_attack_duration, num_attacks)
            
            #final_attacks_duration = np.ceil(attack_durations * scaled_importance_vales[i]).astype(int)
            final_attacks_duration = np.ceil(attack_durations).astype(int)
            
            attacks_tracker_dict[str(target_nodes_location[i])] = list(zip([attack_start_times, final_attacks_duration]))
        
        return attacks_tracker_dict


# Example usage
num_attacks = 10
episode_duration = 500
max_attack_duration = 50  # Adjust this value as needed

att_dict = gen_target_attacks_start_and_length_values(target_nodes_locations, num_attacks=10, episode_duration=500, max_attack_duration=50)
for key, values in att_dict.items():
    print(values)
    
print("Attacchi poisson")
attacchi = np.random.poisson(10, 10)
print(attacchi)

def gen_attacks_with_poisson(attack_frequency):
    np.random.poisson(10, 10)
    
def generate_attacks_using_poisson(_planned_attacks_on_target_nodes, episode_duration, average_attacks_per_step, max_attack_duration):
        attacks_tracker_dict = {}
        
        for target in range(_planned_attacks_on_target_nodes):
            # Calculate the number of attacks for this target using a Poisson distribution
            num_attacks_for_target = np.random.poisson(average_attacks_per_step * episode_duration)
            
            # Generate random attack start times for this target
            attack_start_times = np.sort(np.random.uniform(0, episode_duration, num_attacks_for_target))
            
            # Extract random attack lengths for this target
            attack_durations = np.random.randint(1, max_attack_duration, num_attacks_for_target)
            
            #Round attacks start times
            final_attacks_start_times = np.ceil(attack_start_times).astype(int)
            # Round attacks to integers
            final_attacks_duration = np.ceil(attack_durations).astype(int)
            
            # Build planned attack dict for the current target
            attacks_tracker_dict[str(target)] = list(zip(final_attacks_start_times, final_attacks_duration))
        
        return attacks_tracker_dict
    
risultati = generate_attacks_using_poisson(10, 500, 0.02, 10)
print(risultati)
    

        