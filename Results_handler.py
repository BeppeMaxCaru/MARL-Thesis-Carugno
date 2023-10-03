import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

#filepath_experiment1 = os.path.abspath(r"C:\Users\gmaxc\OneDrive - Università degli Studi di Milano\Thesis experiments\RPatrolling\Cooperative\Non-adversarial\AVG idleness\IPPO\No params sharing\Renamed results file\RPatrolling IPPO coop 256-128 AVG idleness no params sharing.csv")
#filepath_experiment2 = os.path.abspath(r"C:\Users\gmaxc\OneDrive - Università degli Studi di Milano\Thesis experiments\RPatrolling\Cooperative\Non-adversarial\AVG idleness\IPPO\Params sharing\Renamed results file\RPatrolling IPPO coop 256-128 AVG idleness with params sharing.csv")
#print(filepath_experiment1)

#algo_name = "IPPO"
#filepath_experiment1 = os.path.abspath(r"C:\Users\gmaxc\OneDrive - Università degli Studi di Milano\Thesis experiments\RPatrolling\Cooperative\Non-adversarial\MAX idleness\IPPO\No params sharing\Renamed results file\RPatrolling IPPO coop 256-128 MAX idleness no params sharing.csv")
#filepath_experiment2 = os.path.abspath(r"C:\Users\gmaxc\OneDrive - Università degli Studi di Milano\Thesis experiments\RPatrolling\Cooperative\Non-adversarial\MAX idleness\IPPO\Params sharing\Renamed results file\RPatrolling IPPO coop 256-128 MAX idleness with params sharing.csv")

algo_name = "MAPPO"
#filepath_experiment1 = os.path.abspath(r"C:\Users\gmaxc\OneDrive - Università degli Studi di Milano\Thesis experiments\RPatrolling\Cooperative\Non-adversarial\AVG idleness\MAPPO\No params sharing\Renamed results file\RPatrolling MAPPO coop 256-128 AVG idleness no params sharing.csv")
#filepath_experiment2 = os.path.abspath(r"C:\Users\gmaxc\OneDrive - Università degli Studi di Milano\Thesis experiments\RPatrolling\Cooperative\Non-adversarial\AVG idleness\MAPPO\Params sharing\Renamed results file\RPatrolling MAPPO coop 256-128 AVG idleness with params sharing.csv")

filepath_experiment1 = os.path.abspath(r"C:\Users\gmaxc\OneDrive - Università degli Studi di Milano\Thesis experiments\RPatrolling\Cooperative\Non-adversarial\MAX idleness\MAPPO\No params sharing\Renamed results file\RPatrolling MAPPO coop 256-128 MAX idleness no params sharing.csv")
filepath_experiment2 = os.path.abspath(r"C:\Users\gmaxc\OneDrive - Università degli Studi di Milano\Thesis experiments\RPatrolling\Cooperative\Non-adversarial\MAX idleness\MAPPO\Params sharing\Renamed results file\RPatrolling MAPPO coop 256-128 MAX idleness with params sharing.csv")

dataframe_experiment1 = pd.read_csv(filepath_experiment1)
dataframe_experiment2 = pd.read_csv(filepath_experiment2)
#print(dataframe_experiment1.iloc[:,[2]])

df1 = dataframe_experiment1.loc[:,["episode_reward_mean", "timesteps_total"]]
df2 = dataframe_experiment2.loc[:,["episode_reward_mean", "timesteps_total"]]
#print(df1)

#Find max and min timesteps to place reward lines in same window with same x-axis lenght
min_timesteps = max(df1["timesteps_total"].min(), df2["timesteps_total"].min())
max_timesteps = min(df1["timesteps_total"].max(), df2["timesteps_total"].max())
print(max_timesteps)

#Add raw figures
fig_final, ax_final = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig_final.suptitle(algo_name)

ax_final[0].plot(df1["timesteps_total"], df1["episode_reward_mean"], label = "raw results without parameters sharing", color="r")
ax_final[0].set_xlim(0, 50000000)
ax_final[0].legend(loc="lower right")
ax_final[0].grid()
ax_final[0].yaxis.set_tick_params(labelbottom=True)

ax_final[1].plot(df2["timesteps_total"], df2["episode_reward_mean"], label = "raw results with parameters sharing", color="b")
ax_final[1].set_xlim(0, 50000000)
ax_final[1].legend(loc="lower right")
ax_final[1].grid()
ax_final[1].yaxis.set_tick_params(labelbottom=True)

#Alternative to polynomila regression can be moving averages
z = np.polyfit(df1["timesteps_total"], df1["episode_reward_mean"], 3)
p = np.poly1d(z)
ax_final[2].plot(df1["timesteps_total"], p(df1["timesteps_total"]), label = "smoothed results without parameters sharing", color="r")

t = np.polyfit(df2["timesteps_total"], df2["episode_reward_mean"], 3)
k = np.poly1d(t)
ax_final[2].plot(df2["timesteps_total"], k(df2["timesteps_total"]), label = "smoothed results with parameters sharing", color="b")
#Test SMA
#ax_final[2].plot(df2["timesteps_total"], df2["episode_reward_mean"].expanding().mean())

ax_final[2].legend(loc="lower right")
ax_final[2].grid()

# Set the x-axis limits based on the minimum and maximum timesteps
ax_final[2].set_xlim(0, 50000000)
ax_final[2].yaxis.set_tick_params(labelbottom=True)

fig_final.supxlabel("Training iterations")
fig_final.supylabel("Mean reward per episode")

ax_final[0].set_ylim(ax_final[2].get_ylim()[0], ax_final[2].get_ylim()[1])
ax_final[1].set_ylim(ax_final[2].get_ylim()[0], ax_final[2].get_ylim()[1])


plt.tight_layout()
plt.show()
