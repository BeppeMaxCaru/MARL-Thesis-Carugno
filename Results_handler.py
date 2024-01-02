import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

algo_name = "IPPO"
filepath_experiment1 = os.path.abspath(r"C:\Users\gmaxc\OneDrive - Università degli Studi di Milano\Thesis experiments\RWARE\IPPO\No params sharing\Trial AISLAB VM 40M interrupted ippo cooperative no params sharing mlp 258-128 tiny 4p 11x11 medium difficulty.xlsx")
filepath_experiment2 = os.path.abspath(r"C:\Users\gmaxc\OneDrive - Università degli Studi di Milano\Thesis experiments\RWARE\IPPO\Params sharing\Trial AISLAB VM 40M interrupted ippo cooperative with params sharing mlp 258-128 tiny 4p 11x11 medium difficulty.xlsx")
#print(filepath_experiment1)

#For CSV
#dataframe_experiment1 = pd.read_csv(filepath_experiment1)
#dataframe_experiment2 = pd.read_csv(filepath_experiment2)
#print(dataframe_experiment1.iloc[:,[2]])

#For Excel
dataframe_experiment1 = pd.read_excel(filepath_experiment1)
dataframe_experiment2 = pd.read_excel(filepath_experiment2)

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
