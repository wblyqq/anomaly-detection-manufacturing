#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 11:47:19 2018

@author: ramanathan
"""

import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
df_actions = pd.read_csv("../data/output/v2/results/CU36_CU110/20180910_033055_154_2_analysis.csv")

color = {"undefined":"orange", "push part with fingers":"crimson", "access box":"blue",
         "push part with tool":"green", "turn phone by 90": "turquoise"}
fig_gantt, ax_gantt = plt.subplots(figsize=(16,9))


action_timeline = defaultdict(list)
for i, action in enumerate(df_actions.groupby([(df_actions.Action != df_actions.Action.shift()).cumsum()])):
    start_tuple = (action[1].values[0][0], action[1].values[-1][0] - action[1].values[0][0])
    action_timeline[action[1].values[0][1]].append(start_tuple)

#i = 0
for k, v in action_timeline.items():
    ax_gantt.broken_barh(v, (0,1), color=color[k])
    #i += 1

labels=list(action_timeline.keys())
#ax_gantt.set_yticks(range(len(labels)))
#ax_gantt.set_yticklabels(labels) 
ax_gantt.set_xlabel("Time")
ax_gantt.set_ylabel("Action")
ax_gantt.legend(labels, ncol = 5, loc = 8)

plt.tight_layout()       
plt.savefig("../data/output/v2/results/CU36_CU110/20180910_033055_154_2_actions_gantt_test_v2.png")
plt.show()

fig_hist, ax_hist = plt.subplots(figsize=(16,9))
ax_hist = df_actions.groupby(df_actions["Action"]).count()['Frame'].plot(kind="bar")
ax_hist.set_xlabel("Action")
ax_hist.set_ylabel("Time")
plt.tight_layout()
plt.savefig("../data/output/v2/results/CU36_CU110/20180910_033055_154_2_actions_hist_test_v2.png")


plt.show()
