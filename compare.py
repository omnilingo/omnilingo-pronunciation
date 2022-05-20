import sys
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import json

F='gold'
g_data = json.loads(open(F + '.json').read())
g_array = g_data["logits"]
g_df_cm = pd.DataFrame(g_array, index = [i for i in range(0, len(g_data["logits"]))],
                  columns = g_data["alphabet"])
g_df_cm2 = g_df_cm.swapaxes(1, 0, copy=True)

F='test-1'
t1_data = json.loads(open(F + '.json').read())
t1_array = t1_data["logits"]
t1_df_cm = pd.DataFrame(t1_array, index = [i for i in range(0, len(t1_data["logits"]))],
                  columns = t1_data["alphabet"])
t1_df_cm2 = t1_df_cm.swapaxes(1, 0, copy=True)

F='test-2'
t2_data = json.loads(open(F + '.json').read())
t2_array = t2_data["logits"]
t2_df_cm = pd.DataFrame(t2_array, index = [i for i in range(0, len(t2_data["logits"]))],
                  columns = t2_data["alphabet"])
t2_df_cm2 = t2_df_cm.swapaxes(1, 0, copy=True)

F='test-3'
t3_data = json.loads(open(F + '.json').read())
t3_array = t3_data["logits"]
t3_df_cm = pd.DataFrame(t3_array, index = [i for i in range(0, len(t3_data["logits"]))],
                  columns = t3_data["alphabet"])
t3_df_cm2 = t3_df_cm.swapaxes(1, 0, copy=True)


####

fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(72,50))

for ax in axes:
    ax.set_anchor('E', share=True)

ax1, ax2, ax3, ax4 = axes

im1 = ax1.imshow(g_df_cm2)
im2 = ax2.imshow(t1_df_cm2)
im3 = ax3.imshow(t2_df_cm2)
im4 = ax4.imshow(t3_df_cm2)


ax1.set_xticks(range(len(g_data["logits"])))
ax1.set_yticks(range(len(g_data["alphabet"])))
ax1.set_xticklabels(range(len(g_data["logits"])))
ax1.set_yticklabels(g_data["alphabet"])
ax1.set_title(' '.join([i["word"] for i in g_data["words"]]), fontsize=40)
plt.setp(ax1.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')
plt.colorbar(im1, fraction=0.045, pad=0.05, ax=ax1)

ax2.set_xticks(range(len(t1_data["logits"])))
ax2.set_yticks(range(len(t1_data["alphabet"])))
ax2.set_xticklabels(range(len(t1_data["logits"])))
ax2.set_yticklabels(t1_data["alphabet"])
ax2.set_title(' '.join([i["word"] for i in t1_data["words"]]), fontsize=40)
plt.setp(ax2.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')
plt.colorbar(im2, fraction=0.045, pad=0.05, ax=ax2)

ax3.set_xticks(range(len(t2_data["logits"])))
ax3.set_yticks(range(len(t2_data["alphabet"])))
ax3.set_xticklabels(range(len(t2_data["logits"])))
ax3.set_yticklabels(t2_data["alphabet"])
ax3.set_title(' '.join([i["word"] for i in t2_data["words"]]), fontsize=40)
plt.setp(ax3.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')
plt.colorbar(im3, fraction=0.045, pad=0.05, ax=ax3)

ax4.set_xticks(range(len(t3_data["logits"])))
ax4.set_yticks(range(len(t3_data["alphabet"])))
ax4.set_xticklabels(range(len(t3_data["logits"])))
ax4.set_yticklabels(t3_data["alphabet"])
ax4.set_title(' '.join([i["word"] for i in t3_data["words"]]), fontsize=40)
plt.setp(ax4.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')
plt.colorbar(im4, fraction=0.045, pad=0.05, ax=ax4)

plt.xlabel("Timesteps")
plt.ylabel("Symbol")

fig.tight_layout()

fig.savefig("output.png")
