import pickle
from collections import Counter
import matplotlib.pyplot as plt

sensitivity = 0.05

with open('sk_hell_predictions.binary', 'rb') as in_file:
    predictions = pickle.load(in_file)[18]

with open('sk_predictions.binary', 'rb') as in_file:
    base_predictions = pickle.load(in_file)[0]

hel_all = []
jsd_all = []
xen_all = []
cos_right_1 = []
cos_right_0 = []
cos_wrong_1 = []
cos_wrong_0 = []
cos_both_right_1 = []
cos_both_right_0 = []
cos_both_wrong_1 = []
cos_both_wrong_0 = []
for x in range(5000):
    # test and cos are different
    # if predictions[0][1][x] != predictions[1][1][x]:
    #     if predictions[1][1][x] == predictions[1][2][x]: # cos got it right
    #         if predictions[1][2][x] == 1: # correct answer is 1
    #             cos_right_1.append(x)
    #         else: # correct answer is 0
    #             cos_right_0.append(x)
    #     else: # cos got it wrong
    #         if predictions[1][2][x] == 1: # correct answer is 1
    #             cos_wrong_1.append(x)
    #         else: # correct answer is 0
    #             cos_wrong_0.append(x)
    # else: # test and cos are the same
    #     if predictions[1][1][x] == predictions[1][2][x]: # both got it right
    #         if predictions[1][2][x] == 1: # correct answer is 1
    #             cos_both_right_1.append(x)
    #         else: # correct answer is 0
    #             cos_both_right_0.append(x)
    #     else: # both got it wrong
    #         if predictions[1][2][x] == 1:
    #             cos_both_wrong_1.append(x)
    #         else:
    #             cos_both_wrong_0.append(x)
    hel_all.extend([pair for pair in zip(base_predictions[0][x], predictions[0][x])])


fig7, ax7 = plt.subplots()
fig7.suptitle('Baseline vs Hellinger')
ax7.scatter([x[0] for x in hel_all], [x[1] for x in hel_all], s = 1, c = '#990000')
ax7.axline((0, 1), slope = -1, c = 'black')
ax7.set_xlabel('Baseline Score')
ax7.set_ylabel('Hellinger Score')

plt.show()
