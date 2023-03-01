import pickle
from collections import Counter
import matplotlib.pyplot as plt

sensitivity = 0.1

with open('sk_predictions.binary', 'rb') as in_file:
    predictions = pickle.load(in_file)

# Sanity check that the predictions are all the same order
assert predictions[0][2] == predictions[1][2] == predictions[2][2]

interesting = []
cos_right = []
cos_right_1 = []
cos_right_0 = []
cos_wrong = []
cos_wrong_1 = []
cos_wrong_0 = []
cos_both_right = []
cos_both_right_1 = []
cos_both_right_0 = []
cos_both_wrong_1 = []
cos_both_wrong_0 = []
jsd_right = []
jsd_wrong = []
jsd_wrong_1 = []
jsd_wrong_0 = []
jsd_both_right = []
for x in range(5000):
    # test and cos are different
    if predictions[0][1][x] != predictions[1][1][x]:
        if predictions[1][1][x] == predictions[1][2][x]: # cos got it right
            if predictions[1][2][x] == 1: # correct answer is 1
                cos_right_1.append(x)
            else: # correct answer is 0
                cos_right_0.append(x)
            cos_right.append(x)
        else: # cos got it wrong
            if predictions[1][2][x] == 1: # correct answer is 1
                cos_wrong_1.append(x)
            else: # correct answer is 0
                cos_wrong_0.append(x)
            cos_wrong.append(x)
        interesting.append(x)
    else: # test and cos are the same
        if predictions[1][1][x] == predictions[1][2][x]: # both got it right
            if predictions[1][2][x] == 1: # correct answer is 1
                cos_both_right_1.append(x)
            else: # correct answer is 0
                cos_both_right_0.append(x)
        else: # both got it wrong
            if predictions[1][2][x] == 1:
                cos_both_wrong_1.append(x)
            else:
                cos_both_wrong_0.append(x)
        cos_both_right.append(x)
    # test and jsd are different
    if predictions[0][1][x] != predictions[2][1][x]:
        if predictions[2][1][x] == predictions[2][2][x]: # jsd got it right
            jsd_right.append(x)
        else: # jsd got it wrong
            if predictions[2][2][x] == 1: # correct answer is 1
                jsd_wrong_1.append(x)
            else: # correct answer is 0
                jsd_wrong_0.append(x)
            jsd_wrong.append(x)
        interesting.append(x)
    else:
        jsd_both_right.append(x)

# with open('error_analysis.txt', 'w') as out_file:
#     print('Cosine right, test wrong', file = out_file)
#     for x in cos_right:
#         t = '\t'.join([str(n).rjust(5) for n in predictions[1][0][x] if isinstance(n, float)])
#         c = '\t'.join([str(n).rjust(5) for n in predictions[0][0][x] if isinstance(n, float)])
#         print(f'Test:   {t}', file = out_file)
#         print(f'Cosine: {c}', file = out_file)
#     print('Cosine wrong, test right', file = out_file)
#     for x in cos_wrong:
#         t = '\t'.join([str(n).rjust(5) for n in predictions[0][0][x] if isinstance(n, float)])
#         c = '\t'.join([str(n).rjust(5) for n in predictions[1][0][x] if isinstance(n, float)])
#         print(f'Test:   {t}', file = out_file)
#         print(f'Cosine: {c}', file = out_file)
#     print('JSD right, test wrong', file = out_file)
#     for x in jsd_right:
#         t = '\t'.join([str(n).rjust(5) for n in predictions[0][0][x] if isinstance(n, float)])
#         j = '\t'.join([str(n).rjust(5) for n in predictions[2][0][x] if isinstance(n, float)])
#         print(f'Test: {t}', file = out_file)
#         print(f'JSD:  {j}', file = out_file)
#     print('JSD wrong, test right', file = out_file)
#     for x in jsd_wrong:
#         t = '\t'.join([str(n).rjust(5) for n in predictions[0][0][x] if isinstance(n, float)])
#         j = '\t'.join([str(n).rjust(5) for n in predictions[2][0][x] if isinstance(n, float)])
#         print(f'Test: {t}', file = out_file)
#         print(f'JSD:  {j}', file = out_file)

deltas_cr = []
deltas_cr_0 = []
deltas_cr_1 = []
deltas_cw = []
deltas_cw_0 = []
deltas_cw_1 = []
deltas_cbr = []
deltas_bw_0 = []
deltas_bw_1 = []
deltas_br_0 = []
deltas_br_1 = []
# for x in cos_right:
#     # negative numbers, cos is higher score than baseline
#     deltas_cr.extend([pair[0] - pair[1] for pair in zip(predictions[0][0][x], predictions[1][0][x]) if abs(pair[0] - pair[1]) > sensitivity])
# for x in cos_wrong:
#     # negative numbers, cos is higher score than baseline
#     deltas_cw.extend([pair[0] - pair[1] for pair in zip(predictions[0][0][x], predictions[1][0][x]) if abs(pair[0] - pair[1]) > sensitivity])
# for x in cos_both_right:
#     # negative numbers, cos is higher score than baseline
#     deltas_cbr.extend([pair[0] - pair[1] for pair in zip(predictions[0][0][x], predictions[1][0][x]) if abs(pair[0] - pair[1]) > sensitivity])
for x in cos_wrong_0:
    # negative numbers, cos is higher score than baseline
    deltas_cw_0.extend([pair[0] - pair[1] for pair in zip(predictions[0][0][x], predictions[1][0][x]) if abs(pair[0] - pair[1]) > sensitivity])
for x in cos_wrong_1:
    # negative numbers, cos is higher score than baseline
    deltas_cw_1.extend([pair[0] - pair[1] for pair in zip(predictions[0][0][x], predictions[1][0][x]) if abs(pair[0] - pair[1]) > sensitivity])
for x in cos_right_0:
    # negative numbers, cos is higher score than baseline
    deltas_cr_0.extend([pair[0] - pair[1] for pair in zip(predictions[0][0][x], predictions[1][0][x]) if abs(pair[0] - pair[1]) > sensitivity])
for x in cos_right_1:
    # negative numbers, cos is higher score than baseline
    deltas_cr_1.extend([pair[0] - pair[1] for pair in zip(predictions[0][0][x], predictions[1][0][x]) if abs(pair[0] - pair[1]) > sensitivity])

for x in cos_both_wrong_0:
    # negative numbers, cos is higher score than baseline
    deltas_bw_0.extend([pair[0] - pair[1] for pair in zip(predictions[0][0][x], predictions[1][0][x]) if abs(pair[0] - pair[1]) > sensitivity])
for x in cos_both_wrong_1:
    # negative numbers, cos is higher score than baseline
    deltas_bw_1.extend([pair[0] - pair[1] for pair in zip(predictions[0][0][x], predictions[1][0][x]) if abs(pair[0] - pair[1]) > sensitivity])
for x in cos_both_right_0:
    # negative numbers, cos is higher score than baseline
    deltas_br_0.extend([pair[0] - pair[1] for pair in zip(predictions[0][0][x], predictions[1][0][x]) if abs(pair[0] - pair[1]) > sensitivity])
for x in cos_both_right_1:
    # negative numbers, cos is higher score than baseline
    deltas_br_1.extend([pair[0] - pair[1] for pair in zip(predictions[0][0][x], predictions[1][0][x]) if abs(pair[0] - pair[1]) > sensitivity])

# cr_counter = Counter(deltas_cr)
# print(' '.join([f'{k}: {v}' for k, v in sorted(cr_counter.items())]))
# cw_counter = Counter(deltas_cw)
# print(' '.join([f'{k}: {v}' for k, v in sorted(cw_counter.items())]))
# cbr_counter = Counter(deltas_cbr)
# print(' '.join([f'{k}: {v}' for k, v in sorted(cw_counter.items())]))

# make the graphs
fig, ax1 = plt.subplots()
labels = ['Base Right\nAnswer 0', 'Base Right\nAnswer 1', 'Cos Right\nAnswer 0', 'Cos Right\nAnswer 1']
ax1.violinplot([deltas_cw_0, deltas_cw_1, deltas_cr_0, deltas_cr_1], widths=0.9, showmeans=True, showextrema=False)
ax1.set_xticks(list(range(1, len(labels)+1)))
ax1.set_xticklabels(labels)

fig2, ax2 = plt.subplots()
labels = ['Both Wrong\nAnswer 0', 'Both Wrong\nAnswer 1', 'Both Right\nAnswer 0', 'Both Right\nAnswer 1']
ax2.violinplot([deltas_bw_0, deltas_bw_1, deltas_br_0, deltas_br_1], widths=0.9, showmeans=True, showextrema=False)
ax2.set_xticks(list(range(1, len(labels)+1)))
ax2.set_xticklabels(labels)
plt.show()
pass
