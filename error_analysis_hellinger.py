import pickle
from collections import Counter
import matplotlib.pyplot as plt

sensitivity = 0.05

with open('sk_predictions.binary', 'rb') as in_file:
    predictions = pickle.load(in_file)

# Sanity check that the predictions are all the same order
assert predictions[0][2] == predictions[1][2] == predictions[2][2] == predictions[3][2]

cos_all = []
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
    if predictions[0][1][x] != predictions[1][1][x]:
        if predictions[1][1][x] == predictions[1][2][x]: # cos got it right
            if predictions[1][2][x] == 1: # correct answer is 1
                cos_right_1.append(x)
            else: # correct answer is 0
                cos_right_0.append(x)
        else: # cos got it wrong
            if predictions[1][2][x] == 1: # correct answer is 1
                cos_wrong_1.append(x)
            else: # correct answer is 0
                cos_wrong_0.append(x)
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
    cos_all.extend([pair for pair in zip(predictions[0][0][x], predictions[1][0][x])])

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

################################## COS ##################################
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
    deltas_cw_0.extend([pair for pair in zip(predictions[0][0][x], predictions[1][0][x])])# if abs(pair[0] - pair[1]) > sensitivity])
for x in cos_wrong_1:
    # negative numbers, cos is higher score than baseline
    deltas_cw_1.extend([pair for pair in zip(predictions[0][0][x], predictions[1][0][x])])# if abs(pair[0] - pair[1]) > sensitivity])
for x in cos_right_0:
    # negative numbers, cos is higher score than baseline
    deltas_cr_0.extend([pair for pair in zip(predictions[0][0][x], predictions[1][0][x])])# if abs(pair[0] - pair[1]) > sensitivity])
for x in cos_right_1:
    # negative numbers, cos is higher score than baseline
    deltas_cr_1.extend([pair for pair in zip(predictions[0][0][x], predictions[1][0][x])])# if abs(pair[0] - pair[1]) > sensitivity])

for x in cos_both_wrong_0:
    # negative numbers, cos is higher score than baseline
    deltas_bw_0.extend([pair for pair in zip(predictions[0][0][x], predictions[1][0][x])])# if abs(pair[0] - pair[1]) > sensitivity])
for x in cos_both_wrong_1:
    # negative numbers, cos is higher score than baseline
    deltas_bw_1.extend([pair for pair in zip(predictions[0][0][x], predictions[1][0][x])])# if abs(pair[0] - pair[1]) > sensitivity])
for x in cos_both_right_0:
    # negative numbers, cos is higher score than baseline
    deltas_br_0.extend([pair for pair in zip(predictions[0][0][x], predictions[1][0][x])])# if abs(pair[0] - pair[1]) > sensitivity])
for x in cos_both_right_1:
    # negative numbers, cos is higher score than baseline
    deltas_br_1.extend([pair for pair in zip(predictions[0][0][x], predictions[1][0][x])])# if abs(pair[0] - pair[1]) > sensitivity])

# cr_counter = Counter(deltas_cr)
# print(' '.join([f'{k}: {v}' for k, v in sorted(cr_counter.items())]))
# cw_counter = Counter(deltas_cw)
# print(' '.join([f'{k}: {v}' for k, v in sorted(cw_counter.items())]))
# cbr_counter = Counter(deltas_cbr)
# print(' '.join([f'{k}: {v}' for k, v in sorted(cw_counter.items())]))

# make the graphs
# fig, ax1 = plt.subplots()
fig, ((ax1a, ax1b), (ax1c, ax1d)) = plt.subplots(nrows=2, ncols=2)
# labels = ['Base Right\nAnswer 0', 'Base Right\nAnswer 1', 'Cos Right\nAnswer 0', 'Cos Right\nAnswer 1']
# violin1 = ax1.violinplot([deltas_cw_0, deltas_cw_1, deltas_cr_0, deltas_cr_1], widths=0.9, showmeans=True, showextrema=False)
# for b in violin1['bodies']:
#     b.set_color('#990000')
# violin1['cmeans'].set_color('#990000')
# ax1.set_xticks(list(range(1, len(labels)+1)))
# ax1.set_xticklabels(labels)
# x is baseline y is cos
fig.suptitle('Baseline vs Cosine with Different Answers')
ax1a.scatter([x[0] for x in deltas_cw_0], [x[1] for x in deltas_cw_0], s = 1, c = '#990000')
ax1a.axline((0, 0), slope = 1, c = 'black')
ax1a.set_xlabel('Baseline Score')
ax1a.set_ylabel('Cosine Score')
ax1a.set_title('Cosine Wrong Answer 0')
ax1b.scatter([x[0] for x in deltas_cw_1], [x[1] for x in deltas_cw_1], s = 1, c = '#990000')
ax1b.axline((0, 0), slope = 1, c = 'black')
ax1b.set_xlabel('Baseline Score')
ax1b.set_ylabel('Cosine Score')
ax1b.set_title('Cosine Wrong Answer 1')
ax1c.scatter([x[0] for x in deltas_cr_0], [x[1] for x in deltas_cr_0], s = 1, c = '#990000')
ax1c.axline((0, 0), slope = 1, c = 'black')
ax1c.set_xlabel('Baseline Score')
ax1c.set_ylabel('Cosine Score')
ax1c.set_title('Cosine Right Answer 0')
ax1d.scatter([x[0] for x in deltas_cr_1], [x[1] for x in deltas_cr_1], s = 1, c = '#990000')
ax1d.axline((0, 0), slope = 1, c = 'black')
ax1d.set_xlabel('Baseline Score')
ax1d.set_ylabel('Cosine Score')
ax1d.set_title('Cosine Right Answer 1')
# plt.show()
# exit()

# fig2, ax2 = plt.subplots()
fig2, ((ax2a, ax2b), (ax2c, ax2d)) = plt.subplots(nrows=2, ncols=2)
# labels = ['Both Wrong\nAnswer 0', 'Both Wrong\nAnswer 1', 'Both Right\nAnswer 0', 'Both Right\nAnswer 1']
# violin2 = ax2.violinplot([deltas_bw_0, deltas_bw_1, deltas_br_0, deltas_br_1], widths=0.9, showmeans=True, showextrema=False)
# for b in violin2['bodies']:
#     b.set_color('#990000')
# violin2['cmeans'].set_color('#990000')
# ax2.set_xticks(list(range(1, len(labels)+1)))
# ax2.set_xticklabels(labels)
# ax2.set_ylabel('Baseline Score - Cos Score')
# ax2.set_title('Difference Between Cosine and Baseline\nwith Same Answers')
# x is baseline y is cos
fig2.suptitle('Baseline vs Cosine with Same Answers')
ax2a.scatter([x[0] for x in deltas_bw_0], [x[1] for x in deltas_bw_0], s = 1, c = '#990000')
ax2a.axline((0, 0), slope = 1, c = 'black')
ax2a.set_xlabel('Baseline Score')
ax2a.set_ylabel('Cosine Score')
ax2a.set_title('Both Wrong Answer 0')
ax2b.scatter([x[0] for x in deltas_bw_1], [x[1] for x in deltas_bw_1], s = 1, c = '#990000')
ax2b.axline((0, 0), slope = 1, c = 'black')
ax2b.set_xlabel('Baseline Score')
ax2b.set_ylabel('Cosine Score')
ax2b.set_title('Both Wrong Answer 1')
ax2c.scatter([x[0] for x in deltas_br_0], [x[1] for x in deltas_br_0], s = 1, c = '#990000')
ax2c.axline((0, 0), slope = 1, c = 'black')
ax2c.set_xlabel('Baseline Score')
ax2c.set_ylabel('Cosine Score')
ax2c.set_title('Both Right Answer 0')
ax2d.scatter([x[0] for x in deltas_br_1], [x[1] for x in deltas_br_1], s = 1, c = '#990000')
ax2d.axline((0, 0), slope = 1, c = 'black')
ax2d.set_xlabel('Baseline Score')
ax2d.set_ylabel('Cosine Score')
ax2d.set_title('Both Right Answer 1')

fig7, ax7 = plt.subplots()
fig7.suptitle('Baseline vs Cosine')
ax7.scatter([x[0] for x in cos_all], [x[1] for x in cos_all], s = 1, c = '#990000')
ax7.axline((0, 0), slope = 1, c = 'black')
ax7.set_xlabel('Baseline Score')
ax7.set_ylabel('Cosine Score')

