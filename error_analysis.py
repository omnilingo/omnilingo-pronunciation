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
jsd_right_1 = []
jsd_right_0 = []
jsd_wrong_1 = []
jsd_wrong_0 = []
jsd_both_right_1 = []
jsd_both_right_0 = []
jsd_both_wrong_1 = []
jsd_both_wrong_0 = []
xen_right_1 = []
xen_right_0 = []
xen_wrong_1 = []
xen_wrong_0 = []
xen_both_right_1 = []
xen_both_right_0 = []
xen_both_wrong_1 = []
xen_both_wrong_0 = []
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

    # test and jsd are different
    if predictions[0][1][x] != predictions[2][1][x]:
        if predictions[2][1][x] == predictions[2][2][x]: # jsd got it right
            if predictions[2][2][x] == 1: # correct answer is 1
                jsd_right_1.append(x)
            else: # correct answer is 0
                jsd_right_0.append(x)
        else: # jsd got it wrong
            if predictions[2][2][x] == 1: # correct answer is 1
                jsd_wrong_1.append(x)
            else: # correct answer is 0
                jsd_wrong_0.append(x)
    else: # test and jsd are the same
        if predictions[2][1][x] == predictions[2][2][x]: # both got it right
            if predictions[2][2][x] == 1: # correct answer is 1
                jsd_both_right_1.append(x)
            else: # correct answer is 0
                jsd_both_right_0.append(x)
        else: # both got it wrong
            if predictions[2][2][x] == 1:
                jsd_both_wrong_1.append(x)
            else:
                jsd_both_wrong_0.append(x)
    jsd_all.extend([pair for pair in zip(predictions[0][0][x], predictions[2][0][x])])

    # test and xen are different
    if predictions[0][1][x] != predictions[3][1][x]:
        if predictions[3][1][x] == predictions[3][2][x]:  # xen got it right
            if predictions[3][2][x] == 1:  # correct answer is 1
                xen_right_1.append(x)
            else:  # correct answer is 0
                xen_right_0.append(x)
        else:  # xen got it wrong
            if predictions[3][2][x] == 1:  # correct answer is 1
                xen_wrong_1.append(x)
            else:  # correct answer is 0
                xen_wrong_0.append(x)
    else:  # test and xen are the same
        if predictions[3][1][x] == predictions[3][2][x]:  # both got it right
            if predictions[3][2][x] == 1:  # correct answer is 1
                xen_both_right_1.append(x)
            else:  # correct answer is 0
                xen_both_right_0.append(x)
        else:  # both got it wrong
            if predictions[3][2][x] == 1:
                xen_both_wrong_1.append(x)
            else:
                xen_both_wrong_0.append(x)
    xen_all.extend([pair for pair in zip(predictions[0][0][x], predictions[3][0][x])])

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

# fig3, ax3 = plt.subplots()
# labels = ['Answer 0', 'Answer 1']
# ax3.violinplot([deltas_bw_0, deltas_bw_1, deltas_br_0, deltas_br_1], widths=0.9, showmeans=True, showextrema=False)
# ax3.set_xticks(list(range(1, len(labels)+1)))
# ax3.set_xticklabels(labels)
# plt.show()

################################## JSD ##################################
deltas_jr = []
deltas_jr_0 = []
deltas_jr_1 = []
deltas_jw = []
deltas_jw_0 = []
deltas_jw_1 = []
deltas_jbr = []
deltas_jbw_0 = []
deltas_jbw_1 = []
deltas_jbr_0 = []
deltas_jbr_1 = []
# for x in jsd_right:
#     # negative numbers, jsd is higher score than baseline
#     deltas_jr.extend([pair[0] - pair[1] for pair in zip(predictions[0][0][x], predictions[2][0][x]) if abs(pair[0] - pair[1]) > sensitivity])
# for x in jsd_wrong:
#     # negative numbers, jsd is higher score than baseline
#     deltas_jw.extend([pair[0] - pair[1] for pair in zip(predictions[0][0][x], predictions[2][0][x]) if abs(pair[0] - pair[1]) > sensitivity])
# for x in jsd_both_right:
#     # negative numbers, jsd is higher score than baseline
#     deltas_jbr.extend([pair[0] - pair[1] for pair in zip(predictions[0][0][x], predictions[2][0][x]) if abs(pair[0] - pair[1]) > sensitivity])
for x in jsd_wrong_0:
    # negative numbers, jsd is higher score than baseline
    deltas_jw_0.extend([pair for pair in zip(predictions[0][0][x], predictions[2][0][x])])# if abs(pair[0] - pair[1]) > sensitivity])
for x in jsd_wrong_1:
    # negative numbers, jsd is higher score than baseline
    deltas_jw_1.extend([pair for pair in zip(predictions[0][0][x], predictions[2][0][x])])# if abs(pair[0] - pair[1]) > sensitivity])
for x in jsd_right_0:
    # negative numbers, jsd is higher score than baseline
    deltas_jr_0.extend([pair for pair in zip(predictions[0][0][x], predictions[2][0][x])])# if abs(pair[0] - pair[1]) > sensitivity])
for x in jsd_right_1:
    # negative numbers, jsd is higher score than baseline
    deltas_jr_1.extend([pair for pair in zip(predictions[0][0][x], predictions[2][0][x])])# if abs(pair[0] - pair[1]) > sensitivity])

for x in jsd_both_wrong_0:
    # negative numbers, jsd is higher score than baseline
    deltas_jbw_0.extend([pair for pair in zip(predictions[0][0][x], predictions[2][0][x])])# if abs(pair[0] - pair[1]) > sensitivity])
for x in jsd_both_wrong_1:
    # negative numbers, jsd is higher score than baseline
    deltas_jbw_1.extend([pair for pair in zip(predictions[0][0][x], predictions[2][0][x])])# if abs(pair[0] - pair[1]) > sensitivity])
for x in jsd_both_right_0:
    # negative numbers, jsd is higher score than baseline
    deltas_jbr_0.extend([pair for pair in zip(predictions[0][0][x], predictions[2][0][x])])# if abs(pair[0] - pair[1]) > sensitivity])
for x in jsd_both_right_1:
    # negative numbers, jsd is higher score than baseline
    deltas_jbr_1.extend([pair for pair in zip(predictions[0][0][x], predictions[2][0][x])])# if abs(pair[0] - pair[1]) > sensitivity])

# cr_counter = Counter(deltas_jr)
# print(' '.join([f'{k}: {v}' for k, v in sorted(cr_counter.items())]))
# cw_counter = Counter(deltas_jw)
# print(' '.join([f'{k}: {v}' for k, v in sorted(cw_counter.items())]))
# cbr_counter = Counter(deltas_jbr)
# print(' '.join([f'{k}: {v}' for k, v in sorted(cw_counter.items())]))

# make the graphs
# fig3, ax3 = plt.subplots()
fig3, ((ax3a, ax3b), (ax3c, ax3d)) = plt.subplots(nrows=2, ncols=2)
# labels = ['Base Right\nAnswer 0', 'Base Right\nAnswer 1', 'JSD Right\nAnswer 0', 'JSD Right\nAnswer 1']
# violin3 = ax3.violinplot([deltas_jw_0, deltas_jw_1, deltas_jr_0, deltas_jr_1], widths=0.9, showmeans=True, showextrema=False)
# for b in violin3['bodies']:
#     b.set_color('#990000')
# violin3['cmeans'].set_color('#990000')
# ax3.set_xticks(list(range(1, len(labels)+1)))
# ax3.set_xticklabels(labels)
# ax3.set_ylim(-1, 1)
# ax3.set_ylabel('Baseline Score - JSD Score')
# ax3.set_title('Difference Between Jensen Shannon and Baseline\nwith Different Answers')
fig3.suptitle('Baseline vs JSD with Different Answers')
ax3a.scatter([x[0] for x in deltas_jw_0], [x[1] for x in deltas_jw_0], s = 1, c = '#990000')
ax3a.axline((0, 0), slope = 1, c = 'black')
ax3a.set_xlabel('Baseline Score')
ax3a.set_ylabel('Jensen-Shannon Score')
ax3a.set_title('JSD Wrong Answer 0')
ax3b.scatter([x[0] for x in deltas_jw_1], [x[1] for x in deltas_jw_1], s = 1, c = '#990000')
ax3b.axline((0, 0), slope = 1, c = 'black')
ax3b.set_xlabel('Baseline Score')
ax3b.set_ylabel('Jensen-Shannon Score')
ax3b.set_title('JSD Wrong Answer 1')
ax3c.scatter([x[0] for x in deltas_jr_0], [x[1] for x in deltas_jr_0], s = 1, c = '#990000')
ax3c.axline((0, 0), slope = 1, c = 'black')
ax3c.set_xlabel('Baseline Score')
ax3c.set_ylabel('Jensen-Shannon Score')
ax3c.set_title('JSD Right Answer 0')
ax3d.scatter([x[0] for x in deltas_jr_1], [x[1] for x in deltas_jr_1], s = 1, c = '#990000')
ax3d.axline((0, 0), slope = 1, c = 'black')
ax3d.set_xlabel('Baseline Score')
ax3d.set_ylabel('Jensen-Shannon Score')
ax3d.set_title('JSD Right Answer 1')

# fig4, ax4 = plt.subplots()
fig4, ((ax4a, ax4b), (ax4c, ax4d)) = plt.subplots(nrows=2, ncols=2)
# labels = ['Both Wrong\nAnswer 0', 'Both Wrong\nAnswer 1', 'Both Right\nAnswer 0', 'Both Right\nAnswer 1']
# violin2 = ax4.violinplot([deltas_jbw_0, deltas_jbw_1, deltas_jbr_0, deltas_jbr_1], widths=0.9, showmeans=True, showextrema=False)
# for b in violin2['bodies']:
#     b.set_color('#990000')
# violin2['cmeans'].set_color('#990000')
# ax4.set_xticks(list(range(1, len(labels)+1)))
# ax4.set_xticklabels(labels)
# ax4.set_ylim(-1, 1)
# ax4.set_ylabel('Baseline Score - JSD Score')
# ax4.set_title('Difference Between Jensen Shannon and Baseline\nwith Same Answers')
fig4.suptitle('Baseline vs JSD with Same Answers')
ax4a.scatter([x[0] for x in deltas_jbw_0], [x[1] for x in deltas_jbw_0], s = 1, c = '#990000')
ax4a.axline((0, 0), slope = 1, c = 'black')
ax4a.set_xlabel('Baseline Score')
ax4a.set_ylabel('Jensen-Shannon Score')
ax4a.set_title('Both Wrong Answer 0')
ax4b.scatter([x[0] for x in deltas_jbw_1], [x[1] for x in deltas_jbw_1], s = 1, c = '#990000')
ax4b.axline((0, 0), slope = 1, c = 'black')
ax4b.set_xlabel('Baseline Score')
ax4b.set_ylabel('Jensen-Shannon Score')
ax4b.set_title('Both Wrong Answer 1')
ax4c.scatter([x[0] for x in deltas_jbr_0], [x[1] for x in deltas_jbr_0], s = 1, c = '#990000')
ax4c.axline((0, 0), slope = 1, c = 'black')
ax4c.set_xlabel('Baseline Score')
ax4c.set_ylabel('Jensen-Shannon Score')
ax4c.set_title('Both Right Answer 0')
ax4d.scatter([x[0] for x in deltas_jbr_1], [x[1] for x in deltas_jbr_1], s = 1, c = '#990000')
ax4d.axline((0, 0), slope = 1, c = 'black')
ax4d.set_xlabel('Baseline Score')
ax4d.set_ylabel('Jensen-Shannon Score')
ax4d.set_title('Both Right Answer 1')

fig8, ax8 = plt.subplots()
fig8.suptitle('Baseline vs Jensen-Shannon')
ax8.scatter([x[0] for x in jsd_all], [x[1] for x in jsd_all], s = 1, c = '#990000')
ax8.axline((0, 0), slope = 1, c = 'black')
ax8.set_xlabel('Baseline Score')
ax8.set_ylabel('Jensen-Shannon Score')

################################## XEN ##################################
deltas_xr = []
deltas_xr_0 = []
deltas_xr_1 = []
deltas_xw = []
deltas_xw_0 = []
deltas_xw_1 = []
deltas_xbr = []
deltas_xbw_0 = []
deltas_xbw_1 = []
deltas_xbr_0 = []
deltas_xbr_1 = []

for x in xen_wrong_0:
    # negative numbers, xen is higher score than baseline
    deltas_xw_0.extend([pair for pair in zip(predictions[0][0][x], predictions[3][0][x])])# if abs(pair[0] - pair[1]) > sensitivity])
for x in xen_wrong_1:
    # negative numbers, xen is higher score than baseline
    deltas_xw_1.extend([pair for pair in zip(predictions[0][0][x], predictions[3][0][x])])# if abs(pair[0] - pair[1]) > sensitivity])
for x in xen_right_0:
    # negative numbers, xen is higher score than baseline
    deltas_xr_0.extend([pair for pair in zip(predictions[0][0][x], predictions[3][0][x])])# if abs(pair[0] - pair[1]) > sensitivity])
for x in xen_right_1:
    # negative numbers, xen is higher score than baseline
    deltas_xr_1.extend([pair for pair in zip(predictions[0][0][x], predictions[3][0][x])])# if abs(pair[0] - pair[1]) > sensitivity])

for x in xen_both_wrong_0:
    # negative numbers, xen is higher score than baseline
    deltas_xbw_0.extend([pair for pair in zip(predictions[0][0][x], predictions[3][0][x])])# if abs(pair[0] - pair[1]) > sensitivity])
for x in xen_both_wrong_1:
    # negative numbers, xen is higher score than baseline
    deltas_xbw_1.extend([pair for pair in zip(predictions[0][0][x], predictions[3][0][x])])# if abs(pair[0] - pair[1]) > sensitivity])
for x in xen_both_right_0:
    # negative numbers, xen is higher score than baseline
    deltas_xbr_0.extend([pair for pair in zip(predictions[0][0][x], predictions[3][0][x])])# if abs(pair[0] - pair[1]) > sensitivity])
for x in xen_both_right_1:
    # negative numbers, xen is higher score than baseline
    deltas_xbr_1.extend([pair for pair in zip(predictions[0][0][x], predictions[3][0][x])])# if abs(pair[0] - pair[1]) > sensitivity])

# make the graphs
# fig5, ax5 = plt.subplots()
fig5, ((ax5a, ax5b), (ax5c, ax5d)) = plt.subplots(nrows=2, ncols=2)
# labels = ['Base Right\nAnswer 0', 'Base Right\nAnswer 1', 'Cos Right\nAnswer 0', 'Cos Right\nAnswer 1']
# violin5 = ax5.violinplot([deltas_xw_0, deltas_xw_1, deltas_xr_0, deltas_xr_1], widths=0.9, showmeans=True, showextrema=False)
# for b in violin5['bodies']:
#     b.set_color('#990000')
# violin5['cmeans'].set_color('#990000')
# ax5.set_xticks(list(range(1, len(labels)+1)))
# ax5.set_xticklabels(labels)
# ax5.set_ylabel('Baseline Score - Cross Entropy Score')
# ax5.set_title('Difference Between Cross Entropy and Baseline\nwith Different Answers')
fig5.suptitle('Baseline vs Cross Entropy with Different Answers')
ax5a.scatter([x[0] for x in deltas_xw_0], [(x[1] - 1) * -1 for x in deltas_xw_0], s = 1, c = '#990000')
ax5a.set_xlabel('Baseline Score')
ax5a.set_ylabel('Cross Entropy Score')
ax5a.set_title('Cross Entropy Wrong Answer 0')
ax5b.scatter([x[0] for x in deltas_xw_1], [(x[1] - 1) * -1 for x in deltas_xw_1], s = 1, c = '#990000')
ax5b.set_xlabel('Baseline Score')
ax5b.set_ylabel('Cross Entropy Score')
ax5b.set_title('Cross Entropy Wrong Answer 1')
ax5c.scatter([x[0] for x in deltas_xr_0], [(x[1] - 1) * -1 for x in deltas_xr_0], s = 1, c = '#990000')
ax5c.set_xlabel('Baseline Score')
ax5c.set_ylabel('Cross Entropy Score')
ax5c.set_title('Cross Entropy Right Answer 0')
ax5d.scatter([x[0] for x in deltas_xr_1], [(x[1] - 1) * -1 for x in deltas_xr_1], s = 1, c = '#990000')
ax5d.set_xlabel('Baseline Score')
ax5d.set_ylabel('Cross Entropy Score')
ax5d.set_title('Cross Entropy Right Answer 1')

# fig6, ax6 = plt.subplots()
# labels = ['Both Wrong\nAnswer 0', 'Both Wrong\nAnswer 1', 'Both Right\nAnswer 0', 'Both Right\nAnswer 1']
# violin6 = ax6.violinplot([deltas_xbw_0, deltas_xbw_1, deltas_xbr_0, deltas_xbr_1], widths=0.9, showmeans=True, showextrema=False)
# for b in violin6['bodies']:
#     b.set_color('#990000')
# violin6['cmeans'].set_color('#990000')
# ax6.set_xticks(list(range(1, len(labels)+1)))
# ax6.set_xticklabels(labels)
# ax6.set_ylabel('Baseline Score - Cross Entropy Score')
# ax6.set_title('Difference Between Cross Entropy and Baseline\nwith Same Answers')
fig6, ((ax6a, ax6b), (ax6c, ax6d)) = plt.subplots(nrows=2, ncols=2)
fig6.suptitle('Baseline vs Cross Entropy with Same Answers')
ax6a.scatter([x[0] for x in deltas_xbw_0], [(x[1] - 1) * -1 for x in deltas_xbw_0], s = 1, c = '#990000')
ax6a.set_xlabel('Baseline Score')
ax6a.set_ylabel('Cross Entropy Score')
ax6a.set_title('Both Wrong Answer 0')
ax6b.scatter([x[0] for x in deltas_xbw_1], [(x[1] - 1) * -1 for x in deltas_xbw_1], s = 1, c = '#990000')
ax6b.set_xlabel('Baseline Score')
ax6b.set_ylabel('Cross Entropy Score')
ax6b.set_title('Both Wrong Answer 1')
ax6c.scatter([x[0] for x in deltas_xbr_0], [(x[1] - 1) * -1 for x in deltas_xbr_0], s = 1, c = '#990000')
ax6c.set_xlabel('Baseline Score')
ax6c.set_ylabel('Cross Entropy Score')
ax6c.set_title('Both Right Answer 0')
ax6d.scatter([x[0] for x in deltas_xbr_1], [(x[1] - 1) * -1 for x in deltas_xbr_1], s = 1, c = '#990000')
ax6d.set_xlabel('Baseline Score')
ax6d.set_ylabel('Cross Entropy Score')
ax6d.set_title('Both Right Answer 1')

fig9, ax9 = plt.subplots()
fig9.suptitle('Baseline vs Cross Entropy')
ax9.scatter([x[0] for x in xen_all], [(x[1] - 1) * -1 for x in xen_all], s = 1, c = '#990000')
ax9.set_xlabel('Baseline Score')
ax9.set_ylabel('Cross Entropy Score')

plt.show()
pass
