import pickle
import csv
import matplotlib.pyplot as plt
import random
import re

duplicates = {}
alphabet = [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'", "	"]

#client_id	path	sentence	up_votes	down_votes	age	gender	accents	locale	segment
with open('../STT/data/en/sampled_duplicates.tsv', newline='') as validated:
    reader = csv.reader(validated, delimiter='\t')
    header = next(reader)
    # print('file opened')
    for row in reader:
        # hashmap based on file path
        duplicates[row[1]] = row

p = open('balanced_sample.pickle', 'rb')
data_files = pickle.load(p)
p.close()

gold_scores = []
test_scores = []
cos_scores = []
jsd_scores = []
xen_scores = []
hell_scores = []
transcripts = []
test_js = []
cos_js = []
jsd_js = []
xen_js = []
test_qs = []
cos_qs = []
jsd_qs = []
xen_qs = []
test_ns = []
cos_ns = []
jsd_ns = []
xen_ns = []
test_zs = []
cos_zs = []
jsd_zs = []
xen_zs = []
for pair in data_files:
    gold, test = pair.split('/')[-1].split('.mp3.')[:2]
    with open(pair) as in_file:
        transcript = ''.join([c for c in duplicates[test+'.mp3'][2].lower() if c in alphabet])
        transcripts.append(transcript)

        gold_score = [float(x) for x in in_file.readline().split(',')]
        gold_scores.append(gold_score)

        test_score = [float(x) for x in in_file.readline().split(',')]
        test_scores.append(test_score)

        cos_score = [float(x) for x in in_file.readline().split(',')]
        cos_scores.append(cos_score)

        jsd_score = [float(x) for x in in_file.readline().split(',')]
        jsd_scores.append(jsd_score)

        xen_score = [(float(x) - 1) * -1 for x in in_file.readline().split(',')]
        xen_scores.append(xen_score)

        hell_score = [float(x) for x in in_file.readline().split(',')]
        hell_scores.append(hell_score)

        # for x in [m.start() for m in re.finditer('j', transcript)]:
        #     test_js.append(test_score[x])
        #     cos_js.append(cos_score[x])
        #     jsd_js.append(jsd_score[x])
        #     xen_js.append(xen_score[x])
        # for x in [m.start() for m in re.finditer('q', transcript)]:
        #     test_qs.append(test_score[x])
        #     cos_qs.append(cos_score[x])
        #     jsd_qs.append(jsd_score[x])
        #     xen_qs.append(xen_score[x])
        # for x in [m.start() for m in re.finditer('n', transcript)]:
        #     test_ns.append(test_score[x])
        #     cos_ns.append(cos_score[x])
        #     jsd_ns.append(jsd_score[x])
        #     xen_ns.append(xen_score[x])
        # for x in [m.start() for m in re.finditer('z', transcript)]:
        #     test_zs.append(test_score[x])
        #     cos_zs.append(cos_score[x])
        #     jsd_zs.append(jsd_score[x])
        #     xen_zs.append(xen_score[x])


print('Got this far')
characters = sorted(set(''.join(transcripts)))
print(characters)

gold_char_scores = {k: [] for k in characters}
test_char_scores = {k: [] for k in characters}
hell_char_scores = {k: [] for k in characters}
jsd_char_scores = {k: [] for k in characters}
xen_char_scores = {k: [] for k in characters}
for sentence, gold, test, hell, jsd, xen in zip(transcripts, gold_scores, test_scores, hell_scores, jsd_scores, xen_scores):
    try:
        assert len(sentence) == len(gold) == len(test) == len(hell) == len(jsd) == len(xen)
    except AssertionError:
        print(sentence)
        continue
    for c, g, t, s, j, x in zip(sentence, gold, test, hell, jsd, xen):
        gold_char_scores[c].append(g)
        test_char_scores[c].append(t)
        hell_char_scores[c].append(s)
        jsd_char_scores[c].append(j)
        xen_char_scores[c].append(x)

fig, ax1 = plt.subplots()
print([k for k, v in sorted(test_char_scores.items())][2:])
violin1 = ax1.violinplot([v for k, v in sorted(test_char_scores.items())][2:], widths=0.9, showmeans=True, showextrema=False)
for b in violin1['bodies']:
    b.set_color('#990000')
violin1['cmeans'].set_color('#990000')
ax1.set_xticks(list(range(1, len(characters[2:]) + 1)))
ax1.set_xticklabels(characters[2:])
# ax1.set_ylabel('Baseline Score - Cross Entropy Score')
ax1.set_title('Baseline Scores per Character')

fig2, ax2 = plt.subplots()
print([k for k, v in sorted(hell_char_scores.items())][2:])
violin2 = ax2.violinplot([v for k, v in sorted(hell_char_scores.items())][2:], widths=0.9, showmeans=True, showextrema=False)
for b in violin2['bodies']:
    b.set_color('#990000')
violin2['cmeans'].set_color('#990000')
ax2.set_xticks(list(range(1, len(characters[2:]) + 1)))
ax2.set_xticklabels(characters[2:])
# ax2.set_ylabel('Baseline Score - Cross Entropy Score')
ax2.set_title('Hellinger Scores per Character')
plt.show()

fig3, ax3 = plt.subplots()
print([k for k, v in sorted(hell_char_scores.items())][2:])
violin3 = ax3.violinplot([v for k, v in sorted(jsd_char_scores.items())][2:], widths=0.9, showmeans=True, showextrema=False)
for b in violin3['bodies']:
    b.set_color('#990000')
violin3['cmeans'].set_color('#990000')
ax3.set_xticks(list(range(1, len(characters[2:]) + 1)))
ax3.set_xticklabels(characters[2:])
# ax3.set_ylabel('Baseline Score - Cross Entropy Score')
ax3.set_title('Jensen-Shannon Scores per Character')
plt.show()

fig4, ax4 = plt.subplots()
print([k for k, v in sorted(hell_char_scores.items())][2:])
violin4 = ax4.violinplot([v for k, v in sorted(xen_char_scores.items())][2:], widths=0.9, showmeans=True, showextrema=False)
for b in violin4['bodies']:
    b.set_color('#990000')
violin4['cmeans'].set_color('#990000')
ax4.set_xticks(list(range(1, len(characters[2:]) + 1)))
ax4.set_xticklabels(characters[2:])
# ax4.set_ylabel('Baseline Score - Cross Entropy Score')
ax4.set_title('Cross Entropy Scores per Character')

fig5, ax5 = plt.subplots()
fig5.suptitle('Character Frequencies')
ax5.bar(range(len(characters[2:])), [len(test_char_scores[x]) for x in characters[2:]], color='#990000')
ax5.set_xticks(list(range(0, len(characters[2:]))))
ax5.set_xticklabels(characters[2:])
ax5.set_ylabel('Count')

sample_len = min([len(x) for x in test_char_scores.values()])
print(f'Character Sample Size: {sample_len}')
fig6, ax6 = plt.subplots()
violin6 = ax6.violinplot([random.sample(v, sample_len) for k, v in sorted(test_char_scores.items())][2:], widths=0.9, showmeans=True, showextrema=False)
for b in violin6['bodies']:
    b.set_color('#990000')
violin6['cmeans'].set_color('#990000')
ax6.set_xticks(list(range(1, len(characters[2:]) + 1)))
ax6.set_xticklabels(characters[2:])
# ax1.set_ylabel('Baseline Score - Cross Entropy Score')
ax6.set_title('Baseline Scores Equal Sample Size')

# fig5, ((ax5a, ax5b), (ax5c, ax5d)) = plt.subplots(nrows=2, ncols=2)
# fig5.suptitle('Scatter Plots for J')
# ax5a.scatter(test_js, cos_js, s = 1, c = '#990000')
# ax5a.set_xlabel('Baseline Score')
# ax5a.set_ylabel('Cosine Score')
# ax5a.set_title('Baseline vs Cosine')
# ax5b.scatter(test_js, jsd_js, s = 1, c = '#990000')
# ax5b.set_xlabel('Baseline Score')
# ax5b.set_ylabel('Jensen-Shannon Score')
# ax5b.set_title('Baseline vs Jensen-Shannon')
# ax5c.scatter(test_js, xen_js, s = 1, c = '#990000')
# ax5c.set_xlabel('Baseline Score')
# ax5c.set_ylabel('Cross Entropy Score')
# ax5c.set_title('Baseline vs Cross Entropy')
#
# fig6, ((ax6a, ax6b), (ax6c, ax6d)) = plt.subplots(nrows=2, ncols=2)
# fig6.suptitle('Scatter Plots for Q')
# ax6a.scatter(test_qs, cos_qs, s = 1, c = '#990000')
# ax6a.set_xlabel('Baseline Score')
# ax6a.set_ylabel('Cosine Score')
# ax6a.set_title('Baseline vs Cosine')
# ax6b.scatter(test_qs, jsd_qs, s = 1, c = '#990000')
# ax6b.set_xlabel('Baseline Score')
# ax6b.set_ylabel('Jensen-Shannon Score')
# ax6b.set_title('Baseline vs Jensen-Shannon')
# ax6c.scatter(test_qs, xen_qs, s = 1, c = '#990000')
# ax6c.set_xlabel('Baseline Score')
# ax6c.set_ylabel('Cross Entropy Score')
# ax6c.set_title('Baseline vs Cross Entropy')
#
# fig7, ((ax7a, ax7b), (ax7c, ax7d)) = plt.subplots(nrows=2, ncols=2)
# fig7.suptitle('Scatter Plots for N')
# ax7a.scatter(test_ns, cos_ns, s = 1, c = '#990000')
# ax7a.set_xlabel('Baseline Score')
# ax7a.set_ylabel('Cosine Score')
# ax7a.set_title('Baseline vs Cosine')
# ax7b.scatter(test_ns, jsd_ns, s = 1, c = '#990000')
# ax7b.set_xlabel('Baseline Score')
# ax7b.set_ylabel('Jensen-Shannon Score')
# ax7b.set_title('Baseline vs Jensen-Shannon')
# ax7c.scatter(test_ns, xen_ns, s = 1, c = '#990000')
# ax7c.set_xlabel('Baseline Score')
# ax7c.set_ylabel('Cross Entropy Score')
# ax7c.set_title('Baseline vs Cross Entropy')
#
# fig8, ((ax8a, ax8b), (ax8c, ax8d)) = plt.subplots(nrows=2, ncols=2)
# fig8.suptitle('Scatter Plots for Z')
# ax8a.scatter(test_zs, cos_zs, s = 1, c = '#990000')
# ax8a.set_xlabel('Baseline Score')
# ax8a.set_ylabel('Cosine Score')
# ax8a.set_title('Baseline vs Cosine')
# ax8b.scatter(test_zs, jsd_zs, s = 1, c = '#990000')
# ax8b.set_xlabel('Baseline Score')
# ax8b.set_ylabel('Jensen-Shannon Score')
# ax8b.set_title('Baseline vs Jensen-Shannon')
# ax8c.scatter(test_zs, xen_zs, s = 1, c = '#990000')
# ax8c.set_xlabel('Baseline Score')
# ax8c.set_ylabel('Cross Entropy Score')
# ax8c.set_title('Baseline vs Cross Entropy')

plt.show()
