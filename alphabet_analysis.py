import pickle
import csv
import matplotlib.pyplot as plt

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
transcripts = []
for pair in data_files[:5000]:
    gold, test = pair.split('/')[-1].split('.mp3.')[:2]
    with open(pair) as in_file:
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

        transcript = ''.join([c for c in duplicates[test+'.mp3'][2].lower() if c in alphabet])
        transcripts.append(transcript)

print('Got this far')
characters = sorted(set(''.join(transcripts)))
print(characters)

gold_char_scores = {k: [] for k in characters}
test_char_scores = {k: [] for k in characters}
cos_char_scores = {k: [] for k in characters}
jsd_char_scores = {k: [] for k in characters}
xen_char_scores = {k: [] for k in characters}
for sentence, gold, test, cos, jsd, xen in zip(transcripts, gold_scores, test_scores, cos_scores, jsd_scores, xen_scores):
    try:
        assert len(sentence) == len(gold) == len(test) == len(cos) == len(jsd) == len(xen)
    except AssertionError:
        print(sentence)
        continue
    for c, g, t, s, j, x in zip(sentence, gold, test, cos, jsd, xen):
        gold_char_scores[c].append(g)
        test_char_scores[c].append(t)
        cos_char_scores[c].append(s)
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
print([k for k, v in sorted(cos_char_scores.items())][2:])
violin2 = ax2.violinplot([v for k, v in sorted(cos_char_scores.items())][2:], widths=0.9, showmeans=True, showextrema=False)
for b in violin2['bodies']:
    b.set_color('#990000')
violin2['cmeans'].set_color('#990000')
ax2.set_xticks(list(range(1, len(characters[2:]) + 1)))
ax2.set_xticklabels(characters[2:])
# ax2.set_ylabel('Baseline Score - Cross Entropy Score')
ax2.set_title('Cosine Scores per Character')
plt.show()

fig3, ax3 = plt.subplots()
print([k for k, v in sorted(cos_char_scores.items())][2:])
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
print([k for k, v in sorted(cos_char_scores.items())][2:])
violin4 = ax4.violinplot([v for k, v in sorted(xen_char_scores.items())][2:], widths=0.9, showmeans=True, showextrema=False)
for b in violin4['bodies']:
    b.set_color('#990000')
violin4['cmeans'].set_color('#990000')
ax4.set_xticks(list(range(1, len(characters[2:]) + 1)))
ax4.set_xticklabels(characters[2:])
# ax4.set_ylabel('Baseline Score - Cross Entropy Score')
ax4.set_title('Cross Entropy Scores per Character')
plt.show()
