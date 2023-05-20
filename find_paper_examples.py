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

lengths = []
for pair in data_files:
    gold, test = pair.split('/')[-1].split('.mp3.')[:2]
    with open(pair) as in_file:
        transcript = ''.join([c for c in duplicates[test+'.mp3'][2].lower() if c in alphabet])
        if len(transcript) > 15:
            continue
        gold_score = [float(x) for x in in_file.readline().split(',')]
        if len([x for x in gold_score if x < 0.8]):
            continue # gold is confidently correct
        test_score = [float(x) for x in in_file.readline().split(',')]
        hell_score = [float(x) for x in in_file.readline().split(',')]
        diffs = [abs(x-y) for x, y in zip(test_score, hell_score) if abs(x-y) < 0.7]
        if len(diffs) > 2:
            lengths.append((str(len(transcript)), pair))
        lengths.sort()

with open('paper_examples.txt', 'w') as out_file:
    print('\n'.join([','.join(x) for x in lengths]), file = out_file)

# print(len(lengths))

# fig, ax0 = plt.subplots()
# ax0.hist(lengths)
# plt.show()