import glob
import math
import csv
import random
import pickle
import torch
from collections import Counter

max_score_len = 152 # retrieved from dists.txt

duplicates = {}

#client_id	path	sentence	up_votes	down_votes	age	gender	accents	locale	segment
with open('../STT/data/en/sampled_duplicates.tsv', newline='') as validated:
    reader = csv.reader(validated, delimiter='\t')
    header = next(reader)
    # print('file opened')
    for row in reader:
        # hashmap based on file path
        duplicates[row[1]] = row


results_files = list(glob.glob('../STT/data/en/results_diff/*'))
valid_files = []
valid_downvoted = []
valid_upvoted = []
for f in results_files:
    gold, test = f.split('/')[-1].split('.mp3.')[:2]
    # verify files are in duplicates and skip files where gold has downvotes
    if gold+'.mp3' in duplicates and test+'.mp3' in duplicates and int(duplicates[gold+'.mp3'][4]) == 0:
        valid_files.append(f)
        if int(duplicates[test+'.mp3'][4]) > 0:
            valid_downvoted.append(f)
        else:
            valid_upvoted.append(f)
    else:
        continue # common_voice_en_17270482.mp3 doesn't show up the map
data_files = random.sample(valid_files, 10000)
# data_files = random.sample(valid_downvoted, 5000) + random.sample(valid_upvoted, 5000)

# with open("diff_sample.pickle", 'wb') as p:
#     pickle.dump(data_files, p)


# this way I get the same sample every time (and don't have to rerun the sampling
# p = open('balanced_sample.pickle', 'rb')
# data_files = pickle.load(p)
# p.close()

gold_scores = []
test_scores = []
cos_scores = []
jsd_scores = []
max_score_len = 0
downvotes = Counter()
gold_downvotes = Counter()
gold_scores_counter = Counter()
test_scores_counter = Counter()
cos_scores_counter = Counter()
jsd_scores_counter = Counter()
test_up_scores_counter = Counter()
cos_up_scores_counter = Counter()
jsd_up_scores_counter = Counter()
test_down_scores_counter = Counter()
cos_down_scores_counter = Counter()
jsd_down_scores_counter = Counter()

for pair in data_files:
    gold, test = pair.split('/')[-1].split('.mp3.')[:2]
    with open(pair) as in_file:
        gold_score = [float(x) for x in in_file.readline().split(',')]
        if len(gold_score) > max_score_len: max_score_len = len(gold_score)
        gold_scores.append([gold_score, int(duplicates[gold+'.mp3'][4])])
        gold_downvotes.update([int(duplicates[gold+'.mp3'][4])])
        gold_scores_counter.update(gold_score)

        test_score = [float(x) for x in in_file.readline().split(',')]
        test_scores.append([test_score, int(duplicates[test+'.mp3'][4])])
        downvotes.update([int(duplicates[test+'.mp3'][4])])
        test_scores_counter.update(test_score)
        if int(duplicates[test+'.mp3'][4]) > 0:
            test_down_scores_counter.update(test_score)
        else:
            test_up_scores_counter.update(test_score)

        cos_score = [float(x) for x in in_file.readline().split(',')]
        cos_scores.append([cos_score, int(duplicates[test+'.mp3'][4])])
        cos_scores_counter.update(cos_score)
        if int(duplicates[test+'.mp3'][4]) > 0:
            cos_down_scores_counter.update(cos_score)
        else:
            cos_up_scores_counter.update(cos_score)

        jsd_score = [float(x) for x in in_file.readline().split(',')]
        jsd_scores.append([jsd_score, int(duplicates[test+'.mp3'][4])])
        jsd_scores_counter.update(jsd_score)
        if int(duplicates[test+'.mp3'][4]) > 0:
            jsd_down_scores_counter.update(jsd_score)
        else:
            jsd_up_scores_counter.update(jsd_score)


# saving the distributions and data to a file so I can access them again later
with open("results/diff_dists.txt", 'w') as out_file:
    print("Max length: " + str(max_score_len), file = out_file)
    print("Gold Downvotes: " + str(gold_downvotes), file = out_file)
    print("Test Downvotes: " + str(downvotes), file = out_file)
    print("Gold Scores Dist: " + str(gold_scores_counter), file = out_file)
    print("Test Scores Dist: " + str(test_scores_counter), file = out_file)
    print("Cos Scores Dist: " + str(cos_scores_counter), file = out_file)
    print("JSD Scores Dist: " + str(jsd_scores_counter), file = out_file)
    # print("All Scores Dist: " + str(gold_scores_counter + test_scores_counter +
    #                                 cos_scores_counter + jsd_scores_counter), file = out_file)
    print("Test no downvotes: " + str(test_up_scores_counter), file = out_file)
    print("Test with downvotes: " + str(test_down_scores_counter), file = out_file)
    print("Cos no downvotes: " + str(cos_up_scores_counter), file = out_file)
    print("Cos with downvotes: " + str(cos_down_scores_counter), file = out_file)
    print("JSD no downvotes: " + str(jsd_up_scores_counter), file = out_file)
    print("JSD with downvotes: " + str(jsd_down_scores_counter), file = out_file)


# max_downvotes = max(set(downvotes))


def normalize_list_len(in_list, new_len):
    new_list = []
    for i in in_list:
        # normalize length and put in tensor for input
        i_scores = i[0]
        diff = new_len - len(i_scores)
        input_tensor = torch.tensor([1 for x in range(math.floor(diff / 2))] + i_scores + [1 for x in range(math.ceil(diff / 2))])

        # turn labels into one hot encoding
        # i_labels = torch.tensor([i[1]])
        # i_labels = torch.tensor([0.0 if (x != i[1]) else 1.0 for x in range(max_downvotes)])
        i_labels = torch.tensor([1 if i[1] > 0 else 0])

        new_list.append([input_tensor, i_labels])
    return new_list


# gold_scores = normalize_list_len(gold_scores, max_score_len)
# random.shuffle(gold_scores)
# with open('gold_sample.pickle', 'wb') as out_file:
#     pickle.dump(gold_scores, out_file)
test_scores = normalize_list_len(test_scores, max_score_len)
random.shuffle(test_scores)
with open('diff_test_sample.pickle', 'wb') as out_file:
    pickle.dump(test_scores, out_file)
cos_scores = normalize_list_len(cos_scores, max_score_len)
random.shuffle(cos_scores)
with open('diff_cos_sample.pickle', 'wb') as out_file:
    pickle.dump(cos_scores, out_file)
jsd_scores = normalize_list_len(jsd_scores, max_score_len)
random.shuffle(jsd_scores)
with open('diff_jsd_sample.pickle', 'wb') as out_file:
    pickle.dump(jsd_scores, out_file)
