import csv
import pickle
import math

duplicates = {}


# I stole this from https://stackoverflow.com/questions/45741850/python-hellinger-formula-explanation
def hellinger(p, q):
    list_of_squares = []
    for p_i, q_i in zip(p, q):
        # calculate the square of the difference of ith distribution elements
        s = (math.sqrt(p_i) - math.sqrt(q_i)) ** 2

        # append
        list_of_squares.append(s)

    # calculate sum of squares
    sosq = sum(list_of_squares)

    return sosq / math.sqrt(2)


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

for pair in data_files:
    gold, test = pair.split('/')[-1].split('.mp3.')[:2]
    with open('test_filenames.txt', 'w') as out_file:
        print(f'{gold}.mp3\t{test}.mp3')
