import glob
import math
import csv
import random
import pickle
import torch
from collections import Counter
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# hyperparams = [(5e-6, 10), (1e-6, 10), (5e-5, 10), (1e-5, 10), (5e-4, 10), (1e-4, 10),
#                (5e-6, 50), (1e-6, 50), (5e-5, 50), (1e-5, 50), (5e-4, 50), (1e-4, 50)]
hyperparams = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4]
max_score_len = 152 # retrieved from dists.txt

"""

duplicates = {}

#client_id	path	sentence	up_votes	down_votes	age	gender	accents	locale	segment
with open('../STT/data/en/sampled_duplicates.tsv', newline='') as validated:
    reader = csv.reader(validated, delimiter='\t')
    header = next(reader)
    # print('file opened')
    for row in reader:
        # hashmap based on file path
        duplicates[row[1]] = row


results_files = list(glob.glob('../STT/data/en/results/*'))
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
data_files = random.sample(valid_files, 20000)
# data_files = random.sample(valid_downvoted, 10000) + random.sample(valid_upvoted, 10000)

with open("sample.pickle", 'wb') as p:
    pickle.dump(data_files, p)


# this way I get the same sample every time (and don't have to rerun the sampling
p = open('balanced_sample.pickle', 'rb')
data_files = pickle.load(p)
p.close()

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

        cos_score = [float(x) for x in in_file.readline().split(',')]
        cos_scores.append([cos_score, int(duplicates[test+'.mp3'][4])])
        cos_scores_counter.update(cos_score)

        jsd_score = [float(x) for x in in_file.readline().split(',')]
        jsd_scores.append([jsd_score, int(duplicates[test+'.mp3'][4])])
        jsd_scores_counter.update(jsd_score)

# saving the distributions and data to a file so I can access them again later
with open("results/balanced_dists.txt", 'w') as out_file:
    print("Max length: " + str(max_score_len), file = out_file)
    print("Gold Downvotes: " + str(gold_downvotes), file = out_file)
    print("Test Downvotes: " + str(downvotes), file = out_file)
    print("Gold Scores Dist: " + str(gold_scores_counter), file = out_file)
    print("Test Scores Dist: " + str(test_scores_counter), file = out_file)
    print("Cos Scores Dist: " + str(cos_scores_counter), file = out_file)
    print("JSD Scores Dist: " + str(jsd_scores_counter), file = out_file)
    print("All Scores Dist: " + str(gold_scores_counter + test_scores_counter +
                                    cos_scores_counter + jsd_scores_counter), file = out_file)

# max_downvotes = max(set(downvotes))


def normalize_list_len(in_list, new_len):
    new_list = []
    for i in in_list:
        # normalize length and put in tensor for input
        i_scores = i[0]
        diff = new_len - len(i_scores)
        input_tensor = torch.tensor([0 for x in range(math.floor(diff / 2))] + i_scores + [0 for x in range(math.ceil(diff / 2))])

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
with open('balanced_test_sample.pickle', 'wb') as out_file:
    pickle.dump(test_scores, out_file)
cos_scores = normalize_list_len(cos_scores, max_score_len)
random.shuffle(cos_scores)
with open('balanced_cos_sample.pickle', 'wb') as out_file:
    pickle.dump(cos_scores, out_file)
jsd_scores = normalize_list_len(jsd_scores, max_score_len)
random.shuffle(jsd_scores)
with open('balanced_jsd_sample.pickle', 'wb') as out_file:
    pickle.dump(jsd_scores, out_file)
"""

# this way I get the same sample every time (and don't have to rerun the sampling
test_file = open("balanced_test_sample.pickle", 'rb')
test_scores = pickle.load(test_file)
test_file.close()
cos_file = open("balanced_cos_sample.pickle", 'rb')
cos_scores = pickle.load(cos_file)
cos_file.close()
jsd_file = open("balanced_jsd_sample.pickle", 'rb')
jsd_scores = pickle.load(jsd_file)
jsd_file.close()


class MLP(nn.Module):
    '''    Multilayer Perceptron.    '''
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(max_score_len, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


for lr in hyperparams:
    print(f"Beginning training with lr {lr}")
    for n, dataset in [('test_scores', test_scores), ('cos_scores', cos_scores), ('jsd_scores', jsd_scores)]:
        print(f"Training dataset {n}")
        train_data = dataset[:round(len(dataset)*.8)]
        test_data = dataset[round(len(dataset)*.8):]

        mlp = MLP().to("cpu")
        # Define the loss function and optimizer
        loss_function = nn.L1Loss()
        optimizer = torch.optim.Adam(mlp.parameters(), lr = lr)

        trainloader = torch.utils.data.DataLoader(train_data, batch_size = 10, shuffle = True, num_workers = 1)
        testloader = torch.utils.data.DataLoader(test_data)

        # Run the training loop
        for epoch in range(0, 25):

            # Print epoch
            print(f'Starting epoch {epoch + 1}')

            # Set current loss value
            current_loss = 0.0

            # Iterate over the DataLoader for training data
            for i, data in enumerate(trainloader):
                size = len(trainloader.dataset)
                # Get inputs
                inputs, labels = data

                # Zero the gradients
                optimizer.zero_grad()

                # Perform forward pass
                outputs = mlp(inputs)

                # Compute loss
                loss = loss_function(outputs, labels)

                # Perform backward pass
                loss.backward()

                # Perform optimization
                optimizer.step()

                # Print statistics
                # if i % 100 == 0:
                #     loss, current = loss.item(), i * len(inputs)
                    # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                # current_loss += loss.item()
                # if i % 500 == 499:
                #     print('Loss after mini-batch %5d: %.3f' %
                #           (i + 1, current_loss / 500))
                #     current_loss = 0.0

        # Process is complete.
        print('Training process has finished.')

        size = len(testloader.dataset)
        num_batches = len(testloader)
        positive, pos_pred = 0, 0
        test_loss, correct, true_positive, recall = 0, 0, 0, 0

        with torch.no_grad():
            for X, y in testloader:
                pred = mlp(X)
                if test_loss == 0:
                    print(pred)
                    print(y)
                    print(pred.argmax(1))
                test_loss += loss_function(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                if pred.argmax(1) > 0:
                    pos_pred += 1
                    true_positive += (pred.argmax(1) == y).type(torch.float).sum().item()
                if y > 0:
                    positive += 1
                    recall += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Raw Recall: {recall}\tRaw True Positive: {true_positive}")
        recall /= positive
        true_positive /= pos_pred
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        print("Precision: " + str(true_positive) + " from " + str(pos_pred))
        print("Recall: " + str(recall) + " from " + str(positive))
        print("F1: " + str(2 * ((true_positive * recall) / (true_positive + recall))))

