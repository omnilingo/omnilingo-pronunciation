import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# hyperparams = [(5e-6, 10), (1e-6, 10), (5e-5, 10), (1e-5, 10), (5e-4, 10), (1e-4, 10),
#                (5e-6, 50), (1e-6, 50), (5e-5, 50), (1e-5, 50), (5e-4, 50), (1e-4, 50)]
hyperparams = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4]
# hyperparams = [1e-4]

max_score_len = 152 # retrieved from dists.txt

# this way I get the same sample every time (and don't have to rerun the sampling
test_file = open("diff_test_sample.pickle", 'rb')
test_scores = pickle.load(test_file)
test_file.close()
cos_file = open("diff_cos_sample.pickle", 'rb')
cos_scores = pickle.load(cos_file)
cos_file.close()
jsd_file = open("diff_jsd_sample.pickle", 'rb')
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
        dev_data = dataset[round(len(dataset)*.8):round(len(dataset)*.9)]
        test_data = dataset[round(len(dataset)*.9):]

        mlp = MLP().to("cpu")
        # Define the loss function and optimizer
        loss_function = nn.L1Loss()
        optimizer = torch.optim.Adam(mlp.parameters(), lr = lr)

        trainloader = torch.utils.data.DataLoader(train_data, batch_size = 5, shuffle = True, num_workers = 1)
        devloader = torch.utils.data.DataLoader(dev_data)
        testloader = torch.utils.data.DataLoader(test_data)

        dev_size = len(devloader.dataset)
        dev_batches = len(devloader)
        dev_loss, dev_correct = 0, 0

        # Run the training loop
        for epoch in range(0, 50):

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

            # Perform dev testing
            with torch.no_grad():
                for X, y in devloader:
                    pred = mlp(X)
                    dev_loss += loss_function(pred, y).item()
                    dev_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            dev_loss /= dev_batches
            dev_correct /= dev_size
            print(f"Dev Error: \n Accuracy: {(100 * dev_correct):>0.1f}%, Avg loss: {dev_loss:>8f} \n")

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

