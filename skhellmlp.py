import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from collections import Counter


hyperparams = [[1e-5, 0.95], [1e-5, 0.90], [1e-5, 0.85], [1e-5, 0.80], [1e-5, 0.75],
               [5e-5, 0.95], [5e-5, 0.90], [5e-5, 0.85], [5e-5, 0.80], [5e-5, 0.75],
               [1e-4, 0.95], [1e-4, 0.90], [1e-4, 0.85], [1e-4, 0.80], [1e-4, 0.75],
               [5e-4, 0.95], [5e-4, 0.90], [5e-4, 0.85], [5e-4, 0.80], [5e-4, 0.75],
               [1e-3, 0.95], [1e-3, 0.90], [1e-3, 0.85], [1e-3, 0.80], [1e-3, 0.75]]
# hyperparams = [[5e-4, 0.80]] # best results (baseline)

max_score_len = 152 # retrieved from dists.txt

# this way I get the same sample every time (and don't have to rerun the sampling
test_file = open("sk_hell_sample.pickle", 'rb')
hell_scores = pickle.load(test_file)
test_file.close()

# Testing identification of different clips vs same clip
# test_file = open("sk_diff_test_sample.pickle", 'rb')
# test_scores = pickle.load(test_file)
# test_file.close()
# cos_file = open("sk_diff_cos_sample.pickle", 'rb')
# cos_scores = pickle.load(cos_file)
# cos_file.close()
# jsd_file = open("sk_diff_jsd_sample.pickle", 'rb')
# jsd_scores = pickle.load(jsd_file)
# jsd_file.close()

all_predictions = []

for lr, beta in hyperparams:
    print(f"Beginning training with lr {lr} and beta {beta}")
    for n, dataset in [('hell_scores', hell_scores)]:
        print(f"Training dataset {n}")
        X_train, X_test, y_train, y_test = train_test_split(dataset[0], dataset[1], random_state = 47404)

        mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), random_state=42, max_iter=1000, learning_rate_init = lr, beta_1 = beta).fit(X_train, y_train)
        # print(mlp.predict(X_test[:10]))
        # print(y_test[:10])
        # print(mlp.score(X_test, y_test))
        print(mlp.loss_curve_)
        predictions = mlp.predict(X_test)
        all_predictions.append([X_test, predictions, y_test])
        print(Counter(predictions))
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for x, y in zip(predictions, y_test):
            if y == 0: # correct answer is 0
                if x == y: # prediction is 0
                    tn += 1
                else: # prediction is 1
                    fp += 1
            elif y == 1: # correct answer is 1
                if x == y: # prediction is 1
                    tp += 1
                else: # prediction is 0
                    fn += 1
        print("Score: {}".format((tp+tn)/(tp+tn+fp+fn)))
        print("p           truth      \nr      |-true-|-false-|\ne true | {} | {}   |\nd false| {}  | {}  |".format(tp, fp, fn, tn))

with open('sk_hell_predictions.binary', 'wb') as out_file:
    pickle.dump(all_predictions, out_file)
