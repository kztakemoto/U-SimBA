import argparse
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf

def set_seed(seed=200):
    tf.random.set_seed(seed)
    # optional
    # for numpy.random
    np.random.seed(seed)
    # for built-in random
    random.seed(seed)
    # for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

seed = 123
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--nb_samples', type=int)
    args = parser.parse_args()

    # load data
    X_train = np.load(f'./data/{args.dataset}/X_train.npy')
    y_train = np.load(f'./data/{args.dataset}/y_train.npy')
    X_test = np.load(f'./data/{args.dataset}/X_test.npy')
    y_test = np.load(f'./data/{args.dataset}/y_test.npy')

    # load labels
    df = pd.read_csv(f'./data/{args.dataset}/test_pre.csv', header=0, names = ["image_path", "target"])

    # select images randomly
    if args.dataset != "melanoma":
        target_idx = []
        for tl in list(df.target.unique()):
            target_idx.extend(np.array(df[df["target"] == tl].sample(args.nb_samples, random_state=seed).index))
    else:
        set_seed(seed)
        target_idx = np.random.choice(np.arange(len(X_test)), args.nb_samples, replace=False)

    # input data, used to generate a UAP
    X_test_1, y_test_1 = X_test[target_idx], y_test[target_idx]

    # use the rest as the validation data
    rest_idx = np.ones(len(X_test), dtype=bool)
    rest_idx[target_idx] = False
    X_test_2, y_test_2 = X_test[rest_idx], y_test[rest_idx]
    X_test_2 = X_test_2
    y_test_2 = y_test_2

    # save
    np.save(f"./data/{args.dataset}/X_test_1_{args.nb_samples}.npy", X_test_1)
    np.save(f"./data/{args.dataset}/y_test_1_{args.nb_samples}.npy", y_test_1)
    np.save(f"./data/{args.dataset}/X_test_2_{args.nb_samples}.npy", X_test_2)
    np.save(f"./data/{args.dataset}/y_test_2_{args.nb_samples}.npy", y_test_2)
    