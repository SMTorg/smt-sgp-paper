#!/usr/bin/env python

# Standard libraries
import time
import numpy as np
from scipy.cluster.vq import kmeans
import warnings
import csv

warnings.filterwarnings("ignore")

# Database API libraries
import sqlite3
import pandas as pd

# Sparse GP regression model
from smt.surrogate_models import SGP

# Set seed
np.random.seed(0)

# defining global names
LABEL_NAMES = ["campaign_id", "NROT", "NLOT", "CONF"]  # names of some labels
INPUT_NAMES = ["M0C", "RE0C", "ALPHAC", "BETA"]  # names of the inputs
OUTPUT_NAMES = ["CXC", "CYC", "CZC", "CLAAC", "CMAAC", "CNAAC"]  # names of the outputs

SPARSE_METHODS = ["FITC", "VFE"]
INDUCING_METHODS = ["RANDOM", "KMEANS", "NORM_KMEANS"]


def load_database():
    # ## II- Loading WT database with SQLite
    #
    # ### A) Load the full raw database

    # path to the database
    database = "./database.db"

    # connecting to the database via SQLite
    sqdatabase = sqlite3.connect(database)
    cursor = sqdatabase.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")

    # creating dataframe for the database
    raw_df = pd.DataFrame(columns=LABEL_NAMES + INPUT_NAMES + OUTPUT_NAMES)

    # extracting data from the database via cursor.execute and cursor.fetchall
    for name in raw_df.columns:
        string = "SELECT " + name + " FROM data"
        cursor.execute(string)
        var_temp = cursor.fetchall()
        raw_df[name] = np.array(var_temp).flatten()

    # closing the connection
    sqdatabase.close()

    # ### B) Extract subdataset as pandas.dataframe
    #
    # We here consider the subdataset **df:** CONF=BWV
    # path to folder where CONF indices files are stored
    path = "./conf.npz"

    # extract indices for CONF=BWV
    with np.load(path, allow_pickle=True) as file:
        bwv_idx = file["idx"].item()["BWV"]

    # size of subdataset
    print("The reduced database (CONF=BWV) contains %i observations" % len(bwv_idx))

    df = raw_df.loc[bwv_idx]
    return df


def sgp_compute(X, Y, output_name, sparse_method, inducing_method, M):
    N = int(0.9 * X.shape[0])
    random_idx = np.random.choice(X.shape[0], N, replace=False)
    X_train = X[random_idx]
    Y_train = Y[random_idx]
    X_test = np.delete(X, random_idx, axis=0)
    Y_test = np.delete(Y, random_idx, axis=0)

    # Initial guess for lengthscale parameter: standard deviation of training data
    l = np.std(X_train, axis=0)
    # Transform to theta parameter (inverse of lengthscale)
    theta0 = 1 / l**2
    # Specify bounds for theta
    theta_bounds = [1e-16, 1.0]

    start_inducing = time.time()
    if inducing_method == "RANDOM":
        shuffle_idx = np.random.permutation(X_train.shape[0])[:M]
        Z = X_train[shuffle_idx].copy()
    elif inducing_method == "KMEANS":
        data = np.hstack((X_train, Y_train))
        Z = kmeans(data, M)[0][:, :-1]
    elif inducing_method == "NORM_KMEANS":
        min_vals = X_train.min(axis=0)
        max_vals = X_train.max(axis=0)
        X_train_normalized = (X_train - min_vals) / (max_vals - min_vals)
        data = np.hstack((X_train_normalized, Y_train))
        Z_normalized = kmeans(data, M)[0][:, :-1]
        Z = Z_normalized * (max_vals - min_vals) + min_vals
    else:
        raise ValueError(
            f"Bad inducing_method. Should be in {INDUCING_METHODS}, got {inducing_method}"
        )
    inducing_elapsed = time.time() - start_inducing

    # Define model
    sm = SGP(
        method=sparse_method,
        theta0=theta0,
        theta_bounds=theta_bounds,
        print_global=False,
        n_start=1,
    )

    # Assign training data and inducing inputs
    sm.set_training_values(X_train, Y_train)
    sm.set_inducing_inputs(Z=Z)

    # Optimize
    start_training = time.time()
    sm.train()
    training_elapsed = time.time() - start_training

    # Training RMSE
    Y_pred_train = sm.predict_values(X_train)
    training_rmse = np.sqrt(np.mean((Y_pred_train.flatten() - Y_train.flatten()) ** 2))

    # Validate
    start_validation = time.time()
    Y_pred = sm.predict_values(X_test)
    validation_elapsed = time.time() - start_validation

    # Validation RMSE
    validation_rmse = np.sqrt(np.mean((Y_pred.flatten() - Y_test.flatten()) ** 2))

    # RMSE validation

    res = {
        "training_rmse": training_rmse,
        "validation_rmse": validation_rmse,
        "inducing_time": inducing_elapsed,
        "training_time": training_elapsed,
        "validation_time": validation_elapsed,
        "optimal_theta": sm.optimal_theta,
        "gp_variance": sm.optimal_sigma2,
        "noise_variance": sm.optimal_noise,
        "reduced_loglikelihood": float(sm.optimal_rlf_value),
        "output_name": output_name,
        "sparse_method": sparse_method,
        "inducing_method": inducing_method,
        "m": M,
    }

    return res


def save_results(results, M):
    if results:
        fields = results[0].keys()
        with open(f"sgp_wtdata_results_M{M}.csv", "w", newline="") as file:
            writer = csv.DictWriter(file, delimiter=",", fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)


if __name__ == "__main__":
    from optparse import OptionParser
    usage = "usage: %prog [options]"
    parser = OptionParser()
    parser.add_option("-M", type="int", dest="M")

    start = time.time()
    print("Loading data...")
    df = load_database()
    X = np.array(df[INPUT_NAMES])
    print("Data loaded in {:.2f}s".format(time.time() - start))

    print("Computing...")
    start = time.time()

    M = 50
    (options, args) = parser.parse_args()
    if options.M:
        M = options.M

    results = []

    for output_name in OUTPUT_NAMES:
        Y = np.array(df[[output_name]])

        for sparse_method in SPARSE_METHODS:
            for inducing_method in INDUCING_METHODS:
                print(
                    f"*** {output_name} - {sparse_method} - {inducing_method} ******************"
                )
                res = sgp_compute(X, Y, output_name, sparse_method, inducing_method, M)
                print(res)
                results.append(res)

                # save intermediate results
                save_results(results, M)

    elapsed = time.time() - start
    print("Computation in {:.2f}s".format(time.time() - start))
    save_results(results, M)
