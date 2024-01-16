#!/usr/bin/env python

# Standard libraries
import time
import numpy as np
import warnings
import csv

from sparse_gp_wtdata import *

warnings.filterwarnings("ignore")


def save_noseed_results(results, M):
    if results:
        fields = results[0].keys()
        with open(
            f"sgp_wtdata_results_noseed_M{M}_10times.csv", "w", newline=""
        ) as file:
            writer = csv.DictWriter(file, delimiter=",", fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)


if __name__ == "__main__":
    from optparse import OptionParser

    usage = "usage: %prog [options]"
    parser = OptionParser()
    parser.add_option("-M", type="int", dest="M")

    # # Set seed
    # np.random.seed(0)

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
                for i in range(10):
                    print(
                        f"*** {output_name} - {sparse_method} - {inducing_method} ******************"
                    )
                    res = sgp_compute(
                        X, Y, output_name, sparse_method, inducing_method, M
                    )
                    print(res)
                    results.append(res)

                    # save intermediate results
                    save_noseed_results(results, M)

    elapsed = time.time() - start
    print("Computation in {:.2f}s".format(time.time() - start))
    save_noseed_results(results, M)
