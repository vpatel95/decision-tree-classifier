import os
import sys
import json
import subprocess
from itertools import combinations

FEATURE_LIST = ["Light", "CO2", "Humidity", "HumidityRatio", "Temperature"]
def usage():
    print("python3 {} <{}> <{}>".format(sys.argv[0], "config_file", "number_of_feature_sets"))
    exit(1)

def generate_feature_sets(feature_list):
    all_combinations = []
    for i in range(len(feature_list)):
        n_combinations = combinations(feature_list, i+1)
        for combination in list(n_combinations):
            all_combinations.append(list(combination))

    return all_combinations

def data_preprocessing():
    pwd = os.getcwd()
    os.chdir("data")
    command = "python3 data_preprocessing.py"
    output = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    os.chdir(pwd)

    if (output.returncode != 0):
        return [False, output.stderr]

    return [True, output.stdout]

def update_config(config_file, feature_set, algo, trees):
    with open(config_file, "r+") as f:
        config = json.load(f)
        config["verbosity"] = 1
        config["feature_set"] = [feature_set]
        if (algo == "boosted_rf"):
            config["num_trees"] = trees
            config["boosting"] = True
            config["classification_model"] = "random_forest"
        elif (algo == "rf"):
            config["num_trees"] = trees
            config["boosting"] = False
            config["classification_model"] = "random_forest"
        else:
            config["classification_model"] = "decision_tree"

        f.seek(0)
        json.dump(config, f, indent=4)
        f.truncate()


def get_results(config_file, feature_set, algo, tree):
    update_config(config_file, feature_set, algo, tree)
    command = "./app configs/config.json | grep Accuracy | awk '{ print $3 }'"
    output = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if (output.returncode != 0):
        return [False, output.stderr]

    return [True, output.stdout.decode("utf-8").rstrip()]

def generate_results(config_file, feature_sets):
    if not os.path.exists('results'):
        os.makedirs('results')

    with open("results/result.csv", "w") as results_file:
        results_file.write("Feature Set,Decision Tree,Random Forest,Boosted Random Forest\n")
        results_file.flush()
        for feature_set in feature_sets:
            dt_accuracy = get_results(config_file, feature_set, "dt", 200)
            rf_accuracy = get_results(config_file, feature_set, "rf", 200)
            boosted_rf_accuracy = get_results(config_file, feature_set, "boosted_rf", 200)
            results_file.write("{0},{1},{2},{3}\n".format(str(feature_set), dt_accuracy[1], rf_accuracy[1], boosted_rf_accuracy[1]))
            results_file.flush()

def generate_trees_results(config_file, feature_sets):
    feature_set = ['CO2', 'HumidityRatio']

    if not os.path.exists('results'):
        os.makedirs('results')

    with open("results/result_trees.csv", "w") as results_file:
        results_file.write("Feature Set,Num Trees,Random Forest,Boosted Random Forest\n")
        results_file.flush()
        for num_tree in range(1, 100, 1):
            rf_accuracy = get_results(config_file, feature_set, "rf", num_tree)
            boosted_rf_accuracy = get_results(config_file, feature_set, "boosted_rf", num_tree)
            results_file.write("{0},{1},{2}\n".format(str(num_tree), rf_accuracy[1], boosted_rf_accuracy[1]))
            results_file.flush()

if __name__ == "__main__":

    if len(sys.argv) != 3:
        usage()

    config_file = sys.argv[1]
    num_feature_sets = sys.argv[2]

    data_preprocessing()
    feature_sets = generate_feature_sets(FEATURE_LIST)
    # generate_results(config_file, feature_sets)
    generate_trees_results(config_file, feature_sets)
