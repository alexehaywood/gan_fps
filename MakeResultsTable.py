from csv import reader, writer
from os.path import isfile
from os import remove

experiments = {
        "sim": "Simulation_example",
        "microarray": "Microarray",
        "metabolomics": "Lipidomics"
}
methods = ("GAN", "ClassicGAN", "RO", "SMOTE")
classifiers = ("histgradboost", "svm")
groups = {
    "1a": {"n_control": 40, "class_imbalance": 0.4, "alpha1": 0, "alpha2": 0},
    "1b": {"n_control": 40, "class_imbalance": 0.4, "alpha1": 1, "alpha2": 0},
    "1c": {"n_control": 40, "class_imbalance": 0.4, "alpha1": 1, "alpha2": 1},
    "2a": {"n_control": 80, "class_imbalance": 0.4, "alpha1": 0, "alpha2": 0},
    "2b": {"n_control": 80, "class_imbalance": 0.4, "alpha1": 1, "alpha2": 0},
    "2c": {"n_control": 80, "class_imbalance": 0.4, "alpha1": 1, "alpha2": 1},
    "3a": {"n_control": 120, "class_imbalance": 0.4, "alpha1": 0, "alpha2": 0},
    "3b": {"n_control": 120, "class_imbalance": 0.4, "alpha1": 1, "alpha2": 0},
    "3c": {"n_control": 120, "class_imbalance": 0.4, "alpha1": 1, "alpha2": 1},
    "4a": {"n_control": 40, "class_imbalance": 0.5, "alpha1": 0, "alpha2": 0},
    "4b": {"n_control": 40, "class_imbalance": 0.5, "alpha1": 1, "alpha2": 0},
    "4c": {"n_control": 40, "class_imbalance": 0.5, "alpha1": 1, "alpha2": 1},
    "5a": {"n_control": 80, "class_imbalance": 0.5, "alpha1": 0, "alpha2": 0},
    "5b": {"n_control": 80, "class_imbalance": 0.5, "alpha1": 1, "alpha2": 0},
    "5c": {"n_control": 80, "class_imbalance": 0.5, "alpha1": 1, "alpha2": 1},
    "6a": {"n_control": 120, "class_imbalance": 0.5, "alpha1": 0, "alpha2": 0},
    "6b": {"n_control": 120, "class_imbalance": 0.5, "alpha1": 1, "alpha2": 0},
    "6c": {"n_control": 120, "class_imbalance": 0.5, "alpha1": 1, "alpha2": 1},
    "7a": {"n_control": 40, "class_imbalance": 0.6, "alpha1": 0, "alpha2": 0},
    "7b": {"n_control": 40, "class_imbalance": 0.6, "alpha1": 1, "alpha2": 0},
    "7c": {"n_control": 40, "class_imbalance": 0.6, "alpha1": 1, "alpha2": 1},
    "8a": {"n_control": 80, "class_imbalance": 0.6, "alpha1": 0, "alpha2": 0},
    "8b": {"n_control": 80, "class_imbalance": 0.6, "alpha1": 1, "alpha2": 0},
    "8c": {"n_control": 80, "class_imbalance": 0.6, "alpha1": 1, "alpha2": 1},
    "9a": {"n_control": 120, "class_imbalance": 0.6, "alpha1": 0, "alpha2": 0},
    "9b": {"n_control": 120, "class_imbalance": 0.6, "alpha1": 1, "alpha2": 0},
    "9c": {"n_control": 120, "class_imbalance": 0.6, "alpha1": 1, "alpha2": 1},
}
groups_lipid = {
    "1a": {"n_control": None, "class_imbalance": None, "alpha1": 0, "alpha2": 0},
    "1b": {"n_control": None, "class_imbalance": None, "alpha1": 1, "alpha2": 0},
    "1c": {"n_control": None, "class_imbalance": None, "alpha1": 1, "alpha2": 1},
}

PATH = "/Users/cusworsj/Documents/GAN_Paper/Code/gitrepo"
header = [
        "experiment",
        "classifier",
        "n_control",
        "class_imbalance",
        "alpha1",
        "alpha2",
        "method",
        "auc_mean",
        "mean_accuracy",
        "auc_sd",
        "hyp",
]

csv_name = f"{PATH}/results_merged.csv"

if isfile(csv_name):
    remove(csv_name)

with open(csv_name,
        "w",
        newline='') as csv_file:

    results_merged = writer(csv_file)
    results_merged.writerow(header)

    for exp_label, exp_dir in experiments.items():
        path_ = f"{PATH}/{exp_dir}"

        if exp_label=="metabolomics":
            groups_ = groups_lipid.items()
        else:
            groups_ = groups.items()
        for group_label, group_dat in groups_:
            path_group = f"{path_}/{group_label}/Results"

            for classifier_ in classifiers:
                cols = [
                        exp_label,
                        classifier_,
                        group_dat["n_control"],
                        group_dat["class_imbalance"],
                        group_dat["alpha1"],
                        group_dat["alpha2"],
                ]
                with open(f"{path_group}/{classifier_}.csv",
                        "r",
                        newline="") as f:
                    dat_reader = reader(f)
                    next(dat_reader) #skip header
                    for line in dat_reader:
                        results_merged.writerow(cols + line[1:]) #skip index col
