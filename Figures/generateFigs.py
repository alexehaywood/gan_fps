from shutil import copyfile
from os.path import exists
from os import mkdir
import subprocess

experiments = {
        "Lipidomics" : [
            "1a", "1b", "1c",
        ],
        "Lipidomics_lasso" : [
            "1a", "1b", "1c",
        ],
        "Microarray" : [
            "1a", "1b", "1c",
            "2a", "2b", "2c",
            "3a", "3b", "3c",
            "4a", "4b", "4c",
            "5a", "5b", "5c",
            "6a", "6b", "6c",
            "7a", "7b", "7c",
            "8a", "8b", "8c",
            "9a", "9b", "9c",
        ],
        "Microarray_lasso" : [
            "1a", "1b", "1c",
            "2a", "2b", "2c",
            "3a", "3b", "3c",
            "4a", "4b", "4c",
            "5a", "5b", "5c",
            "6a", "6b", "6c",
            "7a", "7b", "7c",
            "8a", "8b", "8c",
            "9a", "9b", "9c",
        ],
        "Simulation_example" : [
            "1a", "1b", "1c",
            "2a", "2b", "2c",
            "3a", "3b", "3c",
            "4a", "4b", "4c",
            "5a", "5b", "5c",
            "6a", "6b", "6c",
            "7a", "7b", "7c",
            "8a", "8b", "8c",
            "9a", "9b", "9c",
        ],
}
graphs = ["svm.png", "histgradboost.png"]

PATH = "/Users/cusworsj/Documents/GAN_Paper/Code/gitrepo"
path_figs = f"{PATH}/Figures"

for g in graphs:
    path_dest = f"{path_figs}"
    if not exists(path_dest):
        mkdir(path_dest)

    for exp, subdirs in experiments.items():
        path_dest = f"{path_figs}/{exp}"
        if not exists(path_dest):
            mkdir(path_dest)
        path_dest = f"{path_figs}/{exp}/{g[:-4]}"
        if not exists(path_dest):
           mkdir(path_dest)

        for sub in subdirs:
            path_graph = f"{PATH}/{exp}/{sub}/Results/{g}"
            copyfile(path_graph, f"{path_dest}/{sub}_{g}")


for g in graphs:
    for exp, subdirs in experiments.items():
        path_dest = f"{path_figs}/{exp}/{g[:-4]}"
        subprocess.run(["image-grid",
                "--folder",
                path_dest,
                "--n",
                str(len(subdirs)),
                "--rows",
                "9",
                "--width",
                "1000",
                "--fill",
                "--out",
                f"{path_dest}/figure_{exp}_{g}",
                "--override"])

subprocess.run(["image-grid",
        "--folder",
        f"{path_figs}/SuppPreTrainingPrep",
        "--n",
        "6",
        "--rows",
        "2",
        "--width",
        "1000",
        "--fill",
        "--out",
        f"{path_figs}/Submit/Supp_Figure_2.png",
        "--override"])

experiment_ = "Simulation_example"
n_suppFig = "3"
subprocess.run(["cp",
                f"{experiment_}/histgradboost/figure_{experiment_}_histgradboost.png",
                f"./Submit/Supp_Figure_{n_suppFig}_1.png"])
subprocess.run(["cp",
                f"{experiment_}/svm/figure_{experiment_}_svm.png",
                f"./Submit/Supp_Figure_{n_suppFig}_2.png"])

experiment_ = "Microarray"
n_suppFig = "4"
subprocess.run(["cp",
                f"{experiment_}/histgradboost/figure_{experiment_}_histgradboost.png",
                f"./Submit/Supp_Figure_{n_suppFig}_1.png"])
subprocess.run(["cp",
                f"{experiment_}/svm/figure_{experiment_}_svm.png",
                f"./Submit/Supp_Figure_{n_suppFig}_2.png"])

experiment_ = "Lipidomics"
n_suppFig = "5"
subprocess.run(["cp",
                f"{experiment_}/histgradboost/figure_{experiment_}_histgradboost.png",
                f"./Submit/Supp_Figure_{n_suppFig}_1.png"])
subprocess.run(["cp",
                f"{experiment_}/svm/figure_{experiment_}_svm.png",
                f"./Submit/Supp_Figure_{n_suppFig}_2.png"])

experiment_ = "Microarray_lasso"
n_suppFig = "6"
subprocess.run(["cp",
                f"{experiment_}/histgradboost/figure_{experiment_}_histgradboost.png",
                f"./Submit/Supp_Figure_{n_suppFig}_1.png"])
subprocess.run(["cp",
                f"{experiment_}/svm/figure_{experiment_}_svm.png",
                f"./Submit/Supp_Figure_{n_suppFig}_2.png"])

experiment_ = "Lipidomics_lasso"
n_suppFig = "7"
subprocess.run(["cp",
                f"{experiment_}/histgradboost/figure_{experiment_}_histgradboost.png",
                f"./Submit/Supp_Figure_{n_suppFig}_1.png"])
subprocess.run(["cp",
                f"{experiment_}/svm/figure_{experiment_}_svm.png",
                f"./Submit/Supp_Figure_{n_suppFig}_2.png"])
