import yaml
import polars as pl

file_micro = "configs/Microarray_config.yml"
file_lipid = "configs/Lipidomics_config.yml"
file_sim = "configs/Simulation_config.yml"

dict_config = {}


def get_config(file):
    """ """

    def del_both(config, key, a="pretraining", b="retraining"):

        if key in config[a].keys():
            del config[a][key]
        if key in config[b].keys():
            del config[b][key]

        return config

    inc_keys = ("pretraining", "retraining")

    with open(file, "r") as config:
        config = yaml.safe_load(config)

    for key_ in tuple(config.keys()):
        if key_ not in inc_keys:
            del config[key_]
            continue

    config["retraining"] = {
        x: (y[0] if isinstance(y, list) else y) for x, y in config["retraining"].items()
    }
    for key_ in tuple(config.keys()):
        for sub_key_ in tuple(config[key_].keys()):
            if isinstance(config[key_][sub_key_], list):
                config[key_][sub_key_] = ", ".join(
                    [str(x) for x in config[key_][sub_key_]]
                )

    config = del_both(config, "alpha")
    config = del_both(config, "data_prep")
    config = del_both(config, "path_prePars")
    config = del_both(config, "n_features")

    for key_ in tuple(config.keys()):
        for subkey_ in config[key_].keys():
            try:
                config[key_][subkey_] = float(config[key_][subkey_])
            except:
                ...
            else:
                continue

            try:
                config[key_][subkey_] = int(config[key_][subkey_])
            except:
                ...
            else:
                continue

            try:
                config[key_][subkey_] = str(config[key_][subkey_])
            except:
                ...
    return config


dict_config["micro"] = get_config(file_micro)
dict_config["lipid"] = get_config(file_lipid)
dict_config["sim"] = get_config(file_sim)


def get_df(dict_config, exp, a="pretraining", b="retraining"):
    dat_pre = pl.DataFrame(
        dict_config[exp][a],
        infer_schema_length=100,
    ).select(pl.Series("id", [f"{exp}_pretraining"]), pl.col("*"))
    dat_re = pl.DataFrame(
        dict_config[exp][b],
        infer_schema_length=100,
    ).select(pl.Series("id", [f"{exp}_retraining"]), pl.col("*"))
    dat = pl.concat([dat_pre, dat_re], how="align")

    return dat


dat_micro = get_df(dict_config, "micro")
dat_lipid = get_df(dict_config, "lipid")
dat_sim = get_df(dict_config, "sim")

dat = pl.concat([dat_micro, dat_lipid, dat_sim], how="align")

dat.write_csv("hyp_table.csv")
