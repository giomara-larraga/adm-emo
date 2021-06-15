import pandas as pd
import numpy as np

data_in = pd.read_csv("./out_cumulatives.csv")

problem_names = ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4"]
n_objs = np.asarray([3, 4, 5, 6, 7, 8, 9])

# the followings are for formatting results
column_names = [
    "problem",
    "num_obj",
    "phase",
    "iRVEA_R_IGD_mean",
    "iRVEA_R_IGD_std",
    "iNSGAIII_R_IGD_mean",
    "iNSGAIII_R_IGD_std",
    "iRVEA_R_HV_mean",
    "iRVEA_R_HV_std",
    "iNSGAIII_R_HV_mean",
    "iNSGAIII_R_HV_std",
    "iRVEA_N_Ss_mean",
    "iRVEA_N_Ss_std",
    "iNSGAIII_N_Ss_mean",
    "iNSGAIII_N_Ss_std",
    "iRVEA_FEs_mean",
    "iRVEA_FEs_std",
    "iNSGAIII_FEs_mean",
    "iNSGAIII_FEs_std",
]

data_out = pd.DataFrame(columns=column_names)
data_row = pd.DataFrame(columns=column_names, index=[1])

x = data_in.values

for problem_name in problem_names:
    for n_obj in n_objs:
        data_row[
            [
                "problem",
                "num_obj",
                "phase",
                "iRVEA_R_IGD_mean",
                "iRVEA_R_IGD_std",
                "iNSGAIII_R_IGD_mean",
                "iNSGAIII_R_IGD_std",
                "iRVEA_R_HV_mean",
                "iRVEA_R_HV_std",
                "iNSGAIII_R_HV_mean",
                "iNSGAIII_R_HV_std",
                "iRVEA_N_Ss_mean",
                "iRVEA_N_Ss_std",
                "iNSGAIII_N_Ss_mean",
                "iNSGAIII_N_Ss_std",
                "iRVEA_FEs_mean",
                "iRVEA_FEs_std",
                "iNSGAIII_FEs_mean",
                "iNSGAIII_FEs_std",
            ]
        ] = [
            problem_name,
            n_obj,
            "learning",
            data_in[data_in["problem"] == problem_name][data_in["num_obj"] == n_obj][
                data_in["phase"] == "learning"
            ]["iRVEA_R_IGD"].values.mean(),
            data_in[data_in["problem"] == problem_name][data_in["num_obj"] == n_obj][
                data_in["phase"] == "learning"
            ]["iRVEA_R_IGD"].values.std(),
            data_in[data_in["problem"] == problem_name][data_in["num_obj"] == n_obj][
                data_in["phase"] == "learning"
            ]["iNSGAIII_R_IGD"].values.mean(),
            data_in[data_in["problem"] == problem_name][data_in["num_obj"] == n_obj][
                data_in["phase"] == "learning"
            ]["iNSGAIII_R_IGD"].values.std(),
            data_in[data_in["problem"] == problem_name][data_in["num_obj"] == n_obj][
                data_in["phase"] == "learning"
            ]["iRVEA_R_HV"].values.mean(),
            data_in[data_in["problem"] == problem_name][data_in["num_obj"] == n_obj][
                data_in["phase"] == "learning"
            ]["iRVEA_R_HV"].values.std(),
            data_in[data_in["problem"] == problem_name][data_in["num_obj"] == n_obj][
                data_in["phase"] == "learning"
            ]["iNSGAIII_R_HV"].values.mean(),
            data_in[data_in["problem"] == problem_name][data_in["num_obj"] == n_obj][
                data_in["phase"] == "learning"
            ]["iNSGAIII_R_HV"].values.std(),
            data_in[data_in["problem"] == problem_name][data_in["num_obj"] == n_obj][
                data_in["phase"] == "learning"
            ]["iRVEA_N_Ss"].values.mean(),
            data_in[data_in["problem"] == problem_name][data_in["num_obj"] == n_obj][
                data_in["phase"] == "learning"
            ]["iRVEA_N_Ss"].values.std(),
            data_in[data_in["problem"] == problem_name][data_in["num_obj"] == n_obj][
                data_in["phase"] == "learning"
            ]["iNSGAIII_N_Ss"].values.mean(),
            data_in[data_in["problem"] == problem_name][data_in["num_obj"] == n_obj][
                data_in["phase"] == "learning"
            ]["iNSGAIII_N_Ss"].values.std(),
            data_in[data_in["problem"] == problem_name][data_in["num_obj"] == n_obj][
                data_in["phase"] == "learning"
            ]["iRVEA_FEs"].values.mean(),
            data_in[data_in["problem"] == problem_name][data_in["num_obj"] == n_obj][
                data_in["phase"] == "learning"
            ]["iRVEA_FEs"].values.std(),
            data_in[data_in["problem"] == problem_name][data_in["num_obj"] == n_obj][
                data_in["phase"] == "learning"
            ]["iNSGAIII_FEs"].values.mean(),
            data_in[data_in["problem"] == problem_name][data_in["num_obj"] == n_obj][
                data_in["phase"] == "learning"
            ]["iNSGAIII_FEs"].values.std(),
        ]
        data_out = data_out.append(data_row, ignore_index=1)

        data_row[
            [
                "problem",
                "num_obj",
                "phase",
                "iRVEA_R_IGD_mean",
                "iRVEA_R_IGD_std",
                "iNSGAIII_R_IGD_mean",
                "iNSGAIII_R_IGD_std",
                "iRVEA_R_HV_mean",
                "iRVEA_R_HV_std",
                "iNSGAIII_R_HV_mean",
                "iNSGAIII_R_HV_std",
                "iRVEA_N_Ss_mean",
                "iRVEA_N_Ss_std",
                "iNSGAIII_N_Ss_mean",
                "iNSGAIII_N_Ss_std",
                "iRVEA_FEs_mean",
                "iRVEA_FEs_std",
                "iNSGAIII_FEs_mean",
                "iNSGAIII_FEs_std",
            ]
        ] = [
            problem_name,
            n_obj,
            "decision",
            data_in[data_in["problem"] == problem_name][data_in["num_obj"] == n_obj][
                data_in["phase"] == "decision"
            ]["iRVEA_R_IGD"].values.mean(),
            data_in[data_in["problem"] == problem_name][data_in["num_obj"] == n_obj][
                data_in["phase"] == "decision"
            ]["iRVEA_R_IGD"].values.std(),
            data_in[data_in["problem"] == problem_name][data_in["num_obj"] == n_obj][
                data_in["phase"] == "decision"
            ]["iNSGAIII_R_IGD"].values.mean(),
            data_in[data_in["problem"] == problem_name][data_in["num_obj"] == n_obj][
                data_in["phase"] == "decision"
            ]["iNSGAIII_R_IGD"].values.std(),
            data_in[data_in["problem"] == problem_name][data_in["num_obj"] == n_obj][
                data_in["phase"] == "decision"
            ]["iRVEA_R_HV"].values.mean(),
            data_in[data_in["problem"] == problem_name][data_in["num_obj"] == n_obj][
                data_in["phase"] == "decision"
            ]["iRVEA_R_HV"].values.std(),
            data_in[data_in["problem"] == problem_name][data_in["num_obj"] == n_obj][
                data_in["phase"] == "decision"
            ]["iNSGAIII_R_HV"].values.mean(),
            data_in[data_in["problem"] == problem_name][data_in["num_obj"] == n_obj][
                data_in["phase"] == "decision"
            ]["iNSGAIII_R_HV"].values.std(),
            data_in[data_in["problem"] == problem_name][data_in["num_obj"] == n_obj][
                data_in["phase"] == "decision"
            ]["iRVEA_N_Ss"].values.mean(),
            data_in[data_in["problem"] == problem_name][data_in["num_obj"] == n_obj][
                data_in["phase"] == "decision"
            ]["iRVEA_N_Ss"].values.std(),
            data_in[data_in["problem"] == problem_name][data_in["num_obj"] == n_obj][
                data_in["phase"] == "decision"
            ]["iNSGAIII_N_Ss"].values.mean(),
            data_in[data_in["problem"] == problem_name][data_in["num_obj"] == n_obj][
                data_in["phase"] == "decision"
            ]["iNSGAIII_N_Ss"].values.std(),
            data_in[data_in["problem"] == problem_name][data_in["num_obj"] == n_obj][
                data_in["phase"] == "decision"
            ]["iRVEA_FEs"].values.mean(),
            data_in[data_in["problem"] == problem_name][data_in["num_obj"] == n_obj][
                data_in["phase"] == "decision"
            ]["iRVEA_FEs"].values.std(),
            data_in[data_in["problem"] == problem_name][data_in["num_obj"] == n_obj][
                data_in["phase"] == "decision"
            ]["iNSGAIII_FEs"].values.mean(),
            data_in[data_in["problem"] == problem_name][data_in["num_obj"] == n_obj][
                data_in["phase"] == "decision"
            ]["iNSGAIII_FEs"].values.std(),
        ]
        data_out = data_out.append(data_row, ignore_index=1)

data_out.to_csv("./out_mean_std.csv", index=False)
