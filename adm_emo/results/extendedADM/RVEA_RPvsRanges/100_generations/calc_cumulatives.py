import pandas as pd
import numpy as np
import glob

all_files = glob.glob("./output*.csv")

problem_names = ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4"]
n_objs = np.asarray([3, 4, 5, 6, 7, 8, 9])
algorithms = ["iRVEA_RP", "iRVEA_Ranges"]  # algorithms to be compared

# the followings are for formatting results
column_names = (
    ["problem", "num_obj", "phase"]
    + [algorithm + "_R_IGD" for algorithm in algorithms]
    + [algorithm + "_R_HV" for algorithm in algorithms]
    + [algorithm + "_N_Ss" for algorithm in algorithms]
    + [algorithm + "_FEs" for algorithm in algorithms]
)
excess_columns = ["_R_IGD", "_R_HV"]
data_out = pd.DataFrame(columns=column_names)
data_row = pd.DataFrame(columns=column_names, index=[1])


iterations_learning = [1, 2, 3, 4]
iterations_decision = [5, 6, 7]
for file_name in all_files:
    data_in = pd.read_csv(file_name)
    for problem_name in problem_names:
        for n_obj in n_objs:
            total_rvea_rigd = 0.0
            total_rvea_rhv = 0.0
            total_rvea_range_rigd = 0.0
            total_rvea_range_rhv = 0.0
            total_rvea_ns = 0
            total_rvea_fes = 0
            total_rvea_range_ns = 0
            total_rvea_range_fes = 0
            for it in iterations_learning:
                rvea_rigd = pd.DataFrame(
                    data_in[data_in["problem"] == problem_name][
                        data_in["num_obj"] == n_obj
                    ][data_in["iteration"] == it]["iRVEA_RP_R_IGD"]
                )
                total_rvea_rigd += rvea_rigd.values[0]

                rvea_rhv = pd.DataFrame(
                    data_in[data_in["problem"] == problem_name][
                        data_in["num_obj"] == n_obj
                    ][data_in["iteration"] == it]["iRVEA_RP_R_HV"]
                )
                total_rvea_rhv += rvea_rhv.values[0]

                rvea_range_rigd = pd.DataFrame(
                    data_in[data_in["problem"] == problem_name][
                        data_in["num_obj"] == n_obj
                    ][data_in["iteration"] == it]["iRVEA_Ranges_R_IGD"]
                )
                total_rvea_range_rigd += rvea_range_rigd.values[0]

                rvea_range_rhv = pd.DataFrame(
                    data_in[data_in["problem"] == problem_name][
                        data_in["num_obj"] == n_obj
                    ][data_in["iteration"] == it]["iRVEA_Ranges_R_HV"]
                )
                total_rvea_range_rhv += rvea_range_rhv.values[0]

                rvea_ns = pd.DataFrame(
                    data_in[data_in["problem"] == problem_name][
                        data_in["num_obj"] == n_obj
                    ][data_in["iteration"] == it]["iRVEA_RP_N_Ss"]
                )
                total_rvea_ns += rvea_ns.values[0]

                rvea_fes = pd.DataFrame(
                    data_in[data_in["problem"] == problem_name][
                        data_in["num_obj"] == n_obj
                    ][data_in["iteration"] == it]["iRVEA_RP_FEs"]
                )
                total_rvea_fes += rvea_fes.values[0]

                rvea_range_ns = pd.DataFrame(
                    data_in[data_in["problem"] == problem_name][
                        data_in["num_obj"] == n_obj
                    ][data_in["iteration"] == it]["iRVEA_Ranges_N_Ss"]
                )
                total_rvea_range_ns += rvea_range_ns.values[0]

                rvea_range_fes = pd.DataFrame(
                    data_in[data_in["problem"] == problem_name][
                        data_in["num_obj"] == n_obj
                    ][data_in["iteration"] == it]["iRVEA_Ranges_FEs"]
                )
                total_rvea_range_fes += rvea_range_fes.values[0]

            data_row[
                [
                    "problem",
                    "num_obj",
                    "phase",
                    "iRVEA_RP_R_IGD",
                    "iRVEA_Ranges_R_IGD",
                    "iRVEA_RP_R_HV",
                    "iRVEA_Ranges_R_HV",
                    "iRVEA_RP_N_Ss",
                    "iRVEA_Ranges_N_Ss",
                    "iRVEA_RP_FEs",
                    "iRVEA_Ranges_FEs",
                ]
            ] = [
                problem_name,
                n_obj,
                "learning",
                total_rvea_rigd[0],
                total_rvea_range_rigd[0],
                total_rvea_rhv[0],
                total_rvea_range_rhv[0],
                total_rvea_ns[0],
                total_rvea_range_ns[0],
                total_rvea_fes[0],
                total_rvea_range_fes[0],
            ]

            data_out = data_out.append(data_row, ignore_index=1)
            total_rvea_rigd = 0.0
            total_rvea_rhv = 0.0
            total_rvea_range_rigd = 0.0
            total_rvea_range_rhv = 0.0
            total_rvea_ns = 0
            total_rvea_fes = 0
            total_rvea_range_ns = 0
            total_rvea_range_fes = 0
            for it in iterations_decision:
                rvea_rigd = pd.DataFrame(
                    data_in[data_in["problem"] == problem_name][
                        data_in["num_obj"] == n_obj
                    ][data_in["iteration"] == it]["iRVEA_RP_R_IGD"]
                )
                total_rvea_rigd += rvea_rigd.values[0]

                rvea_rhv = pd.DataFrame(
                    data_in[data_in["problem"] == problem_name][
                        data_in["num_obj"] == n_obj
                    ][data_in["iteration"] == it]["iRVEA_RP_R_HV"]
                )
                total_rvea_rhv += rvea_rhv.values[0]

                rvea_range_rigd = pd.DataFrame(
                    data_in[data_in["problem"] == problem_name][
                        data_in["num_obj"] == n_obj
                    ][data_in["iteration"] == it]["iRVEA_Ranges_R_IGD"]
                )
                total_rvea_range_rigd += rvea_range_rigd.values[0]

                rvea_range_rhv = pd.DataFrame(
                    data_in[data_in["problem"] == problem_name][
                        data_in["num_obj"] == n_obj
                    ][data_in["iteration"] == it]["iRVEA_Ranges_R_HV"]
                )
                total_rvea_range_rhv += rvea_range_rhv.values[0]

                rvea_ns = pd.DataFrame(
                    data_in[data_in["problem"] == problem_name][
                        data_in["num_obj"] == n_obj
                    ][data_in["iteration"] == it]["iRVEA_RP_N_Ss"]
                )
                total_rvea_ns += rvea_ns.values[0]

                rvea_fes = pd.DataFrame(
                    data_in[data_in["problem"] == problem_name][
                        data_in["num_obj"] == n_obj
                    ][data_in["iteration"] == it]["iRVEA_RP_FEs"]
                )
                total_rvea_fes += rvea_fes.values[0]

                rvea_range_ns = pd.DataFrame(
                    data_in[data_in["problem"] == problem_name][
                        data_in["num_obj"] == n_obj
                    ][data_in["iteration"] == it]["iRVEA_Ranges_N_Ss"]
                )
                total_rvea_range_ns += rvea_range_ns.values[0]

                rvea_range_fes = pd.DataFrame(
                    data_in[data_in["problem"] == problem_name][
                        data_in["num_obj"] == n_obj
                    ][data_in["iteration"] == it]["iRVEA_Ranges_FEs"]
                )
                total_rvea_range_fes += rvea_range_fes.values[0]

            data_row[
                [
                    "problem",
                    "num_obj",
                    "phase",
                    "iRVEA_RP_R_IGD",
                    "iRVEA_Ranges_R_IGD",
                    "iRVEA_RP_R_HV",
                    "iRVEA_Ranges_R_HV",
                    "iRVEA_RP_N_Ss",
                    "iRVEA_Ranges_N_Ss",
                    "iRVEA_RP_FEs",
                    "iRVEA_Ranges_FEs",
                ]
            ] = [
                problem_name,
                n_obj,
                "decision",
                total_rvea_rigd[0],
                total_rvea_range_rigd[0],
                total_rvea_rhv[0],
                total_rvea_range_rhv[0],
                total_rvea_ns[0],
                total_rvea_range_ns[0],
                total_rvea_fes[0],
                total_rvea_range_fes[0],
            ]

            data_out = data_out.append(data_row, ignore_index=1)

data_out.to_csv("./out_cumulatives.csv", index=False)
