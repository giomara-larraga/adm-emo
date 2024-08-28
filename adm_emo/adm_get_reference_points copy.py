import numpy as np
import pandas as pd

import baseADM
from baseADM import *
import generatePreference as gp

from VehicleCrashWorthiness import vehicle_crashworthiness
from RiverPollutionProblem import river_pollution_problem
from CarSideImpact import car_side_impact

from desdeo_emo.utilities.ReferenceVectors import ReferenceVectors

from desdeo_emo.EAs.RVEA import RVEA
from desdeo_emo.EAs.NSGAIII import NSGAIII
import rmetric as rm
from sklearn.preprocessing import Normalizer

problems = [vehicle_crashworthiness(), car_side_impact(), river_pollution_problem()]
n_objs = np.asarray([3, 4, 5])  # number of objectives
names = ["VCW", "CSI", "RPP"]

num_gen_per_iter = [100]  # number of generations per iteration

algorithms = ["iRVEA", "iNSGAIII"]  # algorithms to be compared

# the followings are for formatting results
column_names = (
    ["problem", "num_obj", "iteration", "num_gens", "reference_point"]
    + [algorithm + "_R_IGD" for algorithm in algorithms]
    + [algorithm + "_R_HV" for algorithm in algorithms]
    + [algorithm + "_N_Ss" for algorithm in algorithms]
    + [algorithm + "_FEs" for algorithm in algorithms]
)
excess_columns = ["_R_IGD", "_R_HV"]
data = pd.DataFrame(columns=column_names)
data_row = pd.DataFrame(columns=column_names, index=[1])

# ADM parameters
L = 4  # number of iterations for the learning phase
D = 3  # number of iterations for the decision phase
lattice_resolution = 5  # density variable for creating reference vectors

total_run = 1
for run in range(total_run):
    counter = 1
    for gen in num_gen_per_iter:
        for i in range(len(problems)):
            problem = problems[i]
            problem_name = names[i]
            n_obj = n_objs[i]

            
            int_rvea = RVEA(problem=problem, interact=True, n_gen_per_iter=gen)
            int_nsga = NSGAIII(problem=problem, interact=True, n_gen_per_iter=gen)



            int_rvea.set_interaction_type("Reference point")
            int_nsga.set_interaction_type("Reference point")

            #pbea.set_interaction_type("Reference point")

            pref_int_rvea, plot_rvea = int_rvea.start()
            pref_int_nsga, plot_nsga = int_nsga.start()


            values = np.random.uniform(problem.ideal, problem.nadir, (1, problem.ideal.shape[0]))

            response = values[0]
            print("initial_reference_point",response)

            pref_int_rvea.response = pd.DataFrame([response], columns=problem.objective_names)
            #pref_rvea.response = pd.DataFrame([response], columns=pref_rvea.content['dimensions_data'].columns)
            pref_int_nsga.response = pd.DataFrame([response], columns=problem.objective_names)



            pref_int_rvea, _ = int_rvea.iterate(pref_int_rvea)
            pref_int_nsga, _ = int_nsga.iterate(pref_int_nsga)


            # build initial composite front
            (
                rvea_n_solutions,
                nsga_n_solutions,
                cf,
            ) = generate_composite_front_with_identity(
                int_rvea.population.objectives, int_nsga.population.objectives
            )

            # creates uniformly distributed reference vectors
            reference_vectors = ReferenceVectors(lattice_resolution=lattice_resolution, number_of_objectives=n_obj)

           

            # learning phase
            for i in range(L):
                data_row[["problem", "num_obj", "iteration", "num_gens"]] = [
                    problem_name,
                    n_obj,
                    i + 1,
                    gen,
                ]
                # After this class call, solutions inside the composite front are assigned to reference vectors
                base = baseADM(cf, reference_vectors)
                # generates the next reference point for the next iteration in the learning phase
                response = gp.generateRP4learning(base)

                data_row["reference_point"] = [response]

                # run algorithms with the new reference point
                pref_int_rvea.response = pd.DataFrame([response], columns=problem.objective_names)
                pref_int_nsga.response = pd.DataFrame([response], columns=problem.objective_names)

                previous_RVEA_FEs = int_rvea._function_evaluation_count
                previous_NSGA_FEs = int_nsga._function_evaluation_count

                pref_int_rvea, _ = int_rvea.iterate(pref_int_rvea)
                pref_int_nsga, _ = int_nsga.iterate(pref_int_nsga)

                peritr_RVEA_FEs = (
                    int_rvea._function_evaluation_count - previous_RVEA_FEs
                )
                peritr_NSGA_FEs = (
                    int_nsga._function_evaluation_count - previous_NSGA_FEs
                )

                # extend composite front with newly obtained solutions
                (
                    rvea_n_solutions,
                    nsga_n_solutions,
                    cf,
                ) = generate_composite_front_with_identity(
                    int_rvea.population.objectives,
                    int_nsga.population.objectives,
                    cf,
                )

                data_row["iRVEA_N_Ss"] = [rvea_n_solutions]
                data_row["iNSGAIII_N_Ss"] = [nsga_n_solutions]
                data_row["iRVEA_FEs"] = [peritr_RVEA_FEs * n_obj]
                data_row["iNSGAIII_FEs"] = [peritr_NSGA_FEs * n_obj]

                # R-metric calculation
                ref_point = response.reshape(1, n_obj)

                # normalize reference point
                rp_transformer = Normalizer().fit(ref_point)
                norm_rp = rp_transformer.transform(ref_point)

                rmetric = rm.RMetric(problem, norm_rp)

                # normalize solutions before sending r-metric
                rvea_transformer = Normalizer().fit(int_rvea.population.objectives)
                norm_rvea = rvea_transformer.transform(
                    int_rvea.population.objectives
                )

                nsga_transformer = Normalizer().fit(int_nsga.population.objectives)
                norm_nsga = nsga_transformer.transform(
                    int_nsga.population.objectives
                )

                # R-metric calls for R_IGD and R_HV
                rigd_irvea, rhv_irvea = rmetric.calc(norm_rvea, others=norm_nsga)
                rigd_insga, rhv_insga = rmetric.calc(norm_nsga, others=norm_rvea)

                data_row[
                    ["iRVEA" + excess_col for excess_col in excess_columns]
                ] = [rigd_irvea, rhv_irvea]
                data_row[
                    ["iNSGAIII" + excess_col for excess_col in excess_columns]
                ] = [rigd_insga, rhv_insga]

                data = data.append(data_row, ignore_index=1)

            # Decision phase
            # After the learning phase the reference vector which has the maximum number of assigned solutions forms ROI
            max_assigned_vector = gp.get_max_assigned_vector(base.assigned_vectors)

            for i in range(D):
                data_row[["problem", "num_obj", "iteration", "num_gens"]] = [
                    problem_name,
                    n_obj,
                    L + i + 1,
                    gen,
                ]

                # since composite front grows after each iteration this call should be done for each iteration
                base = baseADM(cf, reference_vectors)

                # generates the next reference point for the decision phase
                response = gp.generatePerturbatedRP4decision(
                    base, max_assigned_vector[0]
                )

                data_row["reference_point"] = [response]

                # run algorithms with the new reference point
                pref_int_rvea.response = pd.DataFrame([response], columns=problem.objective_names)
                pref_int_nsga.response = pd.DataFrame([response], columns=problem.objective_names)

                previous_RVEA_FEs = int_rvea._function_evaluation_count
                previous_NSGA_FEs = int_nsga._function_evaluation_count

                pref_int_rvea, _ = int_rvea.iterate(pref_int_rvea)
                pref_int_nsga, _ = int_nsga.iterate(pref_int_nsga)

                peritr_RVEA_FEs = (
                    int_rvea._function_evaluation_count - previous_RVEA_FEs
                )
                peritr_NSGA_FEs = (
                    int_nsga._function_evaluation_count - previous_NSGA_FEs
                )
                # extend composite front with newly obtained solutions
                (
                    rvea_n_solutions,
                    nsga_n_solutions,
                    cf,
                ) = generate_composite_front_with_identity(
                    int_rvea.population.objectives,
                    int_nsga.population.objectives,
                    cf,
                )

                data_row["iRVEA_N_Ss"] = [rvea_n_solutions]
                data_row["iNSGAIII_N_Ss"] = [nsga_n_solutions]
                data_row["iRVEA_FEs"] = [peritr_RVEA_FEs * n_obj]
                data_row["iNSGAIII_FEs"] = [peritr_NSGA_FEs * n_obj]

                # R-metric calculation
                ref_point = response.reshape(1, n_obj)

                rp_transformer = Normalizer().fit(ref_point)
                norm_rp = rp_transformer.transform(ref_point)

                # for decision phase, delta is specified as 0.2
                rmetric = rm.RMetric(problem, norm_rp, delta=0.2)

                # normalize solutions before sending r-metric
                rvea_transformer = Normalizer().fit(int_rvea.population.objectives)
                norm_rvea = rvea_transformer.transform(
                    int_rvea.population.objectives
                )

                nsga_transformer = Normalizer().fit(int_nsga.population.objectives)
                norm_nsga = nsga_transformer.transform(
                    int_nsga.population.objectives
                )

                rigd_irvea, rhv_irvea = rmetric.calc(norm_rvea, others=norm_nsga)
                rigd_insga, rhv_insga = rmetric.calc(norm_nsga, others=norm_rvea)

                data_row[
                    ["iRVEA" + excess_col for excess_col in excess_columns]
                ] = [rigd_irvea, rhv_irvea]
                data_row[
                    ["iNSGAIII" + excess_col for excess_col in excess_columns]
                ] = [rigd_insga, rhv_insga]

                data = data.append(data_row, ignore_index=1)

    data.to_csv(f"output{run+1}.csv", index=False)