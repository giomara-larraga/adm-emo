import numpy as np
import pandas as pd

import baseADM
from baseADM import *
import generatePreference as gp

from desdeo_problem.testproblems.VehicleCrashworthiness import vehicle_crashworthiness
from desdeo_problem.testproblems.RiverPollution import river_pollution_problem
from desdeo_problem.testproblems.MultipleClutchBrakes import multiple_clutch_brakes
from desdeo_problem.testproblems.CarSideImpact import car_side_impact

from desdeo_emo.utilities.ReferenceVectors import ReferenceVectors

from desdeo_emo.EAs.RVEA import RVEA
from desdeo_emo.EAs.NSGAIII import NSGAIII

from pymoo.util.ref_dirs import get_reference_directions
import rmetric as rm
from sklearn.preprocessing import Normalizer
#from pymoo.configuration import Configuration

#Configuration.show_compile_hint = False

def learning_phase(data_row, i, problem_name, n_obj, cf, reference_vectors):
    # learning phase
    data_row[["problem", "num_obj", "iteration"]] = [
        problem_name,
        n_obj,
        i + 1,
    ]

    # After this class call, solutions inside the composite front are assigned to reference vectors
    base = baseADM(cf, reference_vectors)
    # generates the next reference point for the next iteration in the learning phase
    response = gp.generateRP4learning(base)

    data_row["reference_point"] = [
        response,
    ]

    # run algorithms with the new reference point
    pref_int_rvea.response = pd.DataFrame(
        [response], columns=pref_int_rvea.content["dimensions_data"].columns
    )
    pref_int_nsga.response = pd.DataFrame(
        [response], columns=pref_int_nsga.content["dimensions_data"].columns
    )

    _, pref_int_rvea = int_rvea.iterate(pref_int_rvea)
    _, pref_int_nsga = int_nsga.iterate(pref_int_nsga)

    # extend composite front with newly obtained solutions
    cf = generate_composite_front(
        cf, int_rvea.population.objectives, int_nsga.population.objectives
    )

    # R-metric calculation
    ref_point = response.reshape(1, n_obj)

    # normalize reference point
    rp_transformer = Normalizer().fit(ref_point)
    norm_rp = rp_transformer.transform(ref_point)

    rmetric = rm.RMetric(problemR, norm_rp)

    # normalize solutions before sending r-metric
    rvea_transformer = Normalizer().fit(int_rvea.population.objectives)
    norm_rvea = rvea_transformer.transform(int_rvea.population.objectives)

    nsga_transformer = Normalizer().fit(int_nsga.population.objectives)
    norm_nsga = nsga_transformer.transform(int_nsga.population.objectives)

    # R-metric calls for R_IGD and R_HV
    rigd_irvea, rhv_irvea = rmetric.calc(norm_rvea, others=norm_nsga)
    rigd_insga, rhv_insga = rmetric.calc(norm_nsga, others=norm_rvea)

    data_row[["iRVEA" + excess_col for excess_col in excess_columns]] = [
        rigd_irvea,
        rhv_irvea,
    ]
    data_row[["iNSGAIII" + excess_col for excess_col in excess_columns]] = [
        rigd_insga,
        rhv_insga,
    ]

    data = data.append(data_row, ignore_index=1)
#---------------------------------------

problem_names = ["VCW", "CSI", "RPP", "MCB"]
problems = [vehicle_crashworthiness, car_side_impact, river_pollution_problem, multiple_clutch_brakes]
n_objs = np.asarray([3,3,5,5])  # number of objectives

num_gen_per_iter = [50]  # number of generations per iteration

algorithms = ["iRVEA", "iNSGAIII"]  # algorithms to be compared

# the followings are for formatting results
column_names = (
    ["problem", "num_obj", "iteration", "reference_point"]
    + [algorithm + "_R_HV" for algorithm in algorithms]
)
excess_columns = [
    "_R_HV",
]
data = pd.DataFrame(columns=column_names)
data_row = pd.DataFrame(columns=column_names, index=[1])

# ADM parameters
L = 4  # number of iterations for the learning phase
D = 3  # number of iterations for the decision phase
lattice_resolution = 5  # density variable for creating reference vectors


counter = 1
total_count = len(num_gen_per_iter) * len(n_objs) * len(problem_names)
for gen in num_gen_per_iter:
    for idxp in range(0,len(problems)):
        print(f"Loop {counter} of {total_count}")
        counter += 1
        problem = problems[idxp]
        n_obj = n_objs[idxp]
        problem_name = problem_names[idxp]
        
        # interactive
        int_rvea = RVEA(problem=problem, interact=True, n_gen_per_iter=gen)
        int_nsga = NSGAIII(problem=problem, interact=True, n_gen_per_iter=gen)

        # initial reference point is specified randomly
        response = np.random.rand(n_obj) #TODO: check this (approximate ideal and nadir for each problem)

        # run algorithms once with the randomly generated reference point
        _, pref_int_rvea = int_rvea.requests()
        _, pref_int_nsga = int_nsga.requests()
        pref_int_rvea.response = pd.DataFrame(
            [response], columns=pref_int_rvea.content["dimensions_data"].columns
        )
        pref_int_nsga.response = pd.DataFrame(
            [response], columns=pref_int_nsga.content["dimensions_data"].columns
        )

        _, pref_int_rvea = int_rvea.iterate(pref_int_rvea)
        _, pref_int_nsga = int_nsga.iterate(pref_int_nsga)

        # build initial composite front
        cf = generate_composite_front(
            int_rvea.population.objectives, int_nsga.population.objectives
        )

        # creates uniformly distributed reference vectors
        reference_vectors = ReferenceVectors(lattice_resolution, n_obj)

        

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
            response = gp.generateRP4decision(base, max_assigned_vector[0])

            data_row["reference_point"] = [
                response,
            ]

            # run algorithms with the new reference point
            pref_int_rvea.response = pd.DataFrame(
                [response], columns=pref_int_rvea.content["dimensions_data"].columns
            )
            pref_int_nsga.response = pd.DataFrame(
                [response], columns=pref_int_nsga.content["dimensions_data"].columns
            )

            _, pref_int_rvea = int_rvea.iterate(pref_int_rvea)
            _, pref_int_nsga = int_nsga.iterate(pref_int_nsga)

            # extend composite front with newly obtained solutions
            cf = generate_composite_front(
                cf, int_rvea.population.objectives, int_nsga.population.objectives
            )

            # R-metric calculation
            ref_point = response.reshape(1, n_obj)

            rp_transformer = Normalizer().fit(ref_point)
            norm_rp = rp_transformer.transform(ref_point)

            # for decision phase, delta is specified as 0.2
            rmetric = rm.RMetric(problemR, norm_rp, delta=0.2, pf=pareto_front)

            # normalize solutions before sending r-metric
            rvea_transformer = Normalizer().fit(int_rvea.population.objectives)
            norm_rvea = rvea_transformer.transform(int_rvea.population.objectives)

            nsga_transformer = Normalizer().fit(int_nsga.population.objectives)
            norm_nsga = nsga_transformer.transform(int_nsga.population.objectives)

            rigd_irvea, rhv_irvea = rmetric.calc(norm_rvea, others=norm_nsga)
            rigd_insga, rhv_insga = rmetric.calc(norm_nsga, others=norm_rvea)

            data_row[["iRVEA" + excess_col for excess_col in excess_columns]] = [
                rigd_irvea,
                rhv_irvea,
            ]
            data_row[["iNSGAIII" + excess_col for excess_col in excess_columns]] = [
                rigd_insga,
                rhv_insga,
            ]

            data = data.append(data_row, ignore_index=1)

data.to_csv("./results/EMO2021/50_generations/output21.csv", index=False)