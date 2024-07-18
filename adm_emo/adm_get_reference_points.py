import numpy as np
import pandas as pd

import baseADM
from baseADM import *
import generatePreference as gp

from VehicleCrashWorthiness import vehicle_crashworthiness
from RiverPollutionProblem import river_pollution_problem
from CarSideImpact import car_side_impact
from desdeo_emo.utilities.ReferenceVectors import ReferenceVectors

from desdeo_emo.EAs.NSGAIII import NSGAIII as RVEA
from desdeo_emo.EAs.PBEA import PBEA
from desdeo_emo.EAs.IBEA import IBEA

from pymoo.util.ref_dirs import get_reference_directions
import rmetric as rm
from sklearn.preprocessing import Normalizer

def initialize_pbea(problem, population_size, gens):
    ib = IBEA(problem, population_size=population_size, n_iterations=10, n_gen_per_iter=gens)
    while ib.continue_evolution():
        ib.iterate()
    individuals, objective_values = ib.end()
    ini_pop = ib.population
    return ini_pop

dict_problems = dict([('VCW', vehicle_crashworthiness()), ('CSI', car_side_impact()), ('RPP', river_pollution_problem())])
dict_algorithms = dict([('RVEA', RVEA), ('PBEA', PBEA)])

# the followings are for formatting results
column_names = (
    ["problem", "iteration", "reference_point"]
    + [algorithm + "_R_HV" for algorithm in dict_algorithms.keys()]
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

population_size = 200
gen = 500

for problem_name in dict_problems.keys():
    print("Running for problem", problem_name)
    problem = dict_problems[problem_name]
    n_obj = problem.n_of_objectives
    ideal = np.asarray([0] * n_obj)
    nadir = abs(np.random.normal(size=n_obj, scale=0.15)) + 1

    true_nadir = np.asarray([1] * n_obj)

    # interactive
    ini_pop = initialize_pbea(problem,30,100)
    rvea = RVEA(problem=problem, interact=True, n_gen_per_iter=gen)
    pbea = PBEA(problem=problem, interact=True, n_gen_per_iter=gen, population_size=population_size,initial_population=ini_pop)

    rvea.set_interaction_type("Reference point")
    #pbea.set_interaction_type("Reference point")

    pref_rvea, plot_rvea = rvea.start()
    pref_pbea, plot_pbea = pbea.requests()

    # initial reference point is specified randomly
    response = np.random.rand(n_obj)
    response = problem.ideal

    pref_rvea.response = pd.DataFrame([response], columns=problem.objective_names)
    #pref_rvea.response = pd.DataFrame([response], columns=pref_rvea.content['dimensions_data'].columns)
    pref_pbea.response = pd.DataFrame([response], columns=pref_pbea.content['dimensions_data'].columns)



    _, pref_rvea = rvea.iterate(pref_rvea)
    _, pref_pbea = pbea.iterate(pref_pbea)

    # build initial composite front
    cf = generate_composite_front(
        rvea.population.objectives, pbea.population.objectives
    )

    # creates uniformly distributed reference vectors
    #reference_vectors = get_reference_directions(lattice_resolution, n_obj)
    reference_vectors = get_reference_directions("uniform", n_obj,n_partitions=12)

    # learning phase
    for i in range(L):
        data_row[["problem", "iteration"]] = [
            problem_name,
            i + 1,
        ]
        print ("Learning phase ", i)
        # After this class call, solutions inside the composite front are assigned to reference vectors
        base = baseADM(cf, reference_vectors)
        # generates the next reference point for the next iteration in the learning phase
        response = gp.generateRP4learning(base)

        data_row["reference_point"] = [
            response,
        ]
        rvea.set_interaction_type("Reference point")

        # run algorithms with the new reference point
        pref_rvea.response = pd.DataFrame([response], columns=problem.objective_names)
        #pref_rvea.response = pd.DataFrame([response], columns=pref_rvea.content['dimensions_data'].columns)
        pref_pbea.response = pd.DataFrame([response], columns=pref_pbea.content['dimensions_data'].columns)
        _, pref_rvea = rvea.iterate(pref_rvea)
        _, pref_pbea = pbea.iterate(pref_pbea)

        # extend composite front with newly obtained solutions
        cf = generate_composite_front(
            cf, rvea.population.objectives, pbea.population.objectives
        )

        # R-metric calculation
        ref_point = response.reshape(1, n_obj)

        # normalize reference point
        rp_transformer = Normalizer().fit(ref_point)
        norm_rp = rp_transformer.transform(ref_point)

        rmetric = rm.RMetric(problem, norm_rp)

        # normalize solutions before sending r-metric
        rvea_transformer = Normalizer().fit(rvea.population.objectives)
        norm_rvea = rvea_transformer.transform(rvea.population.objectives)

        pbea_transformer = Normalizer().fit(pbea.population.objectives)
        norm_pbea = pbea_transformer.transform(pbea.population.objectives)

        # R-metric calls for R_IGD and R_HV
        _, rhv_rvea = rmetric.calc(norm_rvea, others=norm_pbea)
        _, rhv_pbea = rmetric.calc(norm_pbea, others=norm_rvea)

        data_row[["RVEA" + excess_col for excess_col in excess_columns]] = [
            rhv_irvea,
        ]
        data_row[["PBEA" + excess_col for excess_col in excess_columns]] = [
            rhv_insga,
        ]

        data = data.append(data_row, ignore_index=1)

    # Decision phase
    # After the learning phase the reference vector which has the maximum number of assigned solutions forms ROI
    max_assigned_vector = gp.get_max_assigned_vector(base.assigned_vectors)

    for i in range(D):
        data_row[["problem", "iteration"]] = [
            problem_name,
            L + i + 1,
        ]
        print ("Decision phase ", i)
        # since composite front grows after each iteration this call should be done for each iteration
        base = baseADM(cf, reference_vectors)

        # generates the next reference point for the decision phase
        response = gp.generateRP4decision(base, max_assigned_vector[0])

        data_row["reference_point"] = [
            response,
        ]

        # run algorithms with the new reference point
        pref_rvea.response = pd.DataFrame([response], columns=problem.objective_names)
        #pref_rvea.response = pd.DataFrame([response], columns=pref_rvea.content['dimensions_data'].columns)
        pref_pbea.response = pd.DataFrame([response], columns=pref_pbea.content['dimensions_data'].columns)

        _, pref_rvea = rvea.iterate(pref_rvea)
        _, pref_pbea = pbea.iterate(pref_pbea)

        # extend composite front with newly obtained solutions
        cf = generate_composite_front(
            cf, rvea.population.objectives, pbea.population.objectives
        )

        # R-metric calculation
        ref_point = response.reshape(1, n_obj)

        rp_transformer = Normalizer().fit(ref_point)
        norm_rp = rp_transformer.transform(ref_point)

        # for decision phase, delta is specified as 0.2
        rmetric = rm.RMetric(problem, norm_rp, delta=0.2)

        # normalize solutions before sending r-metric
        rvea_transformer = Normalizer().fit(rvea.population.objectives)
        norm_rvea = rvea_transformer.transform(rvea.population.objectives)

        pbea_transformer = Normalizer().fit(pbea.population.objectives)
        norm_pbea = pbea_transformer.transform(pbea.population.objectives)

        _, rhv_irvea = rmetric.calc(norm_rvea, others=norm_pbea)
        _, rhv_insga = rmetric.calc(norm_pbea, others=norm_rvea)

        data_row[["iRVEA" + excess_col for excess_col in excess_columns]] = [
            rhv_irvea,
        ]
        data_row[["iNSGAIII" + excess_col for excess_col in excess_columns]] = [
            rhv_insga,
        ]

        data = data.append(data_row, ignore_index=1)

data.to_csv("output21.csv", index=False)
