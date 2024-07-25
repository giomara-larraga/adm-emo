import numpy as np
import pandas as pd

import baseADM
from baseADM import *
import generatePreference as gp

from VehicleCrashWorthiness import vehicle_crashworthiness
from RiverPollutionProblem import river_pollution_problem
from CarSideImpact import car_side_impact


from desdeo_emo.utilities.ReferenceVectors import ReferenceVectors

from desdeo_emo.EAs.AutoRVEA import AutoRVEA
from desdeo_emo.EAs.AutoPBEA import AutoPBEA

from pymoo.util.ref_dirs import get_reference_directions
from sklearn.preprocessing import Normalizer


problem_names = ["VCW", "CSI", "RPP"]
problems = [vehicle_crashworthiness(), car_side_impact(three_obj=False), river_pollution_problem()]
n_objs = np.asarray([3,4,5])  # number of objectives

gen = 300  # number of generations per iteration

algorithms = ["RVEA", "PBEA"]  # algorithms to be compared

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

for idxp in range(0,len(problems)):
    problem = problems[idxp]
    n_obj = n_objs[idxp]
    problem_name = problem_names[idxp]
    
    # interactive
    rvea = AutoRVEA(problem=problem, interact=True, n_gen_per_iter=gen)
    pbea = AutoPBEA(problem=problem, interact=True, n_gen_per_iter=gen)

    # initial reference point is specified randomly
    response = np.random.rand(n_obj) #TODO: check this (approximate ideal and nadir for each problem)


    # run algorithms once with the randomly generated reference point
    print ("Run irace with both algorithms")

    # build initial composite front
    cf = generate_composite_front(
        rvea.population.objectives, pbea.population.objectives
    )

    # creates uniformly distributed reference vectors
    reference_vectors = ReferenceVectors(lattice_resolution, n_obj)

    
    # Learning phase
    for i in range(L):
        # After this class call, solutions inside the composite front are assigned to reference vectors
        base = baseADM(cf, reference_vectors)
        # generates the next reference point for the next iteration in the learning phase
        response = gp.generateRP4learning(base)

        data_row["reference_point"] = [
            response,
        ]
        print ("Run irace with both algorithms for learning phase")

        # extend composite front with newly obtained solutions
        cf = generate_composite_front(
            cf, rvea.population.objectives, pbea.population.objectives
        )

    # Decision phase
    # After the learning phase the reference vector which has the maximum number of assigned solutions forms ROI
    max_assigned_vector = gp.get_max_assigned_vector(base.assigned_vectors)

    for i in range(D):
        # since composite front grows after each iteration this call should be done for each iteration
        base = baseADM(cf, reference_vectors)

        # generates the next reference point for the decision phase
        response = gp.generateRP4decision(base, max_assigned_vector[0])

        data_row["reference_point"] = [
            response,
        ]

        print ("Run irace with both algorithms for decision phase")

        print ("get confing")

        # Run algorithms with config

        #get population

        # extend composite front with newly obtained solutions
        cf = generate_composite_front(
            cf, rvea.population.objectives, pbea.population.objectives
        )

