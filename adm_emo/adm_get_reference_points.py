import numpy as np
import pandas as pd

import baseADM
from baseADM import *
import generatePreference as gp
from desdeo_problem import test_problem_builder

from desdeo_emo.utilities.ReferenceVectors import ReferenceVectors

from desdeo_emo.EAs.NSGAIII import NSGAIII as RVEA
from desdeo_emo.EAs.PBEA import PBEA
from desdeo_emo.EAs.IBEA import IBEA

#from EA import NSGAIII_archive as RVEA
#from EA import PBEA_archive as PBEA
#from EA import archiver

from pymoo.util.ref_dirs import get_reference_directions
from desdeo_emo.utilities import ReferenceVectors
import rmetric as rm
from sklearn.preprocessing import Normalizer
from wfg_problems import wfg_problem_builder


def generate_dict_problems_wfg(problems, objectives):
    # variables = m + k-1
    generated_dict_problems = dict()

    for problem in problems:
        for n_obj in objectives:
            variables = 10 + n_obj -1
            ideal = np.zeros(n_obj)
            nadir = np.ones(n_obj)

            problem_form = wfg_problem_builder(problem, variables, n_obj)
            dict_data = {"problem":  problem_form, "ideal": ideal, "nadir": nadir}
            generated_dict_problems[problem+str(n_obj)] = dict_data

    return generated_dict_problems

def generate_dict_problems_dtlz(problems, objectives):
    # variables = m + k-1
    generated_dict_problems = dict()

    for problem in problems:
        for n_obj in objectives:
            if problem == "DTLZ1":
                variables = 5 + n_obj -1
                ideal = np.zeros(n_obj) 
                nadir = np.ones(n_obj) * 0.5
            else:
                variables = 10 + n_obj -1
                ideal = np.zeros(n_obj)
                nadir = np.ones(n_obj)

            problem_form = test_problem_builder(problem, variables, n_obj)
            dict_data = {"problem":  problem_form, "ideal": ideal, "nadir": nadir}
            generated_dict_problems[problem+str(n_obj)] = dict_data

    return generated_dict_problems

def initialize_pbea(problem, population_size, gens):
    ib = IBEA(problem, population_size=population_size, n_iterations=1, n_gen_per_iter=gens)
    while ib.continue_evolution():
        ib.iterate()
    individuals, objective_values = ib.end()
    ini_pop = ib.population
    return ini_pop

#dict_problems = dict([('VCW', vehicle_crashworthiness()), ('CSI', car_side_impact()), ('RPP', river_pollution_problem())])
dict_algorithms = dict([('RVEA', RVEA), ('PBEA', PBEA)])
problem_names = ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7"]
problem_names_wfg = ["WFG1", "WFG2", "WFG3", "WFG4", "WFG5", "WFG6", "WFG7", "WFG8", "WFG9"]
objectives = [3,5,7]
#dict_problems = generate_dict_problems_dtlz(problem_names, objectives)
dict_problems = generate_dict_problems_wfg(problem_names_wfg, objectives)
# the followings are for formatting results
column_names = (
    ["problem", "iteration", "reference_point"]
    + [algorithm + "_R_HV" for algorithm in dict_algorithms.keys()]
)


data = pd.DataFrame(columns=column_names)
data_row = pd.DataFrame(columns=column_names, index=[1])

# ADM parameters
L = 4  # number of iterations for the learning phase
D = 3  # number of iterations for the decision phase
lattice_resolution = 5  # density variable for creating reference vectors

population_size = 200
gen = 900

print(dict_problems)

for problem_name in dict_problems.keys():
    print("Running for problem", problem_name)
    data_problem = dict_problems[problem_name]
    problem = data_problem["problem"]
    n_obj = problem.n_of_objectives
    ideal = data_problem["ideal"]
    nadir = data_problem["nadir"]

    #true_nadir = data_problem["nadir"]

    # interactive
    ini_pop = initialize_pbea(problem,50,100)

    #archiver_rvea = archiver("RVEA", problem_name)
    #archiver_pbea = archiver("PBEA", problem_name)
    rvea = RVEA(problem=problem, interact=True, n_gen_per_iter=gen)
    pbea = PBEA(problem=problem, interact=True, n_gen_per_iter=gen, population_size=population_size,initial_population=ini_pop)

    rvea.set_interaction_type("Reference point")
    #pbea.set_interaction_type("Reference point")

    pref_rvea, plot_rvea = rvea.start()
    pref_pbea, plot_pbea = pbea.requests()

    # initial reference point is specified randomly
    #response = np.random.rand(n_obj)
    values = np.random.uniform(ideal, nadir, (1, n_obj))

    response = values[0]
    print("initial_reference_point",response)

    pref_rvea.response = pd.DataFrame([response], columns=problem.objective_names)
    #pref_rvea.response = pd.DataFrame([response], columns=pref_rvea.content['dimensions_data'].columns)
    pref_pbea.response = pd.DataFrame([response], columns=pref_pbea.content['dimensions_data'].columns)



    pref_rvea, _ = rvea.iterate(pref_rvea)
    pref_pbea, _ = pbea.iterate(pref_pbea)

    # build initial composite front
    cf = generate_composite_front(
        rvea.population.objectives, pbea.population.objectives
    )

    # creates uniformly distributed reference vectors
    #reference_vectors = get_reference_directions(lattice_resolution, n_obj)
    reference_vectors = ReferenceVectors(lattice_resolution=12, number_of_objectives=n_obj)

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

        print("learning_reference_point",response)

        data_row["reference_point"] = [
            response,
        ]
        #rvea.set_interaction_type("Reference point")

        #print(pref_rvea.content['message'])

        # run algorithms with the new reference point
        pref_rvea.response = pd.DataFrame([response], columns=problem.objective_names)
        #pref_rvea.response = pd.DataFrame([response], columns=pref_rvea.content['dimensions_data'].columns)
        pref_pbea.response = pd.DataFrame([response], columns=pref_pbea.content['dimensions_data'].columns)
        pref_rvea, _ = rvea.iterate(pref_rvea)
        pref_pbea, _ = pbea.iterate(pref_pbea)

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
        rhv_rvea = rmetric.calc(norm_rvea, others=norm_pbea)
        rhv_pbea = rmetric.calc(norm_pbea, others=norm_rvea)

        data_row["RVEA_R_HV"] = [
            rhv_rvea,
        ]
        data_row["PBEA_R_HV"] = [
            rhv_pbea,
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

        print("decision_reference_point",response)

        data_row["reference_point"] = [
            response,
        ]

        # run algorithms with the new reference point
        pref_rvea.response = pd.DataFrame([response], columns=problem.objective_names)
        #pref_rvea.response = pd.DataFrame([response], columns=pref_rvea.content['dimensions_data'].columns)
        pref_pbea.response = pd.DataFrame([response], columns=pref_pbea.content['dimensions_data'].columns)

        pref_rvea, _ = rvea.iterate(pref_rvea)
        pref_pbea, _ = pbea.iterate(pref_pbea)

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

        rhv_rvea = rmetric.calc(norm_rvea, others=norm_pbea)
        rhv_pbea = rmetric.calc(norm_pbea, others=norm_rvea)

        data_row["RVEA_R_HV"] = [
            rhv_rvea,
        ]
        data_row["PBEA_R_HV"] = [
            rhv_pbea,
        ]

        data = data.append(data_row, ignore_index=1)

data.to_csv("output21.csv", index=False)
