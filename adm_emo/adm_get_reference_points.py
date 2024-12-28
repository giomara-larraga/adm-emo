import numpy as np
import pandas as pd

import baseADM
from baseADM import *
import generatePreference as gp
from desdeo_problem import test_problem_builder

from desdeo_emo.utilities.ReferenceVectors import ReferenceVectors

from desdeo_emo.EAs import RVEA as ALG1
from desdeo_emo.EAs import NSGAIII as ALG2

from pymoo.util.ref_dirs import get_reference_directions
from pymoo.problems import get_problem

from desdeo_emo.utilities import ReferenceVectors
import rmetric as rm
from sklearn.preprocessing import Normalizer
from wfg_problems import wfg_problem_builder




""" def initialize_pbea(problem, population_size, gens):
    ib = IBEA(problem, population_size=population_size, n_iterations=1, n_gen_per_iter=gens)
    while ib.continue_evolution():
        ib.iterate()
    individuals, objective_values = ib.end()
    ini_pop = ib.population
    return ini_pop """

#dict_problems = dict([('VCW', vehicle_crashworthiness()), ('CSI', car_side_impact()), ('RPP', river_pollution_problem())])
dict_algorithms = dict([('RVEA', ALG1), ('PBEA', ALG2)])
#problem_names = ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7"]
problem_names_wfg = ["WFG1", "WFG2", "WFG3", "WFG4", "WFG5", "WFG6", "WFG7", "WFG8", "WFG9"]
objectives = [3,5,7]
#dict_problems = generate_dict_problems_dtlz(problem_names, objectives)
dict_problems = generate_dict_problems_wfg(problem_names_wfg, objectives)
#dict_problems.update(dict_problems_wfg)

#print(dict_problems)

# the followings are for formatting results
column_names = (
    ["problem", "iteration", "reference_point"]
    + [algorithm + "_R_IGD" for algorithm in dict_algorithms.keys()]
)


data = pd.DataFrame(columns=column_names)
data_row = pd.DataFrame(columns=column_names, index=[1])

# ADM parameters
L = 4  # number of iterations for the learning phase
D = 3  # number of iterations for the decision phase

gen = 500

#print(dict_problems)

for problem_name in dict_problems.keys():
    print("Running for problem", problem_name)
    data_problem = dict_problems[problem_name]
    name = data_problem["name"]
    problem = data_problem["problem"]
    n_obj = data_problem["objectives"]
    n_var = data_problem["variables"]
    ideal = data_problem["ideal"]
    nadir = data_problem["nadir"]
    ref_dirs = data_problem["ref_dirs"]

    # the following two lines for getting pareto front by using pymoo framework
    problemR = get_problem(name.lower(), n_var, n_obj)  
    #pareto_front = problemR.pareto_front(ref_dirs)
    #true_nadir = data_problem["nadir"]

    # interactive
    #ini_pop = initialize_pbea(problem,50,100)

    #archiver_alg1 = archiver("ALG1", problem_name)
    #archiver_alg2 = archiver("PBEA", problem_name)
    alg1 = ALG1(problem=problem, interact=True, n_gen_per_iter=gen, population_size=len(ref_dirs))
    alg2 = ALG2(problem=problem, interact=True, n_gen_per_iter=gen, population_size=len(ref_dirs))

    alg1.set_interaction_type("Reference point")
    alg2.set_interaction_type("Reference point")

    pref_alg1, plot_alg1 = alg1.start()
    pref_alg2, plot_alg2 = alg2.start()

    # initial reference point is specified randomly
    #response = np.random.rand(n_obj)
    values = np.random.uniform(ideal, nadir, (1, n_obj))

    response = values[0]
    print("initial_reference_point",response)

    pref_alg1.response = pd.DataFrame([response], columns=problem.objective_names)
    pref_alg2.response = pd.DataFrame([response], columns=problem.objective_names)

    pref_alg1, _ = alg1.iterate(pref_alg1)
    pref_alg2, _ = alg2.iterate(pref_alg2)

    # build initial composite front
    cf = generate_composite_front(
        alg1.population.objectives, alg2.population.objectives
    )

    # learning phase
    for i in range(L):
        data_row[["problem", "iteration"]] = [
            problem_name,
            i + 1,
        ]
        print ("Learning phase ", i)
        # After this class call, solutions inside the composite front are assigned to reference vectors
        base = baseADM(cf, ref_dirs)
        # generates the next reference point for the next iteration in the learning phase
        response = gp.generateRP4learning(base)

        np.savetxt("rp_learning_"+str(i)+"_"+name+"_"+str(n_obj),response)

        data_row["reference_point"] = [
            response,
        ]
        #alg1.set_interaction_type("Reference point")

        #print(pref_alg1.content['message'])

        # run algorithms with the new reference point
        pref_alg1.response = pd.DataFrame([response], columns=problem.objective_names)
        #pref_alg1.response = pd.DataFrame([response], columns=pref_alg1.content['dimensions_data'].columns)
        pref_alg2.response = pd.DataFrame([response], columns=problem.objective_names)
        pref_alg1, _ = alg1.iterate(pref_alg1)
        pref_alg2, _ = alg2.iterate(pref_alg2)

        # extend composite front with newly obtained solutions
        cf = generate_composite_front(
            cf, alg1.population.objectives, alg2.population.objectives
        )

        # R-metric calculation
        ref_point = response.reshape(1, n_obj)

        # normalize reference point
        rp_transformer = Normalizer().fit(ref_point)
        norm_rp = rp_transformer.transform(ref_point)

        rmetric = rm.RMetric(problemR, norm_rp)

        # normalize solutions before sending r-metric
        alg1_transformer = Normalizer().fit(alg1.population.objectives)
        norm_alg1 = alg1_transformer.transform(alg1.population.objectives)

        alg2_transformer = Normalizer().fit(alg2.population.objectives)
        norm_alg2 = alg2_transformer.transform(alg2.population.objectives)

        # R-metric calls for R_IGD and R_HV
        rigd_alg1,_ = rmetric.calc(norm_alg1, others=norm_alg2)
        rigd_alg2,_ = rmetric.calc(norm_alg2, others=norm_alg1)

        data_row["ALG1_R_IGD"] = [
            rigd_alg1,
        ]
        data_row["ALG2_R_IGD"] = [
            rigd_alg2,
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
        base = baseADM(cf, ref_dirs)

        # generates the next reference point for the decision phase
        response = gp.generateRP4decision(base, max_assigned_vector[0])

        #print("decision_reference_point",response)
        np.savetxt("rp_decision_"+str(i)+"_"+name+"_"+str(n_obj),response)

        data_row["reference_point"] = [
            response,
        ]

        # run algorithms with the new reference point
        pref_alg1.response = pd.DataFrame([response], columns=problem.objective_names)
        #pref_alg1.response = pd.DataFrame([response], columns=pref_alg1.content['dimensions_data'].columns)
        pref_alg2.response = pd.DataFrame([response], columns=problem.objective_names)

        pref_alg1, _ = alg1.iterate(pref_alg1)
        pref_alg2, _ = alg2.iterate(pref_alg2)

        # extend composite front with newly obtained solutions
        cf = generate_composite_front(
            cf, alg1.population.objectives, alg2.population.objectives
        )

        # R-metric calculation
        ref_point = response.reshape(1, n_obj)

        rp_transformer = Normalizer().fit(ref_point)
        norm_rp = rp_transformer.transform(ref_point)


   
        # for decision phase, delta is specified as 0.2
        rmetric = rm.RMetric(problemR, norm_rp, delta=0.2)

        # normalize solutions before sending r-metric
        alg1_transformer = Normalizer().fit(alg1.population.objectives)
        norm_alg1 = alg1_transformer.transform(alg1.population.objectives)

        alg2_transformer = Normalizer().fit(alg2.population.objectives)
        norm_alg2 = alg2_transformer.transform(alg2.population.objectives)

        rigd_alg1,_ = rmetric.calc(norm_alg1, others=norm_alg2)
        rigd_alg2,_ = rmetric.calc(norm_alg2, others=norm_alg1)

        data_row["ALG1_R_IGD"] = [
            rigd_alg1,
        ]
        data_row["ALG2_R_IGD"] = [
            rigd_alg2,
        ]

        data = data.append(data_row, ignore_index=1)

#data.to_csv("output21.csv", index=False)
