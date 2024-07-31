from DBConnection import get_reference_points, get_solutions_by_gen
from EA import NSGAIII_archive as NSGAIII
from VehicleCrashWorthiness import vehicle_crashworthiness
from RiverPollutionProblem import river_pollution_problem
from CarSideImpact import car_side_impact
import pandas as pd
import numpy as np
from phi import phi

dict_problems = dict([('VCW', vehicle_crashworthiness()), ('CSI', car_side_impact()), ('RPP', river_pollution_problem())])

problem = dict_problems['VCW']
problem_name = "VCW"
gens_iter = 300

def compute_metric_learning(data, RP, nadir):
    phi_learning = phi(problem.ideal)
    results = phi_learning.get_phi(data, RP, nadir)
    return results


def compute_metric(reference_points):
    #Compute metric
    results_learning_positive_nsga3 = []
    results_learning_negative_nsga3 = []

    for i in range(len(reference_points)):
        reference_point = reference_points[i]
        for j in range(gens_iter):
            data = get_solutions_by_gen(i,j,problem_name,"NSGAIII")
            metric_learning = compute_metric_learning(data, reference_point, problem.nadir)
            results_learning_positive_nsga3.append(metric_learning[0])
            
            results_learning_negative_nsga3.append(metric_learning[2])

    return results_learning_positive_nsga3, results_learning_negative_nsga3

def main():
    #Get reference points
    reference_points = get_reference_points(problem_name, "L")
    evolver_nsga3 = NSGAIII(problem, problem_name=problem_name, phase="L", interact=True, n_gen_per_iter=gens_iter)


    evolver_nsga3.set_interaction_type("Reference point")

    pref_nsga3, _ = evolver_nsga3.start()

    for refPoint in reference_points:
        reference_point = np.array(refPoint)
        pref_nsga3.response = pd.DataFrame([reference_point], columns=problem.objective_names)
        pref_nsga3, _ = evolver_nsga3.iterate(pref_nsga3)

    #Delete rows sql
    #delete from problem;
    #UPDATE SQLITE_SEQUENCE SET SEQ=0 WHERE NAME='Problem';


if __name__ =="__main__":
    main()
