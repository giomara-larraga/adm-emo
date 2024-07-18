
from CarSideImpact import car_side_impact
from RiverPollutionProblem import river_pollution_problem
from VehicleCrashWorthiness import vehicle_crashworthiness

from desdeo_emo.EAs.NSGAIII import NSGAIII
import numpy as np

problems = [car_side_impact(three_obj=False), river_pollution_problem(), vehicle_crashworthiness()]
#algorithms = [IBEA, RVEA]

def compute_front (problem):
    evolver = NSGAIII(
        problem,
        n_iterations=1,
        n_gen_per_iter=500,
        population_size=100,
    )

    #evolver.set_interaction_type("Reference point")

    while evolver.continue_evolution():
        evolver.iterate()

    objectives = evolver.population.objectives
    ideal = evolver.population.problem.ideal
    nadir = evolver.population.problem.nadir

    aprox_ideal = np.min(evolver.population.fitness, axis=0)
    aprox_nadir = np.max(evolver.population.fitness, axis=0)

    print(evolver.population.problem._max_multiplier)
    print(ideal)
    print(aprox_ideal)
    print(aprox_nadir)
    return 0


if __name__== "__main__":
    compute_front(problems[2])
