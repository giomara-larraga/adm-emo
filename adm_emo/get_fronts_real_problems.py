
from CarSideImpact import car_side_impact
from RiverPollutionProblem import river_pollution_problem
from VehicleCrashWorthiness import vehicle_crashworthiness

from desdeo_emo.EAs.AutoIBEA import AutoPBEA

problems = [car_side_impact(three_obj=False), river_pollution_problem(), vehicle_crashworthiness()]
#algorithms = [IBEA, RVEA]

def compute_front (problem):
    evolver = AutoPBEA(
        problem,
        n_iterations=1,
        n_gen_per_iter=300,
        population_size=100,
        crossover_probability = 0.0318,
        crossover_distribution_index=0.5,
        mutation_probability = 0.0328,
        polinomial_mut_dist_index  = 140.3278,
    )

    #evolver.set_interaction_type("Reference point")

    while evolver.continue_evolution():
        evolver.iterate()

    objectives = evolver.population.objectives
    ideal = evolver.population.problem.ideal

    print(ideal)
    return 0


if __name__== "__main__":
    compute_front(problems[1])
