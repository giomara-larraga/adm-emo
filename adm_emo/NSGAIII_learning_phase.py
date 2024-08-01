from DBConnection import get_reference_points, get_solutions_by_gen
from EA import NSGAIII_archive as NSGAIII
from VehicleCrashWorthiness import vehicle_crashworthiness
from RiverPollutionProblem import river_pollution_problem
from CarSideImpact import car_side_impact
import pandas as pd
import numpy as np
from phi import phi

import argparse
import logging
import sys

from scipy.stats import ranksums
from desdeo_emo.EAs.AutoNSGAIII import AutoNSGAIII


dict_problems = dict([('VCW', vehicle_crashworthiness()), ('CSI', car_side_impact()), ('RPP', river_pollution_problem())])

#problem = dict_problems['VCW']
#problem_name = "VCW"
#gens_iter = 300


def main(SEED, PROB, GENS, CROS, CROS_PROB, CROS_REP, CROS_DIST, CROS_ALPHA, MUT, MUT_PROB, MUT_REPAIR, MUT_PMD, MUT_UMP, SEL, SEL_SIZE):
    #Get reference points
    reference_points = get_reference_points(PROB, "L")
    #evolver_nsga3 = NSGAIII(problem, problem_name=problem_name, phase="L", interact=True, n_gen_per_iter=gens_iter)
    problem = dict_problems[PROB]

    evolver_nsga3 = AutoNSGAIII(
        problem, 
        problem_name=PROB,
        phase="L", 
        interact=True,
        n_gen_per_iter=GENS,
        seed= SEED,
        selection_parents = SEL,
        slection_tournament_size = SEL_SIZE,
        crossover = CROS,
        crossover_probability = CROS_PROB,
        crossover_distribution_index = CROS_DIST,
        crossover_repair = CROS_REP,
        blx_alpha_crossover = CROS_ALPHA,
        mutation = MUT,
        mutation_probability = MUT_PROB,
        mutation_repair = MUT_REPAIR,
        uniform_mut_perturbation  = MUT_UMP,
        polinomial_mut_dist_index = MUT_PMD,
    )
    evolver_nsga3.set_interaction_type("Reference point")

    pref_nsga3, _ = evolver_nsga3.start()

    for refPoint in reference_points:
        reference_point = np.array(refPoint)
        evolver_nsga3.set_ref_point(reference_point)
        pref_nsga3.response = pd.DataFrame([reference_point], columns=problem.objective_names)
        pref_nsga3, _ = evolver_nsga3.iterate(pref_nsga3)

    print(np.sum(evolver_nsga3.phi_learning_values))
    
    #Delete rows sql
    #delete from problem;
    #UPDATE SQLITE_SEQUENCE SET SEQ=0 WHERE NAME='Problem';


if __name__ == "__main__":
    # just check if args are ok
    with open('args.txt', 'w') as f:
        f.write(str(sys.argv))
    
    # loading example arguments
    ap = argparse.ArgumentParser(description='Feature Selection using GA with DecisionTreeClassifier')
    ap.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    # 3 args to test values
    ap.add_argument('--seed', dest='seed', type=int, required=True, help='Seed for random numbers')
    ap.add_argument('--prob', dest='prob', type=str, required=True, help='Problem name')
    ap.add_argument('--generations', dest='gens', type=int, required=False, help='Number of generations')

    ap.add_argument('--crossover', dest='cros', type=str, required=False, help='Crossover type (SBX or BLX)')
    ap.add_argument('--crossoverProbability', dest='cros_prob', type=float, required=False, help='Crossover probability')
    ap.add_argument('--crossoverRepairStrategy', dest='cros_rep', type=str, required=False, help='Crossover repair strategy (RANDOM, ROUND, BOUNDS)')
    ap.add_argument('--sbxCrossoverDistributionIndex', dest='cros_dist', type=float, required=False, help='SBX Crossover distribution index')
    ap.add_argument('--blxAlphaCrossoverAlphaValue', dest='cros_alpha', type=float, required=False, help='BLX Crossover alpha value')

    ap.add_argument('--mutation', dest='mut', type=str, required=False, help='Mutation type ("polynomial, uniform")')
    ap.add_argument('--mutationProbability', dest='mut_prob', type=float, required=False, help='Mutation probability')
    ap.add_argument('--mutationRepairStrategy', dest='mut_repair', type=str, required=False, help='Mutation repair strategy (random, rpund, bounds)')
    ap.add_argument('--polynomialMutationDistributionIndex', dest='mut_pmd', type=float, required=False, help='Polynomial Mutation Distribution Index')
    ap.add_argument('--uniformMutationPerturbation', dest='mut_ump', type=float, required=False, help='Uniform Mutation Perturbation')

    ap.add_argument('--selection', dest='sel', type=str, required=False, help='Selection operator (random, tournament)')
    ap.add_argument('--selectionTournamentSize', dest='sel_size', type=int, required=False, help='Size of tournament selection')


    # 1 arg file name to save and load fo value
    #ap.add_argument('--datfile', dest='datfile', type=str, required=False, help='File where it will be save the score (result)')

    args = ap.parse_args()
    logging.debug(args)
    #np.random.seed(args.seed)
    # call main function passing args
    main(args.seed, args.prob, args.id, args.obj, args.var, args.gens, args.cros,args.cros_prob, args.cros_rep, args.cros_dist, args.cros_alpha, args.mut, args.mut_prob, args.mut_repair, args.mut_pmd, args.mut_ump, args.sel, args.sel_size)

