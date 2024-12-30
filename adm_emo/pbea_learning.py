#! /usr/bin/env python3

import argparse
import logging
import sys

import numpy as np
import pandas as pd
from desdeo_emo.EAs import AutoIBEA as IBEA
from Archive_EAs import PBEA_archive as PBEA
from problems import generate_dict_problems_dtlz, generate_dict_problems_wfg
from utils import compute_phi_learning_from_list

problem_names = ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7"]
problem_names_wfg = [
    "WFG1",
    "WFG2",
    "WFG3",
    "WFG4",
    "WFG5",
    "WFG6",
    "WFG7",
    "WFG8",
    "WFG9",
]
objectives = [3, 5, 7]
dict_problems = generate_dict_problems_dtlz(problem_names, objectives)
dict_problems_wfg = generate_dict_problems_wfg(problem_names_wfg, objectives)
dict_problems.update(dict_problems_wfg)


def main(
    SEED,
    PROB,
    OBJ,
    GENS,
    CROS,
    CROS_PROB,
    CROS_REP,
    CROS_DIST,
    CROS_ALPHA,
    MUT,
    MUT_PROB,
    MUT_REPAIR,
    MUT_PMD,
    MUT_UMP,
    SEL,
    SEL_SIZE,
):
    problem_data = dict_problems[PROB + str(OBJ)]
    problem = problem_data["problem"]
    if OBJ == 3:
        POP = 50
        GENS = 100
    elif OBJ == 5:
        POP = 80
        GENS = 200
    elif OBJ == 7:
        POP = 120
        GENS = 300
    else:
        POP = 0
        GENS = 0
    # POP = len(problem_data["ref_dirs"])

    list_ref_points = []
    folder_path = (
        "/projappl/project_2012477/implementation/adm-emo/adm_emo/reference_points/"
    )
    folder_path = "C:/Users/Giomara/Documents/Projects/irace_learning_decision/adm-emo/adm_emo/reference_points/"

    it_learning = 4
    # it_decision = 3
    for i in range(it_learning):
        ref_point = np.loadtxt(
            folder_path + "rp_learning_" + str(i) + "_" + PROB + "_" + str(OBJ),
            dtype=float,
        )
        list_ref_points.append(ref_point)
    # for i in range(it_decision):
    #    ref_point = np.loadtxt(folder_path+"rp_decision_"+str(i)+"_"+PROB+"_"+str(OBJ), dtype=float)
    #    list_ref_points.append(ref_point)

    evolver = PBEA(
        problem,
        n_iterations=it_learning,
        n_gen_per_iter=GENS,
        initial_population=None,
        population_size=POP,
        interact=True,
        seed=SEED,
        selection_parents=SEL,
        slection_tournament_size=SEL_SIZE,
        crossover=CROS,
        crossover_probability=CROS_PROB,
        crossover_distribution_index=CROS_DIST,
        blx_alpha_crossover=CROS_ALPHA,
        crossover_repair=CROS_REP,
        mutation=MUT,
        mutation_probability=MUT_PROB,
        mutation_repair=MUT_REPAIR,
        uniform_mut_perturbation=MUT_UMP,
        polinomial_mut_dist_index=MUT_PMD,
    )

    pref, plot = evolver.requests()
    for refPoint in list_ref_points:
        responses = np.asarray([refPoint])
        pref.response = pd.DataFrame(
            [responses[0]], columns=pref.content["dimensions_data"].columns
        )
        pref, plot = evolver.iterate(pref)

    archive_run = evolver.archives
    i2, obj = evolver.end()

    flattened_matrix = np.vstack(list(archive_run.values()))
    A = np.vstack((flattened_matrix, list_ref_points))

    epsilon = 1e-6  # Small perturbation
    ideal = np.min(A, axis=0) - epsilon  # Slightly shift ideal point
    nadir = np.max(A, axis=0) + epsilon  # Slightly shift nadir point

    RS = compute_phi_learning_from_list(
        archive_run, ideal, nadir, GENS, it_learning, list_ref_points
    )
    print(RS)


if __name__ == "__main__":
    main(
        1254,
        "DTLZ2",
        3,
        50,
        "SBX",
        0.5,
        "bounds",
        10,
        0.2,
        "polynomial",
        0.5,
        "bounds",
        20,
        0.5,
        "random",
        None,
    )


def test():
    # just check if args are ok
    with open("args.txt", "w") as f:
        f.write(str(sys.argv))

    # loading example arguments
    ap = argparse.ArgumentParser(
        description="Feature Selection using GA with DecisionTreeClassifier"
    )
    ap.add_argument(
        "-v", "--verbose", help="increase output verbosity", action="store_true"
    )
    # 3 args to test values
    ap.add_argument(
        "--seed", dest="seed", type=int, required=True, help="Seed for random numbers"
    )
    ap.add_argument("--prob", dest="prob", type=str, required=True, help="Problem name")
    ap.add_argument(
        "--obj", dest="obj", type=int, required=True, help="Number of objectives"
    )
    ap.add_argument(
        "--generations",
        dest="gens",
        type=int,
        required=False,
        help="Number of generations",
    )

    ap.add_argument(
        "--crossover",
        dest="cros",
        type=str,
        required=False,
        help="Crossover type (SBX or BLX)",
    )
    ap.add_argument(
        "--crossoverProbability",
        dest="cros_prob",
        type=float,
        required=False,
        help="Crossover probability",
    )
    ap.add_argument(
        "--crossoverRepairStrategy",
        dest="cros_rep",
        type=str,
        required=False,
        help="Crossover repair strategy (RANDOM, ROUND, BOUNDS)",
    )
    ap.add_argument(
        "--sbxCrossoverDistributionIndex",
        dest="cros_dist",
        type=float,
        required=False,
        help="SBX Crossover distribution index",
    )
    ap.add_argument(
        "--blxAlphaCrossoverAlphaValue",
        dest="cros_alpha",
        type=float,
        required=False,
        help="BLX Crossover alpha value",
    )

    ap.add_argument(
        "--mutation",
        dest="mut",
        type=str,
        required=False,
        help='Mutation type ("polynomial, uniform")',
    )
    ap.add_argument(
        "--mutationProbability",
        dest="mut_prob",
        type=float,
        required=False,
        help="Mutation probability",
    )
    ap.add_argument(
        "--mutationRepairStrategy",
        dest="mut_repair",
        type=str,
        required=False,
        help="Mutation repair strategy (random, rpund, bounds)",
    )
    ap.add_argument(
        "--polynomialMutationDistributionIndex",
        dest="mut_pmd",
        type=float,
        required=False,
        help="Polynomial Mutation Distribution Index",
    )
    ap.add_argument(
        "--uniformMutationPerturbation",
        dest="mut_ump",
        type=float,
        required=False,
        help="Uniform Mutation Perturbation",
    )

    ap.add_argument(
        "--selection",
        dest="sel",
        type=str,
        required=False,
        help="Selection operator (random, tournament)",
    )
    ap.add_argument(
        "--selectionTournamentSize",
        dest="sel_size",
        type=int,
        required=False,
        help="Size of tournament selection",
    )
    # 1 arg file name to save and load fo value
    # ap.add_argument('--datfile', dest='datfile', type=str, required=False, help='File where it will be save the score (result)')

    args = ap.parse_args()
    logging.debug(args)
    # np.random.seed(args.seed)
    # call main function passing args
    main(
        args.seed,
        args.prob,
        args.obj,
        args.gens,
        args.cros,
        args.cros_prob,
        args.cros_rep,
        args.cros_dist,
        args.cros_alpha,
        args.mut,
        args.mut_prob,
        args.mut_repair,
        args.mut_pmd,
        args.mut_ump,
        args.sel,
        args.sel_size,
    )
