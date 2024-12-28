import numpy as np
import pandas as pd

from baseADM import *

from desdeo_problem import test_problem_builder



from pymoo.util.ref_dirs import get_reference_directions

from wfg_problems import wfg_problem_builder

def generate_dict_problems_wfg(problems, objectives):
    # variables = m + k-1
    generated_dict_problems = dict()

    for problem in problems:
        for n_obj in objectives:
            variables = 10 + n_obj -1
            ideal = np.zeros(n_obj)
            nadir = np.ones(n_obj)

            if n_obj == 3:
                ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
            elif n_obj == 5:
                ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=6)
            elif n_obj == 7:
                ref_dirs = get_reference_directions(
                    "multi-layer",
                    get_reference_directions("das-dennis", n_obj, n_partitions=3, scaling=1.0),
                    get_reference_directions("das-dennis", n_obj, n_partitions=2, scaling=0.5)
                )
            else:
                ref_dirs=[]

            problem_form = wfg_problem_builder(problem, variables, n_obj)
            dict_data = {"name": problem, "objectives": n_obj, "variables": variables, "problem":  problem_form, "ideal": ideal, "nadir": nadir, "ref_dirs": ref_dirs}

            #dict_data = {"problem":  problem_form, "ideal": ideal, "nadir": nadir}
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

            if n_obj == 3:
                ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
            elif n_obj == 5:
                ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=6)
            elif n_obj == 7:
                ref_dirs = get_reference_directions(
                    "multi-layer",
                    get_reference_directions("das-dennis", n_obj, n_partitions=3, scaling=1.0),
                    get_reference_directions("das-dennis", n_obj, n_partitions=2, scaling=0.5)
                )
            else:
                ref_dirs=[]

            problem_form = test_problem_builder(problem, variables, n_obj)
            dict_data = {"name": problem, "objectives": n_obj, "variables": variables, "problem":  problem_form, "ideal": ideal, "nadir": nadir, "ref_dirs": ref_dirs}
            generated_dict_problems[problem+str(n_obj)] = dict_data

    return generated_dict_problems