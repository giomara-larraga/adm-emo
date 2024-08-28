from optproblems import wfg

from desdeo_problem.problem.Variable import variable_builder
from desdeo_problem.problem.Objective import VectorObjective
from desdeo_problem.problem.Problem import MOProblem
from desdeo_problem.problem.Problem import ProblemError

def wfg_problem_builder(name: str, n_of_variables: int, n_of_objectives: int) -> MOProblem:
    """Build test problems. Currently supported: ZDT1-4, ZDT6, and DTLZ1-7.

    Args:
        name (str): Name of the problem in all caps. For example: "ZDT1", "DTLZ4", etc.
        n_of_variables (int, optional): Number of variables. Required for DTLZ problems,
            but can be skipped for ZDT problems as they only support one variable value.
        n_of_objectives (int, optional): Required for DTLZ problems,
            but can be skipped for ZDT problems as they only support one variable value.

    Raises:
        ProblemError: When one of many issues occur while building the MOProblem
            instance.

    Returns:
        MOProblem: The test problem object
    """
    problems = {
        "WFG1": wfg.WFG1,
        "WFG2": wfg.WFG2,
        "WFG3": wfg.WFG3,
        "WFG4": wfg.WFG4,
        "WFG5": wfg.WFG5,
        "WFG6": wfg.WFG6,
        "WFG7": wfg.WFG7,
        "WFG8": wfg.WFG8,
        "WFG9": wfg.WFG9,

    }

    if not (name in problems.keys()):
        msg = "Specified Problem not yet supported.\n The supported problems are:" + str(problems.keys())
        raise ProblemError(msg)
    if "WFG" in name:
        if (n_of_variables is None) or (n_of_objectives is None):
            msg = "Please provide both number of variables and objectives" + " for the DTLZ problems"
            raise ProblemError(msg)
        if n_of_objectives == 2:
            k = 4
        else:
            k= k = 2 * (n_of_objectives - 1)
        obj_func = problems[name](n_of_objectives, n_of_variables, k)
    else:
        msg = "How did you end up here?"
        raise ProblemError(msg)


    lower_limits = obj_func.min_bounds
    upper_limits = obj_func.max_bounds
    var_names = ["x" + str(i + 1) for i in range(n_of_variables)]
    obj_names = ["f" + str(i + 1) for i in range(n_of_objectives)]
    variables = variable_builder(
        names=var_names,
        initial_values=lower_limits,
        lower_bounds=lower_limits,
        upper_bounds=upper_limits,
    )

    # Because optproblems can only handle one objective at a time
    def modified_obj_func(x):
        if isinstance(x, list):
            if len(x) == n_of_variables:
                return [obj_func(x)]
            elif len(x[0]) == n_of_variables:
                return list(map(obj_func, x))
        else:
            if x.ndim == 1:
                return [obj_func(x)]
            elif x.ndim == 2:
                return list(map(obj_func, x))
        raise TypeError("Unforseen problem, contact developer")

    objective = VectorObjective(name=obj_names, evaluator=modified_obj_func)
    problem = MOProblem([objective], variables, None)
    return problem