from phi import phi, phi_decision
import numpy as np

def compute_phi_from_list(objective_values_per_iteration, ideal, nadir, generations_per_iter, iterations_learning, iterations_decision, reference_points):
    total_iterations = iterations_learning + iterations_decision
    phi_learning = phi(ideal)

    last_rp  = reference_points[-1:]
    decision_rp = reference_points[-iterations_decision:]

    results_learning_positive = []
    results_decision_positive = []

    for iteration in range(total_iterations):
        iteration_lower_bound = iteration * generations_per_iter
        iteration_upper_bound = iteration_lower_bound + generations_per_iter
        if(iteration < iterations_learning):
            for gen in range(iteration_lower_bound,iteration_upper_bound):
                data = objective_values_per_iteration[gen]
                results_learning_positive.append(phi_learning.get_phi(data, reference_points[iteration], nadir)[0])
                #results_learning_negative.append(phi_learning.get_phi(data, reference_points[iteration], nadir)[2])
        else:
            max_gen_iter_data = objective_values_per_iteration[iteration_upper_bound-1]
            results_decision_positive.append(phi_learning.get_phi(max_gen_iter_data, reference_points[iteration], nadir)[0])
            #results_decision_negative.append(phi_learning.get_phi(max_gen_iter_data, reference_points[iteration], nadir)[2])
    
    phi_decision_phase_positive_nsga3 = phi_decision(iterations_decision, results_decision_positive, nadir)
    FD = phi_decision_phase_positive_nsga3.assess_decision_phase(set_of_RPs=np.asarray(decision_rp), main_RP=last_rp[0])[0]
    RS = np.sum(results_learning_positive)

    max_phi = 2 * generations_per_iter * iterations_learning
    RS_normalized = (RS - 1) / (max_phi- 1)

    return RS_normalized, FD