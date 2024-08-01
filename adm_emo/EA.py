import desdeo_emo.EAs.AutoNSGAIII as AutoNSGAIII
import desdeo_emo.EAs.AutoPBEA as AutoPBEA
from phi import phi

def compute_metric_learning(phi_learning, data, reference_point, nadir):
    results = phi_learning.get_phi(data, reference_point, nadir)
    return results[0]

class NSGAIII_archive(AutoNSGAIII):
    def __init__(self, *args, problem_name, phase, **kwargs):
        super().__init__(*args, **kwargs)
        self.problem_name = problem_name
        self.phase = phase
        self.reference_point = None
        self.phi_learning = None
        self.phi_learning_values = []
        if phase == "L":
            self.phi_learning = phi(self.population.problem.ideal)
        #add_objective_values(self._iteration_counter,self.phase, self.problem_name, "NSGA-III", self.population.objectives, self._gen_count_in_curr_iteration)
       

    def set_ref_point(self, ref_point):
        self.reference_point = ref_point
       
        
    def _next_gen(self):
        self.phi_learning_values.append(compute_metric_learning(self.phi_learning,self.population.objectives, self.reference_point,self.population.problem.nadir))
        super()._next_gen()
        #print("Reference point",self.reference_point)
        #print(len(self.population.objectives))
        #add_objective_values(self._iteration_counter,self.phase, self.problem_name, "NSGA-III", self.population.objectives, self._gen_count_in_curr_iteration)



class PBEA_archive(AutoPBEA):
    def __init__(self, *args, problem_name, phase, **kwargs):
        super().__init__(*args, **kwargs)
        self.problem_name = problem_name
        self.phase = phase
        self.reference_point = None
        self.phi_learning = None
        self.phi_learning_values = []
        if phase == "L":
            self.phi_learning = phi(self.population.problem.ideal)

    def set_ref_point(self, ref_point):
        self.reference_point = ref_point

    def _next_gen(self):
        self.phi_learning_values.append(compute_metric_learning(self.phi_learning,self.population.objectives, self.reference_point,self.population.problem.nadir))
        super()._next_gen()

