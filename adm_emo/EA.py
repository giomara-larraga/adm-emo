import desdeo_emo.EAs.NSGAIII as NSGAIII
import desdeo_emo.EAs.PBEA as PBEA
from DBConnection import add_objective_values


class NSGAIII_archive(NSGAIII):
    def __init__(self, *args, problem_name, phase, **kwargs):
        super().__init__(*args, **kwargs)
        self.problem_name = problem_name
        self.phase = phase
        self.reference_point = None
        #add_objective_values(self._iteration_counter,self.phase, self.problem_name, "NSGA-III", self.population.objectives, self._gen_count_in_curr_iteration)
       

    def iterate(self, preference=None):
        self.reference_point= preference
        print(self.reference_point)
        super().iterate(preference)
       
        
    def _next_gen(self):
        super()._next_gen()
        #print(len(self.population.objectives))
        #add_objective_values(self._iteration_counter,self.phase, self.problem_name, "NSGA-III", self.population.objectives, self._gen_count_in_curr_iteration)



class PBEA_archive(PBEA):
    def __init__(self, *args, archiver, **kwargs):
        super().__init__(*args, **kwargs)
        self.archiver = archiver
        self.archiver(
            self.population.objectives,
            self._current_gen_count,
        )

    def _next_gen(self):
        super()._next_gen()
        self.archiver(
            self.population.objectives,
            self._current_gen_count,
        )

