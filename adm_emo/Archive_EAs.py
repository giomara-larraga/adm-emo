from desdeo_emo.EAs.AutoRVEA import AutoRVEA
from desdeo_emo.EAs.AutoPBEA import AutoPBEA
from desdeo_emo.EAs.RNSGAII import RNSGAII
import pandas as pd
import numpy as np



def archive(
    dec: np.ndarray,
    obj: np.ndarray,
    n_gen: int,
):
    """Stores the results of the optimization process.
    Args:
        dec (np.ndarray): The decision variables.
        obj (np.ndarray): The objective variables.
        n_gen (int): The current generation.
    """
    # These could be function arguments
    dec_names = [f"x{i}" for i in range(len(dec[0]))]
    obj_names = [f"f{i}" for i in range(len(obj[0]))]
    # Create a dataframe
    dec = pd.DataFrame(dec, columns=dec_names)
    obj = pd.DataFrame(obj, columns=obj_names)
    df = pd.concat([dec, obj], axis=1)
    df["n_gen"] = n_gen
    #df.to_csv(file_path + f"_{n_gen}.csv", index=False)
    return df


class RVEA_archive(AutoRVEA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.archives = dict()
        self.archives[self._current_gen_count] = np.copy(
            self.population.objectives
        )
       
    def _next_gen(self):
        super()._next_gen()
        local_archive= np.copy(
            self.population.objectives)
        self.archives[self._current_gen_count] = local_archive

    def reset_archive(self):
        self.archives = []


class PBEA_archive(AutoPBEA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.archives = dict()
        self.archives[self._current_gen_count] = np.copy(
            self.population.objectives
        )
       
    def _next_gen(self):
        super()._next_gen()
        local_archive= np.copy(
            self.population.objectives)
        self.archives[self._current_gen_count] = local_archive

    def reset_archive(self):
        self.archives = []

class NSGAII_archive(RNSGAII):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.archives = dict()
        self.archives[self._current_gen_count] = np.copy(
            self.population.objectives
        )
       
    def _next_gen(self):
        super()._next_gen()
        local_archive= np.copy(
            self.population.objectives)
        self.archives[self._current_gen_count] = local_archive

    def reset_archive(self):
        self.archives = []