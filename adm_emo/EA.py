import desdeo_emo.EAs.NSGAIII as NSGAIII
import desdeo_emo.EAs.PBEA as PBEA
import pandas as pd
import numpy as np
import sqlite3

def add_objective_values(conn, solution, alg_name, n_gen):
    for s in solution:
        row = np.array([alg_name, n_gen])
        row = np.concatenate((row, s))
        sql = ''' INSERT INTO Problem(method,gen,f1,f2,f3)
                VALUES(?,?,?,?,?) '''
        cur = conn.cursor()
        cur.execute(sql, row)
        conn.commit()
        return cur.lastrowid


def archiver(alg_name: str, prob_name:str):
    """Archives the results of the optimization process.

    Args:
        file_path (str): The path to the file where the results are to be stored. Do not add the file extension.
    """

    def archive(
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
        conn = sqlite3.connect('database.db') 
        #print(n_gen,"-",np.shape(obj))
        add_objective_values(conn, obj, alg_name, n_gen)


    return archive


class NSGAIII_archive(NSGAIII):
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

