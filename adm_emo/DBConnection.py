import numpy as np
import sqlite3

def init_db():
    conn = sqlite3.connect('database.db') 
    return conn


def add_objective_values(iteration, phase, problem_name, alg_name, solution, gen):
    if problem_name == "VCW":
        try:
            with sqlite3.connect('database.db') as conn:
                sql = ''' INSERT INTO Problem(iteration,phase,problem,method,f1,f2,f3,gen)
                VALUES(?,?,?,?,?,?,?,?) '''
                for i in range(len(solution)):
                    cur = conn.cursor()
                    cur.execute(sql, (iteration,phase,problem_name,alg_name,solution[i,0],solution[i,1],solution[i,2],gen))
                    conn.commit()
                return cur.lastrowid
        except sqlite3.Error as e:
            print(e)
            return None

    elif problem_name == "CSI":
        try:
            with sqlite3.connect('database.db') as conn:
                sql = ''' INSERT INTO Problem(iteration,phase,problem,method,f1,f2,f3,f4,gen)
                VALUES(?,?,?,?,?,?,?,?,?) '''
                cur = conn.cursor()
                cur.execute(sql, (iteration,phase,problem_name,alg_name,solution[0],solution[1],solution[2],solution[3],gen))
                conn.commit()
                return cur.lastrowid
        except sqlite3.Error as e:
            print(e)
            return None
    elif problem_name == "RPP":
        try:
            with sqlite3.connect('database.db') as conn:
                sql = ''' INSERT INTO Problem(iteration,phase,problem,method,f1,f2,f3,f4,f5,gen)
                VALUES(?,?,?,?,?,?,?,?,?,?) '''
                cur = conn.cursor()
                cur.execute(sql, (iteration,phase,problem_name,alg_name,solution[0],solution[1],solution[2],solution[3],solution[4],gen))
                conn.commit()
                return cur.lastrowid
        except sqlite3.Error as e:
            print(e)
            return None
    else:
        print("Error inserting solution")

def get_reference_points(problem_name, phase):
    if problem_name == "VCW":
        try:
            with sqlite3.connect('database.db') as conn:
                cur = conn.cursor()
                cur.execute('select f1, f2, f3 from ReferencePoints where problem=? and phase =? order by id', (problem_name,phase))
                reference_points = cur.fetchall()
                return reference_points
        except sqlite3.Error as e:
            print(e)
            return None

    elif problem_name == "CSI":
        try:
            with sqlite3.connect('database.db') as conn:
                cur = conn.cursor()
                cur.execute('select f1, f2, f3, f4 from ReferencePoints where problem=? and phase =? order by id', (problem_name,phase))
                reference_points = cur.fetchall()
                return reference_points
        except sqlite3.Error as e:
            print(e)
            return None
    elif problem_name == "RPP":
        try:
            with sqlite3.connect('database.db') as conn:
                cur = conn.cursor()
                cur.execute('select f1, f2, f3, f4, f5 from ReferencePoints where problem=? and phase =? order by id', (problem_name,phase))
                reference_points = cur.fetchall()
                return reference_points
        except sqlite3.Error as e:
            print(e)
            return None
    else:
        print("Error getting reference points")


def get_solutions_by_gen(iteration, n_gen, problem_name, alg_name):
    if problem_name == "VCW":
        try:
            with sqlite3.connect('database.db') as conn:
                cur = conn.cursor()
                cur.execute('select f1, f2, f3 from Problem where iteration=? and problem=? and method =? and gen=? order by id', (iteration,problem_name,alg_name,n_gen))
                reference_points = cur.fetchall()
                return reference_points
        except sqlite3.Error as e:
            print(e)
            return None

    elif problem_name == "CSI":
        try:
            with sqlite3.connect('database.db') as conn:
                cur = conn.cursor()
                cur.execute('select f1, f2, f3, f4 from Problem where iteration=? and problem=? and method =? and gen=? order by id', (iteration,problem_name,alg_name,n_gen))
                reference_points = cur.fetchall()
                return reference_points
        except sqlite3.Error as e:
            print(e)
            return None
    elif problem_name == "RPP":
        try:
            with sqlite3.connect('database.db') as conn:
                cur = conn.cursor()
                cur.execute('select f1, f2, f3, f4, f5 from Problem where iteration=? and problem=? and method =? and gen=? order by id', (iteration,problem_name,alg_name,n_gen))
                reference_points = cur.fetchall()
                return reference_points
        except sqlite3.Error as e:
            print(e)
            return None
    else:
        print("Error getting reference points")

if __name__ == "__main__":
    reference_points = get_reference_points("VCW", "D")
    print(reference_points)