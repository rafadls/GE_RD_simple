import numpy as np
np.seterr(all="raise")
import numexpr as ne
import time

from algorithm.parameters import params
from utilities.fitness.math_functions import *
from utilities.fitness.optimize_constants import *
from utilities.fitness.error_metric import *

from fitness.ModeloBaterias.funciones_fitness_java import check_invalid_result, get_matrix_error, fit_ranking, get_error_by_column, fit_to_cells
from fitness.ModeloBaterias.funciones_fitness_java import standarize_datasets, load_ansys_data, expand_ansys_data, create_array_result
from fitness.base_ff_classes.base_ff import base_ff
from fitness.ModeloBaterias.funcionesEvaluar_ModeloJava import *

import os
from contextlib import contextmanager
import tblib.pickling_support
import sys
import math
from copy import deepcopy

class fitness_modelo_java_curvas(base_ff):
    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()
        df_best = pd.read_csv("../best.csv")
        self.phenotype_cdrag = df_best.values[0,0]
        self.phenotype_ff = df_best.values[0,1]
        self.phenotype_n = df_best.values[0,2]

        self.default_fitness = math.nan
        #Se nesecita para quitar los dataset de la libreria
        self.training_in, self.training_exp, self.test_in, self.test_exp = [1, 1, 1, 1]
        self.data_Current_in  = pd.read_csv("../datasets/ModeloFenomenologicoCurvas/" + 'Current' +"_in.txt", sep=' ')
        self.data_Current_out = pd.read_csv("../datasets/ModeloFenomenologicoCurvas/" + 'Current' +"_out.txt", sep=' ')
        self.data_K_in  = pd.read_csv("../datasets/ModeloFenomenologicoCurvas/" + 'K' +"_in.txt", sep=' ')
        self.data_K_out = pd.read_csv("../datasets/ModeloFenomenologicoCurvas/" + 'K' +"_out.txt", sep=' ')
        self.data_Flujo_in  = pd.read_csv("../datasets/ModeloFenomenologicoCurvas/" + 'Flujo' +"_in.txt", sep=' ')
        self.data_Flujo_out = pd.read_csv("../datasets/ModeloFenomenologicoCurvas/" + 'Flujo' +"_out.txt", sep=' ')
        self.data_t_viento_in  = pd.read_csv("../datasets/ModeloFenomenologicoCurvas/" + 't_viento' +"_in.txt", sep=' ')
        self.data_t_viento_out = pd.read_csv("../datasets/ModeloFenomenologicoCurvas/" + 't_viento' +"_out.txt", sep=' ')
        self.data_Diametro_in  = pd.read_csv("../datasets/ModeloFenomenologicoCurvas/" + 'Diametro' +"_in.txt", sep=' ')
        self.data_Diametro_out = pd.read_csv("../datasets/ModeloFenomenologicoCurvas/" + 'Diametro' +"_out.txt", sep=' ')
        self.target_Current = create_array_result(self.data_Current_out)
        self.target_K = create_array_result(self.data_K_out)
        self.target_Flujo = create_array_result(self.data_Flujo_out)
        self.target_t_viento = create_array_result(self.data_t_viento_out)
        self.target_Diametro = create_array_result(self.data_Diametro_out)
        
        self.n_vars = params["CODON_SIZE"]
        self.length_data = len(self.data_Current_in)
        self.max_columns = int(np.max(self.data_Current_in['col_celda']))
        self.errors_by_column_vf = []
        self.errors_by_column_pf = []
        self.errors_by_column_tc = []
        for i in range(self.max_columns):
            self.errors_by_column_vf.append([])
            self.errors_by_column_pf.append([])
            self.errors_by_column_tc.append([])
        self.cache = {}
        self.max_cache = 100000


    def evaluate(self, ind, **kwargs):
        optimize = kwargs['optimization']
        if optimize:
            custom_optimize_constants2(ind, actualizeGenome=True)    
        else:
            if ind.phenotype_original=='':
                ind.phenotype_original = str(ind.phenotype)
            zipped = get_consts(ind.phenotype)
            if len(zipped) != 0:
                acc_values, init_values, step_values, last_values, constantes = zip(*zipped)
                acc_values = list(acc_values)
                init_values = list(init_values)
                step_values = list(step_values)
                last_values = list(last_values)
                constantes = list(constantes)
                ind.phenotype = replace_consts_no_assumption(ind.phenotype, acc_values, init_values, step_values, last_values, constantes)
        if params["COEFICIENTE"] == 1:
            check_result, fitness = self.fitness_stringPhenotype([ind.phenotype, self.phenotype_ff, self.phenotype_n])
        if params["COEFICIENTE"] == 2:
            check_result, fitness = self.fitness_stringPhenotype([self.phenotype_cdrag, ind.phenotype, self.phenotype_n])
        if params["COEFICIENTE"] == 3:
            check_result, fitness = self.fitness_stringPhenotype([self.phenotype_cdrag, self.phenotype_ff, ind.phenotype])
        return check_result, fitness


    def fitness_stringPhenotype(self, final_trees):
        try: 
            modelResult_Current = eval_allData_multicore(self.data_Current_in, final_trees)
            modelResult_K = eval_allData_multicore(self.data_K_in, final_trees)
            modelResult_Flujo = eval_allData_multicore(self.data_Flujo_in, final_trees)
            modelResult_t_viento = eval_allData_multicore(self.data_t_viento_in, final_trees)
            modelResult_Diametro = eval_allData_multicore(self.data_Diametro_in, final_trees)

            matrix_error_current = get_matrix_error(modelResult_Current, self.target_Current)
            matrix_error_K = get_matrix_error(modelResult_K, self.target_K)
            matrix_error_Flujo = get_matrix_error(modelResult_Flujo, self.target_Flujo)
            matrix_error_t_viento = get_matrix_error(modelResult_t_viento, self.target_t_viento)
            matrix_error_Diametro = get_matrix_error(modelResult_Diametro, self.target_Diametro)

            fitness_calculado_current  = fit_to_cells(matrix_error_current)
            fitness_calculado_K = fit_to_cells(matrix_error_K)
            fitness_calculado_Flujo = fit_to_cells(matrix_error_Flujo)
            fitness_calculado_t_viento = fit_to_cells(matrix_error_t_viento)
            fitness_calculado_Diametro = fit_to_cells(matrix_error_Diametro)

            fitness_calculado = math.sqrt( fitness_calculado_current**2 + fitness_calculado_K**2 + fitness_calculado_Flujo**2 + fitness_calculado_t_viento**2 + fitness_calculado_Diametro**2)
            check = True
        except Exception as e:
            if str(e) == "JVM exception occurred: java.lang.IllegalStateException: invalid ind: Vf is NaN or InF":
                pass
            elif str(e) == "JVM exception occurred: invalid ind: Vf is NaN or InF":
                pass
            elif str(e) == "JVM exception occurred: java.lang.IllegalStateException: invalid ind: Vinit is negative":
                pass
            elif str(e) == "JVM exception occurred: invalid ind: Vinit is negative":
                pass
            else:
                print(e)
            fitness_calculado = math.inf
            check = False
        return check, fitness_calculado


