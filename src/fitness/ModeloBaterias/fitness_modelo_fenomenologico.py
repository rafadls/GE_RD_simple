import numpy as np
np.seterr(all="raise")
import numexpr as ne
import time

from algorithm.parameters import params
from utilities.fitness.math_functions import *
from utilities.fitness.optimize_constants import *
from utilities.fitness.error_metric import *

from fitness.base_ff_classes.base_ff import base_ff

from fitness.ModeloBaterias.funciones_modelo_fenomenologico import check_invalid_result, get_matrix_error, fit_ranking, get_error_by_column, eval_allData_python, eval_dataset_python, standarize_datasets, load_ansys_data, expand_ansys_data, create_array_result
from fitness.ModeloBaterias.model import *

import pandas as pd
import os
from contextlib import contextmanager
import tblib.pickling_support
import sys
import math
from copy import deepcopy

class fitness_modelo_fenomenologico(base_ff):
    def __init__(self, num_celdas=53):
        # Initialise base fitness function class.
        super().__init__()
        self.default_fitness = math.nan
        self.default_mse = [math.nan, math.nan, math.nan]
        #Se nesecita para quitar los dataset de la libreria
        self.training_in, self.training_exp, self.test_in, self.test_exp = [1, 1, 1, 1]

        self.data_in  = pd.read_csv("../datasets/ModeloFenomenologicoBaterias/" + str(num_celdas) +"_in.txt", sep=' ')
        self.data_out = pd.read_csv("../datasets/ModeloFenomenologicoBaterias/" + str(num_celdas) +"_out.txt", sep=' ')
        self.target = create_array_result(self.data_out)

        self.n_vars = params["CODON_SIZE"]
        self.length_data = len(self.data_in)
        self.max_columns = int(np.max(self.data_in['col_celda']))
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
        ind.phenotype_original = str(ind.phenotype)
        ind.phenotype = ind.phenotype_original.replace("&","")

        optimize = kwargs['optimization']
        if optimize:
            custom_optimize_constants2(self.data_in, self.target, ind, actualizeGenome=True)    
        else:
            zipped = get_consts(ind.phenotype)
            if len(zipped) != 0:
                acc_values, init_values, step_values, last_values, constantes = zip(*zipped)
                acc_values = list(acc_values)
                init_values = list(init_values)
                step_values = list(step_values)
                last_values = list(last_values)
                constantes = list(constantes)
                ind.phenotype = replace_consts_no_assumption(ind.phenotype, acc_values, init_values, step_values, last_values, constantes)
        [fitness, mse, invalid_type] = self.fitness_stringPhenotype(ind.phenotype)
        return [fitness, mse, invalid_type]


    def fitness_stringPhenotype(self, phenotype):
        try:
            final_trees = phenotype.split(";")
            modelResult = eval_allData_python(self.data_in, final_trees)
            matrix_error = get_matrix_error(modelResult, self.target)
            fitness_calculado, mse = fit_ranking(matrix_error,0.9)
            return fitness_calculado
        except Exception as e:
            print(e)
            return np.inf


