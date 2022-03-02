import numpy as np
import pandas as pd

np.seterr(all="raise")
import numexpr as ne
import time

from algorithm.parameters import params
from utilities.fitness.math_functions import *
from utilities.fitness.optimize_constants import *
from utilities.fitness.error_metric import *

from fitness.funciones_fitness import eval_allData, get_data, check_correlation, check_minimum_fitness
from fitness.base_ff_classes.base_ff import base_ff

import os
from contextlib import contextmanager
import tblib.pickling_support
import sys
import math
from copy import deepcopy

from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, mean_absolute_percentage_error,r2_score,mean_poisson_deviance,mean_gamma_deviance, mean_tweedie_deviance, d2_tweedie_score, mean_pinball_loss 

class fitness_modelo(base_ff):
    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()
        self.default_fitness = math.nan
        self.data_in, self.target = get_data()

    def evaluate(self, ind, **kwargs):
        ind.phenotype_original = str(ind.phenotype)
        ind.phenotype = ind.phenotype_original.replace("&","").replace("^","**")
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
        ckeck_result, fitness = self.fitness_stringPhenotype(ind.phenotype)
        return ckeck_result,fitness


    def fitness_stringPhenotype(self, phenotype):
        try:
            y_pred = eval_allData(phenotype,self.data_in)
            if np.isnan(y_pred).any():
                return False, math.inf
            elif np.isinf(y_pred).any():
                return False, math.inf
            fitness_calculado = params['loss'](self.target,y_pred)
            if (not check_correlation(self.data_in, y_pred)):
                return False, fitness_calculado
            elif not check_minimum_fitness(fitness_calculado):
                return False, fitness_calculado
            else:
                return True, fitness_calculado
        except Exception as e:
            #print(e)
            return False, math.inf

