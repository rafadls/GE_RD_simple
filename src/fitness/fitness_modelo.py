import numpy as np
import pandas as pd

np.seterr(all="raise")
import numexpr as ne
import time

from algorithm.parameters import params
from utilities.fitness.math_functions import *
from utilities.fitness.optimize_constants import *
from utilities.fitness.error_metric import *

from fitness.funciones_fitness import eval_allData, RMSE
from fitness.base_ff_classes.base_ff import base_ff

import os
from contextlib import contextmanager
import tblib.pickling_support
import sys
import math
from copy import deepcopy

class fitness_modelo(base_ff):
    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()
        self.default_fitness = math.nan
        self.data_in  = pd.read_csv(params['DATASET_TRAIN']).iloc[:,:-1]
        self.target = pd.read_csv(params['DATASET_TRAIN']).iloc[:,-1].values

    def evaluate(self, ind, **kwargs):
        ind.phenotype_original = str(ind.phenotype)
        ind.phenotype = ind.phenotype_original.replace("&","").replace("^","**")
        #try:
        optimize = kwargs['optimization']
        if optimize:
            print('optimizando')
            custom_optimize_constants2(self.data_in, self.target, ind, actualizeGenome=True)    
        else:
            zipped = get_consts(ind.phenotype)
            #print(ind.phenotype)
            #print(zipped)
            #print()
            if len(zipped) != 0:
                acc_values, init_values, step_values, last_values, constantes = zip(*zipped)
                acc_values = list(acc_values)
                init_values = list(init_values)
                step_values = list(step_values)
                last_values = list(last_values)
                constantes = list(constantes)
                ind.phenotype = replace_consts_no_assumption(ind.phenotype, acc_values, init_values, step_values, last_values, constantes)
        fitness = self.fitness_stringPhenotype(ind.phenotype)
        #print(fitness)
        return fitness
        #except Exception as e:
        #    print(e)
        #    return math.nan


    def fitness_stringPhenotype(self, phenotype):
        try: 
            y_pred = eval_allData(phenotype,self.data_in)
            if np.isnan(y_pred).any() or np.isinf(y_pred).any():
                fitness_calculado = math.inf
                return fitness_calculado
            fitness_calculado = RMSE(self.target,y_pred)
        except Exception as e:
            #print(e)
            fitness_calculado = math.inf
        return fitness_calculado


