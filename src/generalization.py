#! /usr/bin/env python

# PonyGE2
# Copyright (c) 2017 Michael Fenton, James McDermott,
#                    David Fagan, Stefan Forstenlechner,
#                    and Erik Hemberg
# Hereby licensed under the GNU GPL v3.
""" Python GE implementation """

from utilities.algorithm.general import check_python_version

check_python_version()

from stats.stats import get_stats
from utilities.stats import trackers
from algorithm.parameters import params, set_params
import sys
import time
import json
import os
import itertools
import subprocess
import pandas as pd
import numpy as np
from utilities.stats.stats_in_excel import list2dic, saveGenerationAsExcel
import shutil
import matplotlib.pyplot as plt
from algorithm.parameters import params

from fitness.funciones_fitness import eval_allData, get_data, check_correlation, check_minimum_fitness, get_all_data
from fitness.fitness_modelo import fitness_modelo
from representation import individual
# Se obtiene el path relativo
mainPath = os.path.abspath("..")

data_25, data_53, data_74, data_102 = get_all_data()

fitness_function = fitness_modelo()

ind = individual.Individual([1,1,1,1,1,1,2,22,2,2,2,2,2,2,2,23,3,3,33,3,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], None)

### 25 ####
fitness_function.data_in, fitness_function.target  =  data_25.iloc[:,:-1], data_25.iloc[:,-1].values
ckeck_result, fitness_25 = fitness_function.evaluate(ind)
print('fitness_25: ' + str(fitness_25))
### 53 ####
fitness_function.data_in, fitness_function.target  =  data_53.iloc[:,:-1], data_53.iloc[:,-1].values
ckeck_result, fitness_53 = fitness_function.evaluate(ind)
print('fitness_53: ' + str(fitness_53))
### 74 ####
fitness_function.data_in, fitness_function.target  =  data_74.iloc[:,:-1], data_74.iloc[:,-1].values
ckeck_result, fitness_74 = fitness_function.evaluate(ind)
print('fitness_74: ' + str(fitness_74))
### 102 ####
fitness_function.data_in, fitness_function.target  =  data_102.iloc[:,:-1], data_102.iloc[:,-1].values
ckeck_result, fitness_102 = fitness_function.evaluate(ind)
print('fitness_102: ' + str(fitness_102))