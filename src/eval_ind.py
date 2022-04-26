import sys
import time
import json
import os
import itertools
import subprocess
import pandas as pd
import numpy as np
from algorithm.parameters import params
from fitness.fitness_modelo import fitness_modelo
from fitness.ModeloBaterias.fitness_modelo_java import fitness_modelo_java
from fitness.ModeloBaterias.fitness_modelo_java_curvas import fitness_modelo_java_curvas

# Se obtiene el path relativo
mainPath = os.path.abspath("..")

df_best = pd.read_csv("../best.csv")
best_cdrag = df_best.values[0,0]
best_ff = df_best.values[0,1]
best_n = df_best.values[0,2]

df_experiments =  pd.read_csv(mainPath  + '/Experiments/data.csv')
df_individuals = pd.DataFrame()

for index, row in df_experiments.iterrows():
    dict_ind = {}
    dict_ind['cdrag'] = best_cdrag
    dict_ind['ff'] = best_ff
    dict_ind['n'] = best_n
    if row['COEFICIENTE']==1:
        dict_ind['cdrag'] = row['Best phenotype']
    if row['COEFICIENTE']==2:
        dict_ind['ff'] = row['Best phenotype']
    if row['COEFICIENTE']==3:
        dict_ind['n'] = row['Best phenotype']

    params['COEFICIENTE'] = 1
    params['N_CELDAS'] = 102
    direct_fitness_function = fitness_modelo()
    _ , dict_ind['cdrag fitness'] = direct_fitness_function.fitness_stringPhenotype(dict_ind['cdrag'].replace("^","**"))

    params['COEFICIENTE'] = 2
    params['N_CELDAS'] = 102
    direct_fitness_function = fitness_modelo()
    _ , dict_ind['ff fitness'] = direct_fitness_function.fitness_stringPhenotype(dict_ind['ff'].replace("^","**"))

    params['COEFICIENTE'] = 3
    params['N_CELDAS'] = 102
    direct_fitness_function = fitness_modelo()
    _ , dict_ind['n fitness'] = direct_fitness_function.fitness_stringPhenotype(dict_ind['n'].replace("^","**"))

    modelo_fenomenologico_fitness = fitness_modelo_java()
    dict_ind['Individual fitness'] = modelo_fenomenologico_fitness.fitness_stringPhenotype([dict_ind['cdrag'], dict_ind['ff'], dict_ind['n']])

    modelo_fenomenologico_fitness_curvas = fitness_modelo_java_curvas()
    _ ,dict_ind['Individual fitness curve'] = modelo_fenomenologico_fitness_curvas.fitness_stringPhenotype([dict_ind['cdrag'], dict_ind['ff'], dict_ind['n']])

    df_individuals = df_individuals.append(dict_ind, ignore_index=True)

df_individuals.to_csv('../individuals.csv', index=False)
    