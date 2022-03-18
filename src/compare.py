import sys
import time
import json
import os
import itertools
import subprocess
import pandas as pd
import numpy as np
import shutil
import matplotlib.pyplot as plt
import math


from fitness.funciones_fitness import eval_all_data_modeloFenomenologico

# Se obtiene el path relativo
mainPath = os.path.abspath("..")
path_compare = mainPath + '/compare/'


df = pd.read_csv(mainPath + '/datasets/Compare/individuals.csv')

try:
    shutil.rmtree(path_compare)
except:
    pass
try:
    os.makedirs(path_compare)
except:
    pass

###### FITNESS COEFICIENTES ######
'''
nrow=3
ncol=1
fig, axes = plt.subplots(nrow, ncol, figsize=(15,15))
fig.suptitle('Comparación de individuos',fontsize=25)
df_cdrag = pd.DataFrame()
df_ff = pd.DataFrame()
df_n= pd.DataFrame()
for index, row in df.iterrows():
    name = row['Name']
    fitness_function = fitness_modelo()
    fitness_25, fitness_53, fitness_74, fitness_102 = eval_data(fitness_function,row['cdrag'],1)
    df_cdrag=df_cdrag.append({'fitness 25 celdas': fitness_25,'fitness 53 celdas': fitness_53,'fitness 74 celdas': fitness_74,'fitness 102 celdas': fitness_102},ignore_index=True)
    fitness_25, fitness_53, fitness_74, fitness_102 = eval_data(fitness_function,row['ffactor'],2)
    df_ff=df_ff.append({'fitness 25 celdas': fitness_25,'fitness 53 celdas': fitness_53,'fitness 74 celdas': fitness_74,'fitness 102 celdas': fitness_102},ignore_index=True)
    fitness_25, fitness_53, fitness_74, fitness_102 = eval_data(fitness_function,row['nusselt'],3)
    df_n=df_n.append({'fitness 25 celdas': fitness_25,'fitness 53 celdas': fitness_53,'fitness 74 celdas': fitness_74,'fitness 102 celdas': fitness_102},ignore_index=True)
    #print(eval_all_data_modeloFenomenologico(phtnotype))
df_cdrag.index = df['Name']
df_ff.index = df['Name']
df_n.index = df['Name']

df_cdrag = df_cdrag[['fitness 25 celdas','fitness 53 celdas','fitness 74 celdas','fitness 102 celdas']]
df_ff = df_ff[['fitness 25 celdas','fitness 53 celdas','fitness 74 celdas','fitness 102 celdas']]
df_n = df_n[['fitness 25 celdas','fitness 53 celdas','fitness 74 celdas','fitness 102 celdas']]

df_cdrag.plot.bar(rot=0, ax=axes[0])
axes[0].set_title('Coeficiente de arrastre',fontsize=15)
axes[0].set_xlabel('Individuals')
axes[0].set_ylabel('Fitness')
axes[0].set_yscale('log')


df_ff.plot.bar(rot=0, ax=axes[1])
axes[1].set_title('Factor de fricción',fontsize=15)
axes[1].set_xlabel('Individuals')
axes[1].set_ylabel('Fitness')
axes[1].set_yscale('log')


df_n.plot.bar(rot=0, ax=axes[2])
axes[2].set_title('Número de Nusselt',fontsize=15)
axes[2].set_xlabel('Individuals')
axes[2].set_ylabel('Fitness')
axes[2].set_yscale('log')

plt.savefig(path_compare + 'coeficientes.png')
'''
###### FITNESS SALIDAS ######
df_25 = pd.DataFrame()
df_53 = pd.DataFrame()
df_74 = pd.DataFrame()
df_102 = pd.DataFrame()
dfs = [df_25,df_53,df_74,df_102]
for index, row in df.iterrows():
    name = row['Name']
    phtnotype = str(row['cdrag']) + ";" + str(row['ffactor']) + ";" +str(row['nusselt'])
    fitness_25,fitness_53,fitness_74,fitness_102 =  eval_all_data_modeloFenomenologico(phtnotype)
    if np.any(np.isinf(fitness_25)) or np.any(np.isinf(fitness_53)) or np.any(np.isinf(fitness_74)) or np.any(np.isinf(fitness_102)):
        continue
    fitness_array = [fitness_25,fitness_53,fitness_74,fitness_102]
    for i in range(4):
        dfs[i][name + '_VF'] = fitness_array[i][0,:]
        dfs[i][name + '_PF'] = fitness_array[i][1,:]
        dfs[i][name + '_TC'] = fitness_array[i][2,:]

print(df_53)