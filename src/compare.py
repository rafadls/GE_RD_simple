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

from algorithm.parameters import params
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, mean_absolute_percentage_error,r2_score,mean_poisson_deviance,mean_gamma_deviance, mean_tweedie_deviance, d2_tweedie_score, mean_pinball_loss 
from fitness.funciones_fitness import eval_allData , eval_all_data_modeloFenomenologico, save_graph_data_outputs
from fitness.funciones_fitness import get_data, get_data_to_curves, prepro
from fitness.funciones_fitness import df_from_output, get_df_to_plot, get_dataFrame, get_data_simple, compare, eval_data
from fitness.ModeloBaterias.funcionesEvaluar_ModeloJava import eval_allData_multicore
from fitness.ModeloBaterias.fitness_modelo_java import fitness_modelo_java
from fitness.fitness_modelo import fitness_modelo

# Se obtiene el path relativo
mainPath = os.path.abspath("..")
path_compare = mainPath + '/compare/'
path_compare_direct = path_compare + 'direct/'
path_compare_outputs = path_compare + 'outputs/'
path_compare_curves = path_compare + 'curves/'
path_compare_generalization = path_compare + 'generalization/'

df = pd.read_csv(mainPath + '/datasets/Compare/individuals.csv')
n_individuals = len(df)
coeficientes_array = ['Coeficiente de arrastre', 'Factor de fricción','Número de Nusselt']
coeficientes_array_short = ['cdrag', 'ffactor','nusselt']

try:
    shutil.rmtree(path_compare)
except:
    pass
try:
    os.makedirs(path_compare)
    os.makedirs(path_compare_direct)
    os.makedirs(path_compare_outputs)
    os.makedirs(path_compare_generalization)
    for intem in coeficientes_array_short:
        os.makedirs(path_compare_curves + intem + '/')
except:
    pass

print()
###### ERROR COEFICIENTES POR COLUMNA ######
print('ERROR COEFICIENTES POR COLUMNA')

for j in range(len(coeficientes_array)):
    print(coeficientes_array[j])
    params['COEFICIENTE'] = j+1
    params['N_CELDAS'] = 25
    X, y = get_data_to_curves()
    df_to_eval = prepro(X)
    X['colIndex'] = X['colIndex'].astype(str)
    X = X[['colIndex']]
    for index, row in df.iterrows():
        y_pred = eval_allData(row[coeficientes_array_short[j]].replace('^','**'), df_to_eval)
        error = np.abs(y_pred - y)
        X[row['Name']] = error
    X = X.groupby('colIndex').mean()
    X.plot(style='o')
    plt.title('Error de ' + coeficientes_array[j] + ' por columna')
    plt.ylabel('Error')
    plt.xlabel('Columna')
    plt.savefig(path_compare_direct + coeficientes_array_short[j] + '.png')

print()
###### CURVAS COEFICIENTES VS INPUTS ######
print('CURVAS COEFICIENTES VS INPUTS')

for j in range(len(coeficientes_array)):
    print(coeficientes_array[j])
    params['COEFICIENTE'] = j+1
    params['N_CELDAS'] = 25
    X, y = get_data_to_curves()
    df_to_eval = prepro(X)
    n_columnas = len(X['colIndex'].unique())
    len_dataset = len(X)
    # Calcular valores por individuo
    X['ANSYS'] = y
    for index, row in df.iterrows():
        X[row['Name']] = eval_allData(row[coeficientes_array_short[j]].replace('^','**'), df_to_eval)
    # Crear dataframe con resultados por columna
    df_output = pd.DataFrame()
    for i in range(len_dataset//n_columnas):
        df_aux = X.iloc[i:i+n_columnas,:]
        dictionary = dict(X.iloc[i,:-2])
        for index, row in df_aux.iterrows():
            dictionary['ANSYS_' + str(int(row['colIndex']))] = row['ANSYS']
            for index_i, row_i in df.iterrows():
                dictionary[row_i['Name'] + '_' + str(int(row['colIndex']))] = row[row_i['Name']]
        df_output = df_output.append(dictionary, ignore_index=True)
    df_output = df_output.dropna()
    # numero de individuos
    n_individuals = len(df)
    # obtener comparacion
    for input_model in ['Current','K','Flujo','t_viento','Diametro']:
        fig, axis = plt.subplots(n_individuals + 1,1, figsize=(18,n_individuals*7))
        fig.suptitle(coeficientes_array[j] + ' vs ' + input_model,fontsize=25)
        df_curva = get_df_to_plot(df_output,input_model, 'ANSYS_').drop_duplicates(subset=input_model)
        columns_to_plot = list(df_curva.columns)
        columns_to_plot.remove(input_model)
        df_curva.plot(x=input_model, y=columns_to_plot, ax=axis[0])
        axis[0].set_title('ANSYS',fontsize=15)
        axis[0].set_xlabel(input_model)
        axis[0].set_ylabel(coeficientes_array[j])
        for index_i, row_i in df.iterrows():
            df_curva = get_df_to_plot(df_output,input_model,row_i['Name'] +'_').drop_duplicates(subset=input_model)
            columns_to_plot = list(df_curva.columns)
            columns_to_plot.remove(input_model)
            df_curva.plot(x=input_model, y=columns_to_plot, ax=axis[index_i+1])
            axis[index_i+1].set_title(row_i['Name'],fontsize=15)
            axis[index_i+1].set_xlabel(input_model)
            axis[index_i+1].set_ylabel(coeficientes_array[j])
        plt.savefig(path_compare_curves + coeficientes_array_short[j] + '/' + input_model + '.png')

print()
###### GENERALIZACIÓN ######
print('GENERALIZACIÓN')

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
    fitness_25, fitness_53, fitness_74, fitness_102 = eval_data(fitness_function,row['cdrag'].replace('^','**'),1)
    df_cdrag=df_cdrag.append({'fitness 25 celdas': fitness_25,'fitness 53 celdas': fitness_53,'fitness 74 celdas': fitness_74,'fitness 102 celdas': fitness_102},ignore_index=True)
    fitness_25, fitness_53, fitness_74, fitness_102 = eval_data(fitness_function,row['ffactor'].replace('^','**'),2)
    df_ff=df_ff.append({'fitness 25 celdas': fitness_25,'fitness 53 celdas': fitness_53,'fitness 74 celdas': fitness_74,'fitness 102 celdas': fitness_102},ignore_index=True)
    fitness_25, fitness_53, fitness_74, fitness_102 = eval_data(fitness_function,row['nusselt'].replace('^','**'),3)
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


df_ff.plot.bar(rot=0, ax=axes[1])
axes[1].set_title('Factor de fricción',fontsize=15)
axes[1].set_xlabel('Individuals')
axes[1].set_ylabel('Fitness')


df_n.plot.bar(rot=0, ax=axes[2])
axes[2].set_title('Número de Nusselt',fontsize=15)
axes[2].set_xlabel('Individuals')
axes[2].set_ylabel('Fitness')

plt.savefig(path_compare_generalization + 'coeficientes.png')

print()


###### ERROR SALIDAS ######
print('ERROR SALIDAS')


df_25 = pd.DataFrame()
df_53 = pd.DataFrame()
df_74 = pd.DataFrame()
df_102 = pd.DataFrame()
dfs = [df_25,df_53,df_74,df_102]
ns_celdas = [25,53,74,102]

for index, row in df.iterrows():
    name = row['Name']
    print(name)
    phtnotype = str(row['cdrag']) + ";" + str(row['ffactor']) + ";" +str(row['nusselt'])
    fitness_25,fitness_53,fitness_74,fitness_102 =  eval_all_data_modeloFenomenologico(phtnotype)
    if np.any(np.isinf(fitness_25)) or np.any(np.isinf(fitness_53)) or np.any(np.isinf(fitness_74)) or np.any(np.isinf(fitness_102)):
        continue
    fitness_array = [fitness_25,fitness_53,fitness_74,fitness_102]
    for i in range(len(dfs)):
        dfs[i][name + '_VF'] = fitness_array[i][0,:]
        dfs[i][name + '_PF'] = fitness_array[i][1,:]
        dfs[i][name + '_TC'] = fitness_array[i][2,:]

for i in range(len(dfs)):
    save_graph_data_outputs(dfs[i], ns_celdas[i], path_compare_outputs)

print()
######### CURVAS ###########
print('CURVAS')

dataset_output_array = []
individuals_array = []


data_in  = pd.read_csv("../datasets/ModeloFenomenologicoBaterias/" + str(102) +"_in.txt", sep=' ')
data_out = pd.read_csv("../datasets/ModeloFenomenologicoBaterias/" + str(102) +"_out.txt", sep=' ')
df_ansys = pd.concat([data_in,data_out], axis=1)

dataset_output_array.append(df_ansys)
individuals_array.append('ANSYS')

for index, row in df.iterrows():
    name = row['Name']
    print(name)
    try:
        modelResult = eval_allData_multicore(data_in, [str(row['cdrag']),str(row['ffactor']),str(row['nusselt'])])
        modelResult = np.asarray(modelResult)
        if (not np.any(np.isinf(modelResult))) and (not np.any(np.isnan(modelResult))):
            df_output = df_from_output(modelResult)
            df_output = pd.concat([data_in,df_output], axis=1)
            dataset_output_array.append(df_output)
            individuals_array.append(name)
    except Exception as e:
        print(e)

for column in ['Current', 'K', 'Flujo', 't_viento', 'Diametro']:
    compare(column, dataset_output_array, individuals_array, path_compare_curves)

print()
