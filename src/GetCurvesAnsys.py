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
path_AA = mainPath + '/ANSYS_ANALISIS/'

coeficientes_array = ['Coeficiente de arrastre', 'Factor de fricción','Número de Nusselt']
coeficientes_array_short = ['cdrag', 'ffactor','nusselt']

translate_input = {
    'Current': 'Corriente',
    'K': 'Separación',
    'Flujo': 'Flujo',
    't_viento': 'Temperatura de ventilación',
    'Diametro': 'Diametro'
}

try:
    shutil.rmtree(path_AA)
except:
    pass
try:
    os.makedirs(path_AA)
except:
    pass

###### CURVAS COEFICIENTES VS INPUTS ######
print('CURVAS COEFICIENTES VS INPUTS')
for j in range(len(coeficientes_array)):
    fig = plt.figure(figsize=(20,20))
    fig.suptitle(coeficientes_array[j],fontsize=25)
    print(coeficientes_array[j])
    params['COEFICIENTE'] = j+1
    params['N_CELDAS'] = 25
    X, y = get_data_to_curves()
    df_to_eval = prepro(X)
    n_columnas = len(X['colIndex'].unique())
    len_dataset = len(X)
    X['ANSYS'] = y
    # Crear dataframe con resultados por columna
    df_output = pd.DataFrame()
    for i in range(len_dataset//n_columnas):
        df_aux = X.iloc[i:i+n_columnas,:]
        dictionary = dict(X.iloc[i,:-2])
        for index, row in df_aux.iterrows():
            dictionary['Columna ' + str(int(row['colIndex']))] = row['ANSYS']
        df_output = df_output.append(dictionary, ignore_index=True)
    df_output = df_output.dropna()
    # obtener comparacion
    ax1 = plt.subplot(3,2,1)
    ax2 = plt.subplot(3,2,2)
    ax3 = plt.subplot(3,2,3)
    ax4 = plt.subplot(3,2,4)
    ax5 = plt.subplot(3,1,3)
    axis = [ax1,ax2,ax3,ax4,ax5]
    count = 0
    for input_model in ['Current','K','Flujo','t_viento','Diametro']:
        df_curva = get_df_to_plot(df_output,input_model, 'Columna ').drop_duplicates(subset=input_model)
        columns_to_plot = list(df_curva.columns)
        columns_to_plot.remove(input_model)
        df_curva.plot(x=input_model, y=columns_to_plot, ax=axis[count])
        axis[count].set_title( coeficientes_array[j] + ' vs ' + translate_input[input_model],fontsize=15)
        axis[count].set_xlabel(input_model)
        axis[count].set_ylabel(coeficientes_array[j])
        count+=1
    plt.savefig(path_AA + coeficientes_array_short[j] + '.png')

print()
######### CURVAS ###########
print('CURVAS')

data_in  = pd.read_csv("../datasets/ModeloFenomenologicoBaterias/" + str(25) +"_in.txt", sep=' ')
data_out = pd.read_csv("../datasets/ModeloFenomenologicoBaterias/" + str(25) +"_out.txt", sep=' ')
df_ansys = pd.concat([data_in,data_out], axis=1)


outputs_names = ['Velocidad', 'Presión', 'Temperatura de celda']

count_out = 0
for output in ['V', 'P', 'TC']:
    fig = plt.figure(figsize=(20,20))
    fig.suptitle(outputs_names[count_out],fontsize=25)
    ax1 = plt.subplot(3,2,1)
    ax2 = plt.subplot(3,2,2)
    ax3 = plt.subplot(3,2,3)
    ax4 = plt.subplot(3,2,4)
    ax5 = plt.subplot(3,1,3)
    axis = [ax1,ax2,ax3,ax4,ax5]
    count = 0
    for column in ['Current', 'K', 'Flujo', 't_viento', 'Diametro']:
        df_i_vs_o = get_df_to_plot(df_ansys,column,output)
        columns_to_plot = list(df_i_vs_o.columns)
        columns_to_plot.remove(column)
        df_i_vs_o.plot(x=column, y=columns_to_plot, ax=axis[count])
        axis[count].set_title(outputs_names[count_out] + ' vs ' + translate_input[column],fontsize=25)
        axis[count].set_xlabel(column, fontsize=20)
        axis[count].set_ylabel(outputs_names[count_out],fontsize=20)
        count+=1
    plt.savefig(path_AA + output + '.png')
    count_out+=1
