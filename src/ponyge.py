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
from fitness.funciones_fitness import eval_allData, get_data, check_correlation, check_minimum_fitness, get_all_data
from fitness.fitness_modelo import fitness_modelo
from representation import individual


# Se obtiene el path relativo
mainPath = os.path.abspath("..")

# Parametros
def obtenerVariasConfiguraciones():
    path_file = os.sep.join([mainPath, "parameters", "listaDeParametros.json"])
    global variableParams
    with open(path_file, 'r') as json_file:
        variableParams = json.load(json_file)["variableParams"]

    #obtengo configuraciones posibles
    listOfConfigurations = []
    for key in variableParams:
        listOfConfigurations.append([])
        for value in variableParams[key]:
            listOfConfigurations[-1].append((key, value))
    configurations = list(itertools.product(*listOfConfigurations))
    return configurations

def correrParametrosVarios(configurations):
    dictionary = {}
    newFile = os.sep.join([mainPath, "parameters", "parametrosVariables.txt"])

    #modifico el diccionario
    numberOfConfigurations = len(configurations)
    contadorConfiguraciones = 1
    contadorErrores = 0
    configuracionesConErrores = []
    configuracionesRealizadas = []
    df = pd.DataFrame() #dataframe de configuraciones
    for configuration in configurations:
        dic_configuration = list2dic(configuration) #dataframe de configuración
        df = df.append(dic_configuration,ignore_index=True)

    try:
        shutil.rmtree(mainPath + '/Experiments')
        os.makedirs(mainPath + '/Experiments')
    except:
        pass

    try:
        shutil.rmtree(mainPath + '/results')
    except:
        pass

    try:
        os.makedirs(mainPath + '/results')
    except:
        pass

    df.to_csv(mainPath + '/Experiments/data.csv', index=False) 

    for configuration in configurations:
        for key, value in configuration:
            dictionary[key] = value
            with open(newFile, 'w') as file:
                for element in dictionary:
                    file.write(element + ': ' + str(dictionary[element]) + '\n')
            with open(newFile, 'r') as file:
                newFileString = file.read()
        print("###############################################################################")
        print("###############################################################################")
        print("Corriendo configuracion numero " + str(contadorConfiguraciones) + "/" + str(numberOfConfigurations))
        print("###############################################################################")
        #print("COLOQUE CHOOSEGEN 1 EN EL CROSSOVER")
        try:
            subprocess.check_call(["python", "ponyge.py", "--parameters", newFile])
            configuracionesRealizadas.append(contadorConfiguraciones)
            with open(os.sep.join([mainPath, "estadoConfiguraciones.txt"]), "w") as state_file:
                state_file.write(str(configuracionesRealizadas) + "\n")
                state_file.write(str(configurations))
        except Exception as inst:
            print("ERROR EN LA CONFIGURACION")

            contadorErrores += 1
            configuracionesConErrores.append(contadorConfiguraciones)
        contadorConfiguraciones += 1
        print(" ")
        print(" ")
        print(" ")
    print("Proceso terminado")
    print("Numero configuraciones realizadas: " + str(numberOfConfigurations))
    print("Numero configuraciones con errores: " + str(contadorErrores))
    print("Configuraciones con errores: " + str(configuracionesConErrores)) 

    # DATA
    df =  pd.read_csv(mainPath  + '/Experiments/data.csv')
    columns_text = df.nunique()
    columns_text = columns_text[columns_text>1]
    columns_text = list(columns_text.index)
    columns_text = [x for x in columns_text if x not in ['Best fitness', 'Best phenotype', 'Total time']]

    # EVOLUTION
    def get_fitness_array(n_gen, index):
        fitness_array = []
        n_valid_array = []
        time_array = []
        duplicated_array = []
        df_inicial = pd.read_excel(mainPath + '/results/'+ str(index) +'/savedPopulations/poblacionInicial.xls')
        tiempo = df_inicial['Time'][0]
        for i in range(1,n_gen+1):
            nombre = 'generation_' + str(i) + '.xls'
            df = pd.read_excel(mainPath + '/results/'+ str(index) +'/savedPopulations/' + nombre)
            fitness_array.append(df['Fitness'].min())
            n_valid_array.append(np.sum(df['Fitness'] != np.inf)*100/len(df['Fitness']))
            time_array.append(df['Time'][0]-tiempo)
            tiempo = df['Time'][0]
            duplicated_array.append(np.sum(df[['Fenotipo']].duplicated()))
        return fitness_array, n_valid_array, time_array, duplicated_array

    # EVOLUTION
    df_fitness = pd.DataFrame()
    df_valid = pd.DataFrame()
    df_time = pd.DataFrame()
    df_duplicated = pd.DataFrame()

    for index, row in df.iterrows():
        text = ''
        for column in columns_text:
            text = text + '(' + column + ':' + str(row[column]) + ') '
        df_fitness[text], df_valid[text], df_time[text], df_duplicated[text]  = get_fitness_array(int(row['GENERATIONS']), index)

    ###################   evolusion   #####################
    fig = plt.figure(figsize=(20,20))
    fig.suptitle('Evolution',fontsize=25) 
    #########
    ax1 = plt.subplot(3,1,1)
    df_invalid.plot(ax=ax1)
    ax1.set_ylabel('Valid percentage')
    ax1.set_xlabel('Generations')
    ax1.set_title('Valid')
    #########
    ax2 = plt.subplot(3,1,2)
    df_time.plot(ax=ax2)
    ax2.set_ylabel('Time')
    ax2.set_xlabel('Generations')
    ax2.set_title('Time per generation')
    ######### 
    ax3 = plt.subplot(3,1,3)
    df_duplicated.plot(ax=ax3)
    ax3.set_ylabel('Duplicates')
    ax3.set_xlabel('Generations')
    ax3.set_title('Duplicateds per generation')
    plt.savefig(mainPath + '/Experiments/evolution.png')

    ###################   Performance   #####################
    fig = plt.figure(figsize=(20,20))
    fig.suptitle('Performance',fontsize=25) 
    #########
    ax1 = plt.subplot(2,1,1) # fig.add_axes([0,0.3,1,0.3])
    df_fitness.plot(ax=ax1)
    ax1.set_ylabel('Fitness')
    ax1.set_xlabel('Generations')
    ax1.set_title('Fitness')
    ########
    ax2 = plt.subplot(2,1,2) #fig.add_axes([0,1,1,0.6]) # [left, bottom, width, height]  
    x = df[['Best fitness']].values
    y = df[['Total time']].values
    ax2.scatter(x,y)
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    for index, row in df.iterrows():
        text = ''
        for column in columns_text:
            text = text + '(' + column + ':' + str(row[column]) + ') '
        ax2.annotate(text, (row['Best fitness'], row['Total time']))  
    ax2.set_ylabel('Total time')
    ax2.set_xlabel('Best fitness')
    ax2.set_title('Performance')

    for index, row in df.iterrows():
        text = ''
        for column in columns_text:
            text = text + '(' + column + ':' + str(row[column]) + ') '
        ax2.annotate(text, (row['Best fitness'], row['Total time']))    

    ### Save multi plot
    plt.savefig(mainPath + '/Experiments/performance.png')

    ########################## Generalization ########################    
    data_25, data_53, data_74, data_102 = get_all_data()

    fitness_function = fitness_modelo()
    text_array = []
    array_25 = []
    array_53 = []
    array_74 = []
    array_102 = []
    for index, row in df.iterrows():
        phenotype = row['Best phenotype']
        text = ''
        for column in columns_text:
            text = text + '(' + column + ':' + str(row[column]) + ') '
        
        params['COEFICIENTE']=row['COEFICIENTE']
        print(params['COEFICIENTE'])
        data_25, data_53, data_74, data_102 = get_all_data()
        ### 25 ####
        fitness_function.data_in, fitness_function.target  =  data_25.iloc[:,:-1], data_25.iloc[:,-1].values
        check_result, fitness_25 = fitness_function.fitness_stringPhenotype(phenotype)
        print('fitness_25: ' + str(fitness_25))
        ### 53 ####
        fitness_function.data_in, fitness_function.target  =  data_53.iloc[:,:-1], data_53.iloc[:,-1].values
        check_result, fitness_53 = fitness_function.fitness_stringPhenotype(phenotype)
        print('fitness_53: ' + str(fitness_53))
        ### 74 ####
        fitness_function.data_in, fitness_function.target  =  data_74.iloc[:,:-1], data_74.iloc[:,-1].values
        check_result, fitness_74 = fitness_function.fitness_stringPhenotype(phenotype)
        print('fitness_74: ' + str(fitness_74))
        ### 102 ####
        fitness_function.data_in, fitness_function.target  =  data_102.iloc[:,:-1], data_102.iloc[:,-1].values
        check_result, fitness_102 = fitness_function.fitness_stringPhenotype(phenotype)
        print('fitness_102: ' + str(fitness_102))
        text_array.append(text)
        array_25.append(fitness_25)
        array_53.append(fitness_53)
        array_74.append(fitness_74)
        array_102.append(fitness_102)
    
    fig = plt.figure(figsize=(12,12))
    fig.suptitle('Generalization',fontsize=25)    
    df_generalization = pd.DataFrame({'fitness 25 celdas': array_25,'fitness 53 celdas': array_53,'fitness 74 celdas': array_74,'fitness 102 celdas': array_102}, index=text_array)
    ax = df_generalization.plot.bar(rot=0)
    ax.set_ylabel('Fitness per distribution')
    ax.set_xlabel('Configuration')
    ax.set_yscale('log')
    ax.set_title('Generalization')
    plt.savefig(mainPath + '/Experiments/generalization.png')



def mane():
    """ Run program """
    # Run evolution
    individuals = params['SEARCH_LOOP']()

    # Print final review
    get_stats(individuals, end=True)
    saveGenerationAsExcel(individuals, params['FILE_PATH'], "poblacion_final.xls")


if __name__ == "__main__":
    if sys.argv[1] == "--variable":
        differentConfigurations = obtenerVariasConfiguraciones()
        correrParametrosVarios(differentConfigurations)
    else:
        set_params(sys.argv[1:])
        params["start-time"] = time.time()
        mane()
