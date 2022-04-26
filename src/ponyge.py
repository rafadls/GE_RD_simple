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
from fitness.funciones_fitness import eval_all_data, get_all_data, get_dataframes
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
    df_fitness, df_valid, df_time, df_duplicated, df_phenotype = get_dataframes(df,columns_text, mainPath)

    ###################   evolution   #####################
    fig = plt.figure(figsize=(15,15))
    fig.suptitle('Evolution',fontsize=25) 
    #########
    ax1 = plt.subplot(3,1,1)
    df_valid.plot(ax=ax1)
    ax1.set_ylabel('Valid percentage')
    ax1.set_xlabel('Generations')
    ax1.set_title('Valid individuals',fontsize=15)
    #########
    ax2 = plt.subplot(3,1,2)
    df_time.plot(ax=ax2)
    ax2.set_ylabel('Time')
    ax2.set_xlabel('Generations')
    ax2.set_title('Time per generation',fontsize=15)
    ######### 
    ax3 = plt.subplot(3,1,3)
    df_duplicated.plot(ax=ax3)
    ax3.set_ylabel('Duplicates')
    ax3.set_xlabel('Generations')
    ax3.set_title('Duplicateds per generation',fontsize=15)
    plt.savefig(mainPath + '/Experiments/evolution.png')



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
