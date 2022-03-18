import numpy as np
import pandas as pd
import jnius_config
import math
import os
from utilities.fitness.error_metric import *

this_file_path = os.path.abspath(__file__)
mainPath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(this_file_path)))) 
path_java_files = os.sep.join([mainPath, "src", "archivos_java"])
jnius_config.add_classpath(os.sep.join([path_java_files, "bin", "Batteries_GP_MO_Java.jar"]))

from jnius import autoclass

INDIV = autoclass('cl.ian.gp.MyGPIndividual')
MODELEV = autoclass('cl.nico.StrModelEvaluator')
#FULL_EVALUATOR = autoclass('com.company.Evaluate_allData')
FULL_EVALUATOR = autoclass('cl.born.Evaluate_allData')

indiv = INDIV()
model_evaluator = MODELEV()
eval_fullData = FULL_EVALUATOR()

def eval_allData(data_in_val, individual,
                          col_fluido,  col_celda,  n_fluido, n_celda):
    I, S, Fin, Tin, D = [], [], [], [], []
    I = data_in_val["Current"].tolist()
    S = data_in_val["K"].tolist()
    Fin = data_in_val["Flujo"].tolist()
    Tin = data_in_val["t_viento"].tolist()
    D = data_in_val["Diametro"].tolist()
    eval_fullData.evaluate(I, S, Fin, Tin, D,
                          col_fluido,  col_celda,  n_fluido, n_celda,
                          individual[0],  individual[1],  individual[2])
    return eval_fullData.get_result()

def eval_allData_multicore(data_in_val, individual):
    #I, S, Fin, Tin, D = [], [], [], [], []
    col_fluido = data_in_val["col_fluido"].tolist()
    col_celda = data_in_val["col_celda"].tolist()
    n_fluido = data_in_val["n_fluido"].tolist()
    n_celda = data_in_val["n_celda"].tolist()
    I = data_in_val["Current"].tolist()
    S = data_in_val["K"].tolist()
    Fin = data_in_val["Flujo"].tolist()
    Tin = data_in_val["t_viento"].tolist()
    D = data_in_val["Diametro"].tolist()
    list_genes = [gene.replace("Sep", "S") for gene in individual]
    eval_fullData.evaluate_multicore(I, S, Fin, Tin, D,
                          col_fluido,  col_celda,  n_fluido, n_celda,
                          list_genes)
    return eval_fullData.get_result()


def get_dataOut_formatted(dataOut_path):
    data_out = pd.read_csv(dataOut_path, sep=',', header=None)

    data_out_aux = [data_out[data_out.columns[2:3]],
                    data_out[data_out.columns[1:2]], 
                    data_out[data_out.columns[0:1]]]

    data_out_vf = np.asarray(data_out_aux[0])
    data_out_pf = np.asarray(data_out_aux[1])
    data_out_tc = np.asarray(data_out_aux[2]-273.15)
    #data_out_tc = np.asarray(data_out_aux[2])
    target = np.hstack([data_out_vf, data_out_pf, data_out_tc])
    return target

def calc_mse(modelResult, target):
    mse_vf = mse(modelResult[:][0], target[:][0])
    mse_pf = mse(modelResult[:][1], target[:][1])
    mse_tc = mse(modelResult[:][2], target[:][2])
    #mse_vf = mse(modelResult[:,0], target[:,0])
    #mse_pf = mse(modelResult[:,1], target[:,1])
    #mse_tc = mse(modelResult[:,2], target[:,2])
    return [mse_vf, mse_pf, mse_tc]

def calc_rmse(modelResult, target):
    mse = calc_mse(modelResult, target)
    return [math.sqrt(mse[0]), math.sqrt(mse[1]), math.sqrt(mse[2])]

def calc_squareError(modelResult, target):
    return np.square(modelResult - target)


def calc_mse_102celdas(modelResult, target):
    #modelResult = np.reshape(modelResult, (3,,14))

    dif_vf = (modelResult[:][0] - target[:][0])
    dif_pf = (modelResult[:][1] - target[:][1])
    dif_tc = (modelResult[:][2] - target[:][2])
    mse_vf = mse(modelResult[:,0:14], target[:,0:14])
    mse_pf = mse(modelResult[:,14:28], target[:,14:28])
    mse_tc = mse(modelResult[:,28:42], target[:,28:42])
    return [mse_vf, mse_pf, mse_tc]