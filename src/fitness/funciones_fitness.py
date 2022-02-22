import numpy as np
np.seterr(all="raise")
import numexpr as ne
import time

from utilities.fitness.optimize_constants import *
import os
from contextlib import contextmanager
import tblib.pickling_support
import sys
import math
from copy import deepcopy     
import pandas as pd 

from sklearn.metrics import mean_squared_error

def check_invalid_result(modelResult):
    for data_row in modelResult:
        for variable_out in data_row:
            for value in variable_out:
                if math.isinf(value) or math.isnan(value):
                    return True
    return False

def if_lower_else(a,b,c,d):
  if a<=b:
    return c
  else:
    return d

def eval_ind_cdrag(seq,ind):
  string = ind.replace('S', '{S}')
  string = string.replace('An', '{An}')
  string = string.replace('fluidColumn', '{fluidColumn}')
  string = string.replace('Rem', '{Rem}')
  string = string.replace('Dfn', '{Dfn}')
  string = string.replace('colIndex', '{colIndex}')
  resultado = eval(string.format(S=seq[0], fluidColumn=seq[1], An=seq[2], Rem=seq[3],Dfn=seq[4],colIndex=seq[5]))
  return resultado

def eval_ind_ff(seq,ind):
  string = ind.replace('S', '{S}')
  string = string.replace('Vmfn', '{Vmfn}')
  string = string.replace('Rem', '{Rem}')
  string = string.replace('Dfn', '{Dfn}')
  string = string.replace('colIndex', '{colIndex}')
  resultado = eval(string.format(S=seq[0], Vmfn=seq[1], Rem=seq[2],Dfn=seq[3],colIndex=seq[4]))
  return resultado

def eval_ind_n(seq,ind):
  string = ind.replace('S', '{S}')
  string = string.replace('Rem', '{Rem}')
  string = string.replace('Prandtl', '{Prandtl}')
  string = string.replace('colIndex', '{colIndex}')
  resultado = eval(string.format(S=seq[0], Rem=seq[1], Prandtl=seq[2],colIndex=seq[3]))
  return resultado


def evaluar_cdrag(exp, df):
  resultado = list(map(lambda x: eval_ind_cdrag(x,exp), df.values))
  return np.asarray(resultado)

def evaluar_ff(exp, df):
  resultado = list(map(lambda x: eval_ind_ff(x,exp), df.values))
  return np.asarray(resultado)

def evaluar_n(exp, df):
  resultado = list(map(lambda x: eval_ind_n(x,exp), df.values))
  return np.asarray(resultado)

def eval_allData(exp, df):
  if params['COEFICIENTE']==1:
    return evaluar_cdrag(exp, df)
  elif params['COEFICIENTE']==2:
    return evaluar_ff(exp, df)
  elif params['COEFICIENTE']==3:
    return evaluar_n(exp, df)

def check(dic, corrmat_pred, col):
  sentido_fisico = []
  no_sentido_fisico = []
  for param in list(dic.keys()):
    if corrmat_pred[col][param] * dic[param] > 0:
      sentido_fisico.append(param)
    else:
      no_sentido_fisico.append(param)
  return len(no_sentido_fisico) == 0

def RMSE(y,y_pred):
  mse = mean_squared_error(y, y_pred)
  rmse = np.sqrt(mse)
  return rmse

def check_correlation(ind):
  if not params["Correlation"]:
    return True
  ind_ph = deepcopy(ind.phenotype)
  ind_ph = str(ind_ph)
  ind_ph = ind_ph.replace("&","")
  ind_ph = ind_ph.replace("^","**")
  zipped = get_consts(ind_ph)
  if len(zipped) != 0:
      acc_values, init_values, step_values, last_values, constantes = zip(*zipped)
      acc_values = list(acc_values)
      init_values = list(init_values)
      step_values = list(step_values)
      last_values = list(last_values)
      constantes = list(constantes)
      ind_ph = replace_consts_no_assumption(ind_ph, acc_values, init_values, step_values, last_values, constantes)
  df_cdrag, df_ff, df_n = get_data_correlation()
  if params['COEFICIENTE'] == 1:
    try:
      df_cdrag['cdrag_pred'] = evaluar_cdrag(ind_ph, df_cdrag)
      corrmat = df_cdrag.corr()
      corrmat_cdrag_pred = corrmat[['cdrag_pred']]
      dic_cdrag = {'S':	0.476021,'colIndex':	0.252838,'fluidColumn':	0.114454,'Rem':	-0.267775,'Dfn':	-0.292734,'An':	-0.481160}
      return check(dic_cdrag, corrmat_cdrag_pred, 'cdrag_pred')
    except:
      return False
  elif params['COEFICIENTE'] == 2:
    try:
      df_ff['ff_pred'] = evaluar_ff(ind_ph, df_ff)
      corrmat = df_ff.corr()
      corrmat_ff_pred = corrmat[['ff_pred']]
      dic_ff = {'colIndex':	0.180658,'Dfn':	0.162611,'Rem':	-0.146201,'Vmfn':	-0.843099,'S':	-0.853635}
      return check(dic_ff, corrmat_ff_pred, 'ff_pred')
    except:
      return False
  elif params['COEFICIENTE'] == 3:
    try:
      df_n['n_pred'] = evaluar_n(ind_ph, df_n)
      corrmat = df_n.corr()
      corrmat_n_pred = corrmat[['n_pred']]
      dic_n = {'Rem':	0.860579,'Prandtl':	-0.024847,'S':	-0.113077,'colIndex':	-0.318418}
      return check(dic_n, corrmat_n_pred, 'n_pred')
    except:
      return False


def get_data_correlation():
  path = '../datasets/DATASET_correlation/'
  df_cdrag = pd.read_csv(path + 'df_cdrag.txt')
  df_ff = pd.read_csv(path + 'df_ff.txt')
  df_n = pd.read_csv(path + 'df_n.txt')
  return df_cdrag, df_ff, df_n