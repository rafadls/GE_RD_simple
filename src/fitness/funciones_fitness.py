import numpy as np
np.seterr(all="raise")

from algorithm.parameters import params
from utilities.fitness.optimize_constants import *
 
import pandas as pd 
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, mean_absolute_percentage_error,r2_score,mean_poisson_deviance,mean_gamma_deviance, mean_tweedie_deviance, d2_tweedie_score, mean_pinball_loss 



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
    return np.real(evaluar_cdrag(exp, df))
  elif params['COEFICIENTE']==2:
    return np.real(evaluar_ff(exp, df))
  elif params['COEFICIENTE']==3:
    return np.real(evaluar_n(exp, df))

def check(dic, corrmat_pred, col):
  sentido_fisico = []
  no_sentido_fisico = []
  for param in list(dic.keys()):
    if corrmat_pred[col][param] * dic[param] > 0:
      sentido_fisico.append(param)
    else:
      no_sentido_fisico.append(param)
  return len(no_sentido_fisico) == 0

def check_minimum_fitness(fitness):
  if params['check_minimum_fitness']:
    if fitness <= params['media_fitness']:
      return True
    else:
      return False
  else:
    return True

def check_correlation(df, y_pred):
  if not params["Correlation"]:
    return True
  df['pred'] = y_pred
  df_corr = df.sample(n=params['N_ROWS_CORR'])
  corrmat = df_corr.corr()
  corrmat_pred = corrmat[['pred']]
  if params['COEFICIENTE'] == 1:
    dic = {'S':	0.476021,'colIndex':	0.252838,'Rem':	-0.267775,'Dfn':	-0.292734,'An':	-0.481160}
  elif params['COEFICIENTE'] == 2:
    dic = {'Vmfn':	-0.843099,'S': -0.853635}
  elif params['COEFICIENTE'] == 3:
    dic = {'Rem':	0.860579,'colIndex':	-0.318418}
  return check(dic, corrmat_pred, 'pred')

def get_data():
  path = '../datasets/ModeloBaterias/'
  if params['COEFICIENTE']==1:
    data = pd.read_csv(path + 'df_' + str(params['N_CELDAS']) + '_cdrag.txt')
  elif params['COEFICIENTE']==2:
    data = pd.read_csv(path + 'df_' + str(params['N_CELDAS']) + '_ff.txt')
  elif params['COEFICIENTE']==3:
    data = pd.read_csv(path + 'df_' + str(params['N_CELDAS']) + '_n.txt')
  data_train = data.sample(n=params['N_ROWS_TRAIN'])
  return data_train.iloc[:,:-1], data_train.iloc[:,-1].values