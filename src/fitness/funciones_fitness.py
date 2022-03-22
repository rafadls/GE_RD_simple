from distutils.command.build_scripts import first_line_re
import numpy as np
np.seterr(all="raise")

from algorithm.parameters import params
from utilities.fitness.optimize_constants import *
 
import pandas as pd 
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, mean_absolute_percentage_error,r2_score,mean_poisson_deviance,mean_gamma_deviance, mean_tweedie_deviance, d2_tweedie_score, mean_pinball_loss 
from fitness.ModeloBaterias.fitness_modelo_java import fitness_modelo_java
import matplotlib.pyplot as plt

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
  data = data.sample(n=params['N_ROWS_TRAIN'])
  return data.iloc[:,:-1], data.iloc[:,-1].values

def get_all_data():
    path = '../datasets/ModeloBaterias/'
    if params['COEFICIENTE']==1:
        text = '_cdrag.txt'
    if params['COEFICIENTE']==2:
        text = '_ff.txt'
    if params['COEFICIENTE']==3:
        text = '_n.txt'
    data_25 = pd.read_csv(path + 'df_' + str(25) + text)
    data_53 = pd.read_csv(path + 'df_' + str(53) + text)
    data_74 = pd.read_csv(path + 'df_' + str(74) + text)
    data_102 = pd.read_csv(path + 'df_' + str(102) + text)
    return data_25, data_53, data_74, data_102


def eval_all_data(fitness_function, phenotypes, coeficiente):
  params['COEFICIENTE']=coeficiente
  data_25, data_53, data_74, data_102 = get_all_data()
  fitness_25_array, fitness_53_array, fitness_74_array, fitness_102_array = [], [], [], []
  for phenotype in phenotypes:
    ### 25 ####
    fitness_function.data_in, fitness_function.target  =  data_25.iloc[:,:-1], data_25.iloc[:,-1].values
    check_result, fitness_25 = fitness_function.fitness_stringPhenotype(phenotype)
    fitness_25_array.append(fitness_25)
    ### 53 ####
    fitness_function.data_in, fitness_function.target  =  data_53.iloc[:,:-1], data_53.iloc[:,-1].values
    check_result, fitness_53 = fitness_function.fitness_stringPhenotype(phenotype)
    fitness_53_array.append(fitness_53)
    ### 74 ####
    fitness_function.data_in, fitness_function.target  =  data_74.iloc[:,:-1], data_74.iloc[:,-1].values
    check_result, fitness_74 = fitness_function.fitness_stringPhenotype(phenotype)
    fitness_74_array.append(fitness_74)
    ### 102 ####
    fitness_function.data_in, fitness_function.target  =  data_102.iloc[:,:-1], data_102.iloc[:,-1].values
    check_result, fitness_102 = fitness_function.fitness_stringPhenotype(phenotype)
    fitness_102_array.append(fitness_102)
  return fitness_25_array, fitness_53_array, fitness_74_array, fitness_102_array

def eval_data(fitness_function, phenotype, coeficiente):
  params['COEFICIENTE']=coeficiente
  data_25, data_53, data_74, data_102 = get_all_data()
  ### 25 ####
  fitness_function.data_in, fitness_function.target  =  data_25.iloc[:,:-1], data_25.iloc[:,-1].values
  check_result, fitness_25 = fitness_function.fitness_stringPhenotype(phenotype)
  ### 53 ####
  fitness_function.data_in, fitness_function.target  =  data_53.iloc[:,:-1], data_53.iloc[:,-1].values
  check_result, fitness_53 = fitness_function.fitness_stringPhenotype(phenotype)
  ### 74 ####
  fitness_function.data_in, fitness_function.target  =  data_74.iloc[:,:-1], data_74.iloc[:,-1].values
  check_result, fitness_74 = fitness_function.fitness_stringPhenotype(phenotype)
  ### 102 ####
  fitness_function.data_in, fitness_function.target  =  data_102.iloc[:,:-1], data_102.iloc[:,-1].values
  check_result, fitness_102 = fitness_function.fitness_stringPhenotype(phenotype)
  return fitness_25, fitness_53, fitness_74, fitness_102

def eval_all_data_modeloFenomenologico(phenotype):
  fitness_array = []
  for num_celdas in [25,53,74,102]:
    params["num_celdas"] = num_celdas
    fitness_function = fitness_modelo_java()
    fitness = fitness_function.fitness_stringPhenotype(phenotype)
    fitness_array.append(fitness)
  return fitness_array


def get_data_outputs(df):
  df_outputs = []
  for output in ["VF",'PF','TC']:
    df_output = df.filter(regex=(output))
    columns = df_output.columns
    for column in columns:
      new_column = column.split('_')[0]
      df_output = df_output.rename(columns={column: new_column})
    df_outputs.append(df_output)
  return df_outputs

######################## CURVAS #######################

def df_from_output(numpy_array):
  string_array = ['V', 'P', 'TC']
  df_salidas = pd.DataFrame()
  print(numpy_array.shape)
  for i in range(3):
    data = numpy_array[:,i,:]
    for j in range(data.shape[1]):
      if j+1<10:
        j_s = '0' + str(j+1)
      else:
        j_s = str(j+1)
      df_salidas[string_array[i] + j_s] = data[:,j]
  return df_salidas

def get_df_to_plot(df,var1,var2):
  col_inputs = ['Current', 'K', 'Flujo', 't_viento', 'Diametro']
  df_aux = pd.concat([df[col_inputs],df.filter(regex=(var2 + "+\d")) ],axis=1)
  col_inputs.remove(var1)
  dir_base_values = dict(df.groupby(by=col_inputs).size().reset_index().rename(columns={0:'records'}).sort_values(by=['records'],ascending=False).reset_index(drop=True).iloc[0,:])
  for col in col_inputs:
    df_aux = df_aux[ df_aux[col]==dir_base_values[col]]
  df_aux.drop(columns=col_inputs, inplace=True)
  df_aux.reset_index(drop=True,inplace=True)
  return df_aux

def get_dataFrame(path_to_file):
  data = pd.read_csv(path_to_file, header=6)
  colCoded = data.columns.tolist()
  nameCode = pd.read_csv(path_to_file, header=3,nrows=1)
  colName = [[col.replace(d+' - ','') for i,col in enumerate(nameCode) if re.search(d, col)] for d in colCoded]
  colName = [temp[0] if temp else 'Name' for temp in colName]
  colName = [temp if (' ' not in temp) else temp[:temp.index(' ')] for temp in colName]
  data.rename(columns=dict(zip(colCoded, colName)), inplace=True)
  data.dropna(inplace=True)
  data.drop_duplicates(inplace=True)
  return data


def get_data_simple(df):
  df_output = df[['Current', 'K', 'Flujo', 't_viento', 'Diametro']]
  for string in ['V','P','TC', 'cdrag','frictionFactor', 'nusselt']:
    if string=='TC':
      df_aux = df.filter(regex=(string + "+\d")) - 273.15
    else:
      df_aux = df.filter(regex=(string + "+\d"))
    df_output = pd.concat([df_output,df_aux], axis=1)
  df_output = df_output[ df_output['Flujo'] > 10]
  df_output.reset_index(drop=True,inplace=True)
  return df_output


def compare(input, dataset_array, individuals_array, path_to_folder):
  n_individuals = len(individuals_array)
  fig, axis = plt.subplots(n_individuals, 3, figsize=(25,n_individuals*7))
  fig.suptitle('Comparación de individuos en cuanto a: ' + input ,fontsize=30)
  output_array = ['V', 'P', 'TC']
  for i in range(len(dataset_array)):
    for j in range(len(output_array)):
      df_i_vs_o = get_df_to_plot(dataset_array[i],input,output_array[j])
      columns_to_plot = list(df_i_vs_o.columns)
      columns_to_plot.remove(input)
      df_i_vs_o.plot(x=input, y=columns_to_plot, ax=axis[i,j])
      if j==1:
        axis[i,j].set_title(individuals_array[i],fontsize=25)
      axis[i,j].set_xlabel(input, fontsize=20)
      axis[i,j].set_ylabel(output_array[j],fontsize=20)
  plt.savefig(path_to_folder + input + '.png')

#######################################################

def save_graph_data_outputs(df, num_celdas, path_to_save):
  df_vf,df_pf,df_tc = get_data_outputs(df)
  nrow=3
  ncol=1
  fig, axes = plt.subplots(nrow, ncol,figsize=(15,nrow*8))
  fig.suptitle('Comparación de individuos para ' + str(num_celdas) + ' celdas',fontsize=30)

  df_vf.plot(rot=0, ax=axes[0])
  axes[0].set_title('Error en cálculo de velocidad de fluido para por columna',fontsize=25)
  axes[0].set_xlabel('Columnas',fontsize=20)
  axes[0].set_ylabel('Error',fontsize=20)

  df_pf.plot(rot=0, ax=axes[1])
  axes[1].set_title('Error en cálculo de presión de fluido para por columna',fontsize=25)
  axes[1].set_xlabel('Columnas',fontsize=20)

  df_tc.plot(rot=0, ax=axes[2])
  axes[2].set_title('Error en cálculo de temperatura de celda para por columna',fontsize=25)
  axes[2].set_xlabel('Columnas',fontsize=20)
  axes[2].set_ylabel('Error',fontsize=20)

  plt.savefig(path_to_save + 'individuals_comparation_' + str(num_celdas) + '_cells.png')

def get_arrays(n_gen, index, mainPath):
    fitness_array = []
    n_valid_array = []
    time_array = []
    duplicated_array = []
    phenotype_array = []
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
        phenotype_array.append(df['Fenotipo'][0])
    return fitness_array, n_valid_array, time_array, duplicated_array, phenotype_array


def get_dataframes(df,columns_text, mainPath):
  df_fitness = pd.DataFrame()
  df_valid = pd.DataFrame()
  df_time = pd.DataFrame()
  df_duplicated = pd.DataFrame()
  df_phenotype = pd.DataFrame()
  for index, row in df.iterrows():
      text = ''
      for column in columns_text:
          text = text + '(' + column + ':' + str(row[column]) + ') '
      df_fitness[text], df_valid[text], df_time[text], df_duplicated[text], df_phenotype[text]  = get_arrays(int(row['GENERATIONS']), index, mainPath)
  return df_fitness, df_valid, df_time, df_duplicated, df_phenotype



