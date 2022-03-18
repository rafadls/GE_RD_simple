import numpy as np
np.seterr(all="raise")
import numexpr as ne
import time

from fitness.ModeloBaterias.funcionesEvaluar_ModeloJava import *
from utilities.fitness.optimize_constants import *
import os
from contextlib import contextmanager
import tblib.pickling_support
import sys
import math
from copy import deepcopy      



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

def check(dic, corrmat_pred, col):
  sentido_fisico = []
  no_sentido_fisico = []
  for param in list(dic.keys()):
    if corrmat_pred[col][param] * dic[param] > 0:
      sentido_fisico.append(param)
    else:
      no_sentido_fisico.append(param)
  return len(no_sentido_fisico) == 0

def eval_total(ind, df_cdrag, df_ff, df_n):
  df_cdrag['cdrag_pred'] = evaluar_cdrag(ind[0], df_cdrag)
  df_ff['ff_pred'] = evaluar_ff(ind[1], df_ff)
  df_n['n_pred'] = evaluar_n(ind[2], df_n)
  ###### cdrag ######
  corrmat = df_cdrag[['S','An','Rem','Dfn','colIndex','cdrag_pred']].corr()
  corrmat_cdrag_pred = corrmat[['cdrag_pred']].sort_values( by=['cdrag_pred'], ascending=False)
  dic_cdrag = { "colIndex":	0.445811,"S":	0.183466,"Dfn":	-0.146940,"An": -0.197465,"Rem": -0.232187}
  ###### ff ######
  corrmat = df_ff[['S','Vmfn','Rem','Dfn','colIndex','ff_pred']].corr()
  corrmat_ff_pred = corrmat[['ff_pred']].sort_values( by=['ff_pred'], ascending=False)
  dic_ff = { "colIndex": 0.227915, "Dfn": 0.147556, "Rem": -0.293133, "S": -0.862079, "Vmfn": -0.876163}
  ###### Nusselt ######
  corrmat = df_n[['S','Rem','Prandtl','colIndex','n_pred']].corr()
  corrmat_n_pred = corrmat[['n_pred']].sort_values( by=['n_pred'], ascending=False)
  dic_n = {"Rem": 0.937536, "S":	0.083368, "Prandtl": -0.078839, "colIndex": -0.307273}
  return check(dic_cdrag, corrmat_cdrag_pred,'cdrag_pred'), check(dic_ff, corrmat_ff_pred,'ff_pred') , check(dic_n, corrmat_n_pred,'n_pred')

def check_correlation(ind):
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
  ind_ph = ind_ph.split(';')
  try:
    df_cdrag, df_ff, df_n = get_data_correlation()
    check_cdrag, check_ff , check_n = eval_total(ind_ph, df_cdrag, df_ff, df_n)
    return check_cdrag, check_ff , check_n
  except:
    return False, False, False

def get_data_correlation():
  path = '../datasets/DATASET_corelation/'
  df_cdrag = pd.read_csv(path + 'df_cdrag.csv')
  df_ff = pd.read_csv(path + 'df_ff.csv')
  df_n = pd.read_csv(path + 'df_n.csv')
  return df_cdrag, df_ff, df_n

def get_matrix_error(modelResult, target, type="abs_dif"):
    if type == "relative":
        return list(np.abs((np.array(modelResult) - np.array(target))/np.array(target)))
    elif type == "abs_dif":
        return list(np.abs((np.array(modelResult) - np.array(target))))
    
def get_list_vf_pf_tc(matrix_error):
    list_vf = []
    list_pf = []
    list_tc = []
    length_data = len(matrix_error)
    for i in range(length_data):
        for vf_output in matrix_error[i][0]:
            list_vf.append(vf_output)
        for pf_output in matrix_error[i][1]:
            list_pf.append(pf_output)
        for tc_output in matrix_error[i][2]:
            list_tc.append(tc_output)
    list_vf.sort()
    list_pf.sort()
    list_tc.sort()
    return [list_vf, list_pf, list_tc]

def fit_to_cells(matrix_error):
    matrix_error_np = np.asarray(matrix_error)
    matrix_error_celdas = matrix_error_np.mean(axis=0)
    return matrix_error_celdas


def fit_ranking(matrix_error, percent):
    print(np.asarray(matrix_error).shape)
    [list_vf, list_pf, list_tc] = get_list_vf_pf_tc(matrix_error)
    list_vf.sort()
    list_pf.sort()
    list_tc.sort()
    fit_vf = list_vf[math.floor(len(list_vf)*percent)]
    fit_pf = list_pf[math.floor(len(list_pf)*percent)]
    fit_tc = list_tc[math.floor(len(list_tc)*percent)]
    #fitness_calculado = np.max([fit_vf, fit_pf, fit_tc])
    # for element in fit_pf:
    #     print(element)
    fitness_calculado = fit_pf
    mse = [fit_vf*fit_vf, fit_pf*fit_pf, fit_tc*fit_tc]
    #print(fitness_calculado, mse)
    return fitness_calculado, mse

def get_error_by_column(matrix_error):
    length_data = len(matrix_error)
    errors_by_column_vf = []
    errors_by_column_pf = []
    errors_by_column_tc = []
    for i in range(length_data):
        errors_by_column_vf.append([])
        errors_by_column_pf.append([])
        errors_by_column_tc.append([])
        for vf_output in matrix_error[i][0]:
            errors_by_column_vf[-1].append(vf_output)
        for pf_output in matrix_error[i][1]:
            errors_by_column_pf[-1].append(pf_output)
        for tc_output in matrix_error[i][2]:
            errors_by_column_tc[-1].append(tc_output)
    return [np.array(errors_by_column_vf), np.array(errors_by_column_pf), np.array(errors_by_column_tc)]

def standarize_datasets(data_out):
    data_out_vf = []
    data_out_pf = []
    data_out_tc = []
    for index in data_out.index:
        n_control_vol = int(data_out.iloc[index,:].count()/3)
        data_out_tc.append(data_out.iloc[index,0*n_control_vol:1*n_control_vol]-273.15)
        data_out_pf.append(data_out.iloc[index,1*n_control_vol:2*n_control_vol])
        #print("#")
        #print(data_out.iloc[index,1*n_control_vol:2*n_control_vol])
        data_out_vf.append(data_out.iloc[index,2*n_control_vol:3*n_control_vol]) 
    #ns = self.col_celda #numero de salidas
    #data_out_aux = [self.data_out[self.data_out.columns[2*ns:3*ns]],
    #                self.data_out[self.data_out.columns[1*ns:2*ns]], 
    #                self.data_out[self.data_out.columns[0*ns:1*ns]]]
    # estos son cuando la salida tiene cosas aparte de la velocidad, presion y temperatura
    #data_out_aux = [self.data_out[self.data_out.columns[3:4]],
    #                self.data_out[self.data_out.columns[2:3]], 
    #                self.data_out[self.data_out.columns[1:2]]]
    
    #for element in data_out_aux_vf:
    #    print(element)
    #data_out_tc = np.asarray(data_out_aux[2])
    target = []
    for i in range(len(data_out)):
        target.append([data_out_vf[i].values, data_out_pf[i].values, data_out_tc[i].values])
    return target


def load_ansys_data(ansys_file):
    lines_to_delete = [0,1,2,4,5,6]

    with open(ansys_file, 'r') as file:
      lines = file.readlines()
    with open("ansys_file", 'w') as new_file:
      for line_index in range(len(lines)):
        if line_index not in lines_to_delete:
          if line_index == 3:
            old_header_list = lines[line_index].split(",")
            new_header_list = []
            for old_header in old_header_list:
              column_name_with_unit = old_header.split("-", 1)[-1].strip(" ")
              column_name_without_unit = column_name_with_unit.split(" ")[0]
              new_header_list.append(column_name_without_unit)
            #print(new_header_list)
            new_file.write(','.join(new_header_list) + "\n") 
          else:
            new_file.write(lines[line_index]) 


    df = pd.read_csv("ansys_file", header=0)
    df = df.dropna().drop_duplicates()
    return df


def expand_ansys_data(df):
    largo = 65E-3
    r = 32E-3
    e = 15E-3
    z = 5E-3
    errmax = 1E-3
    pi = math.pi
    piQuarter = pi/4
    doubleE = 2 * e
    df['n_fluido'] = df['celda1.ny'] + 1
    df['n_celda'] = df['celda2.ny'] + 1
    df['col_fluido'] = df['celda1.nx'] + 1
    df['col_celda'] = df['celda2.nx'] + 1
    df['flux'] = df['Flujo']*0.00047
    df['cellDiameter'] = df['Diametro']/1000
    df['cellArea'] = df['cellDiameter']**2 *piQuarter
    df['vol'] = largo * df['cellArea']
    df['volumetricQdot'] = (df['Current']**2)* r/df['vol']
    df['qdot'] = df['volumetricQdot'] * z * df['cellArea']
    df['diamTimesZ'] = df['cellDiameter'] * z
    df['superficialArea'] =pi * df['diamTimesZ']
    df['height'] = doubleE + df['cellDiameter'] * (df['n_fluido'] + df['K']*df['n_celda'])
    df['entranceArea'] = df['height'] * z
    df['sPlusOne'] = df['K'] + 1
    df['controlVolArea'] = df['sPlusOne'] * df['diamTimesZ']
    df['innerArg'] = df['flux'] * z/(largo*df['entranceArea'])
    df['initVelocity'] = get_a2(df['K']) * df['innerArg']
    df['dfMultiplicationTerm'] = df['diamTimesZ'] * df['initVelocity'] * get_densidad_celcius(df['t_viento'])
    df['m_punto'] = df['sPlusOne'] * df['dfMultiplicationTerm']
    df['sTerm'] = df['K'] / df['sPlusOne']
    df['heatPerArea'] = df['qdot'] / df['superficialArea']
    df['initialFFTerm'] = 0.5* df['dfMultiplicationTerm'] * df['initVelocity']
    df['fluidTempTerm'] = df['qdot'] / df['m_punto']
    df['normalizedArea'] = df['superficialArea'] / df['controlVolArea']
    df['firstRem'] = get_reynolds(df['initVelocity'], df['t_viento'], df['cellDiameter'], get_densidad_celcius(df['t_viento']))


    tf = 'tf_calc'
    df['tf_calc01'] = df['t_viento']
    df['df_calc01'] = get_densidad_celcius(df['tf_calc01'])
    df['FF01'] = (df['initVelocity'] - df['V01'])*df['m_punto']
    df['cdrag00'] = df['FF01']/df['initialFFTerm'] 
    for i in range(1,df['col_fluido'][0]):
        actualCol = str(i) 
        nextCol = str(i+1)
        if len(actualCol)<2: actualCol = '0' + actualCol
        if len(nextCol)<2: nextCol = '0' + nextCol
        df['tf_calc'+nextCol] = df['tf_calc'+actualCol] + (df['fluidTempTerm']-0.5*(df['V'+nextCol]**2 - df['V'+actualCol]**2))/get_cp_celcius(df['tf_calc'+actualCol]);
        df['fluidK'+actualCol] = get_conductividad_celcius(df['tf_calc'+actualCol])
        df['h'+actualCol] = df['heatPerArea']/(df['TC'+actualCol]-273.15 - (df['tf_calc'+actualCol] + df['tf_calc'+nextCol])/2) 
        df['nusselt'+actualCol] = df['h'+actualCol]*df['cellDiameter']/df['fluidK'+actualCol]
        df['df_calc'+nextCol] = get_densidad_celcius(df['tf_calc'+nextCol])
        df['vmf'+actualCol] = df['V'+actualCol]*df['sTerm']
        df['frictionFactor'+actualCol] = (df['P'+actualCol] - df['P'+nextCol])/(0.5 * df['df_calc'+actualCol] * df['vmf'+actualCol]**2)
        df['FF'+nextCol] = df['controlVolArea']*(df['P'+actualCol] - df['P'+nextCol]) - (df['V'+nextCol] - df['V'+actualCol])*df['m_punto']
        df['cdrag'+actualCol] = df['FF'+nextCol]/(0.5 * df['diamTimesZ'] * df['df_calc'+actualCol] * df['V'+actualCol]**2)
    return df


def create_array_result(df):
    df_vf = df.filter(regex=("V+\d"))
    df_vf = df_vf.reindex(sorted(df_vf.columns), axis=1)
    df_pf = df.filter(regex=("P+\d"))
    df_pf = df_pf.reindex(sorted(df_pf.columns), axis=1)
    # print(df_pf.to_numpy())
    #print(df_pf.to_numpy() - df_pf.to_numpy()[:,-1])
    # print(df_pf.to_numpy()[:,:-1])
    df_tc = df.filter(regex=("TC+\d"))
    df_tc = df_tc.reindex(sorted(df_tc.columns), axis=1)

    target = []
    for i in range(len(df_vf.index)):
        vf_values = df_vf.iloc[i,:]
        pf_values = df_pf.iloc[i,:]-df_pf.iloc[i,-1]
        tc_values = df_tc.iloc[i,:]-273.15
        target.append([vf_values.values[1:], pf_values.values[:-1], tc_values.values])
    return target


def interpolate(value_x, list_x, list_y):
  if value_x <= list_x[0]:
    return list_y[0]
  for i in range(len(list_x)):
    if value_x <= list_x[i]:
      return list_y[i-1] + (value_x-list_x[i-1])*(list_y[i]-list_y[i-1])/(list_x[i]-list_x[i-1])
  return list_y[-1]


def densidad_celcius(temp_celcius):
  return interpolate(temp_celcius, [0, 20, 40], [1.293, 1.205, 1.127])
def get_densidad_celcius(series_temp_celcius):
  return series_temp_celcius.apply(densidad_celcius)

def conductividad_celcius(temp_celcius):
  temp_kelvin = temp_celcius + 273.15
  return interpolate(temp_kelvin,  [250, 300, 350, 400, 450], [22.3e-3, 26.3e-3, 30e-3, 33.8e-3, 37.3e-3])
def get_conductividad_celcius(series_temp_celcius):
  return series_temp_celcius.apply(conductividad_celcius)

def viscosidad_celcius(temp_celcius):
  temp_kelvin = temp_celcius + 273.15
  return interpolate(temp_kelvin,  [250, 300, 350, 400, 450], [159.6e-7, 184.6e-7, 208.2e-7, 230.1e-7, 250.7E-7])
def get_viscosidad_celcius(series_temp_celcius):
  return series_temp_celcius.apply(viscosidad_celcius)

def cp_celcius(temp_celcius):
  temp_kelvin = temp_celcius + 273.15
  return interpolate(temp_kelvin,  [250, 300, 350, 400, 450], [1.006e3, 1.007e3, 1.009e3, 1.014e3, 1.021e3])
def get_cp_celcius(series_temp_celcius):
  return series_temp_celcius.apply(cp_celcius)

def a2(separation):
  return interpolate(separation,  [0.1, 0.25, 0.5, 0.75, 1], [3.270, 2.416, 2.907, 2.974, 2.063])
def get_a2(series_temp_celcius):
  return series_temp_celcius.apply(a2)

def get_reynolds(series_velocity, series_temp_celcius, series_diameter, series_density):
  series_viscosity = get_viscosidad_celcius(series_temp_celcius)
  return series_density * series_velocity * series_diameter / series_viscosity