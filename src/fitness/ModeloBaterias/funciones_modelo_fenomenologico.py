import numpy as np
np.seterr(all="raise")
import numexpr as ne
import time

from fitness.ModeloBaterias import parametric,model

import os
from contextlib import contextmanager
import tblib.pickling_support
import sys
import math
from copy import deepcopy 



def if_lower_else(a,b,c,d):
  if a<b:
    return c
  else:
    return d


def eval_dataset_python(m,ind,seq):
    point = {'Current':seq[0],'K': seq[1], 'Flujo':seq[2] * 0.00047,'t_viento':seq[3],'Diametro':seq[4]/1000}
    return m.evolve(point, ind)

def eval_allData_python(data_in, final_trees):
    fenotipo_cd = lambda a1, Rem, An, Dfn : eval(final_trees[0])
    fenotipo_ff = lambda Rem, S, Vmfn, Dfn: eval(final_trees[1])
    fenotipo_nu =  lambda colIndex, Rem, a3: eval(final_trees[2])

    individuo = [fenotipo_cd,fenotipo_ff,fenotipo_nu]
    col_fluido, col_celda, n_fluido,  n_celda = data_in[['col_fluido', 'col_celda', 'n_fluido', 'n_celda']].iloc[0,:]
    mdl = model.ParametricModel(int(col_fluido),int(n_fluido),int(col_celda),int(n_celda))
    result = list(map(lambda x: eval_dataset_python(mdl,individuo,x), data_in.iloc[:,:5].values))
    return result

def check_invalid_result(modelResult):
    have_nan , have_inf = np.any(np.isnan(modelResult)) , np.any(np.isinf(modelResult))
    if have_nan and have_inf:
      return "None e inf en resultado del modelo"
    elif have_inf:
      return "inf en resultado del modelo"
    elif have_nan:
      return "None en resultado del modelo"
    else:
      return 'Sin problema'

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


def fit_ranking(matrix_error, percent):
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