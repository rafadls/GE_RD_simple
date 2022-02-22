import scipy

import re

from algorithm.parameters import params
from utilities.fitness.math_functions import *
#from utilities.fitness.customMinimize.scipy.optimize._shgo import minimize as min_lbfgsb
from utilities.fitness.customMinimize.scipy.optimize import minimize as min_lbfgsb
from algorithm.mapper import map_ind_from_genome
import time
import math
import functools
import operator
from copy import deepcopy


def custom_optimize_constants2(x, y, ind, actualizeGenome=False):
    s, n_ctes, acc_values_int, lower_values_float, step_values_float, last_values_float, init = make_consts_consecutive(ind.phenotype, returnInit=True)
    ind.phenotype_consec_consts = s
    new_phenotype = s
    if n_ctes == 0:
        c = []
        ind.opt_consts = c
        fitness = params['FITNESS_FUNCTION'].fitness_stringPhenotype(new_phenotype)
        return fitness

    [best_result, best_constant, best_val_pos] = seudo_gradient_optimization(acc_values_int, lower_values_float, 
        step_values_float, last_values_float, init, s, grad_calculations=2, grad_descendant=20)
    ind.opt_consts = best_constant

    if actualizeGenome:
        ind.genome_original = deepcopy(ind.genome)
        new_genome = deepcopy(ind.genome)
        consts_location = map_ind_from_genome(ind.genome_original, return_consts_location=True)
        i = 0
        for locus in consts_location:
            new_genome[locus] = int(round((float(ind.opt_consts[i])-lower_values_float[i])/step_values_float[i]))
            i += 1

        ind.genome = deepcopy(new_genome)
        fenotip, genomin,FFERF, nodesd, nse, depthsd, inputasd = map_ind_from_genome(ind.genome)
    ind.phenotype = replace_consts_with_values(new_phenotype, ind.opt_consts, acc_values_int)
    return best_result

def seudo_gradient_optimization(accuracy, min_val, step_val, max_val, actual_val, fun, grad_calculations=3, grad_descendant=50):
    num_ctes = len(actual_val)
    posValues = [[]]*num_ctes
    actual_val_pos = [0]*num_ctes
    #print("#####################################################################")
    final_posValues = [0]*num_ctes
    for i in range(num_ctes):
        posValues[i] = [round(x,accuracy[i]) for x in np.arange(min_val[i], max_val[i]+step_val[i], step_val[i])]
        if posValues[i][-1]>max_val[i]:
            posValues[i] = posValues[i][0:-1]
        actual_val_pos[i] = posValues[i].index(actual_val[i])
        final_posValues[i] = len(posValues[i])-1
    best_val_pos = list(actual_val_pos)
    gradiente = [0]*num_ctes
    str_to_eval = replaceConstants(actual_val, fun)
    prev_result = eval_fun(str_to_eval) 
    actual_result = prev_result
    best_result = prev_result
    #print("ahora", best_result)
    prev_constant = list(actual_val)
    best_constant = list(actual_val)
    actual_constant = list(actual_val)
    counter_calc = 1
    while counter_calc <= grad_calculations:
        actual_constant = list(best_constant)
        actual_val_pos = list(best_val_pos)
        actual_result = best_result
        prev_constant = list(best_constant)
        #obtengo seudo gradiente
        for j in range(num_ctes):
            actual_constant = list(prev_constant)
            # pruebo aumentando las constantes
            next_pos = actual_val_pos[j]+1
            if next_pos > final_posValues[j]:
                next_pos = final_posValues[j]
            actual_constant[j] = posValues[j][next_pos]
            #print(actual_constant)
            result_increment_step = eval_fun(replaceConstants(actual_constant, fun))
            if result_increment_step < best_result:
                best_result = result_increment_step
                best_constant = list(actual_constant)
                best_val_pos = list(actual_val_pos)
                best_val_pos[j] = next_pos

            # pruebo disminuyendo las constantes
            next_pos = actual_val_pos[j]-1
            if next_pos < 0:
                next_pos = 0
            actual_constant[j] = posValues[j][next_pos]
            #print(actual_constant)
            result_decrement_step = eval_fun(replaceConstants(actual_constant, fun))
            if result_decrement_step < best_result:
                best_result = result_decrement_step
                best_constant = list(actual_constant)
                best_val_pos = list(actual_val_pos)
                best_val_pos[j] = next_pos
            # print(actual_result, result_increment_step, result_decrement_step)
            if result_increment_step < result_decrement_step and result_increment_step < actual_result:
                gradiente[j] = 1
            elif result_increment_step > result_decrement_step and result_decrement_step < actual_result:
                gradiente[j] = -1
            else:
                gradiente[j] = 0
                
            if result_increment_step == math.inf and result_decrement_step == math.inf and actual_result == math.inf:
                gradiente[j] = -1
            # print(gradiente)
        if gradiente == [0]*num_ctes:
            break
        #print(gradiente)
        ###########    
        # avanzo en el gradiente
        counter_desc = 1
        while counter_desc <= grad_descendant:
            actual_constant = list(best_constant)
            #print("cte", actual_constant)
            #print("yom")
            actual_val_pos = list(best_val_pos)
            actual_result = best_result
            for x in range(num_ctes):
                next_pos = actual_val_pos[x] + counter_desc*gradiente[x]
                if next_pos < 0:
                    next_pos = 0
                elif next_pos > final_posValues[x]:
                    next_pos = final_posValues[x]
                actual_val_pos[x] = next_pos
                actual_constant[x] = posValues[x][actual_val_pos[x]]
            actual_result = eval_fun(replaceConstants(actual_constant, fun))
            if actual_result <= best_result:
                best_result = actual_result
                best_constant = list(actual_constant)
                best_val_pos = list(actual_val_pos)
            else:
                break
            counter_desc += 1
        counter_calc += 1
    return [best_result, best_constant, best_val_pos]

def replaceConstants(c,string):
    for i in range(len(c)):
        string = string.replace("c["+str(i)+"]", str(c[i]))
    return string

def get_consts(string):
    if params["smartConstant"]:
        p = r"c\[(-\d+|-\d+.\d+|\d+|\d+.\d+)_(-\d+|-\d+.\d+|\d+|\d+.\d+)_(-\d+|-\d+.\d+|\d+|\d+.\d+)_(-\d+|-\d+.\d+|\d+|\d+.\d+)_(-\d+|-\d+.\d+|\d+|\d+.\d+)\]"
        constantes = re.findall(p, string)
    else:
        p = r"c\[(\d+)\]"
        constantes = re.findall(p, string)
    return constantes

def make_consts_consecutive(s, returnInit=False):
    if params["smartConstant"]:
        zipped = get_consts(s)
        if len(zipped) != 0:
            acc_value_actual, lower_value_actual, step_value_actual, last_value_actual, ctes_gene_actual = zip(*zipped)
            acc_value_actual, lower_value_actual, step_value_actual, last_value_actual, ctes_gene_actual = list(acc_value_actual), list(lower_value_actual), list(step_value_actual), list(last_value_actual), list(ctes_gene_actual)
        else:
            acc_value_actual, lower_value_actual, step_value_actual, last_value_actual, ctes_gene_actual = [],[],[],[],[]

        acc_values_int = [int(item) for item in acc_value_actual]
        lower_values_float = [float(item) for item in lower_value_actual]
        step_values_float = [float(item) for item in step_value_actual]
        last_values_float = [float(item) for item in last_value_actual]
        init = [float(item) for item in ctes_gene_actual]
        n_ctes = len(init)
        for i in range(n_ctes):
            c_old = "c[{}_{}_{}_{}_{}]".format(acc_values_int[i], lower_values_float[i], step_values_float[i], last_values_float[i], init[i])
            c_new = "c[{}]".format(i)
            s = s.replace(c_old, c_new, 1)
    else:
        # find the consts, extract idxs as ints, unique-ify and sort
        #const_idxs = sorted(map(int, set(re.findall(p, s))))
        ctes = get_consts(s)
        n_ctes = len(ctes)
        init = [0]*n_ctes
        for i in range(n_ctes):
            init[i] = float(ctes[i])
            c_old = "c[%d]" % int(ctes[i])
            c_new = "c[-]"
            s = s.replace(c_old, c_new, 1)
        for i in range(n_ctes):
            c_old = "c[-]"
            c_new = "c[%d]" % i
            s = s.replace(c_old, c_new, 1)
    if returnInit: 
        return s, n_ctes, acc_values_int, lower_values_float, step_values_float, last_values_float, init 
    else:
        return s, n_ctes
    
def eval_fun(fun):
    fitness = params['FITNESS_FUNCTION'].fitness_stringPhenotype(fun)
    return fitness

def replace_consts_with_values(s, c, acc_values_int):
    """
    Replace the constants in a given string s with the values in a list c.
    
    :param s: A given phenotype string.
    :param c: A list of values which will replace the constants in the
    phenotype string.
    :return: The phenotype string with the constants replaced.
    """
    for i in range(len(c)):
        #print(round(c[i],4))
        s = s.replace("c[%d]" % i, str(round(c[i],acc_values_int[i])), 1)
    return s

def replace_consts_no_assumption(s, acc_values, init_values, step_values, last_values, ctes):
    for i in range(len(ctes)):
        #print(round(c[i],4))
        #s = s.replace("c[{}_{}_{}_{}]".format(max_value[i],ctes[i]), ctes[i])
        s = s.replace("c[{}_{}_{}_{}_{}]".format(acc_values[i], init_values[i], step_values[i], last_values[i], ctes[i]), ctes[i])
    return s
