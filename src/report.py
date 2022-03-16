#! /usr/bin/env python

# PonyGE2
# Copyright (c) 2017 Michael Fenton, James McDermott,
#                    David Fagan, Stefan Forstenlechner,
#                    and Erik Hemberg
# Hereby licensed under the GNU GPL v3.
""" Python GE implementation """

from utilities.algorithm.general import check_python_version

check_python_version()

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fitness.funciones_fitness import eval_all_data, get_all_data, get_dataframes
from fitness.fitness_modelo import fitness_modelo

mainPath = os.path.abspath("..")

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
ax1.set_ylim(0, 100)
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

###################   Performance   #####################

fig = plt.figure(figsize=(15,15))
fig.suptitle('Performance',fontsize=25) 
#########
ax1 = plt.subplot(2,1,1) # fig.add_axes([0,0.3,1,0.3])
df_fitness.plot(ax=ax1)
ax1.set_ylabel('Fitness')
ax1.set_xlabel('Generations')
ax1.set_title('Fitness',fontsize=15)
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
        text = text + '(' + column + ':' + str(row[column]) + ')\n'
    ax2.annotate(text, (row['Best fitness'], row['Total time']))  
ax2.set_ylabel('Total time')
ax2.set_xlabel('Best fitness')
ax2.set_title('Performance',fontsize=15)

### Save multi plot
plt.savefig(mainPath + '/Experiments/performance.png')
'''
########################## Generalization ########################    
nrow=len(df)
ncol=1
fig, axes = plt.subplots(nrow, ncol, figsize=(20,nrow*5))
fig.suptitle('Generalization per configuration',fontsize=25)
fitness_function = fitness_modelo()
for index, row in df.iterrows():
    text = ''
    for column in columns_text:
        text = text + '(' + column + ':' + str(row[column]) + ') '
    print(text)
    fitness_25_array, fitness_53_array, fitness_74_array, fitness_102_array = eval_all_data(fitness_function, df_phenotype[text], row['COEFICIENTE'])
    df_generalization = pd.DataFrame({'fitness 25 celdas': fitness_25_array,'fitness 53 celdas': fitness_53_array,'fitness 74 celdas': fitness_74_array,'fitness 102 celdas': fitness_102_array})
    df_generalization.plot(ax=axes[index])
    axes[index].set_xlabel('Generations')
    axes[index].set_ylabel('Fitness')
    axes[index].set_yscale('log')
plt.savefig(mainPath + '/Experiments/generalization.png')
'''