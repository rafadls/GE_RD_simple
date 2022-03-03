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

mainPath = os.path.abspath("..")

# DATA
df =  pd.read_csv(mainPath + '/Experiments/data.csv')
columns_text = df.nunique()
columns_text = columns_text[columns_text>1]
columns_text = list(columns_text.index)
columns_text = [x for x in columns_text if x not in ['Best fitness', 'Best phenotype', 'Total time']]

# EVOLUTION
def get_fitness_array(n_gen, index):
    fitness_array = []
    n_invalid_array = []
    time_array = []
    df_inicial = pd.read_excel(mainPath + '/results/'+ str(index) +'/savedPopulations/poblacionInicial.xls')
    tiempo = df_inicial['Time'][0]
    for i in range(1,n_gen+1):
        nombre = 'generation_' + str(i) + '.xls'
        df = pd.read_excel(mainPath + '/results/'+ str(index) +'/savedPopulations/' + nombre)
        fitness_array.append(df['Fitness'].min())
        n_invalid_array.append(np.sum(df['Fitness'] != np.inf)*100/len(df['Fitness']))
        time_array.append(df['Time'][0]-tiempo)
        tiempo = df['Time'][0]
    return fitness_array, n_invalid_array, time_array

# EVOLUTION
df_fitness = pd.DataFrame()
df_invalid = pd.DataFrame()
df_time = pd.DataFrame()

for index, row in df.iterrows():
    text = ''
    for column in columns_text:
        text = text + '(' + column + ':' + str(row[column]) + ') '
    df_fitness[text], df_invalid[text], df_time[text]  = get_fitness_array(int(row['GENERATIONS']), index)



fig = plt.figure(figsize=(30,20))
fig.suptitle('Evolution',fontsize=25) 
#########
ax1 = plt.subplot(3,2,1)
df_fitness.plot(ax=ax1)
ax1.set_ylabel('Fitness')
ax1.set_xlabel('Generations')
ax1.set_title('Fitness')
########
ax2 = plt.subplot(3,2,3)
df_invalid.plot(ax=ax2)
ax2.set_ylabel('Valid percentage')
ax2.set_xlabel('Generations')
ax2.set_title('Valid')
########
ax3 = plt.subplot(3,2,5)
df_time.plot(ax=ax3)
ax3.set_ylabel('Time')
ax3.set_xlabel('Generations')
ax3.set_title('Time per generation')
########
ax4 = plt.subplot(1,2,2)
x = df[['Best fitness']].values
y = df[['Total time']].values
ax4.scatter(x,y)
ax4.set_yscale('log')
ax4.set_xscale('log')
for index, row in df.iterrows():
    text = ''
    for column in columns_text:
        text = text + '(' + column + ':' + str(row[column]) + ') '
    ax4.annotate(text, (row['Best fitness'], row['Total time']))  
ax4.set_ylabel('Total time')
ax4.set_xlabel('Best fitness')
ax4.set_title('Performance')

for index, row in df.iterrows():
    text = ''
    for column in columns_text:
        text = text + '(' + column + ':' + str(row[column]) + ') '
    ax4.annotate(text, (row['Best fitness'], row['Total time']))    

### Save multi plot
plt.savefig(mainPath + '/Experiments/evolution.png')