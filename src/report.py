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
    for i in range(1,n_gen+1):
        nombre = 'generation_' + str(i) + '.xls'
        df = pd.read_excel(mainPath + '/results/'+ str(index) +'/savedPopulations/' + nombre)
        fitness_array.append(df['Fitness'].min())
        n_invalid_array.append(np.sum(df['Fitness'] != np.inf))
    return fitness_array, n_invalid_array

# EVOLUTION
df_fitness = pd.DataFrame()
df_invalid = pd.DataFrame()

for index, row in df.iterrows():
    text = ''
    for column in columns_text:
        text = text + '(' + column + ':' + str(row[column]) + ') '
    df_fitness[text], df_invalid[text] = get_fitness_array(int(row['GENERATIONS']), index)



fig = plt.figure(figsize=(24,12))
fig.suptitle('Evolution',fontsize=25) 
#########
ax1 = plt.subplot(2,2,1)
df_fitness.plot(ax=ax1)
ax1.set_ylabel('Fitness')
ax1.set_xlabel('Generations')
ax1.set_title('Fitness')
########
ax2 = plt.subplot(2,2,3)
df_invalid.plot(ax=ax2)
ax2.set_ylabel('Number of Valid')
ax2.set_xlabel('Generations')
ax2.set_title('Valid')
########
ax3 = plt.subplot(1,2,2)
x = df[['Best fitness']].values
y = df[['Total time']].values
ax3.scatter(x,y)
for index, row in df.iterrows():
    text = ''
    for column in columns_text:
        text = text + '(' + column + ':' + str(row[column]) + ') '
    ax3.annotate(text, (row['Best fitness'], row['Total time']))  
ax3.set_ylabel('Total time')
ax3.set_xlabel('Best fitness')
ax3.set_title('Performance')

for index, row in df.iterrows():
    text = ''
    for column in columns_text:
        text = text + '(' + column + ':' + str(row[column]) + ') '
    ax3.annotate(text, (row['Best fitness'], row['Total time']))    

### Save multi plot
plt.savefig(mainPath + '/Experiments/evolution.png')
