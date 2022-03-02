import xlwt

import os

import numpy as np

from algorithm.parameters import params

import time

def list2dic(lista):
  dic = {}
  for item in lista:
    dic[item[0]] = item[1]
  return dic

def saveGenerationAsExcel(individuals, folder_path, file_name, folder_name="savedPopulations"):
  folder_path = os.sep.join([folder_path, folder_name])
  if not os.path.exists(folder_path):
      os.mkdir(folder_path)
  path = os.sep.join([folder_path, file_name])
  generation_workBook = xlwt.Workbook()
  generation_sheet = generation_workBook.add_sheet('stats')
  generation_sheet.write(0,0,"Time")
  generation_sheet.write(0,1,"Genotipo")
  generation_sheet.write(0,2,"Fenotipo")
  generation_sheet.write(0,3,"Depth")
  generation_sheet.write(0,4,"Fitness")

  row = 1
  elapsedTime = time.time()- params["start-time"]
  for ind in individuals:
      generation_sheet.write(row,0,elapsedTime)  
      generation_sheet.write(row,1,str(ind.genome)) 
      generation_sheet.write(row,2,str(ind.phenotype))    
      generation_sheet.write(row,3,str(ind.depth))
      generation_sheet.write(row,4,str(ind.fitness))
      row = row + 1

  if params['check_minimum_fitness']:
    params['media_fitness'] = individuals[params['POPULATION_SIZE']//2-1].fitness
  generation_workBook.save(path)