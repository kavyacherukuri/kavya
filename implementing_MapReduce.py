#!/usr/bin/env python
# coding: utf-8

# In[79]:


import numpy as np
import pandas as pd
import datetime
import csv


# In[45]:


#modifing the text file to make the split of date and temperture pairs.
def textModify():
    with open('temperature.txt', 'r') as file :
        filedata = file.read()
    filedata = filedata.replace('), ', ' $ ')
    with open('temperature.txt', 'w') as file:
        file.write(filedata)


# In[59]:


def dataCleaning(temp):
    
    # processing 2000 rows in a single dataframe and split the date and temperature values into two columns
    temp1 = temp.iloc[:,1]
    temp.drop(1, axis=1, inplace = True)
    temperature_main = pd.concat([temp, temp1], ignore_index=True)
    temperature_main = temperature_main[0].str.split(', ', expand =True)
    
    # converting the date to year and temperature to integer
    for i in range(len(temperature_main[0])):
        #year 
        value1=temperature_main.loc[i,0]
        temperature_main.loc[i,0] = value1.replace(" ","")
        temperature_main.loc[i,0] = value1.lstrip('(')
        temperature_main.loc[i,0] = value1.lstrip(' (')
        temperature_main.loc[i,0] = datetime.datetime.strptime(temperature_main.loc[i,0], "%Y%m").year
        #temperature
        value2= temperature_main.loc[i,1]
        temperature_main.loc[i,1] = value2.replace(" ","")
        temperature_main.loc[i,1] = value2.rstrip(')')
        temperature_main.loc[i,1] = int(temperature_main.loc[i,1])
    
    #rename column headers 
    temperature_main.rename(columns = {0: "year" , 1: "temperature"}, inplace = True)
        
    return temperature_main


# In[47]:


def dataSplit(main_dataSet):
    
    #split the data into two data sets to send to mapper
    #first 1000 rows to mapper1
    dataSet1 = main_dataSet.iloc[0:1000,:]
    #second 1000 rows to mapper2
    dataSet2 = main_dataSet.iloc[1000:2000,:]

    return dataSet1,dataSet2


# In[48]:


def mapper(dataSet):
    #converting the dataframe to list 
    mapperOutput = list(dataSet.to_records(index = False))
    
    return mapperOutput


# In[49]:


def sortFunction(mapping):
    
    #sorting the output of mapping fuction in ascending order
    mapping.sort(key=lambda x:x[0])
    
    #shuffling all the data based on the keys 
    shuffle ={}
    for key, value in mapping:
        shuffle.setdefault(key,[]).append(value)
        
    return shuffle


# In[50]:


def partitionFunction(dataSet):
    #spliting the data to send to reducer
    reducer1 = {}
    reducer2 = {}

    for key, value in dataSet.items():
        if key <= 2015:
            reducer1[key] = value
        else:
            reducer2[key] = value
    
    return reducer1, reducer2


# In[97]:


def reducerFunction(dataSet):
    #aquiring the max value for each year
    for key, values in dataSet.items():
        dataSet[key] = max(values)
    
    return dataSet
    


# In[105]:


def toCsvFile(output):
    #defining header for csv file
    headers= ['year', 'Max Temp']
    with open('max_temp.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames = headers)
        writer.writeheader()
        #writting data into csv files
        for key in output.keys():
            f.write("%s, %s\n" % (key, output[key]))

    


# In[106]:


def mainFunction():
    
    textModify()
    temp=  pd.read_csv('temperature.txt',sep="$", header=None)#reading the text file
    mainDataSet = dataCleaning(temp) #cleaning data
    data1, data2 = dataSplit(mainDataSet) #splitting data
    mapperPart1 = mapper(data1) #output of mapper1
    mapperPart2 = mapper(data2) #output of mapper2
    mapperDataSet = mapperPart1 + mapperPart2 #combining mappers 1 & 2 output
    sortedDataSet = sortFunction(mapperDataSet) # sorting and shuffleing dataset
    reducerInput1, reducerInput2 = partitionFunction(sortedDataSet) # spliting data for reducer input
    reducerOutput1 = reducerFunction(reducerInput1) # max values from reducer1 
    reducerOutput2 = reducerFunction(reducerInput2) # max values from reducer2
    combinedOutput = reducerOutput1 | reducerOutput2 #combining outputs fo reducer1 and reducer2
    toCsvFile(combinedOutput) #writing output into csv file

#Calling main function
mainFunction()
    


# In[ ]:




