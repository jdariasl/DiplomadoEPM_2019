#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 09:15:40 2018

@author: julian
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#Mean Absolute Percentage Error para los problemas de regresión
def MAPE(Y_est,Y):
    N = np.size(Y)
    mape = np.sum(abs((Y_est.reshape(N,1) - Y.reshape(N,1))/Y.reshape(N,1)))/N
    return mape

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# convert an array of values into a dataset matrix
def create_datasetMultipleTimesAhead(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(int(len(dataset)/look_back)-1):
        a = dataset[i*look_back:(i+1)*look_back, 0]
        dataX.append(a)
        dataY.append(dataset[(i+1)*look_back:(i+2)*look_back, 0])
    return np.array(dataX), np.array(dataY)

def create_datasetMultipleTimesBackAhead(dataset, n_steps_out=1, n_steps_in = 1, overlap = 1):
    dataX, dataY = [], []
    tem = n_steps_in + n_steps_out - overlap
    for i in range(int((len(dataset) - tem)/overlap)):
        startx = i*overlap
        endx = startx + n_steps_in
        starty = endx
        endy = endx + n_steps_out
        a = dataset[startx:endx, 0]
        dataX.append(a)
        dataY.append(dataset[starty:endy, 0])
    return np.array(dataX), np.array(dataY)

def SplitTimeseries(df,column,day = 'Monday', ValData = 'steps', TimeAhead = 96, look_back = 1):
    if day == 'All':
        if ValData == 'index':
            df2 = df.loc[df.index < TimeAhead]
            train = df2[column].dropna().values

            df2 = df.loc[df.index >= TimeAhead]
            test = df2[column].dropna().values
            # normalize the dataset
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(np.concatenate((train.reshape(len(train),1),test.reshape(len(test),1)),axis=0))
            dataset = scaler.transform(train.reshape(len(train),1))
            
            trainX, trainY = create_dataset(dataset, look_back)

            test2 = scaler.transform(np.concatenate((train[-look_back:].reshape(-1,1),test.reshape(len(test),1)),axis=0))
            testX, testY = create_dataset(test2, look_back)

            # reshape input to be [samples, time steps, features]
            trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
            testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        elif ValData == 'steps':
            df2 = df
            #TimeAhead = 672 #4semanas
            #Validation should start at 00:00h
            if TimeAhead % 24 != 0:
                 TimeAhead += TimeAhead % 24

            dataset = df2[column].dropna().values
            # normalize the dataset
            scaler = MinMaxScaler(feature_range=(0, 1))
            dataset = scaler.fit_transform(dataset.reshape(len(dataset),1))
            # split into train and test sets
            train_size = len(dataset) - TimeAhead
            #test_size = len(dataset) - train_size
            train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
            # reshape into X=t and Y=t+1
            trainX, trainY = create_dataset(train, look_back)
            testX, testY = create_dataset(np.concatenate((train[-look_back:].reshape(-1,1),test),axis=0), look_back)

            # reshape input to be [samples, time steps, features]
            trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
            testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    else:
         if ValData == 'index':
             df2 = df.loc[(df['day_of_week'] == day) & (df.index < TimeAhead)]
             train = df2[column].dropna().values
             df2 = df.loc[(df['day_of_week'] == day) & (df.index >= TimeAhead)]
             test = df2[column].dropna().values
             # normalize the dataset
             scaler = MinMaxScaler(feature_range=(0, 1))
             scaler.fit(np.concatenate((train.reshape(len(train),1),test.reshape(len(test),1)),axis=0))
             dataset = scaler.transform(train.reshape(len(train),1))
             trainX, trainY = create_dataset(dataset, look_back)

             test2 = scaler.transform(np.concatenate((train[-look_back:].reshape(-1,1),test.reshape(len(test),1)),axis=0))
             testX, testY = create_dataset(test2, look_back)

             # reshape input to be [samples, time steps, features]
             trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
             testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
         elif ValData == 'steps':
             #Validation should start at 00:00h
             if TimeAhead % 24 != 0:
                 TimeAhead += TimeAhead % 24
             df2 = df.loc[df['day_of_week'] == day]
             dataset = df2[column].dropna().values
             # normalize the dataset
             scaler = MinMaxScaler(feature_range=(0, 1))
             dataset = scaler.fit_transform(dataset.reshape(len(dataset),1))
             # split into train and test sets
             train_size = len(dataset) - TimeAhead
             #test_size = len(dataset) - train_size
             train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
             # reshape into X=t and Y=t+1
             trainX, trainY = create_dataset(train, look_back)
             testX, testY = create_dataset(np.concatenate((train[-look_back:].reshape(-1,1),test),axis=0), look_back)

             # reshape input to be [samples, time steps, features]
             trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
             testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    return trainX, trainY, testX, testY, scaler, df2, dataset

def SplitTimeseriesMultipleTimesBackAheadRW(df,column,day = 'Monday', ValData = 'steps', TimeAhead = 96, n_steps_out=1, n_steps_in = 1):
    overlap = 1
    if day == 'All':
        if ValData == 'index':
            df2 = df.loc[df.index < TimeAhead]
            train = df2[column].dropna().values

            df2 = df.loc[df.index >= TimeAhead]
            test = df2[column].dropna().values
            # normalize the dataset
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(np.concatenate((train.reshape(len(train),1),test.reshape(len(test),1)),axis=0))
            dataset = scaler.transform(train.reshape(len(train),1))
            trainX, trainY = create_datasetMultipleTimesBackAhead(dataset, n_steps_out=1, n_steps_in = n_steps_in, overlap = overlap)

            test2 = scaler.transform(np.concatenate((train[-n_steps_in:].reshape(n_steps_in,1),test.reshape(len(test),1)),axis=0))
            testX, testY = create_datasetMultipleTimesBackAhead(test2, n_steps_out=n_steps_out, n_steps_in = n_steps_in, overlap = n_steps_out)

            # reshape input to be [samples, time steps, features]
            trainX = np.reshape(trainX, (trainX.shape[0], n_steps_in, trainX.shape[1]))
            testX = np.reshape(testX, (testX.shape[0], n_steps_in, testX.shape[1]))
            trainY = np.reshape(trainY, (trainY.shape[0], 1, trainY.shape[1]))
            testY = np.reshape(testY, (testY.shape[0], n_steps_out, testY.shape[1]))
        elif ValData == 'steps':
            df2 = df
            #TimeAhead = 2688 #4semanas
            #Validation should start at 00:00h
            if TimeAhead % 24 != 0:
                TimeAhead += TimeAhead % 24

            dataset = df2[column].dropna().values
            # normalize the dataset
            scaler = MinMaxScaler(feature_range=(0, 1))
            dataset = scaler.fit_transform(dataset.reshape(len(dataset),1))
            # split into train and test sets
            train_size = len(dataset) - TimeAhead
            #test_size = len(dataset) - train_size
            train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
            # reshape into X=t and Y=t+1
            
            trainX, trainY = create_datasetMultipleTimesBackAhead(train, n_steps_out=1, n_steps_in = n_steps_in, overlap = overlap)

            test2 = scaler.transform(np.concatenate((train[-n_steps_in:].reshape(n_steps_in,1),test),axis=0))
            testX, testY = create_datasetMultipleTimesBackAhead(test2, n_steps_out=n_steps_out, n_steps_in = n_steps_in, overlap = n_steps_out)
            
            
            
            # reshape input to be [samples, time steps, features]
            trainX = np.reshape(trainX, (trainX.shape[0], n_steps_in, trainX.shape[1]))
            testX = np.reshape(testX, (testX.shape[0], n_steps_in, testX.shape[1]))
            trainY = np.reshape(trainY, (trainY.shape[0], 1, trainY.shape[1]))
            testY = np.reshape(testY, (testY.shape[0], n_steps_out, testY.shape[1]))
    else:
        if ValData == 'index':
            df2 = df.loc[(df['day_of_week'] == day) & (df.index < TimeAhead)]
            train = df2[column].dropna().values
            df2 = df.loc[(df['day_of_week'] == day) & (df.index >= TimeAhead)]
            test = df2[column].dropna().values
            # normalize the dataset
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(np.concatenate((train.reshape(len(train),1),test.reshape(len(test),1)),axis=0))
            dataset = scaler.transform(train.reshape(len(train),1))
            trainX, trainY = create_datasetMultipleTimesBackAhead(dataset, n_steps_out=1, n_steps_in = n_steps_in, overlap = overlap)

            test2 = scaler.transform(np.concatenate((train[-n_steps_in:].reshape(n_steps_in,1),test.reshape(len(test),1)),axis=0))
            testX, testY = create_datasetMultipleTimesBackAhead(test2, n_steps_out=n_steps_out, n_steps_in = n_steps_in, overlap = n_steps_out)

            # reshape input to be [samples, time steps, features]
            
            trainX = np.reshape(trainX, (trainX.shape[0], n_steps_in, 1))
            testX = np.reshape(testX, (testX.shape[0], n_steps_in, 1))
            trainY = np.reshape(trainY, (trainY.shape[0], 1, 1))
            testY = np.reshape(testY, (testY.shape[0], n_steps_out, 1))
        elif ValData == 'steps':
            #Validation should start at 00:00h
            if TimeAhead % 24 != 0:
                TimeAhead += TimeAhead % 24
  
            df2 = df.loc[df['day_of_week'] == day]
            dataset = df2[column].dropna().values
            # normalize the dataset
            scaler = MinMaxScaler(feature_range=(0, 1))
            dataset = scaler.fit_transform(dataset.reshape(len(dataset),1))
            # split into train and test sets
            train_size = len(dataset) - TimeAhead
            #test_size = len(dataset) - train_size
            train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
            # reshape into X=t and Y=t+1
            trainX, trainY = create_datasetMultipleTimesBackAhead(train, n_steps_out=1, n_steps_in = n_steps_in, overlap = overlap)

            test2 = scaler.transform(np.concatenate((train[-n_steps_in:].reshape(n_steps_in,1),test),axis=0))
            testX, testY = create_datasetMultipleTimesBackAhead(test2, n_steps_out=n_steps_out, n_steps_in = n_steps_in, overlap = n_steps_out)
            
            # reshape input to be [samples, time steps, features]
            trainX = np.reshape(trainX, (trainX.shape[0], n_steps_in, trainX.shape[1]))
            testX = np.reshape(testX, (testX.shape[0], n_steps_in, testX.shape[1]))
            trainY = np.reshape(trainY, (trainY.shape[0], 1, trainY.shape[1]))
            testY = np.reshape(testY, (testY.shape[0], n_steps_out, testY.shape[1]))
    return trainX, trainY, testX, testY, scaler, df2, dataset

def SplitTimeseriesMultipleTimesBackAhead(df,column, day = 'Monday', ValData = 'steps', TimeAhead = 96, n_steps_out=1, n_steps_in = 1, overlap = 1):
    if day == 'All':
        if ValData == 'index':
            df2 = df.loc[df.index < TimeAhead]
            train = df2[column].dropna().values

            df2 = df.loc[df.index >= TimeAhead]
            test = df2[column].dropna().values
            # normalize the dataset
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(np.concatenate((train.reshape(len(train),1),test.reshape(len(test),1)),axis=0))
            dataset = scaler.transform(train.reshape(len(train),1))
            trainX, trainY = create_datasetMultipleTimesBackAhead(dataset, n_steps_out=n_steps_out, n_steps_in = n_steps_in, overlap = overlap)

            test2 = scaler.transform(np.concatenate((train[-n_steps_in:].reshape(n_steps_in,1),test.reshape(len(test),1)),axis=0))
            testX, testY = create_datasetMultipleTimesBackAhead(test2, n_steps_out=n_steps_out, n_steps_in = n_steps_in, overlap = overlap)

            # reshape input to be [samples, time steps, features]
            trainX = np.reshape(trainX, (trainX.shape[0], n_steps_in, trainX.shape[1]))
            testX = np.reshape(testX, (testX.shape[0], n_steps_in, testX.shape[1]))
            trainY = np.reshape(trainY, (trainY.shape[0], n_steps_out, trainY.shape[1]))
            testY = np.reshape(testY, (testY.shape[0], n_steps_out, testY.shape[1]))
        elif ValData == 'steps':
            df2 = df
            #TimeAhead = 2688 #4semanas
            #Validation should start at 00:00h
            if TimeAhead % 24 != 0:
                TimeAhead += TimeAhead % 24

            dataset = df2[column].dropna().values
            # normalize the dataset
            scaler = MinMaxScaler(feature_range=(0, 1))
            dataset = scaler.fit_transform(dataset.reshape(len(dataset),1))
            # split into train and test sets
            train_size = len(dataset) - TimeAhead
            #test_size = len(dataset) - train_size
            train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
            # reshape into X=t and Y=t+1
            
            trainX, trainY = create_datasetMultipleTimesBackAhead(train, n_steps_out=n_steps_out, n_steps_in = n_steps_in, overlap = overlap)

            test2 = scaler.transform(np.concatenate((train[-n_steps_in:].reshape(n_steps_in,1),test),axis=0))
            testX, testY = create_datasetMultipleTimesBackAhead(test2, n_steps_out=n_steps_out, n_steps_in = n_steps_in, overlap = overlap)
            
            
            
            # reshape input to be [samples, time steps, features]
            trainX = np.reshape(trainX, (trainX.shape[0], n_steps_in, trainX.shape[1]))
            testX = np.reshape(testX, (testX.shape[0], n_steps_in, testX.shape[1]))
            trainY = np.reshape(trainY, (trainY.shape[0], n_steps_out, trainY.shape[1]))
            testY = np.reshape(testY, (testY.shape[0], n_steps_out, testY.shape[1]))
    else:
        if ValData == 'index':
            df2 = df.loc[(df['day_of_week'] == day) & (df.index < TimeAhead)]
            train = df2[column].dropna().values
            df2 = df.loc[(df['day_of_week'] == day) & (df.index >= TimeAhead)]
            test = df2[column].dropna().values
            # normalize the dataset
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(np.concatenate((train.reshape(len(train),1),test.reshape(len(test),1)),axis=0))
            dataset = scaler.transform(train.reshape(len(train),1))
            trainX, trainY = create_datasetMultipleTimesBackAhead(dataset, n_steps_out=n_steps_out, n_steps_in = n_steps_in, overlap = overlap)

            test2 = scaler.transform(np.concatenate((train[-n_steps_in:].reshape(n_steps_in,1),test.reshape(len(test),1)),axis=0))
            testX, testY = create_datasetMultipleTimesBackAhead(test2, n_steps_out=n_steps_out, n_steps_in = n_steps_in, overlap = overlap)

            # reshape input to be [samples, time steps, features]
            
            trainX = np.reshape(trainX, (trainX.shape[0], n_steps_in, 1))
            testX = np.reshape(testX, (testX.shape[0], n_steps_in, 1))
            trainY = np.reshape(trainY, (trainY.shape[0], n_steps_out, 1))
            testY = np.reshape(testY, (testY.shape[0], n_steps_out, 1))
        elif ValData == 'steps':
            #Validation should start at 00:00h
            if TimeAhead % 24 != 0:
                TimeAhead += TimeAhead % 24
  
            df2 = df.loc[df['day_of_week'] == day]
            dataset = df2[column].dropna().values
            # normalize the dataset
            scaler = MinMaxScaler(feature_range=(0, 1))
            dataset = scaler.fit_transform(dataset.reshape(len(dataset),1))
            # split into train and test sets
            train_size = len(dataset) - TimeAhead
            #test_size = len(dataset) - train_size
            train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
            # reshape into X=t and Y=t+1
            trainX, trainY = create_datasetMultipleTimesBackAhead(train, n_steps_out=n_steps_out, n_steps_in = n_steps_in, overlap = overlap)

            test2 = scaler.transform(np.concatenate((train[-n_steps_in:].reshape(n_steps_in,1),test),axis=0))
            testX, testY = create_datasetMultipleTimesBackAhead(test2, n_steps_out=n_steps_out, n_steps_in = n_steps_in, overlap = overlap)
            
            
            
            # reshape input to be [samples, time steps, features]
            trainX = np.reshape(trainX, (trainX.shape[0], n_steps_in, trainX.shape[1]))
            testX = np.reshape(testX, (testX.shape[0], n_steps_in, testX.shape[1]))
            trainY = np.reshape(trainY, (trainY.shape[0], n_steps_out, trainY.shape[1]))
            testY = np.reshape(testY, (testY.shape[0], n_steps_out, testY.shape[1]))
    return trainX, trainY, testX, testY, scaler, df2, dataset