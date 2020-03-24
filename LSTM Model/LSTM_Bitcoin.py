# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 11:18:15 2019

@author: pranav
"""
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime as DT
import datetime
from datetime import date, timedelta

import numpy as np
import pandas as pd
import statsmodels.api as sm
from dateutil import parser
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
import plotly.offline as py
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
import requests
import lxml.html as lh
import pandas as pd
import arrow

def getDataSet_Scraper():
    date = arrow.now().format('YYYYMMDD')
    url= 'https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end=' + date
    
    #url='https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end=20181118'
    
    #Create a handle, page, to handle the contents of the website
    page = requests.get(url)
    #Store the contents of the website under doc
    doc = lh.fromstring(page.content)
    #Parse data that are stored between <tr>..</tr> of HTML
    tr_elements = doc.xpath('//tr')
    
    #Check the length of the first 12 rows
    print(len(T) for T in tr_elements[:12])
    
    tr_elements = doc.xpath('//tr')
    #Create empty list
    col=[]
    i=0
    #For each row, store each first element (header) and an empty list
    for t in tr_elements[0]:
        i+=1
        name=t.text_content()
        print ('%d:"%s"'%(i,name))
        col.append((name,[]))
        
        print(col)
        
        #Since out first row is the header, data is stored on the second row onwards
    for j in range(1,len(tr_elements)):
        #T is our j'th row
        T=tr_elements[j]
        
        #If row is not of size 10, the //tr data is not from our table 
        #if len(T)!=10:
            #break
        
        #i is the index of our column
        i=0
        
        #Iterate through each element of the row
        for t in T.iterchildren():
            data=t.text_content() 
            #Check if row is empty
            if i>0:
            #Convert any numerical value to integers
                try:
                    data=int(data)
                except:
                    pass
            #Append the data to the empty list of the i'th column
            col[i][1].append(data)
            #Increment i for the next column
            i+=1
            
    Dict = {title:column for (title,column) in col}
    print(len(Dict['Date']))
    print(len(Dict['Open*']))
    print(len(Dict['High']))
    print(len(Dict['Low']))
    print(len(Dict['Close**']))
    print(len(Dict['Volume']))
    print(len(Dict['Market Cap']))
    #print(Dict['Open*'])
    
    Dict['Date'].pop(0)
    Dict['Date'].pop(0)
    Dict['Open*'].pop(0)
    Dict['High'].pop(0)
    Dict['Low'].pop(0)
    Dict['Close**'].pop(0)
    Dict['Volume'].pop(0)
    Dict['Market Cap'].pop(0)
    df=pd.DataFrame(Dict)
    
    df['Open*'] = df['Open*'].apply(lambda x: float(x.split()[0].replace(',', '')))
    df['High'] = df['High'].apply(lambda x: float(x.split()[0].replace(',', '')))
    df['Low'] = df['Low'].apply(lambda x: float(x.split()[0].replace(',', '')))
    df['Close**'] = df['Close**'].apply(lambda x: float(x.split()[0].replace(',', '')))
    
    df.astype({'Open*': 'float64'}).dtypes
    df.astype({'High': 'float64'}).dtypes
    df.astype({'Low': 'float64'}).dtypes
    df.astype({'Close**': 'float64'}).dtypes
    
    print(df)
    print(df.dtypes)
    
    
    df.to_csv("coinmarket.csv")
    print("DataSet Scraped and Downloaded Successfully!! 'coinmarket.csv' ")

def predictions(df, Daily_Price, count, loop_counter):
    
    #Split the dataset to test and train
    #df_test = Daily_Price[:(len(Daily_Price))]
    #df_train= Daily_Price
    #train, test = train_test_split(Daily_Price, test_size=0.2 ,shuffle = False, stratify = None)
    
    df_train = Daily_Price[round((len(Daily_Price)*0.20)):]     #80% data to train
    df_test = Daily_Price[:round((len(Daily_Price)*0.20))]      #20% Data to test on
    #train = Daily_Price[(len(Daily_Price)*0.80):(len(Daily_Price))]


    print(df_train)
    print("df_train Length = " + str(len(df_train)))
        
    print("\n\ndf_test Length = " + str(len(df_test)))
    print(df_test)
    
    top_date = df_test.index[0]
    top_date_1 = parser.parse(top_date)
    plus_two_days_series = pd.Series([0.0],index=[((top_date_1 + datetime.timedelta(days=1)).strftime('%d-%b-%y'))])
    #plus_two_days_series
    df_test = plus_two_days_series.append(df_test)
    print("df_test length: "+str(len(df_test)))
    working_data = [plus_two_days_series,df_train]
    working_data = pd.concat(working_data)
    working_data = working_data.reset_index()
    working_data['index'] = pd.to_datetime(working_data['index'])
    working_data = working_data.set_index('index')
    
        #Seasonal Decomposition
    s = sm.tsa.seasonal_decompose(df.Close.values, freq=60)
    if count == loop_counter-1:
        print("Inside Seasonal Decomposition Graph Preparor: "+ str(count))
        trace1 = go.Scatter(x = np.arange(0, len(s.trend), 1),y = s.trend,mode = 'lines',name = 'Trend',
            line = dict(color = ('rgb(244, 146, 65)'), width = 4))
        trace2 = go.Scatter(x = np.arange(0, len(s.seasonal), 1),y = s.seasonal,mode = 'lines',name = 'Seasonal',
            line = dict(color = ('rgb(66, 244, 155)'), width = 2))
        
        trace3 = go.Scatter(x = np.arange(0, len(s.resid), 1),y = s.resid,mode = 'lines',name = 'Residual',
            line = dict(color = ('rgb(209, 244, 66)'), width = 2))
        
        trace4 = go.Scatter(x = np.arange(0, len(s.observed), 1),y = s.observed,mode = 'lines',name = 'Observed',
            line = dict(color = ('rgb(66, 134, 244)'), width = 2))
        
        data = [trace1, trace2, trace3, trace4]
        layout = dict(title = 'Seasonal Decomposition', xaxis = dict(title = 'Time'), yaxis = dict(title = 'Price, USD'))
        fig = dict(data=data, layout=layout)
        py.plot(fig, filename='Seasonal_Decomposition')

    def create_lookback(dataset, check_previous):
        X, Y = [], []
        for i in range(len(dataset) - check_previous):
            a = dataset[i:(i + check_previous), 0]
            X.append(a)
            Y.append(dataset[i + check_previous, 0])
        return np.array(X), np.array(Y)
    
    training_set = df_train.values
    len(training_set)
    training_set = np.reshape(training_set, (df_train.size , 1))
    test_set = df_test.values
    test_set = np.reshape(test_set, (df_test.size, 1))
    
    #scale datasets
    scaler = MinMaxScaler()
    training_set = scaler.fit_transform(training_set)
    test_set = scaler.transform(test_set)
    
    look_back = 1
    X_train, Y_train = create_lookback(training_set, look_back)
    X_test, Y_test = create_lookback(test_set, look_back)
    X_train = np.reshape(X_train, (len(X_train), 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (len(X_test), 1, X_test.shape[1]))
    
    model = Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(256))
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(X_train, Y_train, epochs=100, batch_size=2000, shuffle=False, 
                        validation_data=(X_test, Y_test), 
                        callbacks = [EarlyStopping(monitor='val_loss', min_delta=9e-9, patience=50, verbose=1)])
    
    #if count == loop_counter-1:
    print("Inside loss graph: "+ str(count))
    trace1 = go.Scatter(
        x = np.arange(0, len(history.history['loss']), 1),
        y = history.history['loss'],
        mode = 'lines',
        name = 'Train loss',
        line = dict(color=('rgb(66, 244, 155)'), width=2, dash='dash')
    )
    trace2 = go.Scatter(
        x = np.arange(0, len(history.history['val_loss']), 1),
        y = history.history['val_loss'],
        mode = 'lines',
        name = 'Test loss',
        line = dict(color=('rgb(244, 146, 65)'), width=2)
    )
    
    data = [trace1, trace2]
    
    if count == loop_counter-1:  #Plotting Graph
        layout = dict(title = 'Train and Test Loss during training', 
                      xaxis = dict(title = 'Epoch number'), yaxis = dict(title = 'Loss'))
        fig = dict(data=data, layout=layout)
        py.plot(fig, filename='Training_process(LOSS_Graph)')
    
    # add one additional data point to align shapes of the predictions and true labels
    print("Printing size 'X_test' " +str(X_test.size))
    #X_test = np.append(X_test, scaler.transform(working_data.iloc[-1][0]))
    X_test = np.reshape(X_test, (len(X_test), 1, 1))
    print("Printing size 'X_Test' " +str(X_test.size))
    print("Printing size 'Working Data' " +str(working_data.size))
    
    # get predictions and then make some transformations to be able to calculate RMSE properly in USD
    prediction = model.predict(X_test)
    prediction_inverse = scaler.inverse_transform(prediction.reshape(-1, 1))
    Y_test_inverse = scaler.inverse_transform(Y_test.reshape(-1, 1))
    #prediction2_inverse = np.array(prediction_inverse[:,0])
    prediction2_inverse = np.array(prediction_inverse[:,0][1:])
    Y_test2_inverse = np.array(Y_test_inverse[:,0])
    

    #PBG_chng
    Y_test2_inv = ([float('naN')])
    Y_test2_inv_new = Y_test2_inverse[0:((len(Y_test2_inverse))-1)]   
    Y_test_inv_test = np.concatenate((Y_test2_inv,Y_test2_inv_new))
    
    #print(prediction2_inverse.shape)
    #print(Y_test2_inverse.shape)
    #print(Y_test_inv_test.shape)
    #print(Y_test2_inv_new.shape)
    #temp = ([float(Y_test2_inverse[0])])
    #prediction2_inv_new = prediction2_inverse[0:((len(prediction2_inverse))-1)]
    #pred2_invrs_pred = np.concatenate((temp , prediction2_inverse))
    #pred2_invrs_pred = prediction2_inverse
    #pred2_invrs_pred[0] = float(Y_test2_inverse[0])
    
    print("Inside result graph: "+str(count))
    trace1 = go.Scatter(
        x = np.arange(0, len(prediction2_inverse), 1),
        y = prediction2_inverse,
        mode = 'lines',
        name = 'Predicted price',
        hoverlabel= dict(namelength=-1),
        line = dict(color=('rgb(244, 146, 65)'), width=2)
    )
    trace2 = go.Scatter(
        x = np.arange(0, len(Y_test2_inv_new), 1),
        y = Y_test2_inv_new,
        mode = 'lines',
        name = 'True price',
        line = dict(color=('rgb(66, 244, 155)'), width=2)
    )
    #print(trace1)
    #print(trace2)
    data = [trace1, trace2]
    #print(data)
    
    if count == loop_counter-1: #Prining Graph 2
        layout = dict(title = 'Comparison of true prices with prices our model predicted (on Entire TEST Data)',
                     xaxis = dict(title = 'Day number'), yaxis = dict(title = 'Price, USD'))
        fig = dict(data=data, layout=layout)
        py.plot(fig, filename='Prediction_On_Training_Data')
        
        #print(Y_test2_inv_new)
        #print(prediction2_inverse)
        #print(Y_test2_inv_new.shape)
        #print(prediction2_inverse.shape)
        error = mean_squared_error(Y_test2_inv_new, prediction2_inverse)
        #error = mean_squared_error(Y_test2_inverse, pred2_invrs_pred)
        print ("\n\nMean Squared Error is :: %.2f " % error)

        #Accuracy = accuracy_score(Y_test2_inv_new,prediction2_inverse)
        #print(Accuracy)
        #print("Accuracy = " + str(Accuracy * 100))
        R2_Score = r2_score(Y_test2_inv_new, prediction2_inverse)
        #R2_Score = r2_score(Y_test2_inverse, pred2_invrs_pred)
        print ("\nR2 Score : %.2f" % R2_Score + "\n\n")

    
    
    #mean = np.mean((Y_test2_inv_new[:10])-((np.round((prediction2_inverse[:(len(prediction2_inverse)-1)]),decimals=2,out=None)))[:10])
    
    
    #prediction2_inverse = prediction2_inverse[3:]
    
    prediction_after_mean = prediction2_inverse[:(loop_counter)]
    Actual_Data = np.roll(Y_test2_inv_new,1, axis=None)
    Actual_Data[0] = np.nan
    Pre_Predicted_Data = (np.round_((prediction2_inverse[:(len(prediction2_inverse)-1)]), decimals=2,out=None))
    #Bias_Adjusted_Prediction = ((((Actual_Data - Pre_Predicted_Data)+Pre_Predicted_Data))-100)
    Bias_Adjusted_Prediction = Pre_Predicted_Data
    temp = ([float('naN')])
    Bias_Adjusted_Prediction = np.concatenate((Bias_Adjusted_Prediction,temp))
    Bias_Adjusted_Prediction[0]=np.round(Pre_Predicted_Data[0],decimals=2,out=None)
    Bias_Adjusted_Prediction_Final = np.concatenate((prediction_after_mean,Bias_Adjusted_Prediction[(loop_counter):]))
    
    for i in range (loop_counter):
        Y_test_inv_test[i] = float('naN')
    
    #df_test = Daily_Price[:2000]
    
        
        
    if count == loop_counter-1: #Printing final Graph
        Test_Dates = df_test.index[:(loop_counter+10)]
        
        Test_Dates = Test_Dates[::-1]
        Y_test_inv_test = Y_test_inv_test[::-1]
        Bias_Adjusted_Prediction_Final = Bias_Adjusted_Prediction_Final[::-1]
        Y_test_inv_test = Y_test_inv_test[(len(Y_test_inv_test))-(len(Test_Dates)):]
        Bias_Adjusted_Prediction_Final = Bias_Adjusted_Prediction_Final[((len(Bias_Adjusted_Prediction_Final))-(len(Test_Dates))):]
        
        Predicted_Price = Daily_Price
        
    
        Predicted_Price = Predicted_Price.append(pd.Series([prediction2_inverse[0]],index=[df_test.index[0]]))
        
        Predicted_Price = np.roll(Predicted_Price,1, axis=None)
        #Predicted_Price[0] = np.nan
        
        Predicted_Price = Predicted_Price[::-1]
        Predicted_Price = Predicted_Price[((len(Predicted_Price))-(len(Test_Dates))):]
        
        length = (len(Predicted_Price)-(loop_counter+1))
        for i in range(length):
            Predicted_Price[i] = np.nan
        
        trace1 = go.Scatter(x=Test_Dates, y=Y_test_inv_test, name= 'Actual Price', 
                           line = dict(color = ('rgb(66, 244, 155)'),width = 2))
        trace2 = go.Scatter(x=Test_Dates, y=Predicted_Price, name= 'Predicted Price',
                           line = dict(color = ('rgb(244, 146, 65)'),width = 2))
        data = [trace1, trace2]
        
       
        layout = dict(title = 'Actual Future Predictions (till the date user requested)',
                     xaxis = dict(title = 'Date'), yaxis = dict(title = 'Price, USD'))
        fig = dict(data=data, layout=layout)
        py.plot(fig, filename='Final_Prediction_for_User')

    
    next_day_predicted =  pd.Series([prediction2_inverse[0]],index=[df_test.index[0]])
    return next_day_predicted


def init(input_date):
    getDataSet_Scraper()
    py.init_notebook_mode(connected=True)
    df = pd.read_csv('coinmarket.csv')
    df.columns=['','Date','Open','High','Low','Close','Volume','Market Cap']
    print(df[:5])
    print("Dataset Size :: " + str(len(df)) + "Rows")
    
    #Series is generated for closing values and Date as index
    Daily_Price = pd.Series(df['Close'].values,index=df['Date'])
    
    now = (DT.today()).date()
    input_date = DT.strptime(input_date,'%Y-%m-%d')
    input_date = input_date.date()

    yesterday = now - timedelta(days = 1) # date - days
    
    print("\nToday's Date : "+str(now))
    #print("type : "+str(type(now)))
    print("Yesterday (Data in dataset till date) : "+str(yesterday))
    print("\n\nDate till which future predictions are to be made : "+str(input_date) + "\n\n")
    #print("Type : "+str(type(input_date)))


    
    #print(df.Close.values)
    #print(len(df.Close.values))
    
    loop_counter = abs((input_date - yesterday).days)
    for i in range(loop_counter):
        print("**$$ Recurrenc No. (Predicting Value for Day->) : "+str(i+1))
        next_day_predicted = predictions(df, Daily_Price, i,loop_counter)
        print("Iteration: "+str(i))
        print("Next Day Predicted: "+ str(next_day_predicted))
        Daily_Price = next_day_predicted.append(Daily_Price)
        array=[]
        array.insert(0, {'': 0, 'Date': next_day_predicted.index[0], 'Open': 0,'High': 0, 'Low':0, 'Close': next_day_predicted[0], 'Volume':0, 'Market Cap': 0})
        df = pd.concat([pd.DataFrame(array),df], ignore_index=True)
        len(df)
        
        if i <= (loop_counter-2) :
            print("Next Dy Predicted Series: \n")
            print(next_day_predicted)
            print("\n\nDaily Price Head :\n")
            print(Daily_Price.head(10))
            print("\n\n DF :\n")
            print(df.head(10))


#now = (DT.today()).date()    
#input_date = now + timedelta(days=1)   #This will be replaced by HTML input date 
#print(str(input_date))
#init(input_date)
input_date = '2019-12-20'
init(input_date)