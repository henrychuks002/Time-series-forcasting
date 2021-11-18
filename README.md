# Time-series-forcasting

### Content
According to Ahmed Adam415, The data was prepared using Alpha Vantage API
The data represent historical daily time series for digital currency (BTC) traded on the Saudi market
The prices and volumes are quoted both in SAR and USD and the date ranges from 7th May 2018 - 30th January 2021

### Objectives
To perform time series analysis/forecasting on the obtained dataset, predicting future daily stock pricees of BTC from February 2021

### Concept
Most of us would have heard and already know about the cryptocurrency buzz in the online market and most of use would've invested in them, including businesses, both small, medium and large enterprices. But is investing in such volatile currency safe? How can one make sure that investing in these currencies won't put them at a loss in the future. We can't be sure, but we can surely generate as approximate value based on the previous prices. Time series modelling/Forecasting is one way to predict time.

### About the data
The dataset originally contains 10 rows and 1000 columns. Data dictionary as follows;
Time: Time from 07-05-2011 to 30-01-2021
Open_SAR: Opening BTC stock in Saudi Riyal currency from each day corresponding to the time column
Open_USD: Opening BTC stock in USD currency from each day corresponding to the time column
High_SAR: Maximum BTC price in SAR currency for each day, highest price at which BTC is trading in SAR
Low_USD: Minimum BTC price in USD currency for each day, highest price at which BTC is trading in USD
Close_SAR: Closing BTC stock price in SAR currency for each day corresponding in the time column
Close_USD: Closing BTC stock price is USD currency for each day corresponding in the time column
Volume: Total number of BTC shares traded in a security over a period (daily, corresponding to the time column)

### Method used and Processes
1. Making the data stationary: Time series requires the data to be stationary. If the time series has a particular behaviour overtime, therer is a very high probability that it will follow the same in the future. So in order to apply time series to dataset, the data has to be stationary. There has to be a constant mean, a constant variance, an auto-covariance that does not depend on time.

There are different ways to test if a dataset is stationary or not, For this project, I have used Rolling statistics and ADCF (Augmented Dickey Fuller) test for stationary to check if the dataset used is stationary or not.
The code snippet below shows the rolling statistics on original dataset;

```
#Test to checking for stationarity using the rolling statistics method
rollmean = dataset.rolling(window=12).mean() #getting the rolling mean for a window of 12 months in a year for a monthly based time series dataset
rollstd = dataset.rolling(window=12).std() #getting the rolling standard deviation
print(rollmean, rollstd)

#Plotting the original data against rolling mean and standard deviation for visualization purpose
original_data = plt.plot(dataset, color='blue', label='Original')
mean = plt.plot(rollmean, color='red', label='Rolling mean')
std = plt.plot(rollstd, color='black', label='Rolling std')
plt.legend(loc='best')
plt.title('Rolling mean and standard deviation')
plt.xlabel('Time')
plt.ylabel('Closing stock')
plt.xticks(rotation='vertical')
plt.grid()
plt.show(block=False)

```
![First rolling mean and std](https://user-images.githubusercontent.com/65792408/142500830-57dd849b-eefc-4a56-8ef0-ef95641f650f.png)

The above plot shows the output of the rolling statistics method of testing for stationarity. This shows that the data is non-stationary because there's no constant mean and variance.

Now using ADCF stationarity test on the original dataset to test for stationarity;
The code snippet below shows this

```
#performing the augmented dickey fuller test to check for stationarity in the data too
from statsmodels.tsa.stattools import adfuller
print('Results of Dickey Fuller Test')
dftest = adfuller(dataset['close_USD'], autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test statistics', 'p-value', '#Lags used', 'Number of Observation used'])
for key, value in dftest[4].items():
    dfoutput['critical value (%s)'%key] = value
    
print(dfoutput)

```

The result of the above code snippet shows that;
Test statistics               2.547
p-value                       0.999
#lags used                    4.00
Number of obervations used    28.00
Critical value (1%)          -3.68889
Critical value (5%)          -2.9719
Critical value (10%)         -2.6252

This results shows that the dataset is non-stationary. Now let me explain this; Testing for stationarity using ADCF test. The test results comprise of a test statistics and some critical values as seen above, now if the test statistics is less than some critical values, then we can reject the null hypothesis and say that the data is stationary. As seen above, the value of the "test statistics" is 2.547 and all negatives for Critical values, meaning that it greater than all critical and however, we can't reject the hypothesis.

Now the aims of this project becomes as follows;
1. Making the data stationary
2. Identifying the values of p, d and q. These are the most essential parameter for one of time series models which I used for this project, the ARIMA model. p and q values are determined using pacf and acf plots respectively and d can take values of other 0, 1 and 2
3. Third is build the model and provide the forecasted resulted ie the forecasted prices of BTC starting from Feb 2021

For this project, I have some time series model for the forecasting, such as ARIMA model, AUTO-ARIMA model, Simple exponential smoothing, Exponential smoothing model, Holt's Linear Trend model.

Check out the entire pipeline in the jupyter notebook attached to this repository

This project is solely authored by me. Still have little issue in it, but I'm open to corrections, suggestions and collaboration!
Contact me me at barrychukwu12@gmail.com
