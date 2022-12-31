import pandas as pd
import numpy as np
import farmhash
import re
import os

from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline



### Date decomposition
def date_components(X,dates):
    
    
    
    X[dates] = pd.to_datetime(X[dates])
    X[dates+"_year"] = pd.DatetimeIndex(X[dates]).year
    X[dates+"_month"] = pd.DatetimeIndex(X[dates]).month
    X[dates+"_week"] = pd.DatetimeIndex(X[dates]).week
    X[dates+"_weekday"] = pd.DatetimeIndex(X[dates]).day_of_week
    X[dates+"_hour"] = pd.DatetimeIndex(X[dates]).hour
    X[dates+"_caldate"] = pd.DatetimeIndex(X[dates]).date
    X[dates+"_day"] = pd.DatetimeIndex(X[dates]).day

    return X




### Avg Daily txn Volume transformer
class Aggregator(BaseEstimator,TransformerMixin):
    
    def __init__(self,partition_col,agg_col = None,date_col = ["trans_date_trans_time_caldate"],agg_type = "mean",agg_value = "count"):
        
        self.partition_col = partition_col
        self.agg_type = agg_type
        self.date_col =  date_col
        self.agg_value = agg_value
        self.agg_col = agg_col
        
        
    def fit(self,X, y = None):
        
        X = X.copy()
        
        container = []
        for i in self.partition_col:
            for d in self.date_col:
            
                if self.agg_col  ==  None:

                    daily_sum = X.groupby([i,d] ).agg(value = (i, self.agg_value )).reset_index()

                else:

                    daily_sum = X.groupby([i,d] ).agg(value = (self.agg_col,self.agg_value )).reset_index()

                avg_daily_vol = daily_sum.groupby([i]).agg(avg_value=("value",self.agg_type)).to_dict()["avg_value"]
                container.append((avg_daily_vol,i,d))

            self.container = container
        
        return self
    
    def transform(self, X):
        
        X = X.copy()
        
        
                             
        for a,b,c in self.container:
            
            date_val = re.findall("(caldate|month|year|hour|weekday|week|day)",c)[0]
        
            X[b+"_"+self.agg_type+"_"+date_val+"_"+self.agg_value+"_"+str(self.agg_col)] = X[b].map(a)
            X[b+"_"+self.agg_type+"_"+date_val+"_"+self.agg_value+"_"+str(self.agg_col)].fillna(1,inplace = True)
        
        return X
        
        
    

### Avg Daily amount per txn


class AggAmtperTxn(BaseEstimator,TransformerMixin):
    
    def __init__(self,partition_col,agg_col = "amt",date_col = ["trans_date_trans_time_caldate"],agg_type = "mean"):
        
        self.partition_col = partition_col
        self.agg_type = agg_type
        self.date_col =  date_col
        self.agg_col = agg_col
        
        
    def fit(self,X, y = None):
        
        X = X.copy()
        
        container = []
        for i in self.partition_col:
            for d in self.date_col:
        
                daily_sum = X.groupby([i,d] ).agg(value = (self.agg_col, 'sum' ),count = (i,'count')).reset_index()

                daily_sum["amt_per_txn"] = daily_sum.apply(lambda x: x["value"]/x["count"], axis = 1)

                avg_daily_sum_vol = daily_sum.groupby([i]).agg(avg_value=("amt_per_txn",self.agg_type)).to_dict()["avg_value"]
                container.append((avg_daily_sum_vol,i,d) )

        self.container = container
        
        return self
    
    def transform(self, X):
        
        X = X.copy()
        
        
                             
        for a,b,c in self.container:
            
            date_val = re.findall("(caldate|month|year|hour|weekday|week|day)",c)[0]
        
            X[b+"_"+self.agg_type+"_"+date_val+"_"+self.agg_col+"_per_txn"]  = X[b].map(a)
            X[b+"_"+self.agg_type+"_"+date_val+"_"+self.agg_col+"_per_txn"] .fillna(1,inplace = True)
        
        return X

    
### hash data 
class FarmHash(BaseEstimator,TransformerMixin):
    
    def __init__(self,col):
        
        self.col = col
        
    
    def fit(self,X,y = None):
        
        return self
            
            
            
    
    def transform(self,X):
        
        
        X = X.copy()
        
        for c in self.col:
            
            X[c] = X[c].map(lambda x: farmhash.fingerprint32(str(x) ))
            
        return X
        
        
#### date decomposition

class DateDecomp(BaseEstimator,TransformerMixin):
    
    def __init__(self,col):
        
        self.col = col
        
    def fit(self,X,y = None):
        
        return self
    
    def transform(self,X):
        
        X = X.copy()
        
        for c in self.col:
            result = date_components(X,c)
        
        return result
    
    

#### fraud frequency 
    
class FraudFreq(BaseEstimator,TransformerMixin):
    
    def __init__(self, col):
        
        
        self.col = col
        
        
    def fit(self,X,y= None):
        
        
        container = []
        X=  X.copy()
        X["is_fraud"] = y
        
        
        for c in self.col:
            
            freq = X.groupby(c)["is_fraud"].mean().to_dict()
            container.append(freq)
        
        self.freq = container
        
        return self
    
    def transform(self,X):
        
        X = X.copy()
        
        for f,c in zip(self.freq,self.col) :
            
            X[c+"_fraud_freq"] = X[c].map(f)
            
        return X
    
    
class ValueLength(BaseEstimator,TransformerMixin):
    
    def __init__(self,col):
        
        self.col = col
        
    def fit(self,X,y = None):
        
        return self
    
    def transform(self,X):
        
        X = X.copy()
        
        for c in self.col:
            
            X[c+"_length"] = X[c].map(lambda x: len(str(x)))
                            
        return X
            
        
        
class WeekCategory(BaseEstimator,TransformerMixin):
    
    def __init__(self,col):
        
        self.col = col
        
    def fit(self,X,y = None):
        
        return self 
    
    def transform(self,X):
        
        X = X.copy()
        
        
        for c in self.col:
            
            X[c+"_category"] =  X[c].map(lambda x: "Weekend" if x in [0,6]  else "start_end_week" if x in [1,5] else "midweek")
        
        return X
        
        
class PurchaseType(BaseEstimator,TransformerMixin):
    
    def __init__(self,col):
        
        self.col = col
    
    def fit(self,X,y = None):
        
        return self
    
    def transform(self,X):
        
        X = X.copy()
        
        for c in self.col:
            
            X[c+"_purchase_type"] = X[c].map(lambda x: "net" if re.search("net",str(x) ) else "pos" if re.search("pos",str(x) ) else "other")
            
        return X

class ReplaceNaN(BaseEstimator,TransformerMixin):
    
    def __init__(self):
        
        pass
        
    def fit(self,X,y=None):
        
        return self
    
    def transform(self,X):
        
        X = X.copy()
        
        for c in X.columns.tolist():
            
            X[c] = X[c].replace(to_replace = [np.inf,-np.inf,np.nan],value = 0)
            
        return X
    
    
class SelectFeatures(BaseEstimator,TransformerMixin):
    
    def __init__(self,cols):
        
        self.cols = cols
        
    def fit(self,X,y = None):
        
        return self
    
    def transform(self,X):
        
        X = X.copy()
        
        return X[self.cols]
    