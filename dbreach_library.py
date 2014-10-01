import numpy as np
import pandas as pd
import pylab as pl
from scipy import stats,optimize,math
import csv
from datetime import datetime
import boto
import json
import pandas
import time
import sys
import re

sys.path.append("/home/ubuntu/python/")
from tm_python_lib import * 
bucketName = "databreaches"
s3 = boto.connect_s3()
bucket = s3.get_bucket(bucketName)

def loadInternetUsers():
    
    key = bucket.get_key("json/internetUsers.csv")
    data = key.get_contents_as_string()
    
    data = re.sub(" ","",data)
    
    date = []
    users = []
    
    for line in data.splitlines():
        l = line.split('\t')
        date.append(l[0])
        users.append(l[1])
    
    users = np.array(map(int,users))
    date = [time.mktime(datetime.strptime(x,"%m/%d/%Y").timetuple()) for x in date]
    date =  np.array(date)/3600./24
    t2000 = time.mktime(datetime.strptime("2000-01-01",'%Y-%m-%d').timetuple())/3600/24
    date = date - t2000

    c = date > 0
    
    return {'date' : date[c],'users' : users[c]}


def loadFacebookUsers():
    
    key = bucket.get_key("json/facebookUsers.csv")
    data = key.get_contents_as_string()


    date = []
    users = []
    
    for line in data.splitlines():
        l = line.split(',')
        date.append(l[0])
        users.append(l[1])

    users = np.array(map(int,users))
    date = [time.mktime(datetime.strptime(x,"%d/%m/%Y").timetuple()) for x in date]
    date =  np.array(date)/3600./24
    t2000 = time.mktime(datetime.strptime("2000-01-01",'%Y-%m-%d').timetuple())/3600/24
    date = date - t2000

    c = date > 0
    
    return {'date' : date[c],'users' : users[c]}


def loadBreachData():
    ''' Load json from S3'''
    bucketName = "databreaches"
    s3 = boto.connect_s3()
    bucket = s3.get_bucket(bucketName)
    key = bucket.get_key("json/data.json")
    data = key.get_contents_as_string()
    data = json.loads(data)
    
    ''' Make pandas dataframe'''
    df = pandas.DataFrame(data).T
    
    '''Replace missing dates by closest date available'''
    df.date[df.date==-1]=df.discovered[df.date==-1]
    df.date[df.date==-1]=df.reported[df.date==-1]
    #df.date[df.date==-1]=df.posted[df.date==-1]
    
    incorrectDates = ['1012-11-07','1903-08-20','3007-12-03','2212-08-13']
    correctedDates = ['2012-11-07',-1,'2007-12-03','2012-08-13']
    
    for i,ix in enumerate(incorrectDates):
        df.date[df.date==ix] = correctedDates[i]    
     
    '''correct specific events'''

    incorrectIds = ['3782','9235'] 
    correctedDates = ['2011-05-27','2013-03-04']
    
    for i,ix in enumerate(incorrectIds):
        df.date[df.id == ix] = correctedDates[i]
    
    '''remove all events for which no loss was recorded''' 
    df = df[df.records>0]
        
    '''Create a new column with datestamps'''
    x= list(df.date.values[df.date.values!=-1])
    inputFormat = '%Y-%m-%d'
    x = map(lambda x: datetime.strptime(x,inputFormat),x)
    timestamps = []
    for i,ix in enumerate(df.date):
        try:
            ix = datetime.strptime(ix,inputFormat)
            ix = time.mktime(ix.timetuple())
            timestamps.append(ix)
        except:
            timestamps.append(ix)
    datestamp = np.array(timestamps)/3600./24
    df['datestamp'] = datestamp
    
    
    '''remove rows for dates < 2000'''
    t2000 = time.mktime(datetime.strptime("2000-01-01",'%Y-%m-%d').timetuple())/3600/24
    df = df[df.datestamp > t2000]
    minDateStamp = min(df['datestamp'])
    df['datestamp'] = df['datestamp']-minDateStamp
    df = df.sort(columns=['datestamp','id'])
    
    df['logLoss'] = np.log10(list(df['records'].values))
    cumLoss =  df['records'].cumsum()
    df['cumLoss'] = map(int,cumLoss)
    
    df.index=np.arange(len(df))
    df['cumEvents'] = df.index + 1

    df.records = map(int,df.records)
    df.id = map(int,df.id)

    df['submissionLag'] = submitDelay(df)
    
    return df


def submitDelay(dataframe):
    '''Compute timelag between event occurrence and submission on datalossdb'''
    B = binning(np.array(map(int,dataframe['id'].values)),np.array(map(int,dataframe['datestamp'].values)),150,confinter=0.01)
    interpolation = np.interp(np.array(map(int,dataframe['id'].values)),B[0],B[-2])
    dt = np.array(interpolation) - np.array(map(int,dataframe.datestamp.values))
    dt = map(int,np.round(dt))
    return dt
    

def linGressFit(x,y,typFit='linlin',plot=False):
    
    if typFit == 'linlog' or typFit == 'loglin':
        c = y > 0
        x = x[c]
        y = np.log10(y[c])


    if typFit == 'loglog':
        c = (x>0)*(y>0)
        x = np.log10(x[c])
        y = np.log10(y[c])

    fit = stats.linregress(x,y)
    
    if plot:
        pl.plot(x,y)
        pl.plot(x,x*fit[0]+fit[1],'k-')
        pl.text(min(x)*1.03,min(y),"slope=%.2f (R=%.2f,p=%.2f)"%(fit[0],fit[2],fit[3]))
    return fit


def fitCumulative(df,threshold=0,normalize=False):
    t = np.array(map(int,df['datestamp'].values))
    e = np.array(map(int,df['cumEvents'].values))
    l = np.array(map(int,df['cumLoss'].values))

    if normalize:
        e = e/np.std(e)
        l = l/np.std(l)
    
    c = t > threshold
    pl.figure(1,(15,7))
    pl.subplot(121)
    fitE = linGressFit(t[c],e[c],typFit='loglog',plot=True)
    print "scaling exponent cumEvents: ",fitE
    pl.xlabel("log10(days)")
    pl.ylabel("log10(countEvents)")
    pl.subplot(122)
    fitL =  linGressFit(t[c],l[c],typFit='loglog',plot=True)
    print "scaling exponent cumLoss:", fitL
    pl.xlabel("log10(days)")
    pl.ylabel("log10(cumulativeLoss)")
    
    return fitE,fitL
    