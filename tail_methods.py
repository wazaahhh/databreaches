'''This script is an implementation of 

Pisarenko, V. F., Sornette, A., Sornette, D. & Rodkin, M. V. 
Characterization of the tail of the distribution of earthquake magnitudes 
by combining the GEV and GPD descriptions of extreme value theory (2008).


'''


import numpy as np
import pandas
from random import choice,randrange, shuffle
import math

def magnitudeQuantiles(H,s,xi,Lambda,t,q):
    '''returns quantiles of the Distribution Function of magnitude [Formula 10]'''
    Q = H -(s/xi)*(1-(Lambda*tau/np.log(1./q))**xi)
    return Q
    
def randGEV(m,s,xi,n):
    '''generates a random sample of the GEV distribution'''
    Fx = np.random.rand(n)
    sample = []
    i=0
    while i<n:
        value = -((np.log(np.random.rand())+1)**(-xi))*s/xi + m
        if not np.isnan(value):
            sample.append(value)
            i+=1
    return np.array(sample)


def findTmaxima(df,T=100):
    '''From input Pandas DataFrame, find maxima in each period of time T '''
    
    dates = np.array(df.datestamp)      
    bins = np.arange(0,len(dates)+1,T)
    groups = df.groupby(np.digitize(df.index, bins))
    Tmaxima = np.array(groups.max().records)
    logTmaxima = np.log10(Tmaxima)
    
    return {'Tmaxima':Tmaxima, 'logTmaxima' : logTmaxima,'T' : T}


def chooseTmaximaRange(self):
    print "to do (c.f. page 11 article)"

def ThreeMoments(sample):
    '''find the three moments of the sample'''
    n = len(sample)
    M1 = (1./n)*np.sum(sample)
    M2 = (1./n)*np.sum((sample-M1)**2)
    M3 = (1./n)*np.sum((sample-M1)**3)
    return {'M1':M1,'M2':M2,'M3':M3}


def fitGEVmoments(logTmaxima,xiRange):
    '''Fit GEV distribution of Tmaxima given a range of xi values, 
    and returns the set of parameters, which minimize the third moment'''
    
    M = ThreeMoments(logTmaxima)
    
    results = {'xi' : 0, 'sME' : 0, 'mME' : 0, 'Mmax' : 0, 'diffM3' : 100}
    
    for xi in xiRange:
        sME = ((M['M2']*xi**2)/(math.gamma(1-2*xi)-(math.gamma(1-xi))**2))**(1/2.)
        M3 = (sME**3)*(-2*(math.gamma(-xi))**3-(6/xi)*math.gamma(-xi)*math.gamma(-2*xi)-(3/xi**2)*math.gamma(-3*xi))
        mME = sME/xi-(sME/xi)*math.gamma(1-xi)+M['M1']
        Mmax = mME - sME/xi
        diff = np.abs(M3-M['M3'])
                
        if diff < results['diffM3']:
            results['diffM3'] = diff
            results['xi'] = xi
            results['mME'] = mME
            results['sME'] = sME
            results['Mmax'] = Mmax          
    return results

def  bootstrapGEVmoments(logTmaxima,xiRange,error=0.01):
    XI = []
    M = []
    S = []
    MAX = []
        
    L = fitGEVmoments(logTmaxima,xiRange)
    n = len(logTmaxima)
    
    N = 1/4.*error**(-2)
    
    i=0
    while i<N:
        i+=1    
        sample = randGEV(L['mME'],L['sME'],L['xi'],n)
        results = fitGEVmoments(sample,xiRange)
        
        XI.append(results['xi'])
        M.append(results['mME'])
        S.append(results['sME'])
        MAX.append(results['Mmax'])
            
    results = {'xiMean': np.mean(XI),
               'xiStd' : np.std(XI),
               'mMean': np.mean(M),
               'mStd' : np.std(M),
               'sMean' : np.mean(S),
               'sStd': np.std(S),
               'maxMean': np.mean(MAX),
               'maxStd' : np.std(MAX),
               'nPoints' : n}    
    return results


def xi_vs_nRange(df,Trange,xiRange):    
    '''General function to compute the GEV method with various sizes of T periods'''
    T = []
    results = {}
    
    for t in Trange:
        periods = findTmaxima(df,T=t)
        logTmaxima = periods['logTmaxima']
        res = bootstrapGEVmoments(logTmaxima,xiRange)        
        results[t] = res
    return results


def estimateGPDparameters(sigma,mu,s,T,zeta,Lambda):
    '''Formula (11) and (12)'''
    fit = S.linregress(np.log(Lambda*T),np.log(sigma))
    xi = fit[0]
    s = np.exp(fit[1])
    H = mu + (s/xi)*(1-(Lambda*T)**xi)
    
    return {'xi':xi,'s':s,'H':H}
    
def psi(Lambda,time_interval,xi,H,s,x):
    '''Formula (13) => not yet sure what it is useful for'''
    p = np.exp(-Lambda*tau*(1+xi*(x-H)/s)**(-1/xi))
    return p
    



def findHsamples(df,Hrange):
    '''Make samples as a function of magnitudes 
    (expressed in logarithmic value of loss)'''
    
    dic = {}
    for H in Hrange:
        dic[H] = df[df['logLoss']>H].logLoss.values
    return dicHsamples
    

def GPDllh(self):
    print "blah"
    
def fitGPD(self):
    print "blah"

def makeBootstrapHsample(dicHsamples):
    
    sample = []
    for k in dicHsamples.keys():
        
        sample = np.append(sample,choice(list))

    return sample

def bootstratpGPDsample(dicHsamples,n):
    '''bootstrap by replacement : 
    pick up randomly r times one value from the set M1,...,Mr'''
    
    for i in range(n):
        '''make a sample'''
        sample = makeBootstrapHsample(dicHsamples)
        '''make GPD estimation of this sample'''
        parameters = fitGPD(sample)

    '''average fitted parameters'''



def estimateGPDparameters(self):
    HminH = H - min(H)
    fit = S.linregress(HminH,s_H)

    s_H1 = fit[1]
    xi = fit[0]

    return {'s_H1':s_H1, 'xi':xi}



