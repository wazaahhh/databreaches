'''This script is an implementation of 

Pisarenko, V. F., Sornette, A., Sornette, D. & Rodkin, M. V. 
Characterization of the tail of the distribution of earthquake magnitudes 
by combining the GEV and GPD descriptions of extreme value theory (2008).


'''

import numpy as np
import pandas
from random import choice,randrange, shuffle
import math
from scipy import stats
import sys

sys.path.append("/home/ubuntu/toyCode/sLibs/")
from tm_python_lib import *

from matplotlib.font_manager import FontProperties
from matplotlib import rc

fig_width_pt = 420.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0 / 72.27  # Convert pt to inch
golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = fig_width  # *golden_mean      # height in inches
fig_size = [fig_width, fig_height]
params = {'backend': 'ps',
          'axes.labelsize': 32,
          'text.fontsize': 32,
          'legend.fontsize': 18,
          'xtick.labelsize': 30,
          'ytick.labelsize': 30,
          'text.usetex': True,
          'figure.figsize': fig_size}
pl.rcParams.update(params)
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

global zetaRange
zetaRange=np.arange(-2,0,0.05)


import boto
import json

def loadBucket():
    global bucket
    bucket = S3connectBucket("databreaches")

def S3connectBucket(bucketName):
    s3 = boto.connect_s3()
    bucket = s3.get_bucket(bucketName)
    return bucket



def magnitudeQuantiles(H,s,xi,Lambda,t,q):
    '''returns quantiles of the Distribution Function of magnitude [Formula 10]'''
    Q = H -(s/xi)*(1-(Lambda*tau/np.log(1./q))**xi)
    return Q

def findLambda(df,H=3.5,minDateStamp=2000):
    '''returns the rate of events above a given magnitude H'''
    dt = df[(df['logLoss'] > H) & (df['datestamp'] > minDateStamp)]['datestamp'].values
    Lambda = 1./np.mean(np.diff(dt))
    return Lambda
    


def replacementShuffle(sample):
    return sample[np.random.randint(len(sample),size=len(sample))]
    

loadBucket()

class methodGEV():
    def __init__(self):
        self.iconn = None
        self.sconn = None
    
    def randGEV(self,m,s,xi,n):
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

    
    def findTmaxima(self,df,T=100):
        '''From input Pandas DataFrame, find maxima in each period of time T '''
        
        dates = np.array(df.datestamp)      
        bins = np.arange(0,len(dates)+1,T)
        groups = df.groupby(np.digitize(df.index, bins))
        Tmaxima = np.array(groups.max().records)
        logTmaxima = np.log10(Tmaxima)
        
        return {'Tmaxima':Tmaxima, 'logTmaxima' : logTmaxima,'T' : T}


    def chooseTmaximaRange(self):
        print "to do (c.f. page 11 article)"

    def ThreeMoments(self,sample):
        '''find the three moments of the sample'''
        n = len(sample)
        M1 = (1./n)*np.sum(sample)
        M2 = (1./n)*np.sum((sample-M1)**2)
        M3 = (1./n)*np.sum((sample-M1)**3)
        return {'M1':M1,'M2':M2,'M3':M3}

    
    def fitGEVmoments(self,logTmaxima,plot=False):
        '''Fit GEV distribution of T-maxima given a range of xi values, 
        and returns the set of parameters, which minimize the third moment'''
        
        
        
        M = self.ThreeMoments(logTmaxima)
        
        results = {'zeta' : 0, 'sigma' : 0, 'mu' : 0, 'diffM3' : 100}
        
        for zeta in zetaRange:
            sigma = ((M['M2']*zeta**2)/(math.gamma(1-2*zeta)-(math.gamma(1-zeta))**2))**(1/2.)
            M3 = (sigma**3)*(-2*(math.gamma(-zeta))**3-(6/zeta)*math.gamma(-zeta)*math.gamma(-2*zeta)-(3/zeta**2)*math.gamma(-3*zeta))
            mu = sigma/zeta-(sigma/zeta)*math.gamma(1-zeta)+M['M1']
            Mmax = mu - sigma/zeta
            diff = np.abs(M3-M['M3'])
                    
            if diff < results['diffM3']:
                results['diffM3'] = diff
                results['zeta'] = zeta
                results['mu'] = mu
                results['sigma'] = sigma
        
        nPoints = len(logTmaxima)
        x,y = rankorder(logTmaxima)
        yNorm = np.array(y/float(np.max(y)))
        cdf = 1-stats.genextreme.cdf(x,results['zeta'],loc=results['mu'],scale=results['sigma'])
        cdf = np.array(cdf)
        
        if plot=='linlin' or plot=='loglin':
            #print H
            pl.figure(1,(8,7))
            
            if plot == 'linlin':
                #pl.plot(x,y,'r-',lw=2,label="Best GPD fit (MLE)")
                pl.plot(x,yNorm,'o',label="Empirical Distribution")
                pl.plot(x,cdf,'r-',lw=2,label="Best GEV fit (Moments)")
            elif plot == 'loglin':
                pl.semilogy(x,yNorm,'o',label="Empirical Distribution")
                pl.semilogy(x,cdf,'r-',lw=2,label="Best GEV fit (Moments)")
            pl.xlabel(r"Magnitude")
            pl.ylabel(r"CCDF")
            pl.legend(loc=0)
            
        KSdist = np.sqrt(nPoints)*max(np.abs(cdf-yNorm))
        results['ks'] = KSdist
        results['nPoints'] = nPoints
        return results

    def  bootstrapGEVmoments(self,logTmaxima,error=0.05, replacement=True):
        XI = []
        M = []
        S = []
        MAX = []
        ks = []
         
        L = self.fitGEVmoments(logTmaxima)
        n = len(logTmaxima)
        
        N = 1/4.*error**(-2)
        
        i=0
        while i<N:
            i+=1
            
            if replacement:
                results = self.fitGEVmoments(replacementShuffle(logTmaxima))
            else:
                sample = self.randGEV(L['mME'],L['sME'],L['xi'],n)  
                results = self.fitGEVmoments(sample,xiRange)
            
            XI.append(results['zeta'])
            M.append(results['mu'])
            S.append(results['sigma'])
            ks.append(results['ks'])
            
        results = {'zetaMean' : np.mean(XI),
                   'zetaStd' : np.std(XI),
                   'muMean': np.mean(M),
                   'muStd' : np.std(M),
                   'sigmaMean' : np.mean(S),
                   'sigmaStd': np.std(S),
                   'nPoints' : n,
                   'ksMean' : np.mean(ks),
                   'ksStd' : np.std(ks)}
        
        return results
    

    def xi_vs_nRange(self,df,Trange):    
        '''General function to compute the GEV method with various sizes of T periods'''
        T = []
        results = {}
        
        for t in Trange:
            periods = self.findTmaxima(df,T=t)
            logTmaxima = periods['logTmaxima']
            res = self.bootstrapGEVmoments(logTmaxima)        
            results[t] = res
            
        #results = pandas.DataFrame(results).transpose()
        #results['Mmax'] = 10**(results['maxMean']+ results['maxStd'])
        dfGEV= pandas.DataFrame(results).T
        return dfGEV


    def plotGEV(self,dfGEV):
        
        pl.close("all")
        pl.figure(1,(15,7))
        pl.subplot(121)
        pl.errorbar(dfGEV.index,dfGEV['zetaMean'],yerr=dfGEV['zetaStd'])
        pl.xlabel("T")
        pl.ylabel("zeta")
        
        
        pl.subplot(122)
        pl.errorbar(dfGEV.index,dfGEV['sigmaMean'],yerr=dfGEV['sigmaStd'])
        pl.xlabel("T")
        pl.ylabel("sigma")



    def estimateGPDparameters(self,sigma,mu,s,T,Lambda):
        '''Formula (11) and (12)'''
        fit = stats.linregress(np.log(Lambda*T),np.log(sigma))
        xi = fit[0]
        s = np.exp(fit[1])
        fit2 = stats.linregress(-(1-(Lambda*T)**xi),mu)
        Hk = mu + (s/xi)*(1-(Lambda*T)**xi)
        return {'xi':xi,'s':s,'Hk':Hk,'H':fit2[1]}
        
    def psi(self,Lambda,time_interval,xi,H1,s1,x):
        '''Formula (13) computes the distribution function of a maximum magnitude M 
        of a Poissonian flow of main shocks over an arbitrary time interval tau'''      
        p = np.exp(-Lambda*time_interval*(1+xi*(x-H1)/s1)**(-1/xi))
        return p
    
#    def Pregress(self,sigmaTk,muTk,Tk,Lambda):
#        np.log(sigmaTk) = np.log(s) + epsilon*log(Lambda*Tk) 
#        H = muTk + s/epsilon *(1-(Lambda*Tk)**epsilon)
#
#        epsilon = (np.log(sigmaTk) - np.log(s))/log(Lambda*Tk) 
#        H(s) = muTk + s/epsilon *(1-(Lambda*Tk)**epsilon)



class methodGPD():
    def __init__(self):
        self.iconn = None
        self.sconn = None
        
    
    def findHsamples(self,df,Hrange):
        '''Make samples as a function of magnitudes 
        (expressed in logarithmic value of loss)'''
        
        dic = {}
        for H in Hrange:
            dic[H] = df[df['logLoss']>H].logLoss.values
        return dicHsamples
        
    
    def mleFit(self,sample,H,plot=False):
        '''Maximum Loglikelihood (MLE) of 
        Generalized Pareto Distribution (GPD) over threshold H'''
        
        H = H + (np.random.random()-0.5)/1000
        
        data = np.array(sample)
        data = data[data >= H]
        
        shape, loc, scale = stats.genpareto.fit(data,loc=H, scale=0.5)
        
        x,y = rankorder(data)
        yNorm = y/float(np.max(y))
        cdf = (1- stats.genpareto.cdf(x,shape,scale=scale,loc=loc))
            
        
        if plot=='linlin' or plot=='loglin':
            #print H
            pl.figure(1,(8,7))
            
            if plot == 'linlin':
                #pl.plot(x,y,'r-',lw=2,label="Best GPD fit (MLE)")
                pl.plot(x,yNorm,'o',label="Empirical Distribution")
                pl.plot(x,cdf,'r-',lw=2,label="Best GPD fit (MLE)")
            elif plot == 'loglin':
                pl.semilogy(x,yNorm,'o',label="Empirical Distribution")
                pl.semilogy(x,cdf,'r-',lw=2,label="Best GPD fit (MLE)")
            pl.xlabel(r"Magnitude")
            pl.ylabel(r"CCDF")
            pl.legend(loc=0)
        
        #KSdist = max(np.abs(cdf-yNorm))
        nPoints = len(data)
        KSdist = np.sqrt(nPoints)*max(np.abs(cdf-yNorm))
        
        #print 'K-S distance: ', KSdist
        #print 'nPoints: ',nPoints
        
        return {'xi': shape, 'loc':loc,'s':scale,'ks':KSdist, 'nPoints' : nPoints}
    
    def bootstrapGPDmle(self,sample,H,nStraps=100,replacement=True,plot=False):
        fitInit = self.mleFit(sample,H)
        #print "fitInit: ",fitInit
        loc = []
        xi= []
        s = []
        ks = []
        
        #N = 1/4.*error**(-2)
        
        for i in range(nStraps):
            if replacement:
                fit = self.mleFit(replacementShuffle(sample),H)
            else:
                rand = stats.genpareto.rvs(fitInit['shape'],loc=fitInit['loc'],scale=fitInit['scale'],size=len(sample))
                fit = self.mleFit(rand,H)
            
            #print i,fit
            loc = np.append(loc,fit['loc'])
            xi = np.append(xi,fit['xi'])
            s = np.append(s,fit['s'])
            ks = np.append(ks,fit['ks'])
            
        return {'nPoints' : fitInit['nPoints'], 
                'ksMean' : np.mean(ks),
                'ksStd': np.std(ks),
                'hMean': np.mean(loc),
                'hStd': np.std(loc),
                'xiMean': np.mean(xi),
                'xiStd': np.std(xi),
                'sMean': np.mean(s),
                'sStd': np.std(s)
                }
        
        
    
    def GPD_vs_Hk(self,sample,Hk,nStraps=100,writeJsonS3=False):
        
        key = bucket.get_key("outputGEVGPD/GPDbootstrapping.json")
            
        if key and not writeJsonS3:
            results = json.loads(key.get_contents_as_string())
        else:
        
            results = {}
            
            
            for H in Hk:
                #fit =  self.mleFit(sample,H)
                fit = self.bootstrapGPDmle(sample,H,nStraps=nStraps)
                
                #print 'hM',H
                #print '  xi',fit['xiMean']
                #print '  s', fit['sMean']
                #print '  ks', fit['ksMean']
                results[H] = fit
            
            #if plot:
            #    pl.plot(dic['Hk'],dic['xi'],'o')
            #    pl.xlabel(r"Threshold $H$")
            #    pl.ylabel(r"$\xi$ (Shape Parameter)")
            if writeJsonS3:
                J = json.dumps(results)
                key = bucket.new_key("outputGEVGPD/GPDbootstrapping.json")
                key.set_contents_from_string(J)
         
           
        dfGPD= pandas.DataFrame(results).T
        return dfGPD
    
    
    def plotGPD(self,dfGPD):
        pl.close("all")
        pl.figure(1,(20,7))
        
        pl.subplot(131)
        pl.errorbar(dfGPD.index,dfGPD['xiMean'],yerr=dfGPD['xiStd'])
        pl.xlabel("Threshold H")
        pl.ylabel(r"\xi")
        pl.subplot(132)
        pl.errorbar(dfGPD.index,dfGPD['sMean'],yerr=dfGPD['sStd'])
        pl.xlabel("Threshold H")
        pl.ylabel("s")
        pl.subplot(133)
        pl.errorbar(dfGPD.index,dfGPD['ksMean'],yerr=dfGPD['ksStd'])
        pl.xlabel("Threshold H")
        pl.ylabel("ks-distance")

    def estimateGPDparameters(self,dfGPD,plot=False,**kwargs):
                
        
        Hk = dfGPD.index
        
        
        if kwargs is not None:
            minH = kwargs['minH']
            maxH = kwargs['maxH']
        else:
            minH = min(Hk)
            maxH = max(Hk) 
        
        
        c = (Hk >= minH) * (Hk <= maxH)

        x = Hk[c]
        
        sHk = []
        
        sHk = dfGPD[c]['sMean']
        
        x = x - min(x)
        
        fit = stats.linregress(x,sHk)
        print fit
        sH1 = fit[1]
        xi = fit[0]
        
        if plot:
            pl.plot(x,sHk,'o')
            pl.plot(x,x*fit[0]+fit[1],'r-')
            pl.xlabel("Hk -H1")
            pl.ylabel('s')

        return {'s_H1':sH1, 'xi':xi}


if __name__ == '__main__':
    loadBucket()


    


