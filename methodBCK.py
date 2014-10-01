'''Method of Moments (c.f. Appendix B, Pisarenko et al. 2008)'''

def ThreeMoments(sample):
    n = len(sample)
    M1 = (1./n)*np.sum(sample)
    M2 = (1./n)*np.sum((sample-M1)**2)
    #print M2,(1./n)*np.sum((sample)**2) - M1**2
    M3 = (1./n)*np.sum((sample-M1)**3)
    #print M1,M2,M3
    return {'M1':M1,'M2':M2,'M3':M3}


def randGEV(m,s,xi,n):
    '''GEV distribution function'''
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
    dates = np.array(df.datestamp)
    
    #bins = np.linspace(0,len(dates),n)
    bins = np.arange(0,len(dates)+1,T)
    
    #T = bins[1]-bins[0]
    groups = df.groupby(np.digitize(df.index, bins))
    
    Tmaxima = np.array(groups.max().records)
    logTmaxima = np.log10(Tmaxima)
    
    return {'Tmaxima':Tmaxima, 'logTmaxima' : logTmaxima,'T' : T}


def findTmaximaKSdistance(df,nIntervals):
    '''Finds most appropriate T-maxima 
    using the Kolmogorov distance'''
    print "to be done"
    
    
def fitGEVmoments(logTmaxima,xiRange):
    
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

def KSdistance(sample,xiRange):
    ''' still a problem there'''
    fit = fitGEVmoments(sample,xiRange) 
    cdf = np.exp( -(1+fit['xi']*(sample-fit['mME'])/fit['sME'])**(-1/fit['xi']))
    l = len(sample)
    KS = l**(0.5)*np.max(np.abs(sample-cdf))

def bootstrapGEVmoments(logTmaxima,xiRange,error=0.01):
    XI = []
    M = []
    S = []
    MAX = []
        
    L = fitGEVmoments(logTmaxima,xiRange)
    n = len(logTmaxima)
    #print L['xi'],L['mME'],L['sME'],L['Mmax'],n
    
    N = 1/4.*error**(-2)
    #print "generating %s bootstraps"%N
    i=0
    while i<N:
        i+=1    
        sample = randGEV(L['mME'],L['sME'],L['xi'],n)
        results = fitGEVmoments(sample,xiRange)
        
        XI.append(results['xi'])
        M.append(results['mME'])
        S.append(results['sME'])
        MAX.append(results['Mmax'])
        
        #print np.round([results['xi'],results['mME'],results['sME'],results['Mmax']],2)
    #print np.round([np.mean(XI),np.mean(M),np.mean(S),np.mean(MAX)],2)
    #print np.round([np.std(XI),np.std(M),np.std(S),np.std(MAX)],2)
    
    #print 10**(np.mean(MAX)+np.array([-0.06,0.06]))
    
    results = {'xiMean': np.mean(XI),'xiStd' : np.std(XI),'mMean': np.mean(M), 'mStd' : np.std(M),'sMean' : np.mean(S), 'sStd': np.std(S),'maxMean': np.mean(MAX),'maxStd' : np.std(MAX),'nPoints' : n}
    
    return results
    

def xi_vs_nRange(df,Trange,xiRange):    
    #N = []
    T = []

    results = {}
    
    for t in Trange:
        
        periods = findTmaxima(df,T=t)
        logTmaxima = periods['logTmaxima']

        res = bootstrapGEVmoments(logTmaxima,xiRange)
        
        results[t] = res
        #print n,t,fitGEVmoments(logTmaxima,xiRange)
        #print n,xi
    return results




def GPDllh(parDic,dataset):
    x = np.array(dataset)
    L = -(parDic['xi']+1)/parDic['xi']*np.log(parDic['xi']*(x-parDic['mu']))/parDic['sigma']
    return L

def GPDllh2(par,dataset,n):
    xi = par[0]
    mu = 0
    sigma = par[1]
    
    a = 1/xi+1
    b = np.sum(xi*(dataset-mu)/sigma)
    c = np.log(sigma)
    llh = -a*np.log(1+b)-c
    return llh

#def fitGPD(par,dataset):
#    #t0=[initDic['xi'],initDic['mu'],parDic['sigma']]
#    optGDP = optimize.fmin(GPDllh2,par,args=(dataset),retall=0,full_output=0,disp=0,xtol=1e-12,ftol=1e-12)
#    return optGDP

def fitGPD(sample,mu=1):
    fit = stats.genpareto.fit(sample,loc=mu,scale=1,shape=1)
    return {'xi':fit[0],'mu':fit[1],'sigma':fit[2],'nPoints':len(sample)}
    
    
def randGPD(parDic,n):
    return stats.genpareto.rvs(parDic['xi'],loc=parDic['mu'],scale=parDic['sigma'],size=n)

def bootstrapGPD(fit,error=0.05):
    xi = []
    mu = []
    sigma = []
    
    n= int(round(1/4.*error**(-2)))
    print "generating %s bootstraps"%n
    
    for i in range(n):
        sample = randGPD(fit,fit['nPoints'])
        bootstrapFit = fitGPD(sample,mu=fit['mu'])
        
        xi.append(bootstrapFit['xi'])
        mu.append(bootstrapFit['mu'])
        sigma.append(bootstrapFit['sigma'])
        
        #print "xi=%.2f, mu=%.2f, sigma=%.2f"%(bootstrapFit['xi'],bootstrapFit['mu'],bootstrapFit['sigma'])
        
    return {'xiMean' : np.mean(xi),
                'xiStd':np.std(xi),
                'muMean':np.mean(mu),
                'muStd':np.std(mu),
                'sigmaMean':np.mean(sigma),
                'sigmaStd':np.std(sigma),
                'nBootstraps' : n}

def fitHrange(df,Hrange,plot=False):
    dic = {}
    for H in Hrange:
        sample = df[df['logLoss']>H].logLoss.values
        
        fit = fitGPD(sample,mu=H)
        B =  bootstrapGPD(fit,error=0.05)
        dic[H]=B
        print "threshold H=%s, sample size n=%s, xi=%.2f(%.2f), sigma=%.2f(%.2f)"%(H,len(sample),B['xiMean'],B['xiStd'],B['sigmaMean'],B['sigmaStd'])
    
        '''
        l = len(sample)
        bootstrap = randGPD(fit,l)
        x,y =rankorder(sample)
        xFit,yFit = rankorder(bootstrap)
        figure(H+10) 
        semilogy(x,y,'b-+')
        semilogy(xFit,yFit,'r-')
        '''
        
    if plot:
        plotFitRange(dic)
    
    return dic

def plotFitRange(dic):
    
    H=dic.keys()
    xiMean = []
    xiStd = []
    sigmaMean = []
    sigmaStd = []
    
    for key in H:
        xiMean.append(dic[key]['xiMean'])
        xiStd.append(dic[key]['xiStd'])
        sigmaMean.append(dic[key]['sigmaMean'])
        sigmaStd.append(dic[key]['sigmaStd'])
    
    pl.close("all")
    pl.figure(1,(15,7))
    pl.subplot(121)
    pl.errorbar(H,xiMean,fmt='o',yerr=xiStd)
    #pl.ylim(-0.5,0.5)
    pl.xlabel("H")
    pl.ylabel("xi")
    
    pl.subplot(122)
    pl.errorbar(H,sigmaMean,fmt='o',yerr=sigmaStd)
    #pl.ylim(-0.5,0.5)
    pl.xlabel("H")
    pl.ylabel("sigma")
    

def avg_s_xi(dic):
    
    Sk = []
    Hk = []
    for key in dic.keys():
        #print dic[key]        
        Sk = np.append(Sk,dic[key]['sigmaMean'])
        Hk = np.append(Hk,key)
    H1 = min(dic.keys())

    fit = stats.linregress(Sk,Hk-H1)
    print fit
    plot(Sk,Hk-H1,'o')
    plot(Sk,Sk*fit[0]+fit[1],'-')
    return {'xi_est' : fit[0], 's_1' : fit[1],'H1':H1}
    
def LomnitzFormulat(parDic,x,rate,tau,n=1):
    '''estimated distribution function of the maximum magnitude M_tau
    of a Poissonian flow of events
    parDic should contain : 
    - 'H1' : the minimum magnitude
    - 's_1': sigma for H1
    - 'xi' : the estimated value of xi
    '''
    PSI_x = np.exp(-rate*tau*(1+parDic['xi_est']*(x-parDic['H1'])/parDic['s_1'])**(-1/parDic['xi_est']))    
    return PSI_x
    
    
def quantileEstimate(parDic,q):
    Q = parDic['mu'] - (parDic['sigma']/parDic['xi'])*(1-(rate*tau/np.log(1./q))**parDic['xi'])
    return Q
