#import urllib2
#import cookielib
#import base64

import os
import numpy as np
import re
import requests
import time
import boto
from datetime import datetime
import json


def loadBucket():
    global bucket
    bucket = S3connectBucket("databreaches")

def S3connectBucket(bucketName):
    s3 = boto.connect_s3()
    bucket = s3.get_bucket(bucketName)
    return bucket

def pushEvents():
    
    dir = "/home/ubuntu/dataloss_db/html/"
    
    fileNames = os.listdir(dir)

    for name in fileNames:
        html = open(dir+name).read()
        
        if len(html) == 14052:
            print "404 not found"
        else:
            print "storing incident number %s (length html: %s)"%(name[:-5],len(html))
            key = bucket.new_key("html/%s"%name)
            key.set_contents_from_string(html)
        
        
def retrieveEvent(incidentNumber,overwrite=False):

    root_url = "http://datalossdb.org/incidents/" 
    credentials = json.load(open("/home/ubuntu/passwd/datalossdb.json",'rb'))
    
    
    #if os.path.exists("%s%s.html" %(outdir,incident_number)):
    #    print incident_number, "Already downloaded"
    #    continue

    key = bucket.get_key("%s/%s.html"%(htmlFolder,incidentNumber))

    if key and overwrite==False:
        print "html for incident id = %s already downloaded"%incidentNumber
        pass
    else:
        url = root_url+str(incidentNumber)+".html"
    
        print url
    
        r = requests.get(url, auth=(credentials['username'], credentials['password']))
        #print r.status_code
        
        html = r.text
        html = html.encode('utf-8')
    
        if len(html) == 14052:
            print "404 not found"
        else:
            print "storing incident number %s (length html: %s)"%(incidentNumber,len(html))
            key = bucket.new_key("%s/%s.html"%(htmlFolder,incidentNumber))
            key.set_contents_from_string(html)
    

def checkLatestIncident():
    
    url = "http://datalossdb.org/"
    
    r = requests.get(url)
    html = r.text
    html = html.encode('utf-8')
    
    latestIncident = np.max(map(int,re.findall("/incidents/(\d*?)-",html)))
    
    return latestIncident

def checkLatestStoredIncident(minId=-1):
    global StoredKeys
    
    try:
        StoredKeys
    except:
        StoredKeys= []
        keys = bucket.list("%s/"%htmlFolder)

        for k in keys:
            StoredKeys.append(int(re.findall("%s/(\d*?)\.html"%htmlFolder,k.name)[0]))
    
    return np.max(StoredKeys)

 
def retrieveLatestEvents(goback = 0,overwriteHtml=False,storeToJson=True):   

    
    '''load latest Json'''
    key = bucket.get_key("json/data.json")
    J = json.loads(key.get_contents_as_string())

    latestStoredKey = checkLatestStoredIncident()-goback
    latestIncident = checkLatestIncident()
    
    print latestStoredKey,latestIncident
    
    if latestIncident <= latestStoredKey:
        print "up to date"
        
    else:    
        incidentsRange = np.arange(latestStoredKey,latestIncident)

        for i in incidentsRange:
            retrieveEvent(i,overwrite=overwriteHtml)
            
            dicEvent = parseEventHtml(i)
            J[str(i)] = dicEvent
        
        if storeToJson:
            Json = json.dumps(J)
            now = datetime.now().strftime("%Y%m%d%H%M%S")
            key = bucket.new_key("json/data%s.json"%now)
            key.set_contents_from_string(Json)
            key = bucket.new_key("json/data.json")
            key.set_contents_from_string(Json)

        print "done"
        return J
    
    
def extractDate(dateType,html):
    try:
        date = re.findall("<b>(\d\d\d\d-\d\d-\d\d)</b></td><td>%s</td>"%dateType,html)[0]
        if date == ('', ''):
            date = "Date Removed"
    except:
        date = -1  
    
    return date      


def parseEventHtml(incidentNumber):
    
    key = bucket.get_key("%s/%s.html"%(htmlFolder,incidentNumber))
        
    if key == None:
        print "no event with id %s found stored on S3"%incidentNumber
        return {}
    
    html = key.get_contents_as_string()
    
    try:
        html2 = re.sub("[\n\t]","",html)
        summary = re.findall('SUMMARY.*?"2">(.*?)</td>',html2)[0]
    except:
        summary = -1
    
    html= re.sub("[\s\n\t]","",html)
    try:
        Incident_ID = re.findall("</span>ShowingIncident(\d*?)</h1>",html)[0]

    except:
        Incident_ID = -1

    try:
        Records = re.findall("Records</th><td>(.*?)</td>",html)[0]
        Records = int(re.sub(",","",Records.__str__()))
        if Records == "Unknown":
            Records = -1
    except:
        Records = -1
    
    if Records == -1:
        try:
            t2 = re.sub(",","",summary)
            Records = re.findall('\s(\d.*?)\s',t2)[0]
            #Records = re.sub(',','',Records)
        except:
            Records = -1
    
    
    #try:
    #    Records = int(re.sub(",","",Records.__str__()))
    #except:
    #    Records = -1
        #return html
    
    IncidentOccurred = extractDate('IncidentOccurred',html)
    IncidentDiscovered = extractDate('IncidentDiscoveredByOrganization',html)
    IncidentReported = extractDate('OrganizationReportsIncident',html)
    
    #try:
    #    Incident_Date= re.findall("<b>(\d\d\d\d-\d\d-\d\d)</b></td><td>IncidentOccurred</td>",html)[0]
    #except:
    #    Incident_Date = -1
        
    
    #try:
    #    Incident_Reported = re.findall("<b>(\d\d\d\d-\d\d-\d\d)</b></td><td>OrganizationReportsIncident</td>",html)[0]
    #except:
    #    Incident_Reported = -1        
    
    
    try:
        Address = re.findall("<b>Address:(.*?)</b>",html)[0]
    except:
        Address = -1
    
    try:
        BreachType = re.findall("BreachType</th><td>(.*?)</td>",html)[0]
    except:
        BreachType = -1
        
        
    try:
        str = re.findall("RecordTypes</th><td>(.*?)</td>",html)[0]
        RecordTypes = re.findall('">([A-Z]{3})</a>',str)
    except:
        RecordTypes = -1
    
    try:
        DataFamily = re.findall("DataFamily</th><td>(.*?)</td>",html)[0]
    except:
        DataFamily = -1
    
    try:
        Source = re.findall("Source</th><td>(.*?)</td>",html)[0]
    except:
        Source = -1
    
    try:
        Organization = re.findall("Organization</th><td>.*?>(.*?)</a></td>",html)[0]
    except:
        Organization = -1
    
    if Organization == -1:
        try:
            Organization = re.findall("title>Incident:(.*?)-",html)[0]
        except:
            Organization = -1
    
    try: 
        str = re.findall("OtherAffected/InvolvedOrganizations</th><td>(.*?)</td>",html)[0]
        OtherAffected = re.findall(">(.*?)</a>",str)[0]
    except:
        OtherAffected = -1
    
    try:
        LawSuit = re.findall("Lawsuit\?</th><td>(.*?)</td>",html)[0]
    except:
        LawSuit = -1
    
    try:
        DataRecovered = re.findall("DataRecovered\?</th><td>(.*?)</td>",html)[0]
    except:
        DataRecovered = -1
    
    try:
        Arrest = re.findall("Arrest\?</th><td>(.*?)</td>",html)[0]
    except:
        Arrest = -1
        
    try:
        SubmittedBy = re.findall("SubmittedBy:</th><td>(.*?)</td>",html)[0]
    except:
        SubmittedBy = -1    

    try:
        References = re.findall('<listyle=""><ahref="(.*?)"popup',html)
    except:
        References = -1
    
     
    if re.findall("FRINGE",html):
        fringe = 1
    else:
        fringe = 0
          
    #RetrieveDate =  datetime.strptime(key.last_modified, '%a, %d %b %Y %H:%M:%S GMT').strftime('%Y-%m-%d')
    
    #if RetrieveDate == "2014-02-09" :
    #    RetrieveDate = -1
    
    dic = {'Id' : Incident_ID, 
           'Records' : Records, 
           'IncidentOccurred' : IncidentOccurred, 
           'IncidentDiscovered' : IncidentDiscovered, 
           'IncidentReported' : IncidentReported,
           'Address' : Address,
           'BreachType' : BreachType,
           'RecordTypes' : RecordTypes,
           'DataFamily' : DataFamily,
           'Source' : Source,
           'Organization' : Organization,
           'OtherAffected' : OtherAffected,
           'LawSuit' : LawSuit,
           'DataRecovered' : DataRecovered,
           'Arrest' : Arrest,
           'SubmittedBy' : SubmittedBy,
           'References' : References,
           'Fringe' : fringe,
           'Summary' : summary}
  
    return dic

#def parseEventsFromId(minId,maxId = -1):
    
       
    
    

def updateFringeIncidents(updateJson=True):

    key = bucket.get_key("json/data.json")
    J = key.get_contents_as_string()
    J = json.loads(J)

    fringeIncidents = []
    for i  in J.keys():
        if J[i]['fringe']:
            fringeIncidents.append(i)
    
    print "%s fringe incidents found"%len(fringeIncidents)
    
    for i,ix in enumerate(fringeIncidents):
        retrieveEvent(i,overwrite=True)
        dic = parseEventHtml(ix)
           
        if dic['fringe']==1:
            print "%s no change made to incident %s"%(i,ix)
            continue
        elif updateJson:
            print "%s updating incident %s"%(i,ix)
            J[i] = dic
            
    remainingFringeIncidents = []
    for i  in J.keys():
        if J[i]['fringe']:
            remainingFringeIncidents.append(i)
    
    print "%s fringe incidents remaining"%len(fringeIncidents)
    
         

def parseAllEventsToJson(store = True):
    checkLatestStoredIncident()

    J = {}
    fringe = []

    for k in StoredKeys:
        print k
        dic = parseEventHtml(k)
        J[k] = dic 
            
    if store:
        J = json.dumps(J)
        now = datetime.now().strftime("%Y%m%d%H%M%S")
        key = bucket.new_key("json/data%s.json"%now)
        key.set_contents_from_string(J)
        key = bucket.new_key("json/data.json")
        key.set_contents_from_string(J)        
    return J
    


if __name__ == '__main__':
    #global StoredKeys

    global bucket
    bucket = S3connectBucket("databreaches")
    
    global htmlFolder
    htmlFolder = "html2"
    
    