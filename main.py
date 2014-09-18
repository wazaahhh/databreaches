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
        key = bucket.new_key("html/%s.html"%incidentNumber)
        key.set_contents_from_string(html)
    

def checkLatestIncident():
    
    url = "http://datalossdb.org/"
    
    r = requests.get(url)
    html = r.text
    html = html.encode('utf-8')
    
    latestIncident = np.max(map(int,re.findall("/incidents/(\d*?)-",html)))
    
    return latestIncident

def checkLatestStoredIncident():
    global StoredKeys
    
    try:
        StoredKeys
    except:
        StoredKeys= []
        keys = bucket.list("html")

        for k in keys:
            StoredKeys.append(int(re.findall("html/(\d*?)\.html",k.name)[0]))
    
    return np.max(StoredKeys)

 
def retrieveLatestEvents(goback = 0,storeToJson=True):   

    
    if storeToJson:
        '''load latest Json'''
        key = bucket.get_key("json/data.json")
        J = key.get_contents_as_string()
        J = json.loads(J)

    latestStoredKey = checkLatestStoredIncident()-goback
    latestIncident = checkLatestIncident()
    
    if latestIncident <= latestStoredKey:
        print "up to date"
        
    else:    
        incidentsRange = np.arange(latestStoredKey,latestIncident)

        for i in incidentsRange:
            retrieveEvent(i,overwrite=True)
            
            if storeToJson:
                dicEvent = parseEventHtml(i)
                J[i] = dicEvent
        
        if storeToJson:
            J = json.dumps(J)
            key = bucket.new_key("json/data.json")
            key.set_contents_from_string(J)
            return J
              
        print "done"
    

def parseEventHtml(incidentNumber):
    
    
    key = bucket.get_key("html/%s.html"%incidentNumber)
        
    if key == None:
        print "no event with id %s found stored on S3"%incidentNumber
        return {}
    
    html= re.sub("[\s\n\t]","",key.get_contents_as_string())
    try:
        Incident_ID = re.findall("</span>ShowingIncident(\d*?)</h1>",html)[0]

    except:
        Incident_ID = -1

    try:
        Records = re.findall("Records</th><td>(.*?)</td>",html)[0]
        if Records == "Unknown":
            Records = -1
    except:
        Records = -1
    
    Records = int(re.sub(",","",Records.__str__()))
    
    try:
        Incident_Date= re.findall("<b>(\d\d\d\d-\d\d-\d\d)</b></td><td>IncidentOccurred</td>",html)[0]
    except:
        Incident_Date = -1
        
    try:
        Incident_Discovered = re.findall("<b>(\d\d\d\d-\d\d-\d\d)</b></td><td>IncidentDiscoveredByOrganization</td>",html)[0]
    except:
        Incident_Discovered = -1        

    try:
        Incident_Reported = re.findall("<b>(\d\d\d\d-\d\d-\d\d)</b></td><td>OrganizationReportsIncident</td>",html)[0]
    except:
        Incident_Reported = -1        
     
    if re.findall("FRINGE",html):
        fringe = 1
    else:
        fringe = 0
    
        
    Retrieve_Date =  datetime.strptime(key.last_modified, '%a, %d %b %Y %H:%M:%S GMT').strftime('%Y-%m-%d')
    
    if Retrieve_Date == "2014-02-09" :
        Retrieve_Date = -1
    
    return {'id' : Incident_ID, 'records' : Records, 'date' : Incident_Date, 'discovered' : Incident_Discovered, 'reported' : Incident_Reported, 'LatestRetrievedFromDataloss' : Retrieve_Date,'fringe': fringe}


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
        now = datetime.now().strftime("%Y-%m-%d")
        key = bucket.new_key("json/data_%s.json"%now)
        key.set_contents_from_string(J)
                
    return J
    





if __name__ == '__main__':
    #global StoredKeys

    global bucket
    bucket = S3connectBucket("databreaches")
    
    