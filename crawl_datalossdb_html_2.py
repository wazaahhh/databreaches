#from urllib2 import Request,urlopen,HTTPError,URLError
#from urllib2 import HTTPPasswordMgrWithDefaultRealm,HTTPBasicAuthHandler,install_opener,build_opener
import urllib2
import cookielib

import requests

import numpy as N
import os
import re
import time

import base64

def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

outdir = "html/"

ensure_dir(outdir)


root_url = "http://datalossdb.org/incidents/" 
username="wazaahhh"
password="carpat391"

range = N.arange(10500,11618)

#errors = []
#e = N.load("errors.npy")

'''
jar = cookielib.FileCookieJar("cookies")
passman = urllib2.HTTPPasswordMgrWithDefaultRealm()
passman.add_password(None, "http://datalossdb.org/session/new", username, password)
authhandler = urllib2.HTTPBasicAuthHandler(passman)
#opener = urllib2.build_opener(authhandler)
opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(jar))
#opener.addheaders = [('User-agent', 'Mozilla/5.0')]
opener.addheaders = [('User-agent', "Mozilla/5.0 (X11; U; Linux i686) Gecko/20071127 Firefox/2.0.0.11")]
urllib2.install_opener(opener)
'''

for incident_number in range:

    if os.path.exists("%s%s.html" %(outdir,incident_number)):
        print incident_number, "Already downloaded"
        continue
    
    url = root_url+str(incident_number)+".html"
    
    '''
    user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
    #user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.8; rv:26.0) Gecko/20100101 Firefox/26.0'
    values = {'name' : 'Michael Foord','location' : 'Northampton','language' : 'Python' }
    
    #try:
    request = Request(url)
    base64string = base64.encodestring('%s:%s' % (username, password)).replace('\n', '')
    request.add_header("Authorization", "Basic %s" % base64string)
    #request.add_header('User-Agent', user_agent)
    #request.add_header('Values',values)
    response = urlopen(request)
    '''
    
    '''
    auth_handler = urllib2.HTTPBasicAuthHandler()
    auth_handler.add_password(None,
                          uri='http://datalossdb.org',
                          user=username,
                          passwd=password)
    opener = urllib2.build_opener(auth_handler)
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    # ...and install it globally so it can be used with urlopen.
    urllib2.install_opener(opener)
    #urllib2.urlopen('https://datalossdb.org/sessions/new')
    urllib2.urlopen(url)
    '''  
    

    #pagehandle = urllib2.urlopen(url)
    
        #response = urlopen(url)
    #except HTTPError:
    #    print "error with incident number %s, skipping" %incident_number
    #    errors.append(incident_number)
    #    continue
    #response = opener.open(url)
    #html= response.read()
    print url
    r = requests.get(url, auth=(username, password))
    print r.status_code
    
    #if r.status_code !=200:
    #    break
    
    html = r.text
    html = html.encode('utf-8')
    
    file_path = outdir+str(incident_number)+".html"    
    #file_path = str(incident_number)+".html" 
    f = open(file_path,'wb') 
    f.write(html)
    print incident_number,"length html: ",len(html) #,url
    #time.sleep(0.5)
    
N.save("errors",errors)