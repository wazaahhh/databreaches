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


def restoreToHtml():
    keys = bucket.list("html/")  
    for k in keys:
        print k.name
        str = k.get_contents_as_string()
        print "html2/%s"%k.name[5:]
        key = bucket.new_key("html2/%s"%k.name[5:])
        key.set_contents_from_string(str)

if __name__ == '__main__':
    #global StoredKeys

    global bucket
    bucket = S3connectBucket("databreaches")