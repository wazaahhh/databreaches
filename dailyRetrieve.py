#!/usr/bin/env python

from main import retrieveLatestEvents,loadBucket

if __name__ == '__main__':
    #bucket = S3connectBucket("databreaches")
    loadBucket()
    retrieveLatestEvents()