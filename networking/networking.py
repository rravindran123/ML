import sys
import time
import numpy as np
from collections import deque
from dataclasses import dataclass
import hashlib
import random
import string
import matplotlib.pyplot as plt

class Ratelimiter:
    def __init__(self, allowedrequest:float, timeduration:float):
        self.ratevalue= allowedrequest/timeduration #number of requests per section
        self.lasttimestamp =0
        self.reqcount=0

    def isallowed(self, timestamp: int)->bool:
        lastrequestduration = timestamp - self.lasttimestamp
        currentRate = 1/lastrequestduration
        if currentRate > self.ratevalue:
            print("Request Dropped", currentRate)
        else:
            print("Request accepted", currentRate)
            self.lasttimestamp= timestamp

class RateLimiterwithBurst:
    def __init__(self, maxallowedrequets, timeduration):
        self.maxallowedrequest = maxallowedrequets
        self.timeduration = timeduration
        self.req = deque()

    def isallowed(self, timestamp)->bool:
        if  self.req :
            print(self.req)
            if self.req[0] <= timestamp - self.timeduration:
                print(f"new request time :{timestamp}")
                self.req.popleft()
        if len(self.req) < self.maxallowedrequest:
            self.req.append(timestamp)
            return True
        else:
            return False
            

@dataclass
class Server:
    name:str
    serverweight: int
    currentweight: int
    utilization: float
    maxcapacity: int

def weightedrrloadbalancer(servers: list, threshold:float=0.5)->str:

    selected_server = None
    #compute the total weight
    total_weight =0
    for server in servers:
        total_weight += server.serverweight

    for server in servers:
        server.currentweight += server.serverweight
        if selected_server is None or server.currentweight > selected_server.currentweight:
            if (server.utilization/server.maxcapacity >=0.5):
                print(f"Server is over utlized - utilization: {server.utilization/server.maxcapacity}")
                continue
            else:
                selected_server = server

    if selected_server is not None:
        selected_server.currentweight -= total_weight
        return selected_server.name

    return None


def testroundrobin():
    serverlist = [Server("1", 5, 0,0,10), Server("2", 1, 0,0,10), Server("3", 1, 0,0,10)]
    
    stats={}

    total_requests =1000
    for _ in range(total_requests):
       serverid= weightedrrloadbalancer(serverlist)
       for server in serverlist:
           if serverid == server.name:
               server.utilization +=1
       #print(f"Selected server: {server}")
       if serverid not in stats:
           stats[serverid]=1
       else:
           stats[serverid]+= 1 
    for key in stats.keys():
        stats[key] /= total_requests

    print(stats)
    for server in serverlist:
        print(f"Server : {server.name}, proportion: {server.serverweight/7}")

# Consistent hashing key/values among a cluste of servers

class serverstore:
    def __init__(self, serverid):
        self.name = serverid
        self.data ={}

    def insert(self, key, value):
        self.data[key]=value

    def remove(self, key):
        if key in self.data:
            try:  
                self.data.pop(key) 
            except KeyError:
                print("key couldnt be found") 

    def __str__(self):
        return f"server: {self.name}, key/values :{self.data}"

class consistenthashing:
    def __init__(self, _serverlist:list, virtualnodelen:int, maxhash:int = 2**32):
        self.ring={}
        self.sortedkey=[]
        self.maxhash =maxhash
        self.virtulnodelen = virtualnodelen
        self.totalkeys =0
        self.serverlist=[]
        for server in _serverlist:
            serverobj = serverstore(server)
            self.serverlist.append(serverobj)
            for i in range(virtualnodelen):
            #    print(server)
                nodeid = server + '-' + str(i) 
                key = (self.computehash(nodeid))%self.maxhash
                self.ring[key] = serverobj
                self.sortedkey.append(key)
                self.sortedkey = sorted(self.sortedkey)
    
    def computehash(self, key):
        key_bytes = key.encode('utf-8')
        md5_hash = hashlib.md5(key_bytes).hexdigest()
        return int(md5_hash, 16)

    def insertkey(self, key, value) -> str:
        serverobj = self.getserver(key)
        serverobj.insert(key, value)
        self.totalkeys+=1

    def getserver(self, key:str) -> str:
        hash = self.computehash(key) % self.maxhash
        nodekey = binarysearch(self.sortedkey, hash)
        if nodekey == len(self.sortedkey):
            nodekey=0
        return self.ring[self.sortedkey[nodekey]]


    def addserver(self, server):
        for i in range(self.virtualnodelen):
            nodeid = server.name + '-' + str(i) 
            key = self.computehash(nodeid)%self.maxhash
            self.ring[key] = nodeid
            self.sortedkey = sorted(self.sortedkey[key])
                                   
    def listservers(self):
        for server in self.serverlist:
            print(f"{server}")

    def plotserverkeys(self):
        servers=[]
        keyporportion =[]
        print(f"Total keys: {self.totalkeys}")
        for serverobj in self.serverlist:
            servers.append(serverobj.name)
            keyporportion.append(len(serverobj.data)/self.totalkeys)
            print(f"Keys in {serverobj.name}: {len(serverobj.data)}")

        plt.plot(servers, keyporportion)
        plt.ylim(0, 1.0)  # Set y-axis range from -1.5 to 1.5
        plt.xlabel('Servers')
        plt.ylabel('Proportion of Keys')
        plt.title('Consistent Hashing plot')
        plt.show()
    
        

def binarysearch(sortedlist:list, key:int):
        low = 0
        high = len(sortedlist) -1

        while (low <= high):
            mid = (low +high)//2
            if key >= sortedlist[mid]:
                low = mid +1
            else:
                high = mid -1
        return low      

def random_string(length=5):
    # Generates a random string of a given length
    return ''.join(random.choices(string.ascii_letters, k=length))
                        
def testconsistenthashing():
    serverlist=['server1', 'server2', 'server3', 'server4','server5']
    hashlist = consistenthashing(serverlist, 3, 2**32)
    hashlist.insertkey("ravi", 12345)
    hashlist.insertkey("sam", 2345)
    hashlist.insertkey("adfadf", 34545)
   

    #generate a random set of strings and numbers and test it
    len = 10000
    for i in range(len):
        hashlist.insertkey(random_string(), random.randint(1000, 100000))

    print("Printing servers...")
    #hashlist.listservers()
    hashlist.plotserverkeys()

def random_string(length=5):
    # Generates a random string of a given length
    return ''.join(random.choices(string.ascii_letters, k=length))

def main():
    requestsallowed = 10.
    timeduraton =10.
    rlimiter = RateLimiterwithBurst(requestsallowed, timeduraton)

    currenttime = 0
    meancustomerrate = 1.
    meaninterarrivaltime = 1/meancustomerrate
    #simulate for 1000 Requests following poisson distribution

    for i in range(100):
        nextRequest = currenttime + np.random.exponential(meaninterarrivaltime)
        if(rlimiter.isallowed(nextRequest)):
            print("Request accepted")
        else:
            print("Request dropped")

        currenttime=nextRequest

    



if __name__=="__main__":
    #main()
    #testroundrobin()
    testconsistenthashing()




