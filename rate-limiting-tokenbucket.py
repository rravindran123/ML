
import time
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

currentTime =0.0
arrivalQueue=list()

@dataclass
class request:
    userId: int
    arrivalTime : float

class tokenBucket:
    def __init__(self, capacity:int, refillRate:int):
        self.bucketCap = capacity
        self.refillRate = refillRate
        self.tokens = capacity
        self.lastRefill = currentTime
       # self.serverQueue= list()
    
    def allowRequest(self):
        returnVal = False
        if self.tokens >0:
            self.tokens -=1
            returnVal= True
        else:
            print("No tokens- request denied")
        return returnVal

    def bucketRefill(self):
        global currentTime
        self.tokens =round(min(self.bucketCap, (self.tokens+ (currentTime - self.lastRefill)*self.refillRate)))
        self.lastRefill = currentTime
        print("Tokens in the bucket :", self.tokens)


class requestQueue:
    def __init__(self, arrivalRate):
        self.interarrivalTime = float(1/arrivalRate)
        self.userId =0

    def addRequest(self):
        global currentTime
        arrivaltime = currentTime + np.random.exponential(self.interarrivalTime)
        self.userId +=1
        userRequest = request(self.userId, arrivaltime)
        arrivalQueue.append(userRequest)
        #sort the arrivalQue based on the time
        arrivalQueue.sort(key=lambda x : x.arrivalTime)
        print("New customer arrival created id/time:",self.userId,"/",arrivaltime)

    def getRequest(self):
        returnItem = None
        if len(arrivalQueue) > 0:
            returnItem = arrivalQueue[0]
        else:
            print("No requests to serve")

        return returnItem

    def delRequest(self):
        if len(arrivalQueue)>0:
            arrivalQueue.pop(0)
        else:
            print("No requests to remove from the arrival queue")
    

def main():
    simulationTime = 10.0
    arrivalRate = 2 
    refillRate = 3
    tokenBucketCap = 10
    
    interArrivalTime = float(1/arrivalRate)

    tbucket = tokenBucket(tokenBucketCap, refillRate)
    requestQ = requestQueue(arrivalRate)

    requestQ.addRequest()
    tbucket.bucketRefill()

    #stat collection
    timeVal = list()
    tokenBucketSize = list()
    arrivalUserList= list()
    global currentTime

    while currentTime <=simulationTime:
        newRequest = requestQ.getRequest()
        
        #check if the request can be allowed
        if tbucket.allowRequest():
            print("Allow user request")
            requestQ.delRequest()
        
        currentTime = newRequest.arrivalTime
        requestQ.addRequest()
        tbucket.bucketRefill()
        
        #update the stats
        timeVal.append(currentTime)
        tokenBucketSize.append(tbucket.tokens)
        arrivalUserList.append(len(arrivalQueue))


    plt.plot(timeVal, tokenBucketSize, marker='o', label="Token bucket size")
    plt.plot(timeVal,  arrivalUserList, marker='x', label="Arrival Queue Size")
    # Adding axis labels
    plt.xlabel('Arrival Time (s)')  # Label for x-axis
    plt.ylabel('Count')             # Label for y-axis

    # Adding a title
    plt.title('Token vs Arrival Queue Over Time')

    # Adding a legend
    plt.legend()
    plt.show()

if __name__=="__main__":
    main()
    


