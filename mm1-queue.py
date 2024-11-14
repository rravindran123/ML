#simulating M/M/1 queue in Python
#queue into which arrivals happen and from which server picks item for service
import time
import sched
import numpy as np
from enum import Enum
from collections import deque
from dataclasses import dataclass
import random
import  matplotlib.pyplot as plt


class eventType(Enum):
    arrival =1
    departure =2

@dataclass
class event:
    timestamp:float
    eventType :  eventType
    customer :  'customer'

class serviceList:
    timestamp: float

@dataclass
class customer:
    identity : int
    arrivalTime : float
    serviceDuration : float # amount of work

class customerArrival:
    def __init__(self, arrivalRate, serviceRate):
        self.arrivalRate = arrivalRate
        self.interArrivalTime = float(1.0/arrivalRate)
        self.serviceRate = serviceRate
        self.avgServiceTime = float(1.0/serviceRate)

    def scheduleArrival(self):
        global currentTime, customerId, eventList
        # schedule the arrival of first customer in the queue
        # generate a arrival time according to expontial distribution(mePanArrivalTime)
        arrivaltime = currentTime + np.random.exponential(self.interArrivalTime)
        servicetime = np.random.exponential(self.avgServiceTime)
        customerId +=1
        new_customer= customer(customerId, arrivaltime, servicetime)
        print(f"Scheduling a new customer arrival {customerId}, arrival: {arrivaltime}, workload: {servicetime}")
        eventItem = event(arrivaltime, eventType.arrival, new_customer)
        eventList.append(eventItem)
        #sort the event queuelist
        #sort(eventList.timestamp)
        eventList.sort(key=lambda x: x.timestamp)
        
        #return new_customer

class customerServiceSched:
    def __init__(self, serviceRate):
        self.avgServiceTime = 1.0/serviceRate

    def scheduleService(self,customer):
        global currentTime
        servicetime =  currentTime + customer.serviceDuration
        print(f"Scheduling service for {customer.identity}, departure time {servicetime}")
        eventItem = event(servicetime, eventType.departure, customer)
        eventList.append(eventItem)
        eventList.sort(key=lambda x: x.timestamp)



currentTime=0.0
customerId =0
eventList = list()
customerQueue = deque()

def main():
    global currentTime, customerId, eventList, customerQueue
    simulationTime:float = 10000.0
    arrivalRate = [0.5, 1.0, 2.0, 3.0]
    serviceRate = 1.0
    queuelen=[]
    
    # get 
    for rate in arrivalRate:
        currentTime=0.0
        eventList.clear()
        customerQueue.clear()
        arrObject = customerArrival(rate, serviceRate)
        serObject = customerServiceSched(serviceRate)
        start = True
        cumulativeQueueLength = 0
        eventCount=0
        while currentTime <= simulationTime:
            if start:
                # create the first arrival in the queue
                print("Starting simulation, adding new customer")
                new_customer = arrObject.scheduleArrival()
                start = False
            else:
                #get the next event to the executed from the sorted eventlist
                event = eventList.pop(0)
                currentTime = event.timestamp
                if event.eventType == eventType.arrival:
                    # check if the cusomter length is 1, in this case schedule the cutomer for departure, or schedule depature for the customer 
                    # at the head of the customer queue 
                    print(f"Event time {event.timestamp}, customer: {event.customer.identity} , type: {event.eventType}")
                    new_customer = event.customer
                    customerQueue.append(new_customer)
                    if len(customerQueue)==1:
                        headCustomer = customerQueue[0]
                        serObject.scheduleService(headCustomer)
                    #schedule next arrival
                    arrObject.scheduleArrival()

                elif event.eventType == eventType.departure:
                    customer = event.customer
                    print(f"Event time {event.timestamp}, customer: {event.customer.identity} type: {event.eventType}")
                    assert customer.identity == customerQueue[0].identity 
                    # try:
                    #     assert customer.identity == customerQueue[0].identity
                    # except AssertionError as e:
                    #     print(f" {customer.identity} is not equal to {customerQueue[0].identity}")
                    customerQueue.popleft()
                    # schedule the next customer for departure
                    if len(customerQueue)>0:
                        headCustomer = customerQueue[0]
                        serObject.scheduleService(headCustomer)
                
                cumulativeQueueLength +=len(customerQueue)
                eventCount +=1
        
        avgQueueLen = cumulativeQueueLength/eventCount if eventCount >0 else 0
        print(f"Simulatiion ended, currenttime: {currentTime}, AvgQueuelength: {avgQueueLen}")
        queuelen.append(avgQueueLen)
    
    # plot Arrival/Service-rate versus length of the queue
    sysUtil = [rate/serviceRate for rate in arrivalRate]
    plt.plot(sysUtil, queuelen, marker='o')
    plt.show()


if __name__ =="__main__":
    main()
    




