import time
import sched
import numpy as np
from enum import Enum
from collections import deque
from dataclasses import dataclass
import random
import  matplotlib.pyplot as plt
import ipaddress

currenttime =0
eventList =[]

class enumType(Enum):
    ingress = 1
    egress =2

#@dataclass
class ipPacket():
    def __init__(self, sourceIP, destIP, payload):
        self.sourceIP = sourceIP
        self.destIP = destIP
        self.payload = payload

    def __str__(self):
        return f"sourceIP: {self.sourceIP}, destIP: {self.destIP}, payload: {self.payload}"

@dataclass
class event():
    timestamp: float
    eventType: Enum
    packet: ipPacket

class port():
    def __init__(self, num_ports, maxQueueSize=10):
        self.num_ports = num_ports
        self.maxQueueSize = 10

class egress_port(port):
    def __init__(self, num_ports, portBw:int):
        super().__init__(num_ports)
        self.routerEgressQueue={i:deque() for i in range(self.num_ports)}
        self.portBw = portBw

    def egressPacketProc(self, packet: ipPacket, egressPort:int):
       # print(f"Packet {packet} being processed in in the egress port")
        _queue=self.routerEgressQueue[egressPort]
        if len(_queue) < self.maxQueueSize:
            _queue.append(packet)
            print(f"Packet {packet} sent to egress port {egressPort}")
        else:
            print(f"Packet {packet} dropped due to queue overflow")
            del packet
            return
        #schedule event to send the packet to the egress port
        scheduleTime = currenttime + (packet.payload)/self.portBw
        eventItem = event(scheduleTime, enumType.egress, packet)
        eventList.append(eventItem)

    def listEgressQueue(self):
        print(f"***Egress Ports Status:****\n")
        for port, _list in self.routerEgressQueue.items():
            print(f"Port {port}")
            if len(_list) > 0:
                for packet in _list:
                    print(f"Packet: {packet}")
            else:
                print(f"No packets in the queue")

class ingress_port(port):
    def __init__(self, num_ports:int, egressFunc: egress_port, maxQueueSize=10):
        super().__init__(num_ports, maxQueueSize)
        self.routerIngressQueue = { i: deque() for i in range(self.num_ports)}
        self.routingTable= dict()
        self.ipPrefixTable=set()
        self.egressFunc = egressFunc

    def addRoutingEntry(self, _ipaddress: str, egressPortList: set ):
        if ipaddress not in self.routingTable:
            self.routingTable[_ipaddress]= set() #initialize if the key is not found
            self.ipPrefixTable.add(_ipaddress)
            for port in egressPortList:
                self.routingTable[_ipaddress].add(port)
        else:
            for port in egressPortList:
                self.routingTable[_ipaddress].append(port)      

    def ingressPacketProc(self, ipPacket, inputPort:int):
        _queue = self.routerIngressQueue[inputPort]
        if len(_queue) > 0:
            packet = _queue.popleft()
            self.processPacket(packet, inputPort)
        #schedule event to send the packet to the egress port
        scheduleTime = currenttime + 0.1
        eventItem = event(scheduleTime, enumType.ingress, ipPacket)
        eventList.append(eventItem)
        self.ingressPacketQueue(ipPacket, inputPort)

    def ingressPacketQueue(self, packet:ipPacket, inputPort:int):
        #check if the length of the queue is less than the max queue size
        _queue = self.routerIngressQueue[inputPort]
        if len(_queue) < self.maxQueueSize:
            _queue.append(packet)
            print(f"Packet {packet} sent to ingress port {inputPort}")
        else:
            print(f"Packet {packet} dropped due to queue overflow")
            del packet
        return   

    def processPacket(self, packet:ipPacket, inputPort:int):
        dest = packet.destIP
       # print(f"Packet {packet} being processed")
        prefixMatch = self.longestPrefixMatch(dest)
        if prefixMatch:
            nextHopSet = self.routingTable[prefixMatch]
            nextHopList = list(nextHopSet)
            nextHopId = random.choice(nextHopList)
            self.egressFunc.egressPacketProc(packet, nextHopId)
            print(f"Packet {packet} sent to egress port {nextHopId}")
        else:
            print(f"No routing entry found for {dest}, check for default route")
            defaultRoute = "0.0.0.0"
            if defaultRoute in self.routingTable:
                nextHopSet = self.routingTable[defaultRoute]
                nextHopList = list(nextHopSet)
                nextHopId = random.choice(nextHopList)
                self.egressFunc.egressPacketProc(packet, nextHopId)
                print(f"Packet {packet} sent to egress port {nextHopId}")     
            else:
                print(f"No default route set, deleting packet")    
                del packet
        return
    
    def longestPrefixMatch(self, _ipaddress):
        prefixMatch = None
        ip = ipaddress.IPv4Address(_ipaddress)
        max_prefix_len=-1
        for prefix in self.ipPrefixTable:
            network = ipaddress.IPv4Network(prefix)
            if ip in network:
                if network.prefixlen > max_prefix_len:
                    prefixMatch = prefix
                    max_prefix_len = network.prefixlen
        return prefixMatch
            

    def printRoutingTable(self):
        print(f"Routing Table:\n")
        for key, value in self.routingTable.items():
            print(f"IP: {key}, Egress Port: {value}")

    def listIngressQueue(self):
        print(f"***Ingress Ports Status:****\n")
        for port, _list in self.routerIngressQueue.items():
            print(f"Port {port}")
            if len(_list) >0:
                for packet in _list:
                    print(f"Packet: {packet}")
            else:
                print(f"No packets in the queue")

    #def scheduleEvent(eventItem: event):    
class router():
    def __init__(self, ingressPorts, egressPorts, portBw):
        self.egressFwd = egress_port(egressPorts, portBw )
        self.ingressFwd = ingress_port(ingressPorts, self.egressFwd)
        self.ingressPorts = ingressPorts
        self.egressPorts = egressPorts
    
    def handleIngressEvent(self, ipPacket, inputPort):
        self.ingressFwd.ingressPacketProc(ipPacket, inputPort)
    
    def handleEgressEvent(self, ipPacket, outputPort):    
        self.egressFwd.egressPacketProc(ipPacket, outputPort)
    
    def addRoutingEntry(self, ipaddress, egressport:set):
        try:
            for x in egressport:
               assert x < self.egressPorts, "Egress port number is out of range"
               self.ingressFwd.addRoutingEntry(ipaddress, egressport)
        except AssertionError as e:
            print(f"Routing entry failes:{e}")


    def printRoutingTable(self):
        print(f"Routing Table:\n")
        self.ingressFwd.printRoutingTable()   

    #list the packets in the ingress queue
    def listIngressQueue(self):
        print(f"Ingress queue:\n")
        self.ingressFwd.listIngressQueue()

    def listEgressQueue(self):
        print(f"Egress queue:\n")
        self.egressFwd.listEgressQueue()

def main():
    simuTime = 10
    portBw = 100
    router1 = router(4,4, portBw)
    router1.addRoutingEntry(ipaddress="10.1.1.0/24", egressport={1,2})
    router1.addRoutingEntry(ipaddress="10.1.2.0/24", egressport={2})
    router1.addRoutingEntry(ipaddress="10.1.3.0/24", egressport={3})
    router1.addRoutingEntry(ipaddress="0.0.0.0", egressport={3})

    router1.printRoutingTable()

    #generate some packets
    packet1 = ipPacket("1.1.1.1",  "10.1.1.1", 100 )
    packet2 = ipPacket("2.2.2.2",  "10.1.2.1", 1000)
    packet3= ipPacket("3.3.3.3", "10.1.4.1", 3000)

    router1.handleIngressEvent(packet1, 0)
    router1.handleIngressEvent(packet2, 1)
    router1.handleIngressEvent(packet3, 2)


    router1.listIngressQueue()
    router1.listEgressQueue()

    # router1.handleIngressEvent(packet1, 0)
    # router1.handleIngressEvent(packet2, 1)
    # router1.handleIngressEvent(packet3, 2)

    # simulate the router with the event list generating the packets till the en do the simulation
    while currenttime < simuTime:
        eventList.sort(key=lambda x: x.timestamp)
        eventItem = eventList.pop(0)
        
        currenttime = eventItem.timestamp
        if eventItem.eventType == enumType.ingress:
            router1.handleIngressEvent(eventItem.packet, 0)
        elif eventItem.eventType == enumType.egress:
            router1.handleEgressEvent(eventItem.packet, 0)
        else:
            print(f"Unknown event type")
        time.sleep(0.1)
        #print(f"Current time: {currenttime}")    



if __name__ == "__main__":
    main()