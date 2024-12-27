import time
import sched
import numpy as np
from enum import Enum
from collections import deque
from dataclasses import dataclass
import random
import  matplotlib.pyplot as plt

currenttime =0
eventList =[]

@dataclass
class ipPacket():
    sourceIP: str
    destIP: str
    payload : int

class port():
    def __init__(self, num_ports):
        self.num_ports = num_ports

class egress_port(port):
    def __init__(self, num_ports, portBw:int):
        super.__init__(num_ports)
        self.routerEgressQueue=[{i,[]} for i in range(self.num_ports)]

    def egressPacketProc(self, packet: ipPacket, egressPort:int):
        self.routerEgressQueue[egressPort].append(ipPacket)

class ingress_port(port):
    def __init__(self, num_ports:int, egressFunc: egress):
        super.__init__(num_ports)
        self.routerIngressQueue = [{i,[]} for i in range(self.num_ports)]
        self.routingTable= dict()
        self.egressFunc = egressFunc

    def addRoutingEntry(self, ipaddress: str, egressport: int ):
        if ipaddress not in self.routingTable:
            self.routingTable[ipaddress]= [] #initialize if the key is not found
        self.routerTable[ipaddress].append(egressport)

    def ingressPacketProc(self, packet:ipPacket):
        src = packet.sourceIP
        if src in self.routingTable:
            nextHopId = self.routingTable[src]
            self.egressFunc.egressPacketProc(ipPacket, nextHopId)


class router():
    def __init__(self, ingressPorts, egressPorts):
        self.egressFwd = egress_port(egressPorts)
        self.ingressFwd = ingress_port(ingressPorts, self.egressFwd)
    
    def handleIngressEvent(self, ipPacket, inputPort):
        self.ingressFwd.ingressPacketProc(ipPacket)
        self
        