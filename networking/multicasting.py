import random
import threading
import time

class multicast_sender:
    def __init__(self,  window_size:int=100):
        self.sequence_number=0
        self.window={}
        self.window_size =100
        self.receiver_list = []

    def add_receiver(self, receiver):
        self.receiver_list.append(receiver)
    
    def send_data(self, data):
        packet = {'seq': self.sequence_number, 'data':data}
        self.window[self.sequence_number]= packet
        self.multicast(packet)
        self.sequence_number+=1

    def multicast(self, packet):
        unreliable_multicast(packet, self.receiver_list )

    def handle_nack(self, seq):
        if seq in self.window:
            packet = self.window[seq]
            self.multicast(packet)
    
class multicast_receiver:
    def __init__(self,  name, sender):
        self.name = name
        self.expected_sequence=0
        self.buffer ={}
        self.delivered_buffer=[]
        self.sender = sender
        self.missing_checker=threading.Timer(0.5, self.detect_missing)
        self.missing_checker.daemon = True
        self.missing_checker.start()

    def deliver_packet(self,packet):
        print(f"Receiver {self.name} delivered packet..{packet['seq']}")
        self.delivered_buffer.append(packet)

    def receive_packet(self, packet):
        sequence_num = packet['seq']
        if self.expected_sequence == sequence_num:
            self.deliver_packet(packet)
            self.expected_sequence +=1
            while len(self.buffer)>0 and self.expected_sequence in self.buffer:
                self.deliver_packet(self.buffer[self.expected_sequence])
                self.buffer[self.expected_sequence].pop()
                self.expected_sequence +=1
        elif packet['seq'] > self.expected_sequence:
            self.buffer[self.expected_sequence]=packet

        self.detect_missing()

    def detect_missing(self):
        time.sleep(random.uniform(0.1, 0.5))
        if self.expected_sequence not in self.buffer:
            send_nack(self.expected_sequence, self.name, self.sender)
        #reschedule the missing packet detection
        self.missing_checker=threading.Timer(0.5, self.detect_missing)
        self.missing_checker.daemon = True
        self.missing_checker.start()


def send_nack(seq, receiver, sender):
    print(f"receiver {receiver} sending NACK to sender")
    sender.handle_nack(seq)

def unreliable_multicast(packet, receivers):
    if random.random() <0.2:
        print(f"Packet dropped: {packet}")
        return
    
    #simulate a delay
    delay = random.uniform(0.05, 0.2)
    threading.Timer(delay, deliver_packet, args=(packet, receivers )).start()

    if random.random()<0.1:
        threading.Timer(delay +0.05, deliver_packet,args=(packet, receivers )).start()

def deliver_packet(packet, receivers):
    for receiver in receivers:
        receiver.receive_packet(packet)


def test_multicast():
    sender = multicast_sender()
    receivers = [multicast_receiver("R1", sender), multicast_receiver("R2", sender)]
    for receiver in receivers:
        sender.add_receiver(receiver)

    messages = [f"Message {i}" for i in range(10)]

    for msg in messages:
        sender.send_data(msg)
        time.sleep(0.1)

    #Allow time for retransmission
    time.sleep(10)

    for receiver in receivers:
        delivered_seq = [pkt['seq'] for pkt in receiver.delivered_buffer]
        print(f"{receiver.name} delivered sequences : {delivered_seq}")

if __name__=="__main__":
    test_multicast()