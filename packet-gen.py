from scapy.all import *

def generate_packet(src_ip, dest_ip, dst_port, packet_count):
    for i in range(packet_count):
        packet = IP(src=src_ip, dst=dest_ip)/TCP(dport=dst_port)/"Hello World"
        send(packet)
        print(f"Packet {i} sent")

generate_packet("192.168.1.1", "192.168.1.2",80, 10)
