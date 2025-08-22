from dataclasses import dataclass
import time
import heapq

@dataclass
class connection_state:
    source_ip:str
    source_port:int
    dest_io:str
    dest_port:int
    last_active_time:float



class connection_tracker():
    def __init__(self, timeout:float=10.0):
        self.connections:dict[tuple[str, str, int, int],connection_state]={}
        self.connection_timeout = timeout
        return

    def add_connection(self, src_ip:str , src_port:int, dst_ip: str, dst_port:int, event_time:float ) -> None:
        if (src_ip, src_port, dst_ip, dst_port) in self.connections:
            print("Connection already existis")
            return
        new_connection = connection_state(src_ip,src_port, dst_ip, dst_port, event_time)
        self.connections[(src_ip, src_port, dst_ip, dst_port )] = new_connection
        return

    def get_active_connections(self, timestamp:float) -> list[tuple[str, int, str, int]]:
        #expire connections that have exceeded the timeout
        active_list=[]
        delete_list=[]
        for key, state in self.connections.items():
            if timestamp - state.last_active_time > self.connection_timeout:
                #remove the state from the dict
                print("connection timer expired..deleting")
                delete_list.append(key)
            else:
                active_list.append(key)

        #return the active connections
        for x in delete_list:
            del self.connections[x]

        return active_list

    def get_top_k_active_ips(self, timestamp:float, k:int) -> list[str]:
        #sort the dic using the activity
        count:dict[str, int] = {}

        for (srcip,  srcport, dstip, dstport), state in self.connections.items():
            if srcip not in count:
                count[srcip] =0
            if dstip not in count:
                count[dstip] =0
            if timestamp - state.last_active_time <= self.connection_timeout:
                count[srcip] +=1
                count[dstip]+=1

        top_k = heapq.nlargest(k, count.items(), key=lambda x:x[1])
        print(f"\n Top {k} IPs in the connnection list are..")
        print(top_k)

        return top_k
        

    def __repr__(self):
        response = f"List of connections..\n"
        for (srcip, dstip, srcport, dstport), state in self.connections.items():
            response = response + f"{srcip}, {dstip}, {srcport}. {dstport}, eventime: {state.last_active_time}\n"
               
        return response

current_time=0.0
track_connection = connection_tracker()
track_connection.add_connection("1.1.1.1", 12002, "2.2.2.2", 12009, current_time)
current_time = 1.0
track_connection.add_connection("2.1.1.1", 12002, "2.2.2.3", 12009, current_time)
print(track_connection)

print("Active connections")
current_time=6
_list= track_connection.get_active_connections(current_time)
print(_list)
current_time =6.0
topk = track_connection.get_top_k_active_ips(current_time, 3)