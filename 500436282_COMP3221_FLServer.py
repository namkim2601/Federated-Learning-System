import json
import numpy as np
import random
import socket
import sys
import threading
import time

class FLServer:
    def __init__(self, port, sub_client):
        self.port = port
        self.sub_client = sub_client
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connections = []
        self.batch_sizes = {}
        self.lock = threading.Lock()

        self.c_models = {}
        self.updates = 0
        self.iterations = 1
        
    def handshake(self, conn): 
        ''' 
        Establish connection with incoming client with 'conn'
        '''
        data = conn.recv(1024).decode()
        client_id, batch_size = data.split(",")
        self.connections.append((conn, client_id))
        self.batch_sizes[client_id] = batch_size
        conn.send(client_id.encode())
    
    def initial_broadcast(self, conn, client_id):
        '''
        Randomly initialise a global model and broadcoast it to client connected on 'conn'
        '''
        batch_size = int(self.batch_sizes[client_id]) 
        w0 = np.random.rand(batch_size, 10)
        gm_json = json.dumps(w0.tolist())
        conn.sendall(gm_json.encode())

    def update_client_model(self, conn, client_id):   
        '''
        Recieve an updated client model from 'conn'
        and store it using 'clientid' as the key
        '''
        # Recieve new client model
        print("Getting local model from client " + client_id[6])
        conn.settimeout(None)
        upd_cm_data = conn.recv(1024)
        conn.settimeout(0.01)
        while True:
            try:
                chunk = conn.recv(1024)
            except socket.timeout:
                break
            upd_cm_data += chunk
        upd_cm_json = upd_cm_data.decode()
        upd_cm = json.loads(upd_cm_json)
        upd_cm_np = np.array(upd_cm)

        # Save the updated client model
        with self.lock:
            self.c_models[client_id] = upd_cm_np
            self.updates += 1

    def aggregate(self):
        '''
        Use the updated client models to aggregate a new global model
        '''
        print("Aggregating new global model")
        if self.sub_client == 0: # M = K = 5
            stacked_models = np.stack([self.c_models["client1"], self.c_models["client5"],
                                        self.c_models["client3"], self.c_models["client4"],
                                        self.c_models["client5"]], axis=-1)
            return np.mean(stacked_models, axis=-1)
        elif self.sub_client == 1: # M = 2
            models = random.sample(list(self.c_models.values()), 2)
            stacked_models = np.stack([models[0], models[1]], axis=-1)
            return np.mean(stacked_models, axis=-1)
        return -1

    def broadcast(self, conn, new_w):
        '''
        Broadcast the updated global model 'new_w' to the client connected on 'conn'
        '''
        gm_json = json.dumps(new_w.tolist())
        conn.sendall(gm_json.encode())    
    
    def communicate(self):
        '''
        Continuously communicate with clients for 100 global communications rounds
        '''
        while True:
            if self.iterations > 100:
                return    

            print("\nGlobal Iteration: " + str(self.iterations))
            print("Total Number of clients: " + str(len(self.connections)))
            for conn, client_id in self.connections:
                update_thread = threading.Thread(target=self.update_client_model, args=(conn, client_id))
                update_thread.start()
        
            while True:
                if self.updates == 5:
                    self.updates = 0
                    break

            print("Broadcasting new global model")
            new_w = self.aggregate() 
            for conn, client_id in self.connections:
                broadcast_thread = threading.Thread(target=self.broadcast, args=(conn, new_w))
                broadcast_thread.start()
            self.iterations += 1

    def run(self):
        print(f"Starting FLServer on localhost:{self.port}, sub_client mode: {self.sub_client}")
        self.sock.bind(('localhost', self.port))
        self.sock.listen(5)
        conn, addr = self.sock.accept() # First connection
        self.handshake(conn)

        # Only accept clients that connect within next 30 seconds
        print("Listening for new connections...")
        self.sock.settimeout(0.1)
        start_time = time.time()
        while time.time() - start_time < 30: 
            try:
                conn, addr = self.sock.accept()
            except socket.timeout:
                continue
            handshake_thread = threading.Thread(target = self.handshake, args=(conn, ))
            handshake_thread.start()
        self.sock.settimeout(None)
        print("Stopped listening for new connections")

        # First Broadcast
        try:
            for conn, client_id in self.connections:
                self.initial_broadcast(conn, client_id)  
        except Exception as e:
            print("Error with initial broadcast: {}".format(str(e)))
  
        self.communicate() # Handler for clients

        # Close all client connections
        for conn, addr in self.connections:
            conn.close()
        self.sock.close()
        print("\nFLServer stopped")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Error: The FLServer program requires 2 arguments")
        exit(1)

    port, sub_client = sys.argv[1], sys.argv[2]

    # Defensive checks for args
    if port != "6000":
        print("Error: 1st arg 'port' must be 6000")
        exit(1)
    if sub_client != '0' and sub_client != '1':
        print("Error: 2nd arg 'sub_client' must be either 0 or 1")
        exit(1)

    # Create and run FLServer
    fl_server = FLServer(int(port), int(sub_client))
    fl_server.run()