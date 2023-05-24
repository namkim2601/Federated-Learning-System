import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import pathlib
import socket
import sys

class MyMCLR: # Methods are from Week 5 Tutorial Solution
    @staticmethod
    def softmax(x, w):
        z = x.dot(w)
        e_z = np.exp(z)
        a = e_z / e_z.sum(axis=1, keepdims=True)
        return a
    @staticmethod
    def softmax_loss(x, y, w, lambd=0.1):
        a = MyMCLR.softmax(x, w)
        id0 = range(x.shape[0])
        loss = -np.mean(np.log(a[id0, y.astype(int)]))
        return loss
    @staticmethod
    def softmax_grad(x, y, w):
        pred = MyMCLR.softmax(x, w)  
        xid = range(x.shape[0])   
        pred[xid, y] -= 1         
        return (x.T.dot(pred) / x.shape[0])
    @staticmethod
    def softmax_fit(x, y, w, type, lr=0.01, n_epochs=50, tol=1e-6, batch_size=10):
        w_old = w.copy()
        epoch = 0 
        loss_hist = [MyMCLR.softmax_loss(x, y, w)]
        n = x.shape[0]
        while epoch < n_epochs: 
            epoch += 1 
            if type == 0: # GD
                w -= lr * MyMCLR.softmax_grad(x, y, w)
            elif type == 1: # mini-batch GD
                idx = np.random.choice(n, batch_size, replace=False)
                x_batch = x[idx]
                y_batch = y[idx]
                w -= lr * MyMCLR.softmax_grad(x_batch, y_batch, w)
            loss_hist.append(MyMCLR.softmax_loss(x, y, w))
            if np.linalg.norm(w - w_old) / w.size < tol:
                break 
            w_old = w.copy()
        return w, loss_hist 
    @staticmethod
    def pred(x, w):
        a = MyMCLR.softmax(x, w)
        return np.argmax(a, axis=1)
    @staticmethod
    def accuracy(x, y, w):
        count = MyMCLR.pred(x, w) == y
        accuracy = count.sum() / len(count)
        return accuracy    
    
class FLClient:
    def __init__(self, client_id, port, opt_method):
        self.id = client_id
        self.port = port
        self.opt_method = opt_method
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.train_data = {}
        self.test_data = {}
        self.accuracies = []

        pathlib.Path('logs').mkdir(parents=True, exist_ok=True) 
        pathlib.Path('graphs').mkdir(parents=True, exist_ok=True) 

        self.log_filename = "logs/" + self.id + "_log.txt"
        try:
            with open(self.log_filename, "r+") as f:
                f.truncate(0)  # clear file contents
        except FileNotFoundError:
            pass
        
    def load_initial_data(self):
        '''
        Load training and testing data from FLdata directory
        '''
        train_path = "FLdata/train/mnist_train_" + self.id + ".json"
        test_path = "FLdata/test/mnist_test_" + self.id + ".json"
        with open(os.path.join(train_path), "r") as f_train:
            train = json.load(f_train)
            self.train_data.update(train['user_data'])
        with open(os.path.join(test_path), "r") as f_test:
            test = json.load(f_test)
            self.test_data.update(test['user_data'])
    
    def save_output_to_file(self, train_loss, test_acc): 
        '''
        Save training results to external files
        '''
        # Write output to log file
        with open(self.log_filename, 'a') as f:
            msg = "{}, {}\n".format(train_loss[len(train_loss) - 1], test_acc)
            f.write(msg)

        # Plot train loss data and save graph 
        plt.plot(train_loss)
        plt.title(self.id + " Training Loss")
        plt.xlabel('number of epoches', fontsize = 13)
        plt.ylabel('loss', fontsize = 13)
        plt.tick_params(axis='both', which='major', labelsize=13)

        plot_f = "graphs/" + self.id + "_loss.jpeg"
        plt.savefig(plot_f)

        # Plot test accuracy data and save graph 
        plt.plot(self.accuracies)
        plt.title(self.id + " Accuracy")
        plt.xlabel('number of iterations', fontsize = 13)
        plt.ylabel('accuracy(%)', fontsize = 13)
        plt.tick_params(axis='both', which='major', labelsize=13)

        plot_f = "graphs/" + self.id + "_accuracy.jpeg"
        plt.savefig(plot_f)

    
    def print_output(self, train_loss, test_acc):
        '''
        Print training results to console
        '''
        output = f"I am client {self.id[6]}\n" 
        output += f"Training loss: {train_loss[len(train_loss) - 1]:.2f}\n" 
        output += f"Testing accuracy: {test_acc * 100:.2f}%\n"
        output += "Local training...\n" 
        output +="Sending new local model"
        print(output)

    def handshake(self): 
        '''
        Establish connection with FLServer
        '''
        self.sock.settimeout(5)
        response  = None
        try:
            self.sock.connect(('localhost' , 6000))
            msg = "{},{}".format(self.id, len(self.train_data['0']['x'][1]))
            self.sock.sendall(msg.encode())
            response = self.sock.recv(1024).decode()
        except socket.timeout:
            print("Handshake with FLserver unsuccessful: Server stopped listening")
            exit(1)
       
        if response:
            print(response + " successful handshake")
        self.sock.settimeout(None)

    def client_update(self, w):  
        '''
        Train models with given global model 'w' and update client model using MyMCLR
        ''' 
        x_train = np.array(self.train_data['0']['x'])
        y_train = np.array(self.train_data['0']['y'])
        x_test = np.array(self.test_data['0']['x'])
        y_test = np.array(self.test_data['0']['y'])
        
        # Train models
        w_new, train_loss = MyMCLR.softmax_fit(x_train, y_train, w, self.opt_method)
        test_acc = MyMCLR.accuracy(x_test, y_test, w_new) # Get testing accuracy
        
        # Save and print training results
        self.accuracies.append(test_acc * 100)
        self.save_output_to_file(train_loss, test_acc)
        self.print_output(train_loss, test_acc)

        return w_new, train_loss, test_acc 

    def communicate(self):
        '''
        Communicate with FLserver until disconnection
        '''
        # Receive aggregated global model
        self.sock.settimeout(None)
        w_data = self.sock.recv(1024)
        self.sock.settimeout(0.01)
        while True:
            try:
                chunk = self.sock.recv(1024)
            except socket.timeout:
                break
            w_data += chunk
        w_json = w_data.decode()
        w = json.loads(w_json)
        w_np = np.array(w)

        # Send updated client model
        new_w, train_loss, test_acc = self.client_update(w_np)
        gm_json = json.dumps(new_w.tolist())
        self.sock.sendall(gm_json.encode())

    def run(self):
        self.load_initial_data()
        self.sock.bind(('localhost', self.port))
        try:
            self.handshake()
        except Exception as e:
            print("Handshake with FLServer unsuccessful: {}".format(str(e)))
    
        while True:
            try:
                self.communicate()
            except KeyboardInterrupt:
                break

        self.sock.close()
        
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Error: The FLClient program requires 3 arguments")
        exit(1)

    client_id, port, opt_method = sys.argv[1], sys.argv[2], sys.argv[3]
    
    # Defensive checks for args
    valid_client_ids = ['client1', 'client2', 'client3', 'client4', 'client5']
    valid_port_ids = ['6001', '6002', '6003', '6004', '6005']
    if client_id not in valid_client_ids:
        print("Error: 1st arg 'client_id' must be either 'client1', 'client2' ... or 'client5")
        exit(1)
    if port not in valid_port_ids:
        print("Error: 2nd arg 'port' must range from 6001 to 6005")
        exit(1) 
    if client_id[6] != port[3]:
        print("Error: client1 must be on port 6001, client2 on port 6002 ...")
        exit(1) 
    if opt_method != '0' and opt_method != '1':
        print("Error: 3rd arg 'opt_method' must be either 0 or 1")
        exit(1)
    
    # Create and run FLClient
    client = FLClient(client_id, int(port), opt_method)
    client.run()
