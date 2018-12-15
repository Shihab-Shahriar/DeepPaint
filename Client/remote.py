import socket
import pickle

HOST,PORT = "10.100.112.3",1539

def remote_call(msg):
	raw_msg = pickle.dumps(msg)
	sock = socket.socket()
	sock.connect((HOST,PORT))
	sock.sendall(raw_msg)
	print("Msg sent")

	file = sock.makefile('b')
	reply = pickle.load(file,encoding='bytes')
	print("Reply received")
	return reply