import socket
import pickle


HOST,PORT = 'localhost',1539

def call(msg):
	raw_msg = pickle.dumps(msg)
	sock = socket.socket()
	sock.sendall(raw_msg)
	raw_reply = []
	while True:
		data = c.recv(4096) 
		if not data: break
		raw_reply.append(data)
	raw_reply = b"".join(raw_reply)

	msg = pickle.loads(raw_reply)
	return msg