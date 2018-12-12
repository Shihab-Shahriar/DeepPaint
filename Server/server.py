import socket 
import pickle
from select import select

from Colorizer.algorithm import colorize
from Stylizer.algorithm import stylize

HOST,PORT = "0.0.0.0",1539
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((HOST,PORT))
sock.listen()

while True:
	readble,_,_ = select([sock],[],[],2)
	if not readble: continue
	c,a = sock.accept()
	print("Conn with ",a)

	file = c.makefile('b')
	msg = pickle.load(file,encoding='bytes')
	print("msg received..")
	
	if msg['type'] == 'colorize':
		out = colorize(msg['img'],msg['points'])

	elif msg['type'] == 'stylize':
		out = stylize(msg['cont'],msg['style'],msg['info'])

	c.sendall(pickle.dumps(out))
	print("REPLY SENT")
	c.close()