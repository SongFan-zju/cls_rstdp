import os
import socket
import time
import struct
import sys
import numpy as np


class Transmitter(object):

    def __init__(self):
        self.socket_inst = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_inst.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def connect_lwip(self, ip_address):
        self.socket_inst.connect(ip_address)

    def close(self):
        self.socket_inst.close()

    def send_flit_from_file(self, flit_file, directions=0):
        with open(flit_file, 'r') as file:
            flit_list = file.readlines()
        # length = len(flit_list)
        # if length > 2**26:
        #     print("===<2>=== %s is larger than 0.25GB" % flit_file)
        #     print("===<2>=== send flit length failed")
        #     return 0
        # j = 0
        # while (j < length):
        #     send_bytes = bytearray()
        #     send_bytes += struct.pack('Q', length)
        #     # send_bytes += struct.pack('I', 0x8000) # TODO tik
        #     for i in range(j, min(j + 16777216 * 4, length)):
        #         send_bytes += struct.pack('Q', int(flit_list[i % length].strip(), 16))
        #     self.socket_inst.sendall(send_bytes)
        #     j = j + 16777216 * 4
        #     if (j <= length):
        #         reply = self.socket_inst.recv(1024)
        #         print("%s" % reply)
        # return 1
        return self.send_flit_from_file_string(flit_list, directions)

    def send_flit_from_string(self, flit_list, directions=0):
        j = 0
        length = len(flit_list)
        while (j < length):
            send_bytes = bytearray()
            send_bytes += struct.pack('Q', length)
            for i in range(j, min(j + 16777216 * 4, len(flit_list))):
                send_bytes += struct.pack('Q', int(flit_list[i % length].strip(), 16))
            self.socket_inst.sendall(send_bytes)
            j = j + 16777216 * 4
            if (j <= length):
                reply = self.socket_inst.recv(1024)
                print("%s" % reply)
        return 1
