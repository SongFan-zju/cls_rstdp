import os
import socket
import time
import struct
import sys
import numpy as np

DEFAULT_FPGA_IP = os.environ.get("PCSS_FPGA_IP", "127.0.0.1")


def binary_to_decimal(binary_str):
    if binary_str[0] == '1':
        inverted = ''.join('1' if bit == '0' else '0' for bit in binary_str)
        decimal_value = -(int(inverted, 2) + 1)
    else:
        decimal_value = int(binary_str, 2)
    return decimal_value


class Transmitter(object):

    def __init__(self):
        self.socket_inst = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_inst.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def connect_lwip(self, ip_address):
        self.socket_inst.connect(ip_address)

    def close(self):
        self.socket_inst.close()

    def send_from_file(self, flit_file, directions=0):
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
        return self.send_from_file_from_string(flit_list, directions)

    def send_from_file_from_string(self, flit_list, directions=0):
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
                # print("%s" % reply)
        return 1


def run_pcss(tc="", filename="", flit_list=[], recv=True, ip=DEFAULT_FPGA_IP, port=1234, flitoutfile=''):
    if tc != "":
        if os.path.exists(tc):
            tc = tc + "\\"
        else:
            tc = ""
    trans = Transmitter()
    ip_address = (ip, port)  #TODO port
    trans.connect_lwip(ip_address)
    if filename != "":
        res = trans.send_from_file(filename)  # send
    else:
        res = trans.send_from_file_from_string(flit_list)  # send
        return 0
    if res == 0:
        return
    hl = b""
    index = 0
    tot = 0
    return_arr = np.zeros([8, 1, 5])
    tik_list = [0, 0, 0, 0, 0]
    tik = 0
    if recv == False:
        return
    while recv:
        request = trans.socket_inst.recv(1024)
        if len(request) <= 0:
            break
        for i in range(len(request)):
            b = b"%02x" % request[i]
            hl = b + hl
            index = index + 1
            if (index == 8):
                #TODO
                if (hl[0:7].decode() == 'fffffff'):
                    break
                if (hl[0:7].decode() == '0300000'):
                    hl = b""
                    index = 0
                    tot = tot + 1
                    tik += 1
                    # print(hl)
                    continue
                hlb = ''.join(format(int(hex_digit, 16), '04b') for hex_digit in hl.decode())
                x = int(hlb[-8:], 2)
                # print(tik, x)
                # print(hl)
                return_arr[tik - 1, 0, x] = 1
                hl = b""
                index = 0
                tot = tot + 1

    trans.socket_inst.close()
    return_arr = return_arr.astype(np.int8)
    # print(np.argmax(return_arr.mean(0)))
    return return_arr, np.argmax(return_arr.mean(0))


def read_wgt(tc="", filename="", flit_list=[], recv=True, ip=DEFAULT_FPGA_IP, port=1234, flitoutfile=''):
    if tc != "":
        if os.path.exists(tc):
            tc = tc + "\\"
        else:
            tc = ""
    trans = Transmitter()
    ip_address = (ip, port)  #TODO port
    trans.connect_lwip(ip_address)
    if filename != "":
        res = trans.send_from_file(filename)  # send
    else:
        res = trans.send_from_file_from_string(flit_list)  # send
        return 0
    if res == 0:
        return
    hl = b""
    index = 0
    tot = 0
    weights = np.zeros([5, 16])
    tik = 0
    if recv == False:
        return
    while recv:
        request = trans.socket_inst.recv(1024)
        if len(request) <= 0:
            break
        for i in range(len(request)):
            b = b"%02x" % request[i]
            hl = b + hl
            index = index + 1
            if (index == 8):
                if (hl[0:7].decode() == '0300000'):
                    tik = tik + 1
                hlb = ''.join(format(int(hex_digit, 16), '04b') for hex_digit in hl.decode())
                ext_raddr_neu = hlb[-33:-21]
                ext_raddr_wgt = hlb[-21:-17]
                ext_rdata = hlb[-16:]
                x = int(ext_raddr_neu, 2)
                y = int(ext_raddr_wgt, 2)
                wgt = binary_to_decimal(ext_rdata[-8:])
                if x > 4000:
                    break
                weights[x, y] = wgt
                hl = b""
                index = 0
                tot = tot + 1

    trans.socket_inst.close()
    return weights


if __name__ == "__main__":
    # socket
    id = 2
    if len(sys.argv) > 1:
        id = int(sys.argv[1])

    dir = 'spike'
    print('flit_config/config_saimsnn.txt')
    run_pcss(filename=dir + 'flit_config/config_saimsnn.txt', port=id)

    filename = f'flit_spk/single/0.txt'
    start_time = time.time_ns()
    run_pcss(filename=dir + filename, port=id, flitoutfile='pcss_o_127_0.txt', tofile=1)
    end_time = time.time_ns()
    print(filename + '\t finished in: ' + f'{(end_time - start_time) / 1000000}' + 'ms')

    print('flit_config/config_saimsnn_127to255.txt')
    run_pcss(filename=dir + 'flit_config/config_saimsnn_127to255.txt', port=id, flitoutfile=dir + 'config_flitout.txt')

    for i in range(500):
        filename = f'flit_spk/single/{i+1}.txt'
        start_time = time.time_ns()
        run_pcss(filename=dir + filename, port=id, flitoutfile=f'pcss_o_255_{i+1}.txt', tofile=1)
        end_time = time.time_ns()
        print(filename + '\t finished in: ' + f'{(end_time - start_time) / 1000000}' + 'ms')

        print('flit_config/config_switch_input.txt')
        run_pcss(filename=dir + 'flit_config/config_switch_input.txt', port=id, flitoutfile=dir + 'sw_input_flitout_c.txt')
