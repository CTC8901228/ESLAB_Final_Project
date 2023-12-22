import socket
import json
import numpy as np
import matplotlib.pyplot as plot
import math
import openpyxl
from collections import deque
import time

HOST = '192.168.1.103' # IP address
PORT = 15000 # Port to listen on (use ports > 1023)
sp_1 = 0
asensitivity = 2048/2 #LSB/g 再除2的原因與靜止實驗結果有關
gsensitivity = 16.4 #LSB/(drg/s)
var = 0.0
newest_10ax = deque(maxlen=10)
newest_10ay = deque(maxlen=10)
newest_10az = deque(maxlen=10)

workbook = openpyxl.Workbook()
sheet = workbook.worksheets[0]
listtitle = ["time","gx", "gy", "gz", "ax", "ay", "az", "var_acc"]
sheet.append (listtitle)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print("Starting server at: ", (HOST, PORT))
    print("Starting Calibrate")
    conn, addr = s.accept()
    with conn:
        print("Connected at", addr)
        while True:
            data = conn.recv(1024*1024).decode('utf-8')
            #print("Received from socket server:", data)
            if (data.count('{') != 1):
                # Incomplete data are received.
                choose = 0
                buffer_data = data.split('}')
                while buffer_data[choose][0] != '{':
                    choose += 1
                data = buffer_data[choose] + '}'
            obj = json.loads(data)

            sp = obj['sp']
            gx = -(obj['gx'])/gsensitivity         #degree/s
            gy = -(obj['gy'])/gsensitivity
            gz = (obj['gz'])/gsensitivity
            ax = obj['ax']*9.81/asensitivity      #unit : m/s^2
            ay = obj['ay']*9.81/asensitivity
            az = -obj['az']*9.81/asensitivity
            sampleFreq = 1/(sp-sp_1)
            #print (gx, gy, gz, ax, ay, az)
            newest_10ax.append(ax)
            newest_10ay.append(ay)
            newest_10az.append(az)
            if len (newest_10ax) == 10:
                var = ((np.var(newest_10ax))**2+(np.var(newest_10ay))**2+(np.var(newest_10az))**2)**0.5 #加速度方差，判斷是否靜止
                if var > 0.005:
                    print ("stop moving,try again")
                    break
                    # stop_delay = 
                    # time.sleep(3)

            sp_1 = sp
            if sp > 51:
                break

            # gyr = np.array([gx,gy,gz])
            # acc = np.array([ax,ay,az])

            print (sp , gx, gy, gz, ax, ay, az)
            listtitle = [sp ,gx, gy, gz, ax, ay, az, var]
            sheet.append (listtitle)


workbook.save('data/static.xlsx')