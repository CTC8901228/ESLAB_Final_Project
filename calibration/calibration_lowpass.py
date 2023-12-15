import socket
import json
import numpy as np
import matplotlib.pyplot as plot
import math
import openpyxl
from collections import deque
import time
import scipy.signal

HOST = '192.168.1.111' # IP address
PORT = 15000 # Port to listen on (use ports > 1023)
sp_1 = 0
asensitivity = 2048/2 #LSB/g 再除2的原因與靜止實驗結果有關
gsensitivity = 16.4 #LSB/(drg/s)
var = 0.0
newest_10ax = deque(maxlen=10)
newest_10ay = deque(maxlen=10)
newest_10az = deque(maxlen=10)


################### lowpass filter
class LiveFilter:
    """Base class for live filters.
    """
    def process(self, x):
        # do not process NaNs
        if np.isnan(x):
            return x

        return self._process(x)

    def __call__(self, x):
        return self.process(x)

    def _process(self, x):
        raise NotImplementedError("Derived class must implement _process")
    
class LiveLFilter(LiveFilter):
    def __init__(self, b, a):
        """Initialize live filter based on difference equation.

        Args:
            b (array-like): numerator coefficients obtained from scipy.
            a (array-like): denominator coefficients obtained from scipy.
        """
        self.b = b
        self.a = a
        self._xs = deque([0] * len(b), maxlen=len(b))
        self._ys = deque([0] * (len(a) - 1), maxlen=len(a)-1)

    def _process(self, x):
        """Filter incoming data with standard difference equations.
        """
        self._xs.appendleft(x)
        y = np.dot(self.b, self._xs) - np.dot(self.a[1:], self._ys)
        y = y / self.a[0]
        self._ys.appendleft(y)

        return y
fs = 30  # sampling rate, Hz
ts = np.arange(0, 5, 1.0 / fs)  # time vector - 5 seconds
# define lowpass filter with 2.5 Hz cutoff frequency
b, a = scipy.signal.iirfilter(4, Wn=2.5, fs=fs, btype="low", ftype="butter")
gx_lfilter = LiveLFilter(b, a)
gy_lfilter = LiveLFilter(b, a)
gz_lfilter = LiveLFilter(b, a)
ax_lfilter = LiveLFilter(b, a)
ay_lfilter = LiveLFilter(b, a)
az_lfilter = LiveLFilter(b, a)
## 前30筆資料不要收
count = 0
# simulate live filter - passing values one by one
# y_live_lfilter = [live_lfilter(y) for y in yraw]

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

            gx_f = [gx_lfilter(gx)]
            gy_f = [gy_lfilter(gy)]
            gz_f = [gz_lfilter(gz)]
            ax_f = [ax_lfilter(ax)]
            ay_f = [ay_lfilter(ay)]
            az_f = [az_lfilter(az)]
            
            newest_10ax.append(ax_f)
            newest_10ay.append(ay_f)
            newest_10az.append(az_f)

            if len (newest_10ax) == 10:
                var = ((np.var(newest_10ax))**2+(np.var(newest_10ay))**2+(np.var(newest_10az))**2)**0.5 #加速度方差，判斷是否靜止
                # if var > 0.005:
                #     print ("stop moving,try again")
                #     break
                    # stop_delay = 
                    # time.sleep(3)

            sp_1 = sp
            if sp > 51:
                break
            
            # gyr = np.array([gx,gy,gz])
            # acc = np.array([ax,ay,az])

            # print (sp , gx, gy, gz, ax, ay, az)
            #listtitle = [sp ,gx, gy, gz, ax, ay, az, var]
            # listtitle = [sp ,gx, gy, gz, ax, ay, az, var,gx_f[0],gy_f[0],gz_f[0],ax_f[0],ay_f[0],az_f[0]]
            #listtitle = [sp ,gx, gy, gz, ax, ay, az, var,gx_f[0],gy_f[0]]
            count = count + 1
            print (sp )
            if count >= 30:
                listtitle = [sp ,gx_f[0],gy_f[0],gz_f[0],ax_f[0],ay_f[0],az_f[0],var]
                sheet.append (listtitle)


workbook.save('data/static.xlsx')