import socket
import numpy as np
import json
import time
import random
import matplotlib.pyplot as plt
from icp import icp
HOST = '192.168.162.246' # IP address 
PORT = 42342 # Port to listen on (use ports > 1023)

def main():
    posid=POS_ID()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))
            while(1):
                sending_pos=False
                s.listen()
                conn, addr = s.accept()
                with conn:
                    print('Connected by', addr)
                    acclist=[]

                    while True:
                        data = conn.recv(1024).decode('utf-8')
                        print(data)

                        try:
                            arr=list(map(int,data.split()))
                        except:
                            print('data split prob, data=',data)
                            continue
                        acc=data['acc']
                        mode=data['mode']
                        if mode==0:
                            if(sending_pos==True):
                                  tra=acc2tra(acclist)
                                  pos_ind,err=POS_ID.Pos_identification(tra)
                            acclist=[]
                            continue  #idle
                        elif mode == 1:
                             acclist.append(acc)
                             #手勢辨識,輸入加速度 放加速度進list 在idle時判斷
                             pass
                        elif mode ==2 :
                        #滑鼠模式 輸入加速度
                            pass
                        else:
                             print('undefine mode:',mode)
class POS_ID:
    def __init__(self):
          self.pos_list=[]
          #記得加想要的軌跡進去
          self.pos_name=[]
          #軌跡名稱
          pass
    def Pos_identification(self,tra):
        minErr=999
        besti=-1
        for i,pos in enumerate(self.pos_list):
            Err=icp(tra,pos)
            if Err<minErr:
                minErr=Err    
                besti=i
        return besti,minErr
    def getPosName(self,i):
         return self.pos_name[i]


                        
                        