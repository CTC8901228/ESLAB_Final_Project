import socket
import numpy as np
import json
import time
import random
import matplotlib.pyplot as plt
from icp import icp
import open3d as o3d
HOST = '192.168.221.246' # IP address 
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
                    gyrlist=[]
                    time_list=[]
                    while True:
                        data = conn.recv(1024*8).decode('utf-8')
                        print(data)

                        try:
                            arr=list(map(int,data.split()[0:7]))
                            float_arr=list(map(float,data.split()))
                            
                     
                        except:
                            print('data split prob, data=',data)
                            continue
                        mode = arr[0]
                        acc=[arr[1],arr[2],arr[3]-1024]
                        gyr=[float_arr[7],float_arr[8],float_arr[9]]
                        t=arr[-1]
                        
                        if mode==0:
                            if(sending_pos==True):
                                    tra=acc2tra(acclist,time_list)
                                    print(tra)
                                    point_cloud = o3d.geometry.PointCloud()
                                    point_cloud.points = o3d.utility.Vector3dVector(tra)
                                    o3d.visualization.draw_geometries([point_cloud])
                                    
                                    pos_ind,err=POS_ID.Pos_identification(tra)
                                    sending_pos=False
                            acclist=[]
                            continue  #idle
                        elif mode == 1:
                            sending_pos=True
                            acclist.append(acc)
                            time_list.append(t)
                            gyrlist.append(gyr)
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

def acc2tra(acc,time):
    acc=np.array(acc)
    time_diff=[]
    for i,t in enumerate(time):
        if i==0:
            continue
        else:
            time_diff.append((time[i]-time[i-1])/1000)
    time_diff.append(0)
    time_diff=np.stack((np.array(time_diff),np.array(time_diff),np.array(time_diff)),axis=-1)
    seqlen=acc.shape[0]
    add=acc*time_diff
    vlist=[add[0,::]]
    for i in range(1,seqlen):
        
        vlist.append(vlist[-1]+add[i,::])
    
    varr = np.array(vlist)

    add=varr*time_diff
    xlist=[add[0,::]]
    for i in range(1,seqlen-1):
        
        xlist.append(xlist[-1]+add[i,::])
    
    return np.array(xlist)
if __name__== '__main__':
    main()
                        