import socket
import numpy as np
import math
import json
import time
import random
import matplotlib.pyplot as plt
from icp import icp
from scipy.signal import butter, lfilter, freqz,filtfilt
# def( np array,'a',資料夾)  save a.npy at folder
# def( 'a',資料夾)  return a.npy at folder
import pyautogui
import screeninfo  # 用于获取屏幕信息的库
import threading

import openpyxl
import open3d as o3d
import imufusion
HOST = '192.168.221.246' # IP address 
PORT = 42342 # Port to listen on (use ports > 1023)
acc_scale=1024/9.8
gyr_scale=16.4
mag_scale=1/6842
workbook = openpyxl.load_workbook('cal/data/calbration_para.xlsx')
sheet = workbook.worksheets[0]
# alpha_yx	alpha_zy	alpha_zx	scale_ax	scale_ay	scale_az	bias_ax	bias_ay	bias_az	r_yz	r_zy	r_xz	r_zx	r_xy	r_yx	s_gx	s_gy	s_gz	gx_avg_bias(use plus)	gy_avg_bias	gz_avg_bias
cali_para = []
for row in sheet.iter_rows(min_row=2, values_only=True):  # 从第二行开始，跳过标题行
    cali_para.append(row)
workbook.close
screen = screeninfo.get_monitors()[0]

# 计算屏幕中央的坐标
center_x = screen.width // 2
center_y = screen.height // 2
def cali_acc (acc):
    K_a = np.array([[cali_para[0][3],0,0],[0,cali_para[0][4],0],[0,0,cali_para[0][5]]])
    T_a = np.array([[1,-cali_para[0][0],cali_para[0][1]],[0,1,-cali_para[0][2]],[0,0,1]])
    b_a = np.array([[cali_para[0][6]],[cali_para[0][7]],[cali_para[0][8]]])
    h = (T_a.dot(K_a)).dot((acc + b_a))
    return h

# func of gyro calibrate
def cali_gyro (gyro):
    T_g = np.array([[1,-cali_para[0][9],cali_para[0][10]],[cali_para[0][11],1,-cali_para[0][12]],[-cali_para[0][13],cali_para[0][14],1]])
    K_g = np.array([[cali_para[0][15],0,0],[0,cali_para[0][16],0],[0,0,cali_para[0][17]]])
    b_g = np.array([[-cali_para[0][18]],[-cali_para[0][19]],[-cali_para[0][20]]])
    h = (T_g.dot(K_g)).dot((gyro +b_g ))
    return h
def click():
    pyautogui.click()

def mousemove(newx,newy):
    pyautogui.moveTo(newx, newy,_pause=False)
        
def main():
    click_thread = threading.Thread(target=click)

    offset_gyr=[0,0,0]
    thr_gyr=[0,0,0]
    offset_acc=[0,0,0]
    thr_acc=[0,0,0]
    init=True
    pre_init=True
    posid=POS_ID()
    x=np.array([[0,0,0]])
    move_scale = 2
    click_thr=-36000
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))

            
# initialize pointcloud instance.
            prev_movex=0
            prev_movey=0
            stable=0
            mode_iter=0
            prevmode=0
            current_x, current_y = pyautogui.position()
            mousemove_thread = threading.Thread(target=mousemove,args=(current_x, current_y))
            mousemove_thread.start()
            while(1):
                sending_pos=False
                s.listen()
                conn, addr = s.accept()
                with conn:
                    print('Connected by', addr)
                    acclist=[]
                    gyrlist=[]
                    time_list=[]
                    maglist=[]
                    
                    while True:
                        data = conn.recv(1024*64).decode('utf-8')
                        
                        datalist=data.split('!')[:-1]
                        for data in datalist:
                            try:
                                # arr=list(map(int,data.split()))
                                float_arr=list(map(float,data.split()))
                                mode = float_arr[0]
                                acc=[float_arr[1]*acc_scale,float_arr[2]*acc_scale,-float_arr[3]*acc_scale]
                                
                                gyr=[-float_arr[7]*gyr_scale,-float_arr[8]*gyr_scale,float_arr[9]*gyr_scale]
                                
                                t=float_arr[-1]/1000
                                mag=[float_arr[4]*mag_scale,float_arr[5]*mag_scale,float_arr[6]*mag_scale]
                                # print(float_arr)
                                print(mode)
                            except:
                                print('data split prob, data=',data)
                                continue
                            if len(time_list)<200 and pre_init:
                                acclist.append(acc)
                                maglist.append(mag)
                                time_list.append(t)
                                gyrlist.append(gyr)
                                continue
                            elif len(time_list)==200 and pre_init:
                                
                                print('pre_init_finish')
                                acclist=[]
                                gyrlist=[]
                                time_list=[]
                                maglist=[]
                                pre_init=False
                                continue
                            if len(time_list)<20 and init:
                                acclist.append(acc)
                                maglist.append(mag)
                                time_list.append(t)
                                gyrlist.append(gyr)
                                continue
                            elif len(time_list)==20 and init:
                                acc_arr=np.array(acclist)
                                gyr_arr=np.array(gyrlist)
                                offset_acc=np.mean(acc_arr,axis=0)
                                # offset_acc[2]=0
                                offset_gyr=np.mean(gyr_arr,axis=0)
                                print('init_finish')
                                acclist=[]
                                gyrlist=[]
                                time_list=[]
                                maglist=[]
                                init=False
                                continue
                                
                                
                            if mode==0:
                                time_list.append(t)
                                
                                if prevmode==1:
                                    prevmode=0
                                    mode_iter=0
                                else:
                                    mode_iter+=1
                                # pyautogui.moveTo(center_x, center_y, _pause=False)
                                x=np.array([[0,0,0],[0,0,0]])

                                # gyrlist=[[0,0,0],[0,0,0]]
                                continue  #idle
                            elif mode == 1:
                                
                                if prevmode!=1:
                                    prevmode=1
                                    mode_iter=0
                                elif prevmode==1:
                                    mode_iter+=1
                                
                                if mode_iter<20:
                                    x=np.array([[0,0,0],[0,0,0]])
                                    acclist=[]
                                    gyrlist=[]
                                    continue
                                # print(mode_iter)
                                sending_pos=True
                                gyr=[a - b for a, b in zip(gyr, offset_gyr)]
                                # gyr_arr=np.array([[gyr[0]],[gyr[1]],[gyr[2]]])
                                gyr_d=cali_gyro (gyr)/1000
                                acc=[a - b for a, b in zip(acc, offset_acc)]
                                # acc_arr=np.array([[acc[0]],[acc[1]],[acc[2]]])
                                acc_d=cali_acc (acc)
                                
                                
                                maglist.append(mag)
                                time_list.append(t)
                                fil_gyr=[gyr_d[0],gyr_d[1],gyr_d[2]]
                                # print([acc_d[0,0],acc_d[0,1],acc_d[0,2]])
                                # print(acc)
                                acclist.append([acc_d[0,0],acc_d[0,1],acc_d[0,2]]  )
                                gyrlist.append([gyr_d[0,0],gyr_d[0,1],gyr_d[0,2]])
                                # print(acc_d)
                                if acc_d[0,2]<click_thr:
                                    mode_iter=0
                                    pyautogui.click()
                                    pass
                                    
                                    
                                if len(acclist)>2 : 
                                    # tra=acc2tra_ahrs(acclist,gyrlist,maglist,time_list)
                                    # tra=acc2tra(gyrlist,time_list)
                                    ori_x=x.copy()
                                    old_x=x[-1,:]
                                    x=draw_and_append(x,np.array(gyrlist),np.array(time_list),False)
                                    now_x=x[-1,:]
                                    dif_x=abs(old_x-now_x)
                                    # print(x)
                                    if sum(dif_x)<0.01:
                                        stable+=1
                                        x=ori_x
                                    if stable>30:
                                        print('stable state x=zero')
                                        stable=0
                                        x=np.array([[0,0,0],[0,0,0]])
                                    
                                    if abs(x[-1,1])<20:
                                        movex =0
                                    else:
                                        movex= (abs(x[-1,1])-10)/5
                                    
                                    if abs(x[-1,0])<20:
                                        movey =0
                                    else:
                                        movey= (abs(x[-1,0])-10)/5
                                    
                                    movex=movex*np.sign(x[-1,1])*move_scale*0.9+0.1*prev_movex
                                    movey=movey*np.sign(x[-1,0])*move_scale*0.9+0.1*prev_movey
                                    
                                    # 计算新的鼠标位置
                                    
                                    new_x =int(current_x + movex) 

                                    new_y = int(current_y + movey)
                                    prev_movex=movex
                                    prev_movey=movey
                                    if new_x<=0 :
                                        new_x=current_x
                                    elif new_x>= screen.width :
                                        new_x=current_x-1
                                    
                                    if new_y<=0 :
                                        new_y=current_y
                                    elif new_y>= screen.height :
                                        new_y=current_y-1

                                    # 移动鼠标到新的位置
                                    if len(acclist)%1==0 and (new_y != current_y  or new_x!=current_x) :
                                        mousemove_thread.join()
                                        mousemove_thread = threading.Thread(target=mousemove,args=(new_x, new_y))
                                        mousemove_thread.start()
                                        # pyautogui.moveTo(new_x, new_y,_pause=False)
                                        current_x, current_y = pyautogui.position()

                                    # pcd=show_ptcloud(tra,pcd,vis)
                                    # print(time_list)
                                #手勢辨識,輸入加速度 放加速度進list 在idle時判斷
                                pass
                            elif mode ==2 :
                            #字母輸入
                                if prevmode!=2:
                                    prevmode=2
                                    mode_iter=0
                                elif prevmode==2:
                                    mode_iter+=1
                                if mode_iter<20:
                                    v=np.array([[0,0,0],[0,0,0]])
                                    acclist=[]
                                    gyrlist=[]
                                    continue
                                if mode_iter==30:
                                    accoffset_mode=np.mean(np.array(acclist),axis=0)
                                    gyroffset_mode=np.mean(np.array(gyrlist),axis=0)
                                    
                                    print('u can draw!')
                                if mode_iter<30:
                                    gyr=[a - b for a, b in zip(gyr, offset_gyr)]
                                    # gyr_arr=np.array([[gyr[0]],[gyr[1]],[gyr[2]]])
                                    gyr_d=cali_gyro (gyr)/1000
                                    acc=[a - b for a, b in zip(acc, offset_acc)]
                                    # acc_arr=np.array([[acc[0]],[acc[1]],[acc[2]]])
                                    acc_d=cali_acc (acc)
                                    
                                    
                                    maglist.append(mag)
                                    time_list.append(t)
                                    fil_gyr=[gyr_d[0],gyr_d[1],gyr_d[2]]
                                    # print([acc_d[0,0],acc_d[0,1],acc_d[0,2]])
                                    # print(acc)
                                    acclist.append([acc_d[0,0],acc_d[0,1],acc_d[0,2]]  )
                                    gyrlist.append([gyr_d[0,0],gyr_d[0,1],gyr_d[0,2]])
                                else:
                                                                       
                                    acc=[a - b for a, b in zip(acc, offset_acc)]
                                    acc_d=cali_acc (acc)
                                    acc_d=[a - b for a, b in zip(acc_d, accoffset_mode)]
                                    
                                    time_list.append(t)
                                    print(acc_d)
                                    acclist.append([acc_d[0][0],acc_d[0][1],acc_d[0][2]]  )
                                    v=draw_and_append(v,np.array(acclist),np.array(time_list),False)
                                    
                                    
                            elif mode ==3:
                                print(v)
                                # print('undefine mode:',mode)
                                pass ##velocity iden
                            else:
                                print('undefined mode')
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
def acc2tra_ahrs(acc,gyr,mag,time):
    
    # return 0
    order = 6
    fs = 30.0
    cutoff = 3.667
    sample_rate=100
    timestamp =np.array(time) # data[:, 0]
    gyroscope =np.array(gyr) #data[:, 1:4]
    accelerometer =np.array(acc) # data[:, 4:7]
    magnetometer =np.array(mag) # data[:, 7:10]
    gyroscope[:,0]= butter_lowpass_filter(gyroscope[:,0], cutoff, fs, order)
    gyroscope[:,1]= butter_lowpass_filter(gyroscope[:,1], cutoff, fs, order)
    gyroscope[:,2]= butter_lowpass_filter(gyroscope[:,2], cutoff, fs, order)
    # gyroscope[:,0]=butter_highpass_filter(gyroscope[:,0], cutoff, fs, order)
    # gyroscope[:,1]=butter_highpass_filter(gyroscope[:,1], cutoff, fs, order)
    # gyroscope[:,2]=butter_highpass_filter(gyroscope[:,2], cutoff, fs, order)
    # # Instantiate algorithms
    offset = imufusion.Offset(sample_rate)
    ahrs = imufusion.Ahrs()

    ahrs.settings = imufusion.Settings(imufusion.CONVENTION_NWU,  # convention
                                    0.5,  # gain
                                    2000,  # gyroscope range
                                    30,  # acceleration rejection
                                    30,  # magnetic rejection
                                    5 * sample_rate)  # recovery trigger period = 5 seconds

    # Process sensor data
    delta_time = np.diff(timestamp, prepend=timestamp[0])

    euler = np.empty((len(timestamp), 3))
    internal_states = np.empty((len(timestamp), 6))
    flags = np.empty((len(timestamp), 4))

    for index in range(len(timestamp)):
        gyroscope[index] = offset.update(gyroscope[index])

        ahrs.update(gyroscope[index], accelerometer[index], magnetometer[index], delta_time[index])

        euler[index] = ahrs.quaternion.to_euler()
        # accelerometer[index]=accelerometer[index]-np.array([
        #     -math.sin(euler[index][1])*9.81,
        #     math.cos(euler[index][1])*math.sin(euler[index][0])*9.81,
        #     math.cos(euler[index][1])*math.cos(euler[index][0])*9.81,
        # ])
        accelerometer[index]=accelerometer[index]-np.array([
            0,
            0,
            9.81,
        ])
    plt.ion()
                    # plt.subplots()
    plt.clf()

    plt.plot(euler[-2000:,0])
    plt.plot(euler[-2000:,1])
    plt.plot(euler[-2000:,2])
    plt.pause(0.0001)
    plt.show()
        # print(accelerometer)
        
        # accelerometer[index][accelerometer[index]<2]=0
    
    order = 6
    fs = 30.0
    cutoff = 3.667
    accelerometer[:,0]= butter_lowpass_filter(accelerometer[:,0], cutoff, fs, order)
    accelerometer[:,1]= butter_lowpass_filter(accelerometer[:,1], cutoff, fs, order)
    accelerometer[:,2]= butter_lowpass_filter(accelerometer[:,2], cutoff, fs, order)


    seqlen=accelerometer.shape[0]
    delta_time=np.stack((np.array(delta_time),np.array(delta_time),np.array(delta_time)),axis=-1)
    
    add=accelerometer*delta_time
    
    vlist=[add[0,::]]
    for i in range(1,seqlen):
        
        vlist.append(vlist[-1]+add[i,::])
    
    varr = np.array(vlist)
    # print(accelerometer)
    
    # print(euler)
    add=varr*delta_time
    xlist=[add[0,::]]
    for i in range(1,seqlen-1):
        
        xlist.append(xlist[-1]+add[i,::])
    
    
    
        ##拿由拉角推算 重力方向

    # plt.clf()

    # plt.plot(accelerometer[:,0])
    # plt.plot(accelerometer[:,1])
    # plt.plot(accelerometer[:,2])
    # plt.pause(0.0001)
    # plt.show()
    return np.array(xlist)
        
######################################################
def show_ptcloud(tra,pcd,vis):
    # vis.remove_geometry(pcd)

   
    pcd.points.extend(tra[-1,:].reshape(-1,3))
    vis.update_geometry(pcd)

    # vis.add_geometry(point_cloud)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(0.005)
    # o3d.visualization.draw_geometries([point_cloud])
    return pcd
#####################################################
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
    varr=np.array(vlist)
    plt.ion()
                    # plt.subplots()
    plt.clf()

    plt.plot(varr[-2000:,0])
    plt.plot(varr[-2000:,1])
    plt.plot(varr[-2000:,2])
    plt.pause(0.0001)
    plt.show()
    return varr
    add=varr*time_diff
    xlist=[add[0,::]]
    for i in range(1,seqlen-1):
        
        xlist.append(xlist[-1]+add[i,::])
    
    return np.array(xlist)
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return b, a
def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
def draw_and_append(x,x_dot,time,draw=True):
    # print(x[-1,:],x_dot[-1,:],(time[-1]-time[-2]))
    x=np.append(x,(x_dot[-1,:]*(time[-1]-time[-2])+x[-1,:]).reshape((-1,3)),axis=0)
    if draw:
        if (time.shape[0]%100==1):
            if (time.shape[0]>2001):
                plt.ion()
                                # plt.subplots()

                plt.clf()

                plt.plot(x_dot[-2000:,0])
                plt.plot(x_dot[-2000:,1])
                plt.plot(x_dot[-2000:,2])
                plt.pause(0.000001)
                plt.show()
            else:    
                plt.ion()
                                # plt.subplots()

                plt.clf()

                plt.plot(x_dot[:,0])
                plt.plot(x_dot[:,1])
                plt.plot(x_dot[:,2])
                plt.pause(0.000001)
                plt.show()
    return x
def get_nxt_x(x,x_dot,time):
    
    # print(x[-1,:],x_dot[-1,:],(time[-1]-time[-2]))
    x=np.append(x,(x_dot[-1,:]*(time[-1]-time[-2])+x[-1,:]).reshape((-1,3)),axis=0)
    if x.shape[0]>2001:
        x=x[-2000:,:]   
        time=time[-2000:]   
        x_dot=x_dot[-2000:,:]
    # if (time.shape[0]%100==1):
    #     if (time.shape[0]>2001):
    #         plt.ion()
    #                         # plt.subplots()

    #         plt.clf()

    #         plt.plot(x[-2000:,0])
    #         plt.plot(x[-2000:,1])
    #         plt.plot(x[-2000:,2])
    #         plt.pause(0.000001)
    #         plt.show()
    #     else:    
    #         plt.ion()
    #                         # plt.subplots()

    #         plt.clf()

    #         plt.plot(x[:,0])
    #         plt.plot(x[:,1])
    #         plt.plot(x[:,2])
    #         plt.pause(0.000001)
    #         plt.show()
    return x


if __name__== '__main__':
    main()
                        