import sys
import numpy as np
from scipy import optimize
import xml.etree.cElementTree as ET
import math


from astropy.io import ascii
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

matplotlib.rcParams.update({'font.size': 22})

DEBUG = False


def skip_bad_lines(self, str_vals, ncols):
    """Simply ignore every line with the wrong number of columns."""
    if DEBUG:
        print('Skipping line:', ' '.join(str_vals))
    return None

def readMotorMap(xml, pid):
    tree = ET.ElementTree(file=xml)

    root=tree.getroot()
    
    j1_fwd_reg=[]
    j1_fwd_stepsize=[]
    for i in root[pid-1][2][0].text.split(',')[2:]:
        if i is not '':
            j1_fwd_reg=np.append(j1_fwd_reg,float(i))

    for i in root[pid-1][2][1].text.split(',')[2:]:
        if i is not '':
            j1_fwd_stepsize=np.append(j1_fwd_stepsize,float(i))

    j1_rev_reg=[]
    j1_rev_stepsize=[]
    for i in root[pid-1][2][2].text.split(',')[2:]:
        if i is not '':
            j1_rev_reg=np.append(j1_rev_reg,float(i))

    for i in root[pid-1][2][3].text.split(',')[2:]:
        if i is not '':
            j1_rev_stepsize=np.append(j1_rev_stepsize,-float(i))


    j2_fwd_reg=[]
    j2_fwd_stepsize=[]
    for i in root[pid-1][2][4].text.split(',')[2:]:
        if i is not '':
            j2_fwd_reg=np.append(j2_fwd_reg,float(i))

    for i in root[pid-1][2][5].text.split(',')[2:]:
        if i is not '':
            j2_fwd_stepsize=np.append(j2_fwd_stepsize,float(i))

    j2_rev_reg=[]
    j2_rev_stepsize=[]
    for i in root[pid-1][2][6].text.split(',')[2:]:
        if i is not '':
            j2_rev_reg=np.append(j2_rev_reg,float(i))

    for i in root[pid-1][2][7].text.split(',')[2:]:
        if i is not '':
            j2_rev_stepsize=np.append(j2_rev_stepsize,-float(i))    
    return j1_fwd_reg,j1_fwd_stepsize,j1_rev_reg,j1_rev_stepsize,\
           j2_fwd_reg,j2_fwd_stepsize,j2_rev_reg,j2_rev_stepsize


def makeSNRplot(fig, data):
    target = []
    for i in data['col1'].data:
        if i not in target:
            target.append(i)

    snr_avg=[]
    snr_num=[]
    n=0
    pixelscale=85
    k_offset=1/(.075)**2
    tmax=105
    tobs=900

    for i in target:
        ind=np.where(data['col1'].data == i)
        it=data['col2'].data[ind]
        #print(np.max(it))
        #tstep=np.array(range(0,np.max(it)+1))*8+12
        tstep=np.array(range(0,10))*8+12
        dist=pixelscale*np.sqrt((data['col11'].data[ind]-data['col5'].data[ind])**2+
                (data['col12'].data[ind]-data['col6'].data[ind])**2)/1000
        
        if dist.size != tstep.size:
            break
        
        snr=(1-k_offset*dist**2)*(np.sqrt((tmax+tobs-tstep)/(tobs)))
        snr[snr<0]=0
        fig.scatter(tstep,snr)
        
        #snr[snr<0]=0
        if n == 0:
            snr_avg=snr
            snr_num=np.zeros(snr.shape[0])
            snr_num[0:snr.shape[0]]=1
            n=n+1
        else:
            num=np.zeros(snr_avg.shape[0])
            num[0:snr.shape[0]]=1
            while (snr_avg.shape[0] != snr.shape[0]):
                #print(snr_avg.shape[0],snr.shape[0])
                snr=np.append(snr,0)
            snr_avg=snr_avg+snr
            
            snr_num=snr_num+num
            n=n+1
    snr_avg=snr_avg/snr_num        
    fig.plot(tstep,snr_avg,color='green',linewidth=6)
    fig.scatter(tstep,snr_avg,color='green',s=100)
    fig.set_title(dataPath1[31:]+' SNR = %.3f'%(np.max(snr_avg)),fontsize=20)
    
def makeArmOneMovement(fig, data, xml, pid, color):
    
    j1_fwd_reg,j1_fwd_stepsize,j1_rev_reg,j1_rev_stepsize,\
            j2_fwd_reg,j2_fwd_stepsize,j2_rev_reg,j2_rev_stepsize=readMotorMap(xml,pid)
    
    target = []
    for i in data['col1'].data:
        if i not in target:
            target.append(i)
    
    for i in target:
        #for i in range(0,1):
            ind=np.where(data['col1'].data == i)
            theta=data['col34'].data[ind]
            step=data['col18'].data[ind]
            thetadeg=[y - x for x,y in zip(theta,theta[1:])]
            for j,t in  enumerate(thetadeg):
                if abs(step[j]) == 0:
                    theta_per_step=0
                else:
                    theta_per_step=t/abs(step[j])
                if theta_per_step > 0:
                    plt.plot([t+theta[j],theta[j]],
                            [theta_per_step,theta_per_step],color='pink',linewidth=2)
                else:
                    plt.plot([t+theta[j],theta[j]],
                            [theta_per_step,theta_per_step],color='blue',linewidth=2)

    fig.plot(j1_fwd_reg,j1_fwd_stepsize,c=color,linewidth=3)
    fig.plot(j1_rev_reg,j1_rev_stepsize,c=color,linewidth=3)
    #fig.xlim((0,360))
    #fig.ylim((-0.6,0.6))
    fig.title('Arm One '+' PID='+str(pid),fontsize=20)

def makeArmTwoMovement(fig, data, xml, pid, color):

    j1_fwd_reg,j1_fwd_stepsize,j1_rev_reg,j1_rev_stepsize,\
            j2_fwd_reg,j2_fwd_stepsize,j2_rev_reg,j2_rev_stepsize=readMotorMap(xml,pid)    

    target = []
    for i in data['col1'].data:
        if i not in target:
            target.append(i)

    for i in target:
        #for i in range(0,2):
        ind=np.where(data['col1'].data == i)
        theta=data['col14'].data[ind]
        step=data['col19'].data[ind]
        thetadeg=[y - x for x,y in zip(theta,theta[1:])]
        for j,t in  enumerate(thetadeg):
            if abs(step[j]) == 0:
                theta_per_step=0
            else:
                theta_per_step=t/abs(step[j])
            if theta_per_step > 0:
                plt.plot([theta[j],theta[j]+t],
                        [theta_per_step,theta_per_step],color='pink',linewidth=2)
            else:
                plt.plot([theta[j],theta[j]+t],
                        [theta_per_step,theta_per_step],color='blue',linewidth=2)


    fig.plot(j2_fwd_reg,j2_fwd_stepsize,c=color,linewidth=4)
    fig.plot(j2_rev_reg,j2_rev_stepsize,c=color,linewidth=4)


if __name__ == '__main__':
    for pid in range(2, 58):
        #if pid not in [1, 16, 24, 26, 30, 40, 41, 42, 43, 45, 48, 51, 56]:
        mid=1
        #pid=5


        dataPath1='/Volumes/Disk/CaltechCobraData/18_01_07_22_45_50_TargetRun/'
        dataPath2='/Volumes/Disk/CaltechCobraData/18_01_11_09_54_58_TargetRun/'

        xml1=dataPath1+'updatedMaps7.xml'
        xml2=dataPath2+'usedXMLFile.xml'


        data_file='mId_'+str(mid)+'_pId_'+str(pid)+'.txt'

        ascii.BaseReader.inconsistent_handler = skip_bad_lines
        data1 = ascii.read(dataPath1+data_file, guess=False, delimiter=',', format='no_header')
        data2 = ascii.read(dataPath2+data_file, guess=False, delimiter=',', format='no_header')


        plt.close('all')

        fig=plt.figure(figsize=(30, 20))
        gs = gridspec.GridSpec(2, 3)
        ax1 = plt.subplot(gs[0, 0])

        makeSNRplot(ax1, data1)
        plt.ylim((0.8,1.05))


        ax2 = plt.subplot(gs[0, 1])
        color='lime'
        makeArmOneMovement(ax2, data1, xml1, pid, color)

        plt.xlim((0,360))
        plt.ylim((-0.6,0.6))
        plt.title(dataPath1[31:]+'Arm One '+' PID='+str(pid),fontsize=20)


        ax3 = plt.subplot(gs[0, 2])
        makeArmTwoMovement(ax3, data1, xml1, pid, color)

        plt.xlim((0,200))
        plt.ylim((-0.6,0.6))
        plt.title(dataPath1[31:]+'Arm Two '+' PID='+str(pid),fontsize=20)

    #
    # ----------------------------------------------------------------------------------
    #

        ax4 = plt.subplot(gs[1, 0])
        makeSNRplot(ax4, data2)
        plt.ylim((0.8,1.05))

        ax5 = plt.subplot(gs[1, 1])
        color='red'
        makeArmOneMovement(ax5, data2, xml2, pid, color)

        plt.xlim((0,360))
        plt.ylim((-0.6,0.6))
        plt.title(dataPath2[31:]+'Arm One '+' PID='+str(pid),fontsize=20)


        ax6 = plt.subplot(gs[1, 2])
        makeArmTwoMovement(ax6, data2, xml2, pid, color)

        plt.xlim((0,200))
        plt.ylim((-0.6,0.6))
        plt.title(dataPath2[31:]+' Arm Two '+' PID='+str(pid),fontsize=20)

        fig.savefig('pid_'+str(pid)+'.png')
        plt.close()


