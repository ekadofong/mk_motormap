#!/usr/bin/env python

from __future__ import print_function

import numpy as np
import george
from . import model, movement

def gen_gproc ( llist ):
    alist = []
    for i in range(len(llist)):
        mm = model.MotorModel ()
        cwait = llist[i]
        bins = np.arange(0, np.nanmax(cwait['startangle']), 10.)
        assns = np.digitize ( cwait['startangle'], bins ) 
        stds = cwait.stepsize.groupby(assns).std()
        stds = stds.replace ( np.NaN, 100.)
        cwait['u_stepsize'] = stds.loc[assns].values    
        mm.read_object(cwait)
        k2 = .5*george.kernels.ExpSquaredKernel(15**2) * george.kernels.ExpSine2Kernel(1.0, 10.0) + \
            0.2 * george.kernels.ExpSquaredKernel(.1**2)
        okernel = mm.set_hyperparam ( k2 )
        mm.model_shape (okernel)
        alist.append(mm)

    return alist



def moveto_stage1testing ( specdir, target_angle=55. ):
    '''
    From "known" stage 2 motor maps, move to 55 deg
    to put motors in a non-colliding position for
    stage 1 testing
    '''
    with open('./msim_scripts/moveto_stage1testing.lst', 'w') as f:
        print('''
{ Init camera and LED }
cmd_createTestDirectoryWithSuffix gotoangle

cmd_zeroPositionerIds
cmd_setPositionerIds_all
cmd_set_Fpga_Motor_Frequency_From_Database
cmd_sendHKCommand_ToFpga 1
cmd_sendHKCommand_ToFpga 2
cmd_showModuleId_Period_Frequency 1

{ ==Camera Setup== }
cmd_remoteEnableCameraRecovery 1,100
cmd_remote_setCameraExposureTime  0.002, 100
cmd_loadDark_Image  2msDarkImage_64.fits
cmd_enableDarkSub  1
cmd_selectCentroiding_Algorithm  6
cmd_enableWindowed_Centroid  0
cmd_setThreshold_NSigma  5

cmd_clearFiducialTable
cmd_setHornMethodFiducialCoordinate 160.639989, 1107.793311, 20
cmd_setHornMethodFiducialCoordinate 540.861509, 1488.688838, 20
cmd_setHornMethodFiducialCoordinate 920.771033, 1869.326100, 20
cmd_enableHornMethod

{ Move Theta and Phi to home positions}
cmd_moveMotor_Steps_all 2, -8000, 1

cmd_zeroPositionerIds
cmd_setPositionerIds_all

cmd_setCentroidsArmLogPrefix   PhiSpecMove
cmd_getImageStart_NoWait
cmd_getImageDone_Wait
cmd_findCentroids
cmd_LogCentroids
cmd_clearOpenCV_Images
cmd_setCurrentPos_all
cmd_getCentroidsArmInverseKin  

{ by integrating inverse of Mmap goto angle }

''', file=f)

        for pid in range(1,58):
            nmdf = movement.read_ctrlstep ( '%s/PhiSpecMove_mId_1_pId_%i.txt' % (specdir,pid),
                                            [50,]*3,
                                            movesize=None).convert_objects ()
            #wlist = [ nmdf.query('iter==%i'%i).dropna() for i in range(nmdf['iter'].max())]
            wlist = [ nmdf.dropna() ]
            alist = gen_gproc ( wlist )
            step_predict = np.mean([gp.movetoangle() for gp in alist])
            step_predict = int(round(step_predict))
            print ( 'cmd_moveMotor_Steps 1, %i, 2, %i, 0' % (pid, step_predict),
                    file=f )
            print ('pid %i predicts %i steps' % (pid, step_predict))

        print('''
cmd_getImageStart_NoWait
cmd_getImageDone_Wait
cmd_findCentroids
cmd_LogCentroids
cmd_clearOpenCV_Images
cmd_setCurrentPos_all
cmd_getCentroidsArmInverseKin 

cmd_moveMotor_Steps_all 1,8000,0

cmd_getImageStart_NoWait
cmd_getImageDone_Wait
cmd_findCentroids
cmd_LogCentroids
cmd_clearOpenCV_Images
cmd_setCurrentPos_all
cmd_getCentroidsArmInverseKin 

cmd_moveMotor_Steps_all 1,-8000,0

cmd_getImageStart_NoWait
cmd_getImageDone_Wait
cmd_findCentroids
cmd_LogCentroids
cmd_clearOpenCV_Images
cmd_setCurrentPos_all
cmd_getCentroidsArmInverseKin 
''', file=f)
