#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read ( fname ):
    df = pd.read_csv ( fname, names = 'x y theta phi time'.split () )
    #df['x'] *= 0.06
    #df['y'] *= 0.06
    home = df.loc[0,'x'], df.loc[0,'y']
    df['dist'] = np.sqrt((df['x'] - home[0])**2 + (df['y']-home[1])**2.)*1000.
    return df

def read_convergencedata ( fname ):
    rawconv = pd.read_csv ( fname, header=None )
    conv = rawconv[[33,13,17,18,1,16,34,7]]
    conv.columns = 'J1 J2 J1_s J2_s iter dist J1_t J2_t'.split ()

    dphi = conv['J2'][1:].values - conv['J2'][:-1]
    dtheta = conv['J1'][1:].values - conv['J1'][:-1]
    conv['J1_stepsize'] = dtheta / conv['J1_s']
    conv['J2_stepsize'] = dphi / conv['J2_s']
    conv['time'] = conv.index
    return conv

def read_ctrlstep ( fname, stepseq = [100.,50.], verbose=False, motor_id=2, movesize=400 ):
    if motor_id == 2:
        aname = 'phi'
    else:
        aname = 'theta'    


    df = read(fname)
    mm = df.apply ( lambda row: 'NAN' in str(row[aname]), axis=1 )
    df = df.loc[~mm].astype(float)
    #print(df['phi'].astype(float))
    mdf = pd.DataFrame(index=df.index[1:], columns=['xpix', 'ypix', 'startangle','d'+aname,'movesize','stepsize'])

    mdf['startangle'] = df[aname][:-1]
    mdf['d' + aname ] = df[aname][1:].values - df[aname][:-1]
    mdf['xpix'] = df['x']
    mdf['ypix'] = df['y']
    mdf['move_phys'] = np.sqrt((df['x'][1:].values - df['x'][:-1])**2 + (df['y'][1:].values - df['y'][:-1])**2)*90.
    mdf['iter'] = 0
    mdf['stepsize'] = 0.


    stake = 0

    if (movesize is None):
        if verbose:
            print('Attempting to infer move size...')
        # Need to account for when motor 1 goes from 360 -> 0
        namask = np.isfinite(mdf['d'+aname])
        if motor_id == 2:
            if stepseq[0] > 0:
                homes = mdf.loc[namask].sort_values('d' + aname).iloc[:len(stepseq)].index.sort_values()
            else:
                homes = mdf.loc[namask].sort_values('d' + aname).iloc[-len(stepseq):].index.sort_values()
        else:
            if stepseq[0] > 0:
                homes = mdf.loc[namask].sort_values('d' + aname).iloc[len(stepseq):2*len(stepseq)].index.sort_values()
            else:
                homes = mdf.loc[namask].sort_values('d' + aname).iloc[-len(stepseq)*2:-len(stepseq)].index.sort_values()
    elif movesize == 0:
        mdf['iter'] = 100
        mdf['movesize'] = stepseq
        mdf['stepsize'] = mdf['d'+aname]/mdf['movesize']
        return mdf
    else:
        lng = np.arange(1,1+len(stepseq))
        homes = np.ones_like(stepseq)*movesize*lng + lng

    for idx,cstep in enumerate(stepseq):
        #nstake = mdf.iloc[stake:].query('dphi<-10.').iloc[0].name
        nstake = homes[idx]
        mdf.loc[mdf.index[stake:nstake],'movesize'] = cstep
        mdf.loc[mdf.index[stake:nstake],'iter'] = idx
        mdf.loc[mdf.index[nstake-1],'stepsize'] = np.NaN
        #print(mdf.loc[mdf.index[nstake],'stepsize'])
        if verbose:
            print('Break @ %i' % nstake)
        stake = nstake

    mdf['stepsize'] = mdf['d'+aname]/mdf['movesize']
    mdf.loc[mdf.index[homes-1], 'stepsize'] = np.NaN
    
    return mdf

def estimate_std ( mdf ):
    angle_grid = np.arange(0,180.,6.)
    assns = np.digitize ( mdf['startangle'], angle_grid )
    stds = mdf.stepsize.groupby(assns).std()
    stds = stds.replace ( np.NaN, 100.)
    mdf['u_stepsize'] = stds.loc[assns].values
    return mdf

def read_mdf ( fname ):
    df = read(fname)

    mdf = pd.DataFrame(index=df.index[1:], columns=['xpix','ypix','startangle','dphi','movesize','stepsize'])
    mdf['startangle'] = df['phi'][:-1]
    mdf['movesize'] = 0.
    #mdf['movesize'] = 15.*(1.-mdf.index%2) -5.*(mdf.index%2)

    mdf['xpix'] = df['x']
    mdf['ypix'] = df['y']

    mdf['dphi'] = df['phi'][1:].values - df['phi'][:-1]
    mdf.loc[mdf['dphi']>0, 'movesize'] = 15.
    mdf.loc[mdf['dphi']<0, 'movesize'] = -5.

    #mdf.loc[mdf.index[::2],'movesize'] =  15.
    #mdf.loc[mdf.index[1::2],'movesize'] = -5.
    mdf['stepsize'] = mdf['dphi']/mdf['movesize']
    mdf['is_fwd'] = np.sign(mdf['movesize']) > 0
    return mdf

def plot ( df, axarr, cmap='PiYG', lbl=None ):    
    axarr[0].scatter ( df['x'], df['y'], s=9, c=df.index % 2,
                       cmap=cmap, alpha=0.8, label=lbl)
    axarr[0].set_aspect('equal','datalim' )
    
    axarr[1].scatter ( df.index, df['dist'], s=18, c=df.index % 2,
                       cmap=cmap, alpha=0.8 )    
    [ axarr[i].grid(alpha=0.4) for i in range(2) ]

    axarr[0].set_xlabel ( 'x (mm)' )
    axarr[0].set_ylabel ( 'y (mm)' )
    axarr[1].set_xlabel ( 'time (step #)')
    axarr[1].set_ylabel ( r'distance from start ($\mu$m)')
    plt.tight_layout ()
    return axarr

def run ( ):
    dirnames = ['18_04_25_10_11_47_erin_test/',
                '18_04_25_11_12_44_erin_test/',
                '18_04_25_12_26_06_erin_test/']
    cmaps=['PiYG','RdBu','PuOr_r']
    labels=['Run 1','Run 2','Run 3']
    for pid in range(1,57):
        fig,axarr = plt.subplots(1,2,figsize=(10,4))
        for i,cdir in enumerate(dirnames):
            fname = './%s/Log/PhiSpecMove_mId_1_pId_%i.txt' % (cdir,pid)
            df = read(fname)
            plot ( df, axarr, cmaps[i] )
        plt.savefig('./timevol_pid%i.png'%pid)
        plt.close ()

def clean_map(ctrlstep):
    ctrlstep = ctrlstep.convert_objects ()
    #// filter ctrlstep based on Johannes' suggestions                   
                      
    lowthresh = 0.01
    ctrlstep.loc[ctrlstep['stepsize'] < 0, 'stepsize'] = np.nan
    ctrlstep.loc[ctrlstep['stepsize'] > 1., 'stepsize'] = np.nan

    bins = np.arange ( 0., 400., 10. )
    assns = np.digitize ( ctrlstep['startangle'], bins )
    grps = ctrlstep.groupby ( assns )
    ssmean = grps.mean()['stepsize']
    sscount = grps.count()['stepsize']

    #// cut on mean change or overpopulation                             
                      
    deltam = abs(ssmean-ssmean.mean()) > 3.*ssmean.std()
    deltact = abs(sscount-sscount.mean()) > 3.*sscount.std()
    to_cut = ssmean.index[deltam|deltact]

    ctrlstep.loc[np.in1d(assns, to_cut),'stepsize'] = np.NaN
    mm = np.isfinite(ctrlstep).all(axis=1)
    return ctrlstep.loc[mm]        

    
def read_tso ( specdir, pid, niter=3 ):
    po = read_ctrlstep ( specdir + '/PhiSpecMove_mId_1_pId_%i.txt'%pid,
                                  movesize=None)
    if not po.iloc[niter]['startangle'] < po.iloc[0]['startangle']:
        #// check to make sure that the motor got all the way back home.
        #// If not, discard
        print ("PID %i did not make it home! :(" % pid )
        return None

    benchmarks = po.iloc[:niter+1]['startangle'].sort_values ()

    diff = benchmarks.diff ()/2

    po['movestate'] = np.nan
    po.loc[:niter,'movestate'] = range(1,niter+1)
    po.loc[niter+1,'movestate'] = 0
    for idx in po.index:
        bench = po.groupby('movestate').apply ( lambda x: x.iloc[-1] )
        po.loc[idx,'movestate'] = abs(po.loc[idx,'startangle'] - bench['startangle']).argmin()
    return po
        
