#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd
import george
import movement,model

def viz_multictrl ( pid,
                    specdirs=['/u/kadofong/work/pfscobras/jpltrip/data/18_04_27_09_35_38_erin_test/Log/'],
                    ax = None,
                    alpha= 1.,
                    normalize=False):
    if ax is None:
        ax = plt.subplot ( 111 )

    colors = [['orange','tomato'],
              ['dodgerblue','darkviolet']]

    mmean = []
    for i,specdir in enumerate(specdirs):
        mdf = movement.read_mdf ( specdir + '/PhiSpecMove_mId_1_pId_%i.txt' % pid )
        mdf = mdf.query('is_fwd')

        cutoff = (mdf.iloc[1:]['startangle'] - mdf['startangle'].iloc[:-1].values).argmin()
        m1 = mdf.loc[:cutoff].index
        m2 = mdf.loc[cutoff:].index

        for cm,color in zip([m1,m2], colors[i]):
            if normalize:
                ax.scatter ( mdf.loc[cm]['startangle'], mdf.loc[cm]['stepsize']/mdf.loc[cm]['stepsize'].median(), 
                             s=7,
                             color=color, alpha=alpha)
            else:
                ax.scatter ( mdf.loc[cm]['startangle'], mdf.loc[cm]['stepsize'], s=7,
                             color=color, alpha=alpha)
        mmean.append(mdf['stepsize'].mean())
    ax.set_ylim ( -0.02, max(mmean)*2.5)
    ax.grid(alpha=0.4)
    ax.set_xlabel (r'$\phi$')
    ax.set_ylabel ('step size (deg)')
    plt.tight_layout ()

def viz_timevar ( pid, specdir='./data/18_05_22_15_20_41_erin_test/Log/',
                  waittime=3600,
                  stepseq=[30.,30.,30.,30.],
                  ax=None ):
    if ax is None:
        ax = plt.subplot ( 111 )

    nmdf = movement.read_ctrlstep ( specdir + '/PhiSpecMove_mId_1_pId_%i.txt' % pid, stepseq=stepseq )

    grps = nmdf.groupby ( 'iter' )

    colors = dict(zip(grps.groups.keys(),gen_colors ( len(stepseq), cmap='viridis')))

    for name, group in grps:        
        ax.scatter ( group.startangle, group.stepsize, s=16, label = '%is' % (waittime*name),)
                     #color=colors[name])
    ax.set_ylim ( -0.02, max(.1,nmdf.query('startangle<150.').stepsize.median()*2.5))
    ax.grid(alpha=0.4)
    ax.set_xlabel (r'$\phi$')
    ax.set_ylabel ('step size (deg)')
    ax.legend(loc=3)
    plt.tight_layout ()


def viz_multipleruns ( pid,
                       specdir ='/u/kadofong/work/pfscobras/jpltrip/data/18_04_27_09_35_38_erin_test/Log/',
                       convdirs=['/u/PFS/PFS_Production/TEST_RESULTS/Spare2/18_04_26_14_45_06_TargetRun/'],
                       minstep = 10,
                       maxstep = 50,
                       axarr = None ):
    mdf = movement.read_mdf ( specdir + '/PhiSpecMove_mId_1_pId_%i.txt' % pid )
    # stepsize cut removes in-between iterations
    mdf = mdf.query('is_fwd')
    if axarr is None:
        fig, axarr = plt.subplots ( 2,1, figsize=(7,7), sharex=True)

    cutoff = (mdf.iloc[1:]['startangle'] - mdf['startangle'].iloc[:-1].values).argmin()
    m1 = mdf.loc[:cutoff].index
    m2 = mdf.loc[cutoff:].index

    for cm,color in zip([m1,m2], ('darkorange', 'dodgerblue')):        
        axarr[0].scatter ( mdf.loc[cm]['startangle'], mdf.loc[cm]['stepsize'], s=20,
                           color=color, marker='+')
                           
                       #c=mdf.index, cmap='bone', label='fixed step run' )
    mbins = np.arange ( mdf['startangle'].min(), mdf['startangle'].max(), 5)
    assns = np.digitize ( mdf['startangle'], mbins )
    grps = mdf.groupby(assns)

    axarr[1].fill_between ( mbins, -grps['stepsize'].std()*3., grps['stepsize'].std()*3.,
                        color='lightcoral', alpha=0.2)

    colors = gen_colors ( len(convdirs)+1, cmap='rainbow' ) [1:]
    all_ym = 0.
    for i,convdir in enumerate(convdirs):
        color = colors[i]
        conv = movement.read_convergencedata ( convdir + '/mId_1_pId_%i.txt' % pid )
        mm = (conv['J2_s']>minstep) & (conv['J2_s']<maxstep) & (abs(conv['J2_stepsize'])<0.5)        
        conv = conv.loc[mm]
        axarr[0].scatter ( conv.loc[:,'J2'], conv.loc[:,'J2_stepsize'], s=10,
                           c=color, label='target run %i' % i )
        cassns = np.digitize ( conv.loc[:,'J2'], mbins )
        y = conv.loc[:,'J2_stepsize'] - grps.mean()['stepsize'].loc[cassns].values

        im = axarr[1].scatter ( conv.loc[:,'J2'], y, s=3, c=color,
                                zorder=10 )         
        ym = np.percentile ( y, 99.9 )
        if ym > all_ym:
            all_ym = ym
    axarr[1].set_ylim(-all_ym, all_ym )    
    axarr[0].set_ylim ( -0.02, np.mean(mdf['stepsize'])*2.5)
    axarr[1].grid(alpha=0.4)
    axarr[0].grid(alpha=0.4)

    axarr[1].set_xlabel ( r'$\phi$ (deg)' )
    axarr[0].set_xlabel ( r'$\phi$ (deg)' )
    axarr[0].set_ylabel ( r'$\frac{d\phi}{ds}$ (deg/step)')
    axarr[1].set_ylabel ( r'$\frac{d\phi}{ds}_{\rm conv} - \langle\frac{d\phi}{ds}_{\rm ctrl}\rangle$ (deg/step)')
    #axarr[0].legend(loc='best')
    plt.tight_layout ()
    
def viz_diff ( pid,
               specdir = '/u/kadofong/work/pfscobras/jpltrip/data/18_04_27_09_35_38_erin_test/Log/',
               convdir = '/u/PFS/PFS_Production/TEST_RESULTS/Spare2/18_04_26_14_45_06_TargetRun/',
               minstep = 10,
               maxstep = 50,
               cval='J2_s'
               ):
    mdf = movement.read_mdf ( specdir + '/PhiSpecMove_mId_1_pId_%i.txt' % pid )
    conv = movement.read_convergencedata ( convdir + '/mId_1_pId_%i.txt' % pid )
    mm = (conv['J2_s']>minstep) & (conv['J2_s']<maxstep) & (abs(conv['J2_stepsize'])<0.5)
    # stepsize cut removes in-between iterations
    mdf = mdf.query('is_fwd')
    conv = conv.loc[mm]
    
    fig, axarr = plt.subplots ( 2,1, figsize=(7,7), sharex=True)
    
    axarr[0].scatter ( mdf['startangle'], mdf['stepsize'], s=1,
                       c=mdf.index, cmap='bone' )
    axarr[0].scatter ( conv.loc[:,'J2'], conv.loc[:,'J2_stepsize'], s=8,
                       c=conv[cval], cmap='viridis' )

    mbins = np.arange ( mdf['startangle'].min(), mdf['startangle'].max(), 5)
    assns = np.digitize ( mdf['startangle'], mbins )
    grps = mdf.groupby(assns)

    axarr[1].fill_between ( mbins, -grps['stepsize'].std()*3., grps['stepsize'].std()*3.,
                        color='lightcoral', alpha=0.2)

    cassns = np.digitize ( conv.loc[:,'J2'], mbins )
    y = conv.loc[:,'J2_stepsize'] - grps.mean()['stepsize'].loc[cassns].values

    #axarr[0].scatter ( mbins, grps.mean()['stepsize'], s=13, c='red')
    im = axarr[1].scatter ( conv.loc[:,'J2'], y, s=8, c=conv[cval], cmap='viridis',
                       zorder=10 )
    ym = np.percentile ( y.dropna(), 99.9 )
    axarr[1].set_ylim(-ym, ym )    
    axarr[0].set_ylim ( -0.02, np.mean(mdf['stepsize'])*2.5)
    axarr[1].grid(alpha=0.4)
    axarr[0].grid(alpha=0.4)

    axarr[1].set_xlabel ( r'$\phi$ (deg)' )
    axarr[0].set_ylabel ( r'$\frac{d\phi}{ds}$ (deg/step)')
    axarr[1].set_ylabel ( r'$\frac{d\phi}{ds}_{\rm conv} - \langle\frac{d\phi}{ds}_{\rm ctrl}\rangle$ (deg/step)')
    fig.tight_layout ()
    fig.subplots_adjust ( right = 0.8 )
    cbar_ax = fig.add_axes ( [0.85,.15,.05,.7] )
    fig.colorbar ( im, cax=cbar_ax, label=cval )
    

    return mdf, conv

def viz_inrunvariation ( conv, cval='time', binsize=0.005, yext=0.05, show_both=False ):
    # first, bin by average motor map step size
    ssbins = np.linspace ( 0., conv['J2'].max(), 40 )
    assns = np.digitize ( conv['J2'], ssbins )
    mstep = conv.groupby ( assns ).mean()["J2_stepsize"]
    y = conv[['J2_stepsize',cval]]
    y['J2_stepsize'] -= mstep.loc[assns].values

    # now check for trend in deviations w.r.t cval
    stbins = np.arange ( -yext, yext, binsize )

    cassns = np.digitize ( y['J2_stepsize'], stbins )
    sgrps = y.groupby ( cassns )

    if show_both:
        fig, axarr = plt.subplots ( 2,1, figsize=(6,6 ), sharex=True)
        axarr[0].scatter ( y['J2_stepsize'],conv['J2'],  c=conv[cval], s=30 )
        axarr[0].set_xlim ( -yext, yext )

        axarr[1].errorbar ( stbins[(np.unique(cassns)-1)[1:]], sgrps.mean().loc[1:][cval],
                           yerr= (sgrps.std()[cval]/sgrps.count()[cval]**.5).loc[1:], fmt='o')
        #axarr[1].hlines ( y[cval].mean(), stbins[np.unique(cassns)-1].min(),
        #                  stbins[np.unique(cassns)-1].max() )
        axarr[1].set_xlabel ( r'step size - $\langle {\rm step size} \rangle$ (deg)' )
        axarr[0].set_ylabel ( r'$\phi$ (deg)' )
        axarr[1].set_ylabel ( r'$\langle {\rm time} \rangle$')
    else:
        ax = plt.subplot ( 111 )
        ax.errorbar ( stbins[(np.unique(cassns)-1)[1:]], sgrps.mean().loc[1:][cval],
                      yerr= (sgrps.std()[cval]/sgrps.count()[cval]**.5).loc[1:], fmt='o')
        ax.set_xlabel ( r'step size - $\langle {\rm step size} \rangle$ (deg)' )
        ax.set_ylabel ( r'$\langle {\rm time} \rangle$')
        
    return sgrps


    
def gen_colors(l, cmap='gist_ncar'):
    '''
    Generates a list of evenly spaced colors from a matplotlib
    colormap.
    
    args:                                                                                                  
        l (int): number of colors to generate        
        cmap (str): colormap to generate colors from

        
    returns:
            clist (list): List of evenly spaced colors
    '''
    import matplotlib.pyplot as plt

    jet = plt.cm.ScalarMappable(cmap=cmap)
    clist = jet.to_rgba(np.arange(l))
    return clist

    

def viz_gproc ( llist, axarr=None, colors=None, kernel=None, angle_grid=None):
    if colors is None:
        colors = ['tomato','dodgerblue','mediumorchid','mediumseagreen','grey','orange','chartreuse','cornflowerblue']
    if axarr is None:
        fig, axarr = plt.subplots ( 2, 1, figsize=(10,6), sharex=True)
    alist = []
    for i in range(len(llist)):
        mm = model.MotorModel (angle_grid)
	     #cwait = wait.query('iter==%i'%i).dropna()
        cwait = llist[i]

        bins = np.arange(0, np.nanmax(cwait['startangle']), 10.)
        assns = np.digitize ( cwait['startangle'], bins ) 
        stds = cwait.stepsize.groupby(assns).std()
        stds = stds.replace ( np.NaN, 100.)
        
        cwait['u_stepsize'] = stds.loc[assns].values    
        mm.read_object(cwait)
        if kernel is None:
            k2 = .5*george.kernels.ExpSquaredKernel(5.**2) * george.kernels.ExpSine2Kernel(3.0, 10.0) + \
                0.2 * george.kernels.ExpSquaredKernel(.05)
        else:
            k2 = kernel

        okernel = mm.set_hyperparam ( k2 )
        mm.model_shape ( okernel )

        axarr[0].plot(  mm.angle_grid, mm.shape_mu*mm.mmean, color=colors[i], alpha=0.5, lw=3 )
        axarr[1].plot(  mm.angle_grid, mm.shape_mu, color=colors[i], alpha=0.5, lw=3
 )
        
        axarr[0].fill_between ( mm.angle_grid, mm.shape_mu*mm.mmean - mm.shape_std*mm.mmean,
                                mm.shape_mu*mm.mmean + mm.shape_std*mm.mmean, alpha=0.08, color=colors[i])
        axarr[1].fill_between ( mm.angle_grid, mm.shape_mu - mm.shape_std,
                                mm.shape_mu + mm.shape_std, alpha=0.08, color=colors[i])
        axarr[0].scatter ( mm.startangle, mm.stepsize, s=1, color=colors[i], alpha=0.3)
        axarr[1].scatter ( mm.startangle, mm.stepsize/mm.mmean, s=5, color=colors[i])
	     
        alist.append(mm)
	     #wait.loc [ cwait.index, 'mval'] = mm.mmean
    axarr[1].set_xlabel ( r'starting angle (deg)' )
    axarr[0].set_ylabel ( 'step size (deg)' )
    axarr[1].set_ylabel ( 'normalized step size')

    ym = np.nanmax(mm.shape_mu+mm.shape_std)*1.5
    #ymin = min(mm.shape_mu+mm.shape_std)*.5
    axarr[1].set_ylim ( 0., ym )

    maxmean = max([mm.mmean for mm in alist])
    
    axarr[0].set_ylim ( -0.02, maxmean*3)
    [ axarr[i].grid(alpha=0.4) for i in range(2) ]
    return alist, axarr

    
def viz_smbdiff ( pid ):
    '''
    Synthesize small steps from larger ones
    '''
    name = 'PhiSpecMove_mId_1_pId_%i.txt' % pid
    allbig = movement.read_ctrlstep('./data/18_05_23_16_59_00_time_variability//Log/' + name,
                                 [30,]*8).convert_objects ()
    big = allbig.query('iter==2')
    #big = movement.read_ctrlstep ( './data/18_05_22_14_46_41_diffstep//Log/' + name ).convert_objects()
    #big = big.query('iter==0')

    small = movement.read_ctrlstep ( './data/18_05_24_08_12_27_small_steps/Log/'+name,
                                     [2,])

    ovldf = []
    for i in range(2):
        idx = np.floor((small.index+i*7)/15.)
        grps = small.groupby(idx)

        df = pd.DataFrame ( index=grps.groups.keys(), columns='sdphi bdphi diff'.split())

        for name, gg in grps:
            gg = gg.dropna()

            if gg.shape[0] == 0:
                continue

            dphi = gg.dphi.sum()
            #dphi = np.sum(np.random.choice(small['dphi'], gg.shape[0])) # test vs. random
            if np.min(abs(big.startangle - gg.startangle.iloc[0])) > 1.:
                continue

            equiv = big.loc[np.argmin(abs(big.startangle - gg.startangle.iloc[0]))]
            df.loc[name,'phi'] = equiv['startangle']
            df.loc[name,'sdphi'] = dphi
            df.loc[name,'bdphi'] = equiv['dphi']
            df.loc[name, 'diff'] = equiv['dphi']-dphi

        df = df.query('phi>10.')
            
        minsize = 0.02 * big['movesize'].mean()
        mm = df.query('bdphi<%f' % minsize).index
        if len(mm) > 0:
            df = df.loc[:mm[0]-1]


        ovldf.append(df)
    df = pd.concat(ovldf).reset_index()

    mbins = np.arange(-5.,5.,.4)

    fig = plt.figure ( figsize=(10,4))
    ax1 = plt.subplot2grid((1,3), (0, 0), colspan=2)
    ax1.scatter ( allbig.startangle, allbig.dphi, s=8, color='lightpink', alpha=0.5)
    ax1.scatter ( big.startangle, big.dphi, s=20, label='real moves', color='indianred' )
    ax1.scatter ( df.phi, df.sdphi, s=20, label='synthesized moves', color='dodgerblue' )
    ax1.scatter ( small.startangle, small.dphi, s=8, color='cornflowerblue', alpha=0.5 )

    ax1.set_ylim(-1.,df.bdphi.mean()*2.5)
    ax1.set_xlim ( 10.,122)
    ax1.set_xlabel ( r'$\phi$ (deg)')
    ax1.set_ylabel ( r'$\Delta \phi$ (deg)')

    #plt.subplot (122)
    ax2 = plt.subplot2grid((1, 3), (0, 2), colspan=1)
    ax2.hist ( [df.bdphi, df.sdphi], bins=8, label='real moves,synthesized moves'.split(','),
               color='indianred dodgerblue'.split())
    ax2.legend ()
    ax2.set_xlabel ( r'$\Delta \phi$ (deg)')
    ax2.set_ylabel ( r'N')
    fig.tight_layout()
    return df.astype(float)

'''fig,axarr = plt.subplots(1,2,figsize=(8,3))
      ...: 
      ...: mbins = np.arange(-2.,2.,.1)
      ...: axarr[0].hist ( odf['diff'], bins=mbins, color='mediumseagreen' )
      ...: axarr[1].hist ( [odf.sdphi, odf.bdphi], bins=np.arange(-.2, 8.,.3), normed=True,
      ...:     label=['small step', 'large step'], color=[c1,c2])
      ...: #plt.plot ( mbins, gauss ( mbins, 654., 0., .5))
      ...: axarr[0].vlines( 0., 0.,654.)
      ...: 
      ...: axarr[0].set_ylim(0.,500.)
      ...: axarr[0].set_xlabel ( r'$d\phi_{\rm large} - d\phi_{\rm small}$ (deg)')
      ...: axarr[0].set_ylabel ( r'$N$')
      ...: axarr[1].set_xlabel ( r'$d\phi$ (deg)' )
      ...: axarr[1].set_ylabel (r'$\frac{dN}{d\phi}$')
      ...: axarr[1].legend(loc='best',frameon=False)
      ...: plt.tight_layout ()
      ...: 

'''


def viz_dphivar ( pid, smm=None, bmm=None ):
    '''
    Compare dphi movements of short and long steps
    '''
    from scipy import stats
    
    fname = 'PhiSpecMove_mId_1_pId_%i.txt' % pid

    if (smm is None)|(bmm is None):
        wait = movement.read_ctrlstep('./data/18_05_23_16_59_00_time_variability//Log/' + fname,
                                      [30,]*8).convert_objects ()
        wait = wait.query('startangle<150.')
        wait['stepsize'] = wait['dphi']
        wait = movement.estimate_std ( wait )


        sml = movement.read_ctrlstep ( './data/18_05_24_08_12_27_small_steps/Log/'+fname,[2,])
        sml = sml.query('startangle<150.')
        sml['stepsize'] = sml['dphi']
        sml = movement.estimate_std ( sml )

        smm = fit_gp ( sml )
        bmm = fit_gp ( wait )

    plt.figure ( figsize=(8.5,4))
    plt.subplot(121)
    
    ys = mkfunc ( smm, color='dodgerblue' )                                                                     
    bys = mkfunc ( bmm, color='tomato' )

    cstd = stats.sigmaclip(ys)[0].std()
    bstd = stats.sigmaclip(bys)[0].std()
    cnaive = np.random.normal( 0., cstd*np.sqrt(15.), smm.startangle.size)
    plt.scatter ( smm.startangle,
                  cnaive,
                  s=8, color='lightgrey', zorder=0)
                                    
                                    
    plt.xlabel ( r'$\phi$ (deg)')                                
    plt.ylabel ( r'$|\vec{\Delta \phi} - \hat{\Delta \phi}| ~(\deg)$')
    #plt.title ( 'pid %i' % pid)
    plt.ylim ( -2.5, 2.5 )
                                               
    plt.subplot(122)
    bins = np.arange(-2.5,2.5,.15)
    plt.hist ( [ys,bys], normed=True, bins=bins,
               color=['dodgerblue','tomato',],
               label=[r'2-step ($\sigma = %.2f~\deg$)'%cstd,
                      r'30-step ($\sigma = %.2f~\deg$)'%bstd])
    plt.hist ( cnaive, normed=True, bins=bins, color='lightgrey',zorder=0, edgecolor='white')
                      
    plt.xlabel ( r'$|\vec{\Delta \phi} - \hat{\Delta \phi}| ~(\deg)$')
    plt.ylabel ( r'$dN / d(|\vec{\Delta \phi} - \hat{\Delta \phi}|)$')
    plt.legend (loc='best')
    plt.tight_layout ()
    return smm, bmm
    

def fit_gp ( wait ):
    smm = model.MotorModel ()                                                               
    smm.read_object ( wait.loc[np.isfinite(wait.astype(float)).all(axis=1)] )

                                                                                           
    k2 = .5*george.kernels.ExpSquaredKernel(15**2) * george.kernels.ExpSine2Kernel(1.0, 5.0) + \
        .2 * george.kernels.ExpSquaredKernel(.1**2)
    okernel = smm.set_hyperparam ( k2 )                           
    smm.model_shape ( okernel ) 
    return smm

def mkfunc ( mm, color='k' ):                                                                     
    func = interpolate.interp1d ( mm.angle_grid, mm.shape_mu*mm.mmean )
    ys = mm.stepsize - func(mm.startangle)                                             
    plt.scatter ( mm.startangle, mm.stepsize - func(mm.startangle),
                  color=color, s=10, alpha=0.6)      
    return ys    


def viz_gotoangle ( specdir, targetangle=55. ):
    df = pd.DataFrame ( index=range(1,58), columns=['phistart','phiend','u_move'])
    for pid in range(1,58):
        try:            
            crec = pd.read_csv ( specdir + 'PhiSpecMove_mId_1_pId_%i.txt' % pid,
                                 header=None )
        except IOError:
            print('No entry for pid %i!' % pid )
            continue
        if crec.shape[0] != 2:
            print('No entry for pid %i!' % pid )
            continue            
        df.loc[pid,'phistart'] = crec.loc[0,3]
        df.loc[pid,'phiend'] = crec.loc[1,3]
        df.loc[pid,'u_move'] = targetangle - crec.loc[1,3]
    return df


def viz_tsodiv ( pid, specdir='./data/SC01/18_06_28_09_06_09_movetostage1-wait/Log/'):
    po = movement.read_tso ( specdir, pid)
    if po is None: 
        return 1
    colors = ['lightgrey','tomato','seagreen','dodgerblue']
    labels = ['@ home', '150 TSO', '300 TSO', '450 TSO']
    for name, group in po.groupby('movestate'):
        if name == 0:
            continue
        elif name==1:
            alpha = 0.3
        else:
            alpha = 1.
        pg = group
        plt.plot ( pg.index, pg.startangle/pg.startangle.mean(), 'o--',
                   alpha=alpha,
                   color=colors[int(name)], label=labels[int(name)])

    plt.legend ()
    plt.xlabel ( 'trial number' )
    plt.ylabel ( r'$\phi/\langle\phi\rangle$')
    plt.grid(linestyle=':')
    return 0
