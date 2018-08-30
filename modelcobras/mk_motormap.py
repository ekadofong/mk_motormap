#!/usr/bin/env python

import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from . import model, movement, control, analyze


def gen_motormap ( pid, savedir ):
    '''
    Generate a controlled step motor map via GP to use to seed the
    convergence runs.

    Parameters
    ==========
    pid : int
     PID of the the cobra for which to generate motor maps
    savedir : str
     The directory path in which to save output.
    '''
    if not type(pid) is int:
        pid, savedir = pid
    mmap = {}
    for key in dirdict.keys ():
        fname = dirname + dirdict[key] + '/Log/PhiSpecMove_mId_1_pId_%i.txt' % pid
        if not os.path.exists ( fname ):
            print('Not found: %s' % fname)
            continue
        movesize, nmoves, niter = paramdict[key]
        
        ctrlstep = movement.read_ctrlstep ( fname, movesize, movesize=0,
                                            verbose=True, motor_id=middict[key])
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

        slow_mask = ctrlstep['stepsize'] < lowthresh
        ctrlstep.loc[slow_mask, 'stepsize' ] += .03
        #mmap[key] = ctrlstep
        #continue

        mm = np.isfinite(ctrlstep).all(axis=1)
        try:
            gpmod, axarr = analyze.viz_gproc ( [ctrlstep.loc[mm]], angle_grid=angbins )
        except ValueError:
            return
        gpmod = gpmod[0]

        #// set no-data to mmap=0.1
        if 'stage2' in key:
            max_angle = 180.
        else:
            max_angle = 365.
        
        gap_thresh = 20.
        gaps = ctrlstep['startangle'].sort_values().diff().dropna() > gap_thresh
        gap_ends = gaps.loc[gaps].index
        gap_stts = set ()
        for eval in gap_ends:
            gap_stts.add(gaps.index[gaps.index.get_loc(eval) - 1])
        gap_ends = ctrlstep.loc[gap_ends,'startangle'].values.tolist()
        gap_stts = ctrlstep['startangle'].loc[gap_stts].values.tolist()
        
        if ctrlstep.loc[gaps.index[0],'startangle'] > gap_thresh:
            gap_stts.append(0)
            gap_ends.append(ctrlstep.loc[gaps.index[0],'startangle'])
        elif ctrlstep.loc[gaps.index[-1], 'startangle'] < (max_angle - gap_thresh):
            gap_stts.append(ctrlstep.loc[gaps.index[-1],'startangle'])
            gap_ends.append(max_angle)

        for start, end in zip ( gap_stts, gap_ends ):
            out_of_bounds = (gpmod.angle_grid>start)&(gpmod.angle_grid<end)
            gpmod.shape_mu[out_of_bounds] = 0.1/gpmod.mmean
            
        gpmod.shape_mu[(gpmod.shape_mu*gpmod.mmean)<.02] = .02/gpmod.mmean

        axarr[0].plot ( gpmod.angle_grid, gpmod.shape_mu*gpmod.mmean,
                        '--', color='dodgerblue')
        axarr[1].plot ( gpmod.angle_grid, gpmod.shape_mu, '--', color='dodgerblue')
        axarr[0].set_ylim(0., 0.25)

        mmap[key] = gpmod
        np.savetxt ( savedir + '/pid%i_%s.dat' % (pid, key ),
                     gpmod.shape_mu * gpmod.mmean)
        plt.savefig(savedir + '/figures/pid%i_%s.png' % (pid, key) )
        plt.close('all')        
    return mmap

def load_directory ( kwargs, key ):
    import glob
    if key not in kwargs:
        try:
            dirname = os.path.basename ( glob.glob("./data/*%s"%key)[0] )
        except IndexError:
            raise OSError ("No directory matching the keyword %s." % key)
    else:
        dirname = kwargs[key]
    return dirname
    
def generate_motormaps (savedir):
    if not os.path.exists ( './motormaps' ):
        os.mkdir('./motormaps')
    if not os.path.exists (savedir):
        os.mkdir(savedir)
    if not os.path.exists (savedir+'/figures'):
        os.mkdir ( savedir+'/figures')

    for pid in range(1,58):
        already_run = os.path.exists ( savedir + '/pid%i_fwd_stage1.dat' % pid )
        if os.path.exists ( savedir + '/pid%i.lock' % pid ) or already_run:
            continue
        else:
            open( savedir + '/pid%i.lock' % pid, 'w').write('locked')
        gen_motormap ( pid, savedir )
        os.remove ( savedir + '/pid%i.lock' % pid )

def update_xml ( savedir,
                 xmlfile = './xml_files/usedXMLFile.xml',
                 savename = './xml_files/ctrlupdate.xml'):
    '''
    For a given directory populated by controlled step motor maps,
    update the MSIM XML file.

    Parameters
    ==========
    savedir : str
     Directory that contains the output of gen_motormap
    xmlfile : str
     Path to the XML file to use as a base.
    '''
    import xml.etree.ElementTree as ET
    tree = ET.parse ( xmlfile )
    root = tree.getroot ()

    xmldict = {1: "Joint1_fwd_stepsizes",
               3: "Joint1_rev_stepsizes",
               5: "Joint2_fwd_stepsizes",
               7: "Joint2_rev_stepsizes" }
    pydict = {1:'fwd_stage1',
              3:'rev_stage1',
              5:'fwd_stage2',
              7:'rev_stage2' }
               
               
    
    for idx in range(len(root)):
        pid = idx + 1
        for calib in [2,3]: #// overwrite both slow & fast calib. table
            for jointset in [1,3,5,7]:

                uname = '%s/pid%i_%s.dat' % (savedir,
                                             pid,
                                             pydict[jointset])

                if not os.path.exists(uname) and (calib==2):
                    print('No data for pid %i, %s!' % (pid, pydict[jointset]))
                    continue
                elif not os.path.exists(uname):
                    continue
                bud = np.array(root[idx][calib][jointset].text.split(',')[:-1], dtype=float)
                header = bud[:2]


                update = np.genfromtxt ( uname )
                deg = np.array(root[idx][calib][jointset-1].text.split(',')[2:-1],
                               dtype=float)
                plt.plot ( deg, update, label='updated map' )
                plt.plot ( deg, bud[2:], label='Newscale map(?)' )
                plt.legend ()
                plt.xlabel ( 'deg' )
                plt.ylabel ( 'speed (deg/step)')
                plt.tight_layout ()
                plt.savefig('%s/figures/pid%i_update_%s.png' % (savedir,
                                                                pid,
                                                                pydict[jointset] ))
                plt.close('all')

                #// actually update xml
                newarray = np.concatenate([header,update]).astype(str)
                newtext = ','.join (newarray) + ','
                root[idx][calib][jointset].text = newtext

    tree.write ( savename )
    return tree


def main (*args, **kwargs):
    global dirname
    global angbins
    global dirdict, paramdict, middict
    dirname = './data/'
    
    dirdict = {}    
    dirdict["rev_stage2"] = load_directory(kwargs,"reverse-stage2")
    dirdict["rev_stage1"] = load_directory(kwargs,"reverse-stage1")
    dirdict["fwd_stage2"] = load_directory(kwargs,"forward-stage2")
    dirdict["fwd_stage1"] = load_directory(kwargs,"forward-stage1")

    binsize = 3.5712 # deg
    angbins = np.concatenate([np.arange( 3.6, 400, binsize ),[400]])

    paramdict = {}
    # ( movesize, nmoves, niter )
    paramdict['fwd_stage2'] = (30, 150, 3)
    paramdict['rev_stage2'] = (-30, 100, 4)
    paramdict['fwd_stage1'] = (30, 400, 3)
    paramdict['rev_stage1'] = (-30, 400, 3)

    middict = {}
    middict['fwd_stage2'] = 2
    middict['rev_stage2'] = 2
    middict['fwd_stage1'] = 1
    middict['rev_stage1'] = 1

    if 'xml' in kwargs.keys():
        xml_file = kwargs['xml']
    else:
        xml_file = './xml_files/usedXMLFile.xml'

    if 'savedir' in kwargs.keys():
        savedir = kwargs['savedir']
    else:
        import time
        ltime = time.localtime ()
        long_date = '%i-%s-%s' % (ltime.tm_year, str(ltime.tm_mon).zfill(2),
                                  str(ltime.tm_mday).zfill(2) )
        savedir = './motormaps/mmap_%s/' % long_date

    if savedir[-1] == '/':
        qs = savedir[:-1]
    else:
        qs = savedir

    save_xml = './xml_files/%s.xml' % os.path.basename(qs)

    # // Display parameters for sanity check
    print('\nUsing controlled step data from:')
    for key,val in dirdict.iteritems ():
        print ('%s :: %s' %(key,val))
    print('Reading XML file :: %s' % xml_file )
    print('Making output directory :: %s\n\n' % savedir)

    generate_motormaps ( savedir )
    update_xml ( savedir, xml_file, save_xml )
    print('Saving to updated XML file: %s' % save_xml)

