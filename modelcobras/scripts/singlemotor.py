#!/usr/bin/env python



specdir = '/Users/kadofong/work/pfscobras//data/SC01/18_06_15_14_07_15_goto-stage1/Log/'

def run():
    wlist = {}
    for pid in range(1,58):
        fname = 'PhiSpecMove_mId_1_pId_%i.txt' % pid
        po = movement.read_ctrlstep ( specdir + fname, movesize=None).query('startangle>10.')
        po['stepsize'] = po.startangle.copy ()
        po['startangle'] = range(po.shape[0])
        if po['stepsize'].shape[0] > 0:
            wlist[pid] = po
    return wlist
