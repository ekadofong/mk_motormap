#!/usr/bin/env python

import numpy as np
from scipy import optimize
import pandas as pd
import george


class MotorModel ( object ):
    def __init__ ( self, angle_grid=None ):
        '''
        Simple model described as
        stepsize(phi,t) = A(t)*f(phi)
        '''
        self.angle_grid = angle_grid

    def read_object ( self, obj ):
        self.stepsize = obj.stepsize
        self.u_stepsize = obj.u_stepsize
        self.startangle = obj.startangle
        #self.mmean = self.stepsize[(self.startangle>0)&(self.startangle<100)].mean()
        fn = lambda x, a: a

        mask = obj.stepsize > 0.02
        self.mmean = optimize.curve_fit ( fn, obj.startangle[mask], obj.stepsize[mask] )[0]

        if obj.startangle.max()>190:
            amax = 370.
        elif obj.startangle.max() < 30:
            amax = 30
        else:
            amax = 180.
        if self.angle_grid is None:
            self.angle_grid = np.arange ( 0., amax, 0.1 )

    def model_shape ( self, kernel ):
        '''
        Model f(phi) as a GP
        '''
        gp = george.GP ( kernel, mean=1.)
        #print(np.isfinite(self.startangle).all())
        #print(np.isfinite(self.u_stepsize).all())
        #print(np.isfinite(self.mmean).all())

        gp.compute ( self.startangle, self.u_stepsize/self.mmean)


        mu,cov = gp.predict ( self.stepsize/self.mmean, self.angle_grid )
        self.shape_mu = mu
        self.shape_cov = cov
        self.shape_std = np.sqrt(np.diag(cov))*3.


    def set_hyperparam ( self, init_kernel, mval=1. ):
        gp = george.GP ( init_kernel, mean=mval )
        def neglnlike ( param ):
            '''
            if min(param) < 0:
                return 1e30
            elif param[0] < 5.:
                return 1e30'''
            gp.kernel.set_parameter_vector ( param )
            lnlike = gp.lnlikelihood ( self.stepsize/self.mmean, quiet=True )
            return -lnlike if np.isfinite(lnlike) else 1e30

        def grad_neglnlike ( param ):
            gp.kernel.set_parameter_vector ( param )
            return -gp.grad_lnlikelihood ( self.stepsize/self.mmean, quiet=True )

        gp.compute ( self.startangle, self.u_stepsize/self.mmean)
        #print( gp.lnlikelihood(self.stepsize/self.mmean) )
        p0 = gp.kernel.get_parameter_vector ()
        #print( p0 )
        results = optimize.minimize ( neglnlike, p0, jac=grad_neglnlike )
        #print( gp.lnlikelihood(self.stepsize/self.mmean) )
        #print( gp.kernel.get_parameter_vector () )
        return gp.kernel


    def movetoangle ( self, pos_start=0., pos_end=55., motorid=2 ):
        from scipy.integrate import simps
        
        mask = (self.angle_grid>=pos_start)&(self.angle_grid<=pos_end)
        fx = self.angle_grid[mask]
        fy = ((self.shape_mu*self.mmean)**-1)[mask]
        min_fy = (((self.shape_mu-self.shape_std)*self.mmean)**-1)[mask]
        max_fy = (((self.shape_mu+self.shape_std)*self.mmean)**-1)[mask]
        nsteps = np.around(simps ( fy, fx)).astype(int)
        max_nsteps = np.around(simps ( min_fy, fx)).astype(int)
        min_nsteps = np.around(simps ( max_fy, fx)).astype(int)
        return (min_nsteps, nsteps, max_nsteps)
                      
