import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import AutoMinorLocator

## This class simulates 2 ultrasonic transudcer seperated by a distance (integer multiple of half wavelength)
## to observe the nodes formed in between them. Each source is treated as a 5-point array to get source directionality.
## A sine wave is used as source.
## Code based on course - Computers, Waves, Simulations: A Practical Introduction to Numerical Methods
class UltraSonicTransducerSim:

    # If we need to test for different frequency, try varying distance_multiplier & x_size/z_size & total time
    # working conditions for 40Khz sine wave - distance_multiplier=7, x_size & z_size = 0.2 meters, runtime = 0.1, dx = 0.0004, array_dist = 10, src_f1=src_f2=40000
    # working conditions for 4Khz sine wave - distance_multiplier=7, x_size & z_size = 2.0 meters, runtime = 0.1, dx = 0.01, array_dist = 4.0, src_f1=src_f2=4000
    def __init__(self):
        # changable params
        self.distance_multiplier = 7.0
    
        ## wave speed params
        self.cmax = 348.5 # max speed in m/s in air
        self.c0 = 348.5 # speed in m/s in air
        
        
        
        ## grid params - start
        self.x_size = 0.2 # in meter ?
        self.z_size = 0.2 # in meter ?
        self.dx = 0.0004
        self.array_dist = int(10.0)
        
        self.nx = int(self.x_size / self.dx)      # grid points in x
        self.nz = int(self.z_size / self.dx)     # grid points in z
        ## grid params - end
        
        
        
        ## emitter params - start        
        # emitter frequency
        self.src_f1 = 40000 # in hz
        self.src_f2 = 40000 # in hz
        
        self.src_T1 = 1.0 / self.src_f1
        self.src_T2 = 1.0 / self.src_f2
        
        self.T = min(self.src_T1, self.src_T2)
        self.calc_dx = self.cmax * self.T / 2.0 # get grid distance required to capture wave
        print('dx is : ' + str(self.dx) + ' and it should be less than ' + str(self.calc_dx))
        
        if self.dx > self.calc_dx:
            raise Exception('Check dx values. dx : ' + str(self.dx) + ' but required is ' + str(self.calc_dx))
            
        # emitter positions
        diff_pos = (self.c0 / (2.0 * self.src_f1)) * self.distance_multiplier
        # offset = -(self.z_size / 2.0) + (20.0 * self.dx)
        offset = 0.0
        self.src_x1 = (self.x_size / 2.0) - (diff_pos / 2.0)
        self.src_z1 = offset + (self.z_size / 2.0)
        self.src_x2 = (self.x_size / 2.0) + (diff_pos / 2.0)
        self.src_z2 = offset + (self.z_size / 2.0)
        
        self.isx1 = int(self.src_x1 / self.dx)
        self.isz1 = int (self.src_z1 / self.dx)
        self.isx2 = int(self.src_x2 / self.dx)
        self.isz2 = int (self.src_z2 / self.dx)
        print('Distance between sources in meters :' + str(self.src_x2 - self.src_x1))
        
        if self.isx2 > self.nx or self.isx1 > self.nx:
            raise Exception('Check distance_multiplier value. Value is high and source is falling outside grid')
        ## emitter params - end
        
        
        
        ## time diff params - start
        self.courantNo = 0.4
        self.dt = self.courantNo * self.dx / self.cmax
        
        self.runtime = 0.1 # in seconds
        self.nt = int(self.runtime / self.dt)     # number of time steps
        ## time diff params - end
        
        
        
        ## computation params - start
        self.nop = 5    # length of differential operator
        
        # Initialize pressure at different time steps and the second
        # derivatives in each direction
        self.p = np.zeros((self.nz, self.nx))
        self.pold = np.zeros((self.nz, self.nx))
        self.pnew = np.zeros((self.nz, self.nx))
        self.pxx = np.zeros((self.nz, self.nx))
        self.pzz = np.zeros((self.nz, self.nx))
        
        # Initialize velocity model for homegenous medium
        self.c = np.zeros((self.nz, self.nx))
        self.c += self.c0
        ## computation params - end
        
        
        
        # Receiver value arrays
        self.irx = self.isx2 - self.isx1 + 1
        self.irz = self.isz1 # assume same z for both transmitters. Measuring along this axis
        self.seis = np.zeros((self.irx, self.nt))
        self.seisPlot = np.zeros((self.irx, self.irx))
        
        
        
        # emitter wave equation
        self.src1 = self.calcsrc(self.src_f1, 15.0, 0.0)
        self.src2 = self.calcsrc(self.src_f2, 15.0, 0.0)
        
    def calcsrc(self, freq, amp, phaseShift):
        lwb = 0
        upb = self.nt
        src = np.empty(self.nt)
        
        for it in range(lwb, upb):
            #src[it] = (-2 / tval) * ((it - phaseShift) * self.dt) * np.exp(-1.0 / tval ** 2 * ((it - phaseShift) * self.dt) ** 2)
            src[it] = amp * np.sin((2 * np.pi * freq * it * self.dt) + phaseShift)
        
        return src
    
    def run(self):
    
        # Plot preparation
        fig, ax = plt.subplots()
        fig_seis, ax_seis = plt.subplots()
        
        v1 = max([np.abs(self.src1.min()), np.abs(self.src1.max())])
        v2 = max([np.abs(self.src2.min()), np.abs(self.src2.max())])
        v = 3.0 * max(v1, v2)
        
        # Initialize animated plot
        self.image = plt.imshow(self.pnew, interpolation='nearest', animated=True,
                        vmin=-v, vmax=+v, cmap=plt.cm.RdBu)
        
        plt.text(self.isx1, self.isz1, 'o')
        plt.text(self.isx2, self.isz2, 'o')
        plt.colorbar()
        plt.xlabel('ix')
        plt.ylabel('iz')
        
        ani = animation.FuncAnimation(fig, self.update,  blit=False, frames = self.nt, interval = 0, repeat = False)
        self.line, = ax.plot(np.arange(1, self.irx + 1), self.seis[:, 0], color='red')
        ax.set_ylim(-3*v, 3*v)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(axis='x', which='both')
        ax.grid(axis='y')
        ani_seis = animation.FuncAnimation(fig_seis, self.update2,  blit=False, frames = self.nt, interval = 0, repeat = False)
        plt.show()
    
    def update2(self, it):
        self.line.set_ydata(self.seis[:, it])
        return self.line,
    
    def update(self, it):
    
        if it == self.nt - 1:
            print('done')
    
        # Time extrapolation
        if self.nop == 3:
            # get values before update
            # pzzLeftBeforeUpdate = np.copy(self.pzz[:, 1])
            # pzzRightBeforeUpdate = np.copy(self.pzz[:, self.nx - 2])
            # pxxUpBeforeUpdate = np.copy(self.pxx[1, :])
            # pzzDownBeforeUpdate = np.copy(self.pxx[self.nx - 2, :])
        
            # calculate partial derivatives, be careful around the boundaries
            for i in range(1, self.nx - 1):
                self.pzz[:, i] = self.p[:, i + 1] - 2 * self.p[:, i] + self.p[:, i - 1]
            for j in range(1, self.nz - 1):
                self.pxx[j, :] = self.p[j - 1, :] - 2 * self.p[j, :] + self.p[j + 1, :]
                
            # factor = (self.courantNo - 1) / (self.courantNo + 1)
            # factor2 = 1.0
            # self.pzz[:, 0] = factor2 * (pzzLeftBeforeUpdate + (factor * (self.pzz[:, 1] - self.pzz[:, 0])))
            # self.pzz[:, self.nx - 1] = factor2 * (pzzRightBeforeUpdate + (factor * (self.pzz[:, self.nx - 2] - self.pzz[:, self.nx - 1])))
            # self.pxx[0, :] = factor2 * (pxxUpBeforeUpdate + (factor * (self.pxx[1, :] - self.pxx[0, :])))
            # self.pxx[self.nx - 1, :] = factor2 * (pzzDownBeforeUpdate + (factor * (self.pxx[self.nx - 2, :] - self.pxx[self.nx - 1, :])))
            
        if self.nop == 5:
            # calculate partial derivatives, be careful around the boundaries
            for i in range(2, self.nx - 2):
                self.pzz[:, i] = -1./12*self.p[:,i+2]+4./3*self.p[:,i+1]-5./2*self.p[:,i]+4./3*self.p[:,i-1]-1./12*self.p[:,i-2]
            for j in range(2, self.nz - 2):
                self.pxx[j, :] = -1./12*self.p[j+2,:]+4./3*self.p[j+1,:]-5./2*self.p[j,:]+4./3*self.p[j-1,:]-1./12*self.p[j-2,:]
                
        self.pxx /= self.dx ** 2
        self.pzz /= self.dx ** 2
        
        # Time extrapolation
        self.pnew = (2 * self.p) - self.pold + ((self.dt ** 2) * (self.c ** 2) * (self.pxx + self.pzz))

        # Add source term at isx, isz - treat each source as an array to simulate directed beam
        self.pnew[self.isz1, self.isx1] = self.pnew[self.isz1, self.isx1] + self.src1[it]
        self.pnew[self.isz1 - self.array_dist, self.isx1] = self.pnew[self.isz1, self.isx1] + self.src1[it]
        self.pnew[self.isz1 - (2 * self.array_dist), self.isx1] = self.pnew[self.isz1, self.isx1] + self.src1[it]
        self.pnew[self.isz1 + self.array_dist, self.isx1] = self.pnew[self.isz1, self.isx1] + self.src1[it]
        self.pnew[self.isz1 + (2 * self.array_dist), self.isx1] = self.pnew[self.isz1, self.isx1] + self.src1[it]
        
        self.pnew[self.isz2, self.isx2] = self.pnew[self.isz2, self.isx2] + self.src2[it]
        self.pnew[self.isz2 - self.array_dist, self.isx2] = self.pnew[self.isz2, self.isx2] + self.src2[it]
        self.pnew[self.isz2 - (2 * self.array_dist), self.isx2] = self.pnew[self.isz2, self.isx2] + self.src2[it]
        self.pnew[self.isz2 + self.array_dist, self.isx2] = self.pnew[self.isz2, self.isx2] + self.src2[it]
        self.pnew[self.isz2 + (2 * self.array_dist), self.isx2] = self.pnew[self.isz2, self.isx2] + self.src2[it]
        
        self.pold, self.p = self.p, self.pnew
        
        # Save seismograms
        self.seis[:, it] = np.transpose(self.p[self.irz, self.isx1 : self.isx2 + 1])
        
        self.image.set_array(self.pnew)
            
        return self.pnew,
    
sim = UltraSonicTransducerSim()
sim.run()