from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import seaborn as sns


class PlotEnv:
    
    def __init__(self,ref_path,c):
        sns.set_theme()
        plt.ion()
        self.fig = plt.figure()
        self.ax = plt.axes()
        self.ax.set_xlim(-40, 40)
        self.ax.set_ylim(-5, 40)
        self.ax.plot(ref_path[:,0],ref_path[:,1],'g',linewidth=3)
   
        self.xdata, self.ydata = [], []
        self.ln, = self.ax.plot([], [], c,linewidth=3)
        self.xdata_r, self.ydata_r = [], []
        self.ln_r, = self.ax.plot([], [], 'b',linewidth=1)
        self.xdata_o, self.ydata_o = [], []
        self.ln_o, = self.ax.plot([], [], 'k',linewidth=3)

    def step(self, x, r, o):
        self.xdata.append(x[0])
        self.ydata.append(x[1])
        
        self.xdata_r = r[:,:,0]
        self.ydata_r = r[:,:,1]  

        self.xdata_o = o[:,0].T
        self.ydata_o = o[:,1].T  

        # self.ax.plot(r[:,:,0].T,r[:,:,1].T,'b')
        # self.ax.plot(o[:,0].T,o[:,1].T,'k')

        self.ln.set_data(self.xdata, self.ydata)
        self.ln_r.set_data(self.xdata_r, self.ydata_r)
        self.ln_o.set_data(self.xdata_o, self.ydata_o)

        # for i in range(self.xdata_r.shape[0]):
            # self.ln_r.set_data(self.xdata_r[i,:], self.ydata_r[i,:])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class PlotEnv3D:
    
    def __init__(self,ref_path,c):
        sns.set_theme()
        plt.ion()
        self.fig = plt.figure()
        self.ax = plt.axes(projection='3d')
        self.ax.set_xlim(-40, 40)
        self.ax.set_ylim(-5, 40)
        self.ax.set_zlim(-5, 40)
        self.ax.plot3D(ref_path[:,0],ref_path[:,1],ref_path[:,1],'g',linewidth=3)
   
        self.xdata, self.ydata, self.zdata = [], [], []
        self.ln, = self.ax.plot3D([], [], [], c, linewidth=3)
        self.xdata_r, self.ydata_r, self.zdata_r = [], [], []
        self.ln_r, = self.ax.plot3D([], [], [], 'b',linewidth=3)
        self.xdata_o, self.ydata_o, self.zdata_o = [], [], []
        self.ln_o, = self.ax.plot3D([], [], [], 'k',linewidth=3)

    def step(self, x, r, o):
        self.xdata.append(x[0])
        self.ydata.append(x[1])
        self.zdata.append(x[2])
        
        self.xdata_r = r[:,:,0]
        self.ydata_r = r[:,:,1] 
        self.zdata_r = r[:,:,2]   

        self.xdata_o = o[:,0].T
        self.ydata_o = o[:,1].T  
        self.zdata_o = o[:,2].T  

        self.ln.set_data(self.xdata, self.ydata)
        self.ln.set_3d_properties(self.zdata)
       
        self.ln_o.set_data(self.xdata_o, self.ydata_o)
        self.ln_o.set_3d_properties(self.zdata_o)

        for i in range(self.xdata_r.shape[0]):
            self.ln_r.set_data(self.xdata_r[i,:], self.ydata_r[i,:])
            self.ln_r.set_3d_properties(self.zdata_r[i,:])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()