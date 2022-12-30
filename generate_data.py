from tkinter import *
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
import sys
import math
import h5py
import os
import numpy as np
import matplotlib.pyplot as plt
#sys.path.append('../WorkingDirectory')
from functions_v6 import get_sigVecInRollCS_from_sigPhi, yld2000_derivative, yld2000_EqvStr , get_R_vs_Theta_Rb_values, get_UniaxYieldStress_vs_Theta_values, get_stress_points_in_Yieldlocus


class widgets(Frame):
    
    def __init__(self, window, picks=[]):
       
      Frame.__init__(self, window)
      
      self.inputs = []
      self.para = ['a2','a3','a4','a5','a6','a7','a8', 'e']
      lf = LabelFrame(window, text='Inputs')
      lf.grid(column=0, row=0)
      value_picks = [[1.0, 1.2, 3], [6,8,3]]
      
      label = Label(lf, text='a1', width=6)
      label.grid(row=0, column=0)      
      var = IntVar()
      var.set(1)
      self.a1 = Entry(lf, textvariable=var, width=6)
      self.a1.grid(row=0, column=1)
      
      label = Label(lf, text='Incre1', width=6)
      label.grid(row=0, column=2)      
      var = IntVar()
      var.set(3)
      self.incre1 = Entry(lf, textvariable=var, width=6)
      self.incre1.grid(row=0, column=3)
      
      label = Label(lf, text='Incre2', width=6)
      label.grid(row=0, column=4)      
      var = IntVar()
      var.set(4)
      self.incre2 = Entry(lf, textvariable=var, width=6)
      self.incre2.grid(row=0, column=5)
    
      for j, para in enumerate(self.para): 
          label = Label(lf, text=self.para[j], width=8)
          label.grid(row=1, column=j+3)
                    
          alpha_range  = []
          for i in range(len(picks)): 
              
              label = Label(lf, text=picks[i], width=6)
              label.grid(row=i+2, column=2)
                                
              var = IntVar()
              if para== 'e':
                  var.set(value_picks[1][i])
              else:                  
                  var.set(value_picks[0][i])
                           
              ent = Entry(lf, textvariable=var, width=8)
              ent.grid(row=i+2, column=j+3) 
              
              alpha_range.append(ent)
              
          self.inputs.append(alpha_range)
              
          Button(lf, text='Get the plot!', command=allstates).grid(row=12,column=1)
           
    def state(self): 
              
      a1 = float(self.a1.get())
      incre1 = float(self.incre1.get()) 
      incre2 = float(self.incre2.get()) 
      
      input_data = []
      for i, alpha_input in enumerate(self.inputs):
          alpha_input = list(map((lambda ent: float(ent.get())), alpha_input))
          n_step = alpha_input[-1]
          alpha_values = alpha_input[:-1]
          
          step_alpha = (alpha_values[1]-alpha_values[0]) / (n_step - 1)
          for j in range(int(n_step-2)): # 2 accounting for the first and the last points
                 
            alpha_values = np.insert(alpha_values, j+1, alpha_values[j]+step_alpha, axis=0)
            input_data.append(alpha_values)
              
      return input_data, a1, n_step, incre1, incre2
                   
def allstates():
     
    global count, canvas, fig, ax1, ax2, ax3, toolbar
    
    count +=1 
    
    if count != 1:
        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax4.cla()
        ax5.cla()
        ax6.cla()
        ax7.cla()
        ax8.cla()
            
    input_data, a1, n_step, incre1, incre2 = lng.state()  
     
    no_of_calcns = 0
    
    for a in input_data[7]: # this is the exponent of the yield locus
        for a2 in input_data[0]: 
            for a3 in input_data[1]:
                for a4 in input_data[2]:
                    for a5 in input_data[3]:
                        for a6 in input_data[4]:
                            for a7 in input_data[5]:
                                for a8 in input_data[6]:
                                    
                                    no_of_calcns += 1 #the first value of the index is always 1
                                    
                                    input_stress = []
                                    stress_points_in_yield_loci = []
                                    output_para = []
                                    
                                    alphaOrj = [a1,a2,a3,a4,a5,a6,a7,a8] 
                                    factor = ( (abs((2.0*alphaOrj[0]+alphaOrj[1])/3.0))**a + (abs((2.0*alphaOrj[2]-2.0*alphaOrj[3])/3.0))**a + (abs((4.0*alphaOrj[4]-alphaOrj[5])/3.0))**a  ) / 2.0
                                    factor = factor ** (1.0/a)
                                    alpha = [i/factor for i in alphaOrj]  # here the normalized values
                                    #print(alpha)
                                    #print(a)
                                                                        
                                    yl = get_stress_points_in_Yieldlocus(alpha,a,shear=0)                                         
                                    rad_dis_yl = np.sqrt( (yl[0][0][0]**2) +  (yl[0][1][0]**2))        
                                    stress_points_in_yield_loci.extend(rad_dis_yl[315:360:int(incre1)])  #set increment as input variable in GUI
                                    stress_points_in_yield_loci.extend(rad_dis_yl[0:136:int(incre1)])
                                    #print(np.shape(stress_points_in_yield_loci))
                                    input_stress.extend(stress_points_in_yield_loci) 
                                    
                                    stress = get_UniaxYieldStress_vs_Theta_values(alpha, a, Y_0=1.0, NormalizeAlpha=False)
                                    stress_in_roll_plane = stress[0][1:90:int(incre2)] #0 and 90 degree excluded beacuase they are already included above
                                  
                                    #print(np.shape(stress_in_roll_plane))
                                    input_stress.extend(stress_in_roll_plane) 
                                    
                                    output_para.extend(alphaOrj[1:])
                                    output_para.append(a)
                                    output_para = np.array(output_para)
                                    
                                    #print(np.shape(output_para))
                                    #print(np.shape(input_stress))
                                    
                                    if no_of_calcns ==1:
                                        directory = 'data/'
                                        filename =  f'yld2000_data_input={np.shape(input_stress)[0]}_output={np.shape(output_para)[0]}.h5'
                                        path = os.path.join(directory, filename)
                                        try:
                                            os.makedirs(directory, exist_ok = True)
                                            print("Directory '%s' created successfully" %directory)
                                        except OSError as error:
                                            print("Directory '%s' can not be created")
                                            
                                        F = h5py.File(path, 'a')
                                        group1 = F.require_group('input_data')  
                                        group2 = F.require_group('output_data') 
                                        
                                        print('Groups created!')
                                                                                                                            
                                    group1.require_dataset(f'd_{no_of_calcns}', data= input_stress, shape=np.shape(input_stress), dtype=np.float64, compression='gzip')
                                    group2.require_dataset(f'd_{no_of_calcns}', data= output_para, shape=np.shape(output_para), dtype=np.float64, compression='gzip')
                                    
                                    #break
    F.close()    
    print('Training Dataset Created!')                    
    print('Dataset Size', no_of_calcns)  
    print('filename', filename)               
           
    canvas.draw()
    canvas.get_tk_widget().grid(row = 10, column=0)

    
                                
if __name__ == '__main__':
             
    root = Tk() 
    root.title('GENERATE DATA!')
    root.geometry("1280x800")
    #root.geometry("500x500")
    
    count = 0
    
    content = ttk.Frame(root)
    frame1 = ttk.Frame(content, borderwidth=5, relief="ridge")
    frame3 = ttk.Frame(content, borderwidth=5, relief="ridge")
    #width and height here sets the dimension of the frame , width=1250, height=550
    content.grid(column=0, row=0)
    
    frame1.grid(column=0, row=0)
    frame3.grid(column=0, row=10, columnspan=70, rowspan=70)
    
    fig = plt.figure()
    fig.set_figheight(8) #sets the height and width of the white space inside frame3
    fig.set_figwidth(18)
         
    ax1 = plt.subplot2grid(shape=(2,4), loc=(0, 0), rowspan = 1, colspan = 1) #shape=Shape of grid in which to place axis
    ax2 = plt.subplot2grid(shape=(2,4), loc=(0, 1), rowspan = 1, colspan = 1)
    ax3 = plt.subplot2grid(shape=(2,4), loc=(0, 2), rowspan = 1, colspan = 1)
    ax4 = plt.subplot2grid(shape=(2,4), loc=(0, 3), rowspan = 1, colspan = 1)
    ax5 = plt.subplot2grid(shape=(2,4), loc=(1, 0), rowspan = 1, colspan = 1)
    ax6 = plt.subplot2grid(shape=(2,4), loc=(1, 1), rowspan = 1, colspan = 1)
    ax7 = plt.subplot2grid(shape=(2,4), loc=(1, 2), rowspan = 1, colspan = 1)
    ax8 = plt.subplot2grid(shape=(2,4), loc=(1, 3), rowspan = 1, colspan = 1)
    
    #fig.tight_layout()
    
    canvas = FigureCanvasTkAgg(fig, master = frame3)
    
    toolbarFrame = Frame(master=frame1)
    toolbarFrame.grid(row=25,column=0)
    toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)
    
    lng = widgets(frame1, ['min', 'max', 'NPoints'])
    
    root.mainloop()