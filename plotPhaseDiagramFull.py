import numpy as np                                                              
import matplotlib.pyplot as plt                                                 
from tkinter import filedialog                                                  
from mpl_toolkits.mplot3d import Axes3D                                         
from matplotlib.ticker import MaxNLocator                                       
                                                                                
filename = filedialog.askopenfilename()                                         
dat_name = filename                                                             
dat_all = np.load(dat_name, allow_pickle=True)                                  
dat = dat_all[:-4]                                                              
sol = dat_all[-2]                                                               
min_values = np.array(dat_all[-4])#[1:,:-15]                                    
min_values_diq = np.array(dat_all[-3])#[1:,:-15]                                
                                                                                
T_min, T_max = dat[14], dat[15]                                                 
mu_min, mu_max = dat[16], dat[17]                                               
N_T, N_mu = dat[18], dat[19]                                                    
T_array = np.linspace(T_min, T_max, N_T)#[1:]                                   
mu_array = np.linspace(mu_min, mu_max, N_mu)#[:-15]                             
mu_ax, T_ax = np.meshgrid(mu_array, T_array)                                    
print(T_array)                                                                  
print(mu_array)                                                                 
                                                                                
fig, ax1 = plt.subplots(nrows=1)                                                
levels = MaxNLocator(nbins=32).tick_values(min_values.min(),                    
                                           min_values.max())                    
CS = ax1.contourf(mu_ax, T_ax, min_values, levels=levels)                       
fig.colorbar(CS, ax=ax1)                                                        
ax1.set_xlabel('mu (MeV)')                                                      
ax1.set_ylabel('T (MeV)')                                                       
plt.title('Chiral condensate (MeV)')                                            
####                                                                            
fig, ax1 = plt.subplots(nrows=1)                                                
levels = MaxNLocator(nbins=32).tick_values(min_values_diq.min(),                
                                           min_values_diq.max())                
CS = ax1.contourf(mu_ax, T_ax, min_values_diq, levels=levels)                   
fig.colorbar(CS, ax=ax1)                                                        
ax1.set_xlabel('mu (MeV)')                                                      
ax1.set_ylabel('T (MeV)')                                                       
plt.title('Diquark condensate (MeV)')                                           
###                                                                             
fig = plt.figure()                                                              
ax = fig.gca(projection='3d')                                                   
ax.plot_wireframe(mu_ax, T_ax, min_values, color='red')                         
ax.plot_wireframe(mu_ax, T_ax, min_values_diq)                                  
ax.set_xlabel('mu (MeV)')                                                       
ax.set_ylabel('T (MeV)')                                                        
ax.set_zlabel('$\sigma$/|$\Delta$| (MeV)')                                      
plt.title('Full phase diagram')                                                 
plt.show() 
