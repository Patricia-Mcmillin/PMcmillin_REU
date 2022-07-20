import sys, os
from icecube import icetray, dataclasses, dataio
from I3Tray import I3Tray
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import random 

gcdFile = dataio.I3File('/cvmfs/icecube.opensciencegrid.org/data/GCD/GeoCalibDetectorStatus_2020.Run134142.Pass2_V0.i3.gz')
frame = gcdFile.pop_frame()

while not frame.Has('I3Geometry'):
    frame = gcdFile.pop_frame()
geometry = frame.Get('I3Geometry')
om_geometry = geometry.omgeo

dom_x_positions=np.zeros((87,67))
dom_y_positions=np.zeros((87,67))
dom_z_positions=np.zeros((87,67))

for om, geo_info in om_geometry:
    dom_x_positions[om[0],om[1]]=geo_info.position.x
    dom_y_positions[om[0],om[1]]=geo_info.position.y
    dom_z_positions[om[0],om[1]]=geo_info.position.z
    
#places the selected string numbers of x, y, z positions into a new array
gen2_dom_x_positions = np.concatenate((dom_x_positions[1:6:2], dom_x_positions[14:21:2], dom_x_positions[31:40:2], dom_x_positions[51:60:2], dom_x_positions[68:75:2]))
gen2_dom_y_positions = np.concatenate((dom_y_positions[1:6:2], dom_y_positions[14:21:2], dom_y_positions[31:40:2], dom_y_positions[51:60:2], dom_y_positions[68:75:2]))
gen2_dom_z_positions = np.concatenate((dom_z_positions[1:6:2], dom_z_positions[14:21:2], dom_z_positions[31:40:2], dom_z_positions[51:60:2], dom_z_positions[68:75:2]))

"""#used to set the range for random.uniform()
print(np.max(gen2_dom_x_positions))
print(np.min(gen2_dom_x_positions))
print(np.max(gen2_dom_y_positions))
print(np.min(gen2_dom_y_positions))
"""

""" low: low end of range for random number generator, inclusive
    high: high end of range for random number generator, exclusive
    coordinate_array: DOM positions array to be transformed to randomly generated positions by string
    split: the second dimension in a 2d array (x, split)
"""
#creates random x,y positions for each string
def uniform_random_string(low, high, coordinate_array, split):
    index = 0
    rand_uniform_positions = []
    for string_num in coordinate_array:
        string_num = random.uniform(low, high)
        rand_uniform_positions = np.insert(rand_uniform_positions,index, string_num)
        index = index + 1
        
    rand_uniform_positions = np.repeat(rand_uniform_positions, 67)
    rand_dom_list = [rand_uniform_positions[x:x+split] for x in range(0, len(rand_uniform_positions), split)]
    rand_uni_positions = np.asarray(rand_dom_list)
    return rand_uni_positions

#plots the IceCube detector area boundaries
def IC_bounds():
    x31_x75 = np.array([dom_x_positions[31][1],dom_x_positions[75][1]])
    y31_y75 = np.array([dom_y_positions[31][1],dom_y_positions[75][1]])
    
    x75_x78 = np.array([dom_x_positions[75][1],dom_x_positions[78][1]])
    y75_y78 = np.array([dom_y_positions[75][1],dom_y_positions[78][1]])
    
    x78_x72 = np.array([dom_x_positions[78][1],dom_x_positions[72][1]])
    y78_y72 = np.array([dom_y_positions[78][1],dom_y_positions[72][1]])
    
    x72_x74 = np.array([dom_x_positions[72][1],dom_x_positions[74][1]])
    y72_y74 = np.array([dom_y_positions[72][1],dom_y_positions[74][1]])
    
    x74_x50 = np.array([dom_x_positions[74][1],dom_x_positions[50][1]])
    y74_y50 = np.array([dom_y_positions[74][1],dom_y_positions[50][1]])
    
    x50_x6 = np.array([dom_x_positions[50][1],dom_x_positions[6][1]])
    y50_y6 = np.array([dom_y_positions[50][1],dom_y_positions[6][1]])
    
    x6_x1 = np.array([dom_x_positions[6][1],dom_x_positions[1][1]])
    y6_y1 = np.array([dom_y_positions[6][1],dom_y_positions[1][1]])
    
    x1_x31 = np.array([dom_x_positions[1][1],dom_x_positions[31][1]])
    y1_y31 = np.array([dom_y_positions[1][1],dom_y_positions[31][1]])
    
    plt.plot(x31_x75, y31_y75, color='red')
    plt.plot(x75_x78, y75_y78, color='red')
    plt.plot(x78_x72, y78_y72, color='red')
    plt.plot(x72_x74, y72_y74, color='red')
    plt.plot(x74_x50, y74_y50, color='red')
    plt.plot(x50_x6, y50_y6, color='red')
    plt.plot(x6_x1, y6_y1, color='red')
    plt.plot(x1_x31, y1_y31, color='red')
    

"""  x: choosen x cooordinates
     y: choosen y coordinates
     z: choosen z coordinates
     random_gen: random number generator used to create x,y coordinates
"""
def plot_3d_icetop(x, y, z, random_gen):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X-axis (m)')
    ax.set_ylabel('Y-axis (m)')
    ax.set_zlabel('Z-axis (m)')
    ax.set_title("3D Gen2 Randomized Geometry with IceTop \n" + random_gen) 
    ax.dist = 11
    ax.scatter3D(x, y, z, s =7, c='blue', depthshade=True)
    return fig
    
def plot_3d_no_icetop(x, y, z, random_gen):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X-axis (m)')
    ax.set_ylabel('Y-axis (m)')
    ax.set_zlabel('Z-axis (m)')
    ax.set_title("3D Gen2 Randomized Geometry without IceTop \n" + random_gen) 
    ax.dist = 11
    ax.scatter3D(x[:,0:61], y[:,0:61], z[:,0:61], s =7, c='green', depthshade=True)
    return fig

"""  x: choosen x cooordinates
     y: choosen y coordinates
     random_gen: random number generator used to create x,y coordinates
"""
def plot_2d_icetop(x, y, random_gen):
    fig = plt.figure()
    ax = plt.axes()
    ax.set_xlabel('X-axis (m)')
    ax.set_ylabel('Y-axis (m)')
    ax.set_title("2D Gen2 Randomized Geometry with IceTop \n" + random_gen) 
    IC_bounds()
    ax.scatter(x, y, s =7, c='blue')
    return fig
    
def plot_2d_no_icetop(x, y, random_gen):
    fig = plt.figure()
    ax = plt.axes()
    ax.set_xlabel('X-axis (m)')
    ax.set_ylabel('Y-axis (m)')
    ax.set_title("2D Gen2 Randomized Geometry without IceTop \n" + random_gen)
    IC_bounds()
    ax.scatter(x, y, s =7, c='green')
    return fig
    
rand_x_positions = uniform_random_string(-600., 550, gen2_dom_x_positions, 67)
rand_y_positions = uniform_random_string(-600., 550, gen2_dom_y_positions, 67)

with open('random_uniform_x_coordinates2.txt', 'w') as my_file:
    for i in rand_x_positions:
        np.savetxt(my_file, i)

with open('random_uniform_y_coordinates2.txt', 'w') as my_file:
    for i in rand_y_positions:
        np.savetxt(my_file, i)
        
with open('gen2_z_coordinates2.txt', 'w') as my_file:
    for i in gen2_dom_z_positions:
        np.savetxt(my_file, i) 
        
print('array exported to file')
fig1 = plot_3d_icetop(rand_x_positions, rand_y_positions, gen2_dom_z_positions, "using uniform distribution")

fig2 = plot_2d_icetop(rand_x_positions, rand_y_positions, "using uniform distribution")

fig3 = plot_3d_no_icetop(rand_x_positions, rand_y_positions, gen2_dom_z_positions, "using uniform distribution")

fig4 = plot_2d_no_icetop(rand_x_positions, rand_y_positions, "using uniform distribution")
plt.show()



fig1.savefig('uniform_3d_icetop.png')
fig2.savefig('uniform_2d_icetop.png')
fig3.savefig('uniform_3d_no_icetop.png')
fig4.savefig('uniform_2d_no_icetop.png')
