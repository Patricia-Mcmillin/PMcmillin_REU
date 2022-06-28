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


""" coordinate_array: DOM positions array to be transformed to randomly generated positions by string
    split: the second dimension in a 2d array (x, split)
"""
#creates random x,y positions for each string
def normalvariate_random_string(coordinate_array, split):
    index = 0
    rand_normalvariate_positions = []
    for string_num in coordinate_array:
        string_num = random.normalvariate(np.average(coordinate_array), np.std(coordinate_array))
        rand_normalvariate_positions = np.insert(rand_normalvariate_positions,index, string_num)
        index = index + 1
        
    rand_normalvariate_positions = np.repeat(rand_normalvariate_positions, 67)    
    random_list = [rand_normalvariate_positions[x:x+split] for x in range(0, len(rand_normalvariate_positions), split)]
    rand_norm_positions = np.asarray(random_list)
    return rand_norm_positions

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
    ax.scatter(x, y, s =7, c='blue')
    return fig
    
def plot_2d_no_icetop(x, y, random_gen):
    fig = plt.figure()
    ax = plt.axes()
    ax.set_xlabel('X-axis (m)')
    ax.set_ylabel('Y-axis (m)')
    ax.set_title("2D Gen2 Randomized Geometry without IceTop \n" + random_gen) 
    ax.scatter(x, y, s =7, c='green')
    return fig

rand_x_positions = normalvariate_random_string(gen2_dom_x_positions, 67)
rand_y_positions = normalvariate_random_string(gen2_dom_y_positions, 67)

fig1 = plot_3d_icetop(rand_x_positions, rand_y_positions, gen2_dom_z_positions, "using normal distribution")

fig2 = plot_2d_icetop(rand_x_positions, rand_y_positions, "using normal distribution")

fig3 = plot_3d_no_icetop(rand_x_positions, rand_y_positions, gen2_dom_z_positions, "using normal distribution")

fig4 = plot_2d_no_icetop(rand_x_positions, rand_y_positions, "using normal distribution")
plt.show()


fig1.savefig('normal_3d_icetop.png')
fig2.savefig('normal_2d_icetop.png')
fig3.savefig('normal_3d_no_icetop.png')
fig4.savefig('normal_2d_no_icetop.png')
