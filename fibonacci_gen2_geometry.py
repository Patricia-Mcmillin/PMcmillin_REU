import sys, os
from icecube import icetray, dataclasses, dataio
from I3Tray import I3Tray
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

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
    
#plotting functions to create 2d and 3d plots of IC gen2 geometry, both with and without icetop   
"""  x: choosen x cooordinates
     y: choosen y coordinates
     z: choosen z coordinates
     sequence: sequence used to create x,y coordinates
"""
def plot_3d_icetop(x, y, z, sequence):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X-axis (m)')
    ax.set_ylabel('Y-axis (m)')
    ax.set_zlabel('Z-axis (m)')
    ax.set_title("3D Gen2 Randomized Geometry with IceTop \n" + sequence) 
    ax.dist = 11
    ax.scatter3D(x, y, z, s =7, c='blue', depthshade=True)
    return fig
    
def plot_3d_no_icetop(x, y, z, sequence):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X-axis (m)')
    ax.set_ylabel('Y-axis (m)')
    ax.set_zlabel('Z-axis (m)')
    ax.set_title("3D Gen2 Randomized Geometry without IceTop \n" + sequence) 
    ax.dist = 11
    ax.scatter3D(x[:,0:61], y[:,0:61], z[:,0:61], s =7, c='green', depthshade=True)
    return fig

"""  x: choosen x cooordinates
     y: choosen y coordinates
     sequence: sequence used to create x,y coordinates
"""
def plot_2d_icetop(x, y, sequence):
    fig = plt.figure()
    ax = plt.axes()
    ax.set_xlabel('X-axis (m)')
    ax.set_ylabel('Y-axis (m)')
    ax.set_title("2D Gen2 Randomized Geometry with IceTop \n" + sequence) 
    IC_bounds()
    ax.scatter(x, y, s =7, c='blue')
    return fig
    
def plot_2d_no_icetop(x, y, sequence):
    fig = plt.figure()
    ax = plt.axes()
    ax.set_xlabel('X-axis (m)')
    ax.set_ylabel('Y-axis (m)')
    ax.set_title("2D Gen2 Randomized Geometry without IceTop \n" + sequence)
    IC_bounds()
    ax.scatter(x, y, s =7, c='green')
    return fig
    
#method 1 to create sunflower geometry using Luca Sabbatini's pattern from 2013/2014 slides 
"""  start_n: initial value of n
     stop_n: end value of n
     s: scale factor to define string spacing
"""

def sunflower_coords(start_n, stop_n, s):
    #golden ratio
    g = (1 + math.sqrt(5))/2
    
    r_positions = []
    for n in range(start_n,stop_n):
        r = s*((n)**(1/2))
        r_positions = np.append(r_positions, r)
    #print('this is r_positions')
    #print(r_positions)
    
    #convert (r,theta) to (x, y)
    x_points = []
    y_points = []
    t = 0
    t_add = ((2*math.pi)/(g**2))
    #print("initial t")
    #print(t)
    for r in r_positions:
        x = r*(math.cos(t))
        y = r*(math.sin(t))
        t = t + t_add
        #print(t)
        x_points = np.append(x_points, x)
        y_points = np.append(y_points, y)
        
    split_size = 67
    x_points = np.repeat(x_points, 67)
    y_points = np.repeat(y_points, 67)
    xx_1 = [x_points[i:i+split_size] for i in range(0, len(x_points), split_size)]
    yy_1 = [y_points[i:i+split_size] for i in range(0, len(y_points), split_size)]
    xx_2 = np.asarray(xx_1)
    yy_2 = np.asarray(yy_1)
    #print("x points")
    #print(x_points)
    #print('y points')
    #print(y_points)
    
    return xx_2, yy_2

#method 2 to create sunflower pattern
# chromoSpirals.py
# ----------------
# Code written by Peter Derlien, University of Sheffield, March 2013
# Draws spiralling patterns of circles using the Golden Angle.
# ----------------
"""  n_strings: number of strings to create coordinates for
     theta_initial: where to start count of theta (eg. 0 to start at 0 degrees) 
     s: scale factor to define string spacing
"""

def chromo_spiral_coords(n_strings, theta_initial, s):
    
    tau=(1+5**0.5)/2.0 # golden ratio approx = 1.618033989
    #(2-tau)*2*np.pi is golden angle = c. 2.39996323 radians, or c. 137.5 degrees
    inc = (2-tau)*2*np.pi
    drad=(1+5**0.5)/8.0 # radius of each disc

    # now collect in list 'patches' the locations of all the discs
    x_pos = []
    y_pos =[]
    for j in range(1,n_strings+1):
        r = j**0.5
        theta_initial += inc
        x = r*np.cos(theta_initial)
        y = r*np.sin(theta_initial)
        x_pos = np.insert(x_pos, j-1, x)
        y_pos = np.insert(y_pos, j-1, y)
        
    x_coords = s*x_pos
    y_coords = s*y_pos
    
    split_size = 67
    x_coords = np.repeat(x_coords, 67)
    y_coords = np.repeat(y_coords, 67)
    xx_1 = [x_coords[i:i+split_size] for i in range(0, len(x_coords), split_size)]
    yy_1 = [y_coords[i:i+split_size] for i in range(0, len(y_coords), split_size)]
    xx_2 = np.asarray(xx_1)
    yy_2 = np.asarray(yy_1)
    #print("x positions")
    #print(xx_2)
    #print("y positions")
    #print(yy_2)
    
    return xx_2, yy_2
    

x , y = sunflower_coords(0,21,120)
xx, yy = chromo_spiral_coords(21, 0, 120)

with open('fibonacci_m1_x_coordinates.txt', 'w') as my_file:
    for i in x:
        np.savetxt(my_file, i)

with open('fibonacci_m1_y_coordinates.txt', 'w') as my_file:
    for i in y:
        np.savetxt(my_file, i)
        
with open('fibonacci_m2_x_coordinates.txt', 'w') as my_file:
    for i in xx:
        np.savetxt(my_file, i)

with open('fibonacci_m2_y_coordinates.txt', 'w') as my_file:
    for i in yy:
        np.savetxt(my_file, i)

with open('gen2_z_coordinates2.txt', 'w') as my_file:
    for i in gen2_dom_z_positions:
        np.savetxt(my_file, i) 
        
print('arrays exported to file')

fig1 = plot_3d_icetop(x, y, gen2_dom_z_positions, "using the Fibonacci sequence method 1")

fig2 = plot_2d_icetop(x, y, "using the Fibonacci sequence method 1")

fig3 = plot_3d_no_icetop(x, y, gen2_dom_z_positions, "using the Fibonacci sequence method 1")

fig4 = plot_2d_no_icetop(x, y, "using the Fibonacci sequence method 1")

fig5 = plot_3d_icetop(xx, yy, gen2_dom_z_positions, "using the Fibonacci sequence method 2")

fig6 = plot_2d_icetop(xx, yy, "using the Fibonacci sequence method 2")

fig7 = plot_3d_no_icetop(xx, yy, gen2_dom_z_positions, "using the Fibonacci sequence method 2")

fig8 = plot_2d_no_icetop(xx, yy, "using the Fibonacci sequence method 2")
plt.show()



fig1.savefig('fibonacci_3d_icetop_m1.png')
fig2.savefig('fibonacci_2d_icetop_m1.png')
fig3.savefig('fibonacci_3d_no_icetop_m1.png')
fig4.savefig('fibonacci_2d_no_icetop_m1.png')
fig5.savefig('fibonacci_3d_icetop_m2.png')
fig6.savefig('fibonacci_2d_icetop_m2.png')
fig7.savefig('fibonacci_3d_no_icetop_m2.png')
fig8.savefig('fibonacci_2d_no_icetop_m2.png')
