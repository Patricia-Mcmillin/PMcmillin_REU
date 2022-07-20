import sys, os
from icecube import icetray, dataclasses, dataio
from I3Tray import I3Tray
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math

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
    
#next 3 functions output halton sequence
#from tupui/halton.py on github

def primes_from_2_to(n):
    """Prime number from 2 to n.
    From `StackOverflow <https://stackoverflow.com/questions/2068372>`_.
    :param int n: sup bound with ``n >= 6``.
    :return: primes in 2 <= p < n.
    :rtype: list
    """
    sieve = np.ones(n // 3 + (n % 6 == 2), dtype=np.bool)
    for i in range(1, int(n ** 0.5) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[k * k // 3::2 * k] = False
            sieve[k * (k - 2 * (i & 1) + 4) // 3::2 * k] = False
    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]


def van_der_corput(n_sample, base=2):
    """Van der Corput sequence.
    :param int n_sample: number of element of the sequence.
    :param int base: base of the sequence.
    :return: sequence of Van der Corput.
    :rtype: list (n_samples,)
    """
    sequence = []
    for i in range(n_sample):
        n_th_number, denom = 0., 1.
        while i > 0:
            i, remainder = divmod(i, base)
            denom *= base
            n_th_number += remainder / denom
        sequence.append(n_th_number)

    return sequence


def halton(dim, n_sample):
    """Halton sequence.
    :param int dim: dimension
    :param int n_sample: number of samples.
    :return: sequence of Halton.
    :rtype: array_like (n_samples, n_features)
    """
    big_number = 10
    while 'Not enought primes':
        base = primes_from_2_to(big_number)[:dim]
        if len(base) == dim:
            break
        big_number += 1000

    # Generate a sample using a Van der Corput sequence per dimension.
    sample = [van_der_corput(n_sample + 1, dim) for dim in base]
    sample = np.stack(sample, axis=-1)[1:]
    
    return sample
#transforms a set of points from that sets center to a new center defined by M
""" coordinate_matrix: set of points to be transformed, of type [[x1 y1 1],...[xn yn 1]]
                       where n is the n_sample in next function
    M: transformation matrix, of type [[1 0 change_in_x],
                                      [0 1 change_in_y]]
    """
def point_transform(coordinate_matrix, M):
    points = []
    for point in coordinate_matrix:
        transform_point = np.dot(M, point)
        points = np.append(points, transform_point)
    return points

"""Halton sequence coordinates
    dim: dimension
    n_sample: number of samples
    s: scale factor 
    """
def halton_coords(dim, n_sample, s):
    #generate halton sequence as (x, y) points
    coords = halton(dim, n_sample)
    
    #scale the (x,y) points to desired size
    halton_coords = s*coords
    
    #insert a column of 1's 
    ones_array =np.ones((21,1))
    coord_matrix = np.insert(halton_coords, [2], ones_array, axis =1)
    #print('coord_matrix')
    #print(coord_matrix)
    #print(len(coord_matrix))
    
    #calculate center of halton_coords
    x, y = halton_coords.T
    center_x = sum(x)/len(x)
    center_y = sum(y)/len(y)
    #print("center")
    #print(center_x, center_y)
    
    #create transformation matrix to move points to be centered at (0,0)
    M = np.array([[1,0,-center_x],[0,1,-center_y]])
    #print('transformation matrix')
    #print(M)
    
    #translate halton_coords and format to be plotted 
    point_array = point_transform(coord_matrix, M)
    #print('point_array')
    #print(point_array)
    
    #splits into (x,y) points
    split = 2
    placeholder = [point_array[x:x+split] for x in range(0, len(point_array), split)]
    split_points = np.asarray(placeholder)
    #print('split_points')
    #print(split_points)
    
    xx, yy = split_points.T
    #print('xx, and  yy')
    #print(xx,yy)
    
    split_size = 67
    xx = np.repeat(xx, 67)
    yy = np.repeat(yy, 67)
    xx_1 = [xx[i:i+split_size] for i in range(0, len(xx), split_size)]
    yy_1 = [yy[i:i+split_size] for i in range(0, len(yy), split_size)]
    xx_2 = np.asarray(xx_1)
    yy_2 = np.asarray(yy_1)
    #print("xx_2")
    #print(xx_2)
    #print("yy_2")
    #print(yy_2)
    return xx_2, yy_2

x, y = halton_coords(2,21,907)

with open('halton_x_coordinates.txt', 'w') as my_file:
    for i in x:
        np.savetxt(my_file, i)


with open('halton_y_coordinates.txt', 'w') as my_file:
    for i in y:
        np.savetxt(my_file, i)
        
with open('coordinates_txt_files/gen2_z_coordinates2.txt', 'w') as my_file:
    for i in gen2_dom_z_positions:
        np.savetxt(my_file, i) 
        
print('array exported to file')
fig1 = plot_3d_icetop(x, y, gen2_dom_z_positions, "using the Halton sequence ")

fig2 = plot_2d_icetop(x, y, "using the Halton sequence ")

fig3 = plot_3d_no_icetop(x, y, gen2_dom_z_positions, "using the Halton sequence ")

fig4 = plot_2d_no_icetop(x, y, "using the Halton sequence ")
plt.show()



fig1.savefig('halton_3d_icetop.png')
fig2.savefig('halton_2d_icetop.png')
fig3.savefig('halton_3d_no_icetop.png')
fig4.savefig('halton_2d_no_icetop.png')