import matplotlib.pyplot
import pylab
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def TRANSPOSE_1DMATRIX(matrix):
    "Returns Transpose of a vector"
    return (matrix[:, None]).T


def FEATURE_MATRIX_FROM_FILE(file_name):
    ''' Return the feature matrix from a  file '''

    #open the file
    opened_file = open(file_name)
    # Get Feature array
    line = opened_file.readline()
    Feature_array = np.zeros((1,len(line.strip().split())) , dtype=int)
    while True:
        try:
            line  = np.array( map(int  , opened_file.next().strip().split()) )
            line  = TRANSPOSE_1DMATRIX(line).T
            Feature_array = np.vstack( (Feature_array,line) )
        except (StopIteration):
            break
        except:
            raise Exception, e
    #delete first row since its not a part of actual data points
    Feature_array = np.delete(Feature_array , 0,0)
    return Feature_array


def PLOT_3D(X , Y , Z):
    ''' Plots the 3D data
        INPUT:
            X ,Y ,Z values
    '''
    try:
        fig = pylab.figure()
        ax = Axes3D(fig)
        ax.scatter(X ,Y ,Z, c='b', marker='o')
        pylab.show()
    except:
        try:
            PLOT_2D(X,Y)
        except:
            print "Data matrix has some error"


def PLOT_2D(X , Y ):
    ''' Plots the 3D data
        INPUT:
            X ,Y ,Z values
    '''

    plt.plot(X,Y)
    pylab.show()
