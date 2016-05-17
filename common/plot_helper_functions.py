from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


'''
    Library to plot data:
    ***In end, always write plt.show() to show the plots***

    It contains the following functions:
        <1> SCATTER_PLOT_3D : used to plot 3d scatter data
            Useage : SCATTER_PLOT_3D(X_values,Y_values,Z_values)

        <2> WIREFRAME_PLOT_3D : used to plot 3d wireframe data
            Useage :    x_grid , y_grid = np.meshgrid(X_values,Y_values)
                        Z_values = function of(x_grid,y_grid)
                        WIREFRAME_PLOT_3D(x_grid,y_grid,Z_values)

        <3> SURFACE_PLOT_3D : used to plot 3d surface_plot_3d data
            Useage :    x_grid , y_grid = np.meshgrid(X_values,Y_values)
                        Z_values = function of(x_grid,y_grid)
                        SURFACE_PLOT_3D(x_grid,y_grid,Z_values)

        <4> SCATTER_PLOT_2D : used to plot 2d scatter data - Y VALUES ARE NOT MANDATORY
            Useage :    SCATTER_PLOT_2D(X_values) or
                        SCATTER_PLOT_2D(X_values,Y_values)

        <5> LINE_PLOT_2D : used to plot 2d line of data - Y VALUES ARE NOT MANDATORY
            Useage :    LINE_PLOT_2D(X_values) or
                        LINE_PLOT_2D(X_values,Y_values)

        <6> HIST_PLOT_1D : used to plot histogram of 1D data
            Useage :    HIST_PLOT_1D(X_values)

        <6> HIST_PLOT_2D : used to plot 2D histogram of 2D data
            Useage :    HIST_PLOT_2D(X_values,Y_values)

'''


def SCATTER_PLOT_3D(x,y,z,figure_no=1,marker='o',color='b',xlabel="X-Axis",ylabel = "Y-Axis",zlabel="Z-Axis",title = '3D SCATTER PLOT', labs = 'Datapoint_1' ):
    ''' 3d scatter plot of data points'''
    fig = plt.figure(figure_no)
    ax = fig.add_subplot(111,projection='3d')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    scatter = ax.scatter(x,y,z,color=color,marker=marker, label = labs)
    return scatter , figure_no

def LINE_PLOT_3D(x,y,z,figure_no=1,marker='o',color='b',xlabel="X-Axis",ylabel = "Y-Axis",zlabel="Z-Axis",title = '3D LINE PLOT'):
    ''' 3d line plot of data points'''
    fig = plt.figure(figure_no)
    ax = fig.add_subplot(111,projection='3d')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    line_plot = ax.plot(x,y,z,color=color,marker=marker)
    return line_plot , figure_no

def WIREFRAME_PLOT_3D(x,y,z,figure_no=1,rstride=1,cstride=1,xlabel="X-Axis",ylabel = "Y-Axis",zlabel="Z-Axis" , title = '3D WIREFRAME PLOT'):
    ''' 3d wireframe plot of data'''
    fig = plt.figure(figure_no)
    ax = fig.add_subplot(111,projection='3d')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    wireplot = ax.plot_wireframe(x,y,z,rstride=rstride,cstride=cstride)
    return wireplot , figure_no

def SURFACE_PLOT_3D(x,y,z,figure_no=1,rstride=1,cstride=1,xlabel="X-Axis",ylabel = "Y-Axis",zlabel="Z-Axis",title = '3D SURFACE PLOT'):
    ''' 3d surface plot of data'''
    fig = plt.figure(figure_no)
    ax = fig.add_subplot(111,projection='3d')
    # X,Y = np.meshgrid(x,y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    surf = ax.plot_surface(x,y,z,rstride=rstride,cstride=cstride,cmap=cm.jet)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    return surf, figure_no

def CONTOUR_PLOT_3D(x,y,z,figure_no=1,cmap=cm.coolwarm,xlabel="X-Axis",ylabel = "Y-Axis",zlabel="Z-Axis",title='3D CONTOUR PLOT'):
    ''' 3d wireframe plot of data'''
    fig = plt.figure(figure_no)
    ax = fig.add_subplot(111,projection='3d')
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    cont = ax.plot_surface(x,y,z,cmap=cm.coolwarm)
    return cont, figure_no

def SCATTER_PLOT_2D(x,y=None,figure_no=1,marker='o',color='g',xlabel="X-Axis",ylabel = "Y-Axis",title = '2D SCATTER PLOT', labs = 'Datapoint_1' ):
    ''' 2d scatter plot of data--y values not necessary'''
    fig = plt.figure(figure_no)
    ax = ax = fig.add_subplot(111)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if y == None:
        y = np.arange(len(x))
        scatter = ax.scatter(y,x,marker=marker,color=color, label = labs )
        ax.set_ylabel(ylabel+'--your values are here')
        return scatter , figure_no
    else:
        scatter = ax.scatter(x,y,marker=marker,color=color, label = labs )
        return scatter, figure_no

# def SCATTER_PLOT_2D(x,y,figure_no=1,xlabel="X-Axis",ylabel = "Y-Axis",title = '2D SCATTER PLOT'):
#     ''' 2d scatter plot of data'''
#     plt.figure(figure_no)
#     plt.subplot(111)
#     plt.ylabel(ylabel)
#     plt.xlabel(xlabel)
#     plt.title(title)
#     scatter = plt.scatter(x,y)
#     return scatter

def LINE_PLOT_2D(x,y=None,figure_no=1,marker='o',color='b',xlabel="X-Axis",ylabel = "Y-Axis",title = '2D LINE PLOT'):
    ''' 2d line plot of data--y values not necessary'''
    fig = plt.figure(figure_no)
    ax = fig.add_subplot(111)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if y == None:
        y = np.arange(len(x))
        line = ax.plot(y,x,marker=marker,color=color)
        ax.set_ylabel(ylabel+'--your values are here')
        return line , figure_no
    else:
        line = ax.plot(x,y,marker=marker,color=color)
        return line , figure_no

# def HISTOGRAM_PLOT_1D(x,bins=20,figure_no=1,normed=False,color='b',xlabel="Data points",ylabel = "Counts",title = "1D HISTOGRAM"):
#     ''' Plots 1d histogram'''
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.set_ylabel(ylabel)
#     ax.set_xlabel(xlabel)
#     ax.set_title(title)
#     ax.grid(True)
#     n ,bins,patches = ax.hist(x,bins=bins,normed=normed,color=color)
#     if normed:
#         ax.set_title('NORMALISED '+title)
#         ax.set_ylabel('Percentage of occurance')
#     else:
#         ax.set_title(title)
#     hist = ax.plot(n)
#     return hist

def HIST_PLOT_1D(x,figure_no=1,n_bins=20,normed=False,color='b',xlabel="Data points",ylabel = "Counts",title = "1D HISTOGRAM"):
    ''' Plots 1d histogram'''
    fig = plt.figure(figure_no)
    ax = fig.add_subplot(111)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True)
    count = np.histogram(x,bins=np.size(x),normed=normed)
    if normed:
        ax.set_title('NORMALISED '+title)
        ax.set_ylabel('Percentage of occurance')
    else:
        ax.set_title(title)
    bars = ax.bar(x,count[0],color=color)
    return bars

def HIST_PLOT_2D(x,y,bins=20,figure_no=2,normed=False,xlabel="X-Data point",ylabel = "Y-Data point",title = "2D HISTOGRAM"):
    ''' plots 2d histogram'''
    plt.figure(figure_no)
    plt.subplot(111)
    H, xedges, yedges = np.histogram2d(x,y,bins=bins,normed=normed)
    H = np.rot90(H) # H needs to be rotated and flipped
    H = np.flipud(H)
    # Mask zeros
    Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
    plt.pcolormesh(xedges,yedges,Hmasked)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    cbar = plt.colorbar()
    if normed:
        plt.title('NORMALISED '+title)
        cbar.ax.set_ylabel('Percentage')
    else:
        plt.title(title)
        cbar.ax.set_ylabel('Counts')
    return H

# if __name__ == '__main__':
    # pass

    # # x= np.linspace(0,20000,12)
    # # # y= np.linspace(-200,20,12)
    # # # data = np.random.uniform(low=-3, high=3, size=1000)
    # # # z= np.linspace(-600,200,12)
    # X = np.linspace(-350,12310,7)
    # Y = np.linspace(-3550,310,7)
    # x,y = np.meshgrid(X,Y)
    # R = y**2 + x**2
    # # Z = np.sin(X)*2+np.tan(Y)+25
    # # # SCATTER_PLOT_3D(X,Y,Z)
    # # # LINE_PLOT_3D(X,Y,Z)
    # WIREFRAME_PLOT_3D(x,y,R)
    # SURFACE_PLOT_3D(x,y,R)
    # # # SCATTER_PLOT_2D(X*-1)
    # # # LINE_PLOT_2D(X,Y)
    # # # HIST_PLOT_1D(Y,normed=False,color='g')
    # plt.show()
