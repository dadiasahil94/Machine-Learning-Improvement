
'''

=======================
KALMAN FILTER
=======================

ONLY NEED  "F, Z ,H" TO CALCULATE/ confirm dimensions of all other parameters

    X == state VECTOR                                      --------|
    F == Prediction Matrix                                 --------|---- Fundamental requirement
    H == Prediction to Measurement Transformation Matrix   --------|
    Z == Observed Measurements VECTOR                      --------|
    P == Covariance matrix
    B == Control Matrix
    U == Control VECTOR
    Q == Noise of Covariance
    K == Kalman Gain
    R == Noise in Measurement
    NOTE : HERE  F`  indicates Transpose


    Prediction Equation:
        X = FX + BU
        P = FPF` + Q  <-------------|
                                    |
    Updation Equation:              |
        K = PH`(HpH` + R)^-1        |
        P = P - KHP                 |
        X = X + K(Z-HK)             ^
                                    |
            |                       |
            |                       |
            |--------> -------------|


    After updation Equation repeat Prediction Equation until convergance

    Dimensionality (Rows*Columns):
        X = M*1
        F = M*M
        H = L*M
        Z = L*1
----------------------------------------------------------------------
        P = M*M
        B = M*N
        U = N*1
        Q = M*M
        K = M*L
        R = L*L
'''

import numpy as np
from random import uniform
import sys
sys.path.insert(0,"/home/sahil/Documents/machine learning algorithm/common")
import helper_functions as hf
import plot_helper_functions as plhf
import matplotlib.pyplot as plt


class Kalman():

    parameters = {'F': None, \
                  'B':None,\
                  'U':None,\
                  'Q':None,
                  'x_start':None,\
                  'p_start':None,\
                  'X_PREDICT_':None, \
                  'P_PREDICT_':None, \
                  'H':None,\
                  'R':None,\
                  'Z':None,\
                  'KALMAN_GAIN_':None,\
                  'X_UPDATED_':None,\
                  'P_UPDATED_':None\
                  }

    data_points =[]
    def __init__(self, no_states, no_iteration = 1000 ):
        ''' Initialization of class'''
        self.no_states = no_states
        self.no_iteration = no_iteration

    def set_parameters(self , parametes_dictinary):
        ''' Set values for the parameters'''
        assert isinstance(parametes_dictinary , dict)
        for key in parametes_dictinary.keys():
            self.parameters[key] = np.array(parametes_dictinary[key])

            #correct for 2-Dimesnion
            if self.parameters[key].ndim <2 and self.parameters[key].ndim >=1 :
                self.parameters[key] = np.expand_dims(self.parameters[key] , 1)
            elif self.parameters[key].ndim <1 :
                self.parameters[key] = np.expand_dims(self.parameters[key] , 0)
                self.parameters[key] = np.expand_dims(self.parameters[key] , 1)
            print self.parameters[key].ndim , self.parameters[key].shape , key

    def get_parameters(self):
        ''' Returns parameters'''
        return self.parameters

    def check_dimensionality(self):
        ''' Checks dimesnionality of parameters'''
        M = self.parameters['F'].shape[0]
        L = self.parameters['H'].shape[0]
        N = self.parameters['B'].shape[1]

        #check the dimensions
        if (self.parameters['X_PREDICT_'].shape != (M,1)):
            print "Dimesnion of X (state vector) is not correct. It should be "+str(M)+str("*")+str("1")+". Currently it is " + str(self.parameters['X_PREDICT_'].shape)
        if (self.parameters['X_UPDATED_'].shape != (M,1)):
            print "Dimesnion of X (state vector) is not correct. It should be "+str(M)+str("*")+str("1")+". Currently it is " + str(self.parameters['X_UPDATED_'].shape)
        if (self.parameters['x_start'].shape != (M,1)):
            print "Dimesnion of X0 (state vector) is not correct. It should be "+str(M)+str("*")+str("1")+". Currently it is " + str(self.parameters['x_start'].shape)
        if (self.parameters['p_start'].shape != (M,M)):
            print "Dimesnion of p_start is not correct. It should be "+str(M)+str("*")+str(M)+" Currently it is " + str(self.parameters['p_start'].shape)
        if (self.parameters['P_UPDATED_'].shape != (M,M)):
            print "Dimesnion of P is not correct. It should be "+str(M)+str("*")+str(M)+" Currently it is " + str(self.parameters['P_UPDATED_'].shape)
        if (self.parameters['P_PREDICT_'].shape != (M,M)):
            print "Dimesnion of P is not correct. It should be "+str(M)+str("*")+str(M)+" Currently it is " + str(self.parameters['P_PREDICT_'].shape)
        if (self.parameters['F'].shape != (M,M)):
            print "Dimesnion of F is not correct. It should be "+str(M)+str("*")+str(N)+". Currently it is " + str(self.parameters['F'].shape)
        if (self.parameters['B'].shape != (M,N)):
            print "Dimesnion of B is not correct. It should be "+str(M)+str("*")+str(N)+". Currently it is " + str(self.parameters['B'].shape)
        if (self.parameters['U'].shape != (N,1)):
            print "Dimesnion of U is not correct. It should be "+str(N)+str("*")+str("1")+". Currently it is " + str(self.parameters['U'].shape)
        if (self.parameters['Q'].shape != (M,M)):
            print "Dimesnion of Q is not correct. It should be "+str(M)+str("*")+str(M)+". Currently it is " + str(self.parameters['Q'].shape)
        if (self.parameters['H'].shape != (L,M)):
            print "Dimesnion of H is not correct. It should be "+str(L)+str("*")+str(M)+". Currently it is " + str(self.parameters['H'].shape)
        if (self.parameters['KALMAN_GAIN_'].shape != (M,L)):
            print "Dimesnion of Kalmanin Gain is not correct. It should be "+str(M)+str("*")+str(L)+". Currently it is " + str(self.parameters['KALMAN_GAIN_'].shape)
        if (self.parameters['R'].shape != (L,L)):
            print "Dimesnion of R is not correct. It should be "+str(L)+str("*")+str(L)+". Currently it is " + str(self.parameters['R'].shape)
        if (self.parameters['Z'][0].shape != (self.no_states,1)):
            print "Dimesnion of Z is not correct. It should be "+str(L)+str("*")+str(1)+". Currently it is " + str(self.parameters['Z'][0].shape)

    def adjust_matrix(self):
        ''' Adjust the parameters before starting the calculations'''
        empty_keys = [key for key in self.parameters.keys() if self.parameters[key]==None]
        if ('B' in empty_keys):
            N = int(5)
        else:
            N = int(self.parameters['B'].shape[1])
        M = int(self.parameters['F'].shape[0])
        L = int(self.parameters['H'].shape[0])

        for key in empty_keys:
            # print empty_keys
            if key == 'x_start':
                self.parameters['x_start'] = np.zeros((M,1))
            if key == 'X_UPDATED_':
                self.parameters['X_UPDATED_'] = np.zeros((M,1))
            if key == 'X_PREDICT_':
                self.parameters['X_PREDICT_'] = np.zeros((M,1))
            if key == 'p_start':
                self.parameters['p_start'] = 1000 * np.identity((M)) #size is M*M
            if key == 'P_PREDICT_':
                self.parameters['P_PREDICT_'] = 1000 * np.identity((M)) #size is M*M
            if key == 'P_UPDATED_':
                self.parameters['P_UPDATED_'] = 1000 * np.identity((M)) #size is M*M
            if key == 'B':
                self.parameters['B'] = np.zeros((M,N))
            if key == 'U':
                self.parameters['U'] = np.zeros((N,1))
            if key == 'Q':
                self.parameters['Q'] = 0.05 * np.identity((M)) #size is M*M
            if key == 'R':
                self.parameters['R'] = 0.1 * np.identity((L)) #size is L*L
            if key == 'KALMAN_GAIN_':
                self.parameters['KALMAN_GAIN_'] = np.zeros((M,L))
            # empty_keys.remove(key)
        print "++++++++++++++++++++++++"

        self.check_dimensionality()

    def prediction_step(self):
        part1 = np.dot(self.parameters['F'],self.parameters['X_UPDATED_'])
        part2 = np.dot(self.parameters['B'] , self.parameters['U'])
        self.parameters['X_PREDICT_'] = part1+part2
        # print part1.shape,part2.shape

        part3 = np.dot(self.parameters['F'],np.dot(self.parameters['P_UPDATED_'],self.parameters['F'].T))
        part4 = self.parameters['Q']
        self.parameters['P_PREDICT_'] = part3 + part4
        # print part3.shape,part4.shape

    def updation_step(self,count):
        part5 = np.dot(self.parameters['P_PREDICT_'],self.parameters['H'].T)
        part6 = np.linalg.inv(np.dot(self.parameters['H'], part5) +self.parameters['R'])
        self.parameters['KALMAN_GAIN_'] = np.dot(part5,part6)
        # print part5.shape,part6.shape

        part7 = self.parameters['P_PREDICT_']
        part8 = np.dot(self.parameters['KALMAN_GAIN_'],np.dot(self.parameters['H'],self.parameters['P_PREDICT_']))
        self.parameters['P_UPDATED_'] = part7 - part8
        # print part7.shape,part8.shape

        part9 = self.parameters['X_PREDICT_']
        part10 = self.parameters['Z'][count,0] - np.dot(self.parameters['H'],self.parameters['KALMAN_GAIN_'])
        # print part9.shape,part10.shape
        part11 = np.dot(self.parameters['KALMAN_GAIN_'],part10)
        # print "Done"
        self.parameters['X_UPDATED_'] = part9 + part11

        print "-------------------------------------"

    def solve(self):
        count = 0
        while count<self.no_iteration-1:
            print count
            self.prediction_step()
            self.updation_step(count)
            print self.parameters['X_UPDATED_']
            self.data_points.append(self.parameters['X_UPDATED_'])
            count+=1
        return self.data_points

if __name__ == '__main__':

    mango = Kalman(1, no_iteration =30)
    F = np.array([1])
    H = np.array([1])
    # Z = np.array([0.39,0.50,0.48,0.29,0.25,0.32,0.34,0.48,0.41,0.45])
    Z1 = np.array([uniform(0, 1) for i in range(2500)])

    # Z = np.array([1.1,1,0.95,1.05,1.2,0.9,0.84,1.15,0.41,0.45])
    Z = [x*0.1 for x in range(30)]
    # Z = Z[:]+Z1[0:10]
    True_vals = [1 for i in range(len(Z))]
    # Z = np.array(Z)
    parameters = {'F':F, \
                  'H': H,\
                  'Z':Z,\
                  }
    mango.set_parameters(parameters)
    mango.adjust_matrix()
    # mango.set_parameters({'x_start':10})
    # mango.adjust_matrix()
    print mango.parameters['x_start']
    j = mango.solve()
    j = [float(x) for x in j]
    # print j
    m ,figure_no = plhf.LINE_PLOT_2D(j,figure_no=1)
    plhf.LINE_PLOT_2D(True_vals,figure_no=1,color='g')
    plhf.SCATTER_PLOT_2D(Z,figure_no=1,color='r')

    plt.show()
