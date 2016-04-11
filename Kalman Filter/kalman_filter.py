import random
import time
import sys
sys.path.insert(0, '/home/sahil/Documents/ML_improvement/LDA')
import helper_functions as hf
import numpy as np

class Kalman:
    ''' Object for creating kalman filter'''

    kalman_count=0
    parameters = {'F':'', \
                  'B':'',\
                  'U':'',\
                  'Q':'',
                  'x_start':'',\
                  'p_start':'',\
                  'X_PREDICT_':'', \
                  'P_PREDICT_':'', \
                  'H':'',\
                  'R':'',\
                  'Z':'',\
                  'KALMAN_GAIN_':'',\
                  'X_UPDATED_':'',\
                  'P_UPDATED_':''\
                  }

    def __init__(self, n_state):
        ''' Class constructor '''
        self.n_state = np.zeros((n_state,1))
        Kalman.kalman_count+=1

    def set_Parameters(self,parameter_dict):
        ''' Sets parameters of the class '''
        assert isinstance(parameter_dict,dict)
        for key in parameter_dict.keys():
            # print  np.array(parameter_dict[key])
            Kalman.parameters[key] = np.array(parameter_dict[key])
            if Kalman.parameters[key].ndim ==1:
                Kalman.parameters[key][:, None]
            # print Kalman.parameters[key].ndim

    def print_parameters(self):
        ''' Prints parameters of the class '''
        for key in self.parameters.keys():
            if self.parameters[key] !='':
                print str(key)  ,'\t----- ',self.parameters[key]

    def solve(self,measurements_array, control_array):
        print "iter_no",' ','X_PREDICT_',' ' ,'P_PREDICT_',' ','KALMAN_GAIN_',' ','X_UPDATED_',' ','P_UPDATED_'
        # print ('{} {.3f} {.3f} {.3f} {.3f} {.3f}' .format(iter_no , self.parameters['X_PREDICT_'] ,self.parameters['P_PREDICT_'],self.parameters['KALMAN_GAIN_'] ,self.parameters['X_UPDATED_'],self.parameters['P_UPDATED_']) )
        ''' solve '''
        iter_no = 1
        measurements_array = np.array(measurements_array)
        control_array = np.array(control_array)
        if measurements_array.ndim == 1:
            measurements_array = measurements_array[:,None]
        if control_array.ndim ==1:
            control_array = control_array[:,None]
            # For starting, set updated_values == start_values
        self.parameters['X_UPDATED_'] = self.parameters['x_start']
        self.parameters['P_UPDATED_'] = self.parameters['p_start']

        B = self.parameters['B']
        F = self.parameters['F']
        H = self.parameters['H']
        H_trans = H.T
        R = self.parameters['R']
        F_trans = F.T
        while iter_no<=25:
            #-----------PREDICTION STEP------------#
            x_prev = self.parameters['X_UPDATED_'] ; p_prev = self.parameters['P_UPDATED_']
            self.parameters['X_PREDICT_'] = np.dot(F,x_prev) + np.dot( B,control_array[iter_no-1, : ] )         #F(X_old) + BU
            self.parameters['P_PREDICT_'] = np.dot( F,np.dot(p_prev,F_trans)) + self.parameters['Q']            #F(P_old)F' + Q

            #---------UPDATUION STEP-----------#
            x_prdt = self.parameters['X_PREDICT_'] ; p_prdt = self.parameters['P_PREDICT_']
            mid1 = np.dot(p_prdt , H_trans)
            mid2 = np.dot( H, np.dot(p_prdt,H_trans)) + R
            if mid2.ndim ==1 or mid2.shape==():
                mid2 = 1/mid2
            else:
                mid2 = np.linalg.inv(mid2)
            self.parameters['KALMAN_GAIN_'] = np.dot(mid1,mid2)                                                 #(P_prdt)H ((HPH'+R))^-1
            self.parameters['X_UPDATED_'] = self.parameters['X_PREDICT_'] +  np.dot(self.parameters['KALMAN_GAIN_'], (measurements_array[iter_no-1 , :] - np.dot(H,self.parameters['X_PREDICT_'])))
            self.parameters['P_UPDATED_'] = p_prdt - np.dot(self.parameters['KALMAN_GAIN_'],np.dot(H,p_prdt))

            # print iter_no , self.parameters['X_PREDICT_'] ,self.parameters['P_PREDICT_'],self.parameters['KALMAN_GAIN_'] ,self.parameters['X_UPDATED_'],self.parameters['P_UPDATED_']
            self.show_result(iter_no)
            # print('{0:.3f}'.format((self.parameters['X_PREDICT_'])))
            iter_no+=1
            # time.sleep(3)


    def show_result(self,iter_no):
        # cos = ['X_PREDICT_' ,'P_PREDICT_','KALMAN_GAIN_','X_UPDATED_','P_UPDATED_']
        cos = ["iter_no",'X_PREDICT_' ,'P_PREDICT_','KALMAN_GAIN_','X_UPDATED_','P_UPDATED_']
        # print (iter_no) ,
        for key in cos:
            try:
                print self.parameters[key] ,
            except:
                print iter_no ,
        print "\n"
Kalman = Kalman(1)
parameters = {'F':1, \
              'B':0,\
              'U':0,\
              'Q':0.001,\
              'x_start':0,\
              'p_start':1000,\
              'X_PREDICT_':0, \
              'P_PREDICT_':0, \
              'H':1,\
              'R':0.1,\
              'Z':'',\
              'KALMAN_GAIN_':0,\
              'X_UPDATED_':0,\
              'P_UPDATED_':0\
              }

Kalman.set_Parameters(parameters)
# Kalman.print_parameters()
measurements_array = [0.9,0.8,1.1,1,0.95,1.05,1.2,0.9,0.85,1.15]
# random = random.randrange(0,3)
for i in range(100):
    measurements_array.append(random.randrange(0,3))
control_array = np.zeros_like(measurements_array)
Kalman.solve(measurements_array,control_array)
