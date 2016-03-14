from sklearn import lda
import pylab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import helper_functions as hf
import helper_function_LDA as hf_LDA
import numpy as np

def GENERATE(mean_vector , cov_matrix, no_pts=100,class_label=1):
    ''' Generates multivariate Gaussian data with given parameters
        INPUT:
            mean_vector : COLUMN mean_vector with 'd' rows
            cov_matrix : dxd size covariance matrix
            no_pts = no of points to generate

        OUTPUT:
            data : 2D - Numpy matrix of multivariate data
    '''
    data = np.random.multivariate_normal(mean_vector , cov_matrix , no_pts)
    class_parameters = (class_label,mean_vector,cov_matrix,no_pts)
    return data , class_parameters

def GENERATE_RANDOM_MULTIVARIATE(no_class=2 , no_features = 2 ,no_pts=100):
    ''' Generates multivariate Gaussian data with random parameters
        INPUT:
            no_class = Number of different classes
            no_pts = no of points to generate

        OUTPUT:
            data : 2D - Numpy matrix of multivariate data
            no_class : Number of classed generated
            no_features : Number of feature generated
            class_parameters : (label, mean , covariance , class priors )
    '''
    #create the total data storage matrix
    data = np.zeros((1,no_features+1))
    #create the class parametes array
    class_parameters = []
    for i in xrange(1,no_class+1):
        # genrerate current set
        mean = np.random.randint(-1000,1000 ,size = (no_features,))
        cov = np.random.randint(-1000,1000 ,size = (no_features ,no_features))
        data_gen = np.random.multivariate_normal(mean , cov , no_pts)
        label_gen = np.ones((no_pts,1) ,dtype=np.uint8)*int(i)
        #add the current set to the overall data
        data = np.vstack((data , np.hstack((label_gen , data_gen))))
        class_parameters.append(( int(i) ,mean[:,None] , cov , int(no_pts) )) # (label , mean , cov , class priors)
    #delete first row since its not a part of actual data points
    data = np.delete(data , 0,0)
    return data , no_class , no_features , class_parameters

def DISCRIMINANT_FUNCTION(class_parameters):
    ''' Finds the discriminant function for each class
        INPUT:
            class_parameters : List of Tuple of (label,mean,cov,class_priors)

        OUTPUT:
            List of tuple of (classlabel ,A, B ,C) of each class
            DISCRIMINANT FUNCTION FORMULA
            g(x) = x'Ax + Bx + C  where x is COLUMN vector and x' = ROW vector
    '''
    no_class = len(class_parameters)
    ABC_list = []
    total_no_points = sum([class_parameters[i][3] for i in range(len(class_parameters))])*1.0

    for i in xrange(no_class):
        # parametes for current class
        mean_vector = np.array(class_parameters[i][1])
        sigma = class_parameters[i][2]
        sig_inv = np.linalg.inv(sigma)
        current_class_prior = class_parameters[i][3]/total_no_points

        # DISCRIMINANT FUNCTION FORMULA
        # g(x) = x'Ax + Bx + C  where x is COLUMN vector and x' = ROW vector
        # A = 2x2 matrix
        # B = 1x2 matrix
        # C = 1x1 matrix

        A = -0.5*sig_inv
        B = np.dot(sig_inv,mean_vector).T
        C = -0.5*np.dot(np.dot(mean_vector.T,sig_inv),mean_vector) -0.5*np.log(np.linalg.det(sigma))+ np.log(current_class_prior)
        ABC_list.append((class_parameters[i][0],A,B,C))    #(classlabel ,A, B ,C)

    return ABC_list

# def FIND_PARAMETERS(data):
#     ''' Estimates the mean, covariance,class priors of a data matrix
#     '''
#     no_class = np.unique(data[:,0])
#     for i in xrang(no_class):
#         part = data[data[:,0]==np_class[i]]
#         mean = [ avg for cols in  ]
#
#         class_parameters.append(( int(i) ,mean , cov , int(no_pts) )) # (label , mean , cov , class priors)

def APPLY_DISCRIMINANT_FUNC(ABC_list , vector):
    ''' Applies all discriminat function to each vector sample
        INPUT:
            vector : row vector of Xbar
        OUTPUT:
            A tuple of values given by each function
    '''

    vector = vector[:,None]  #Add a new axis and make it column vector
    ans = []
    for i in range(len(ABC_list)):
        A,B,C = ABC_list[i][1] , ABC_list[i][2][:,None], ABC_list[i][3]
        ans.append(float(np.dot(np.dot(vector.T,A),vector) + np.dot(B[:,None].T,vector) + C))


    return float(ans.index(min(ans))+1)

def PREDICT_CLASS(funct_output_list):
    ''' Precidts the class label of the list

    '''
    pass



def main():

    #generate set1
    mean1 = np.array([100,120])
    cov1 = np.array([[10,0],[0,10]])
    data1, class1_para = GENERATE( mean1 , cov1 , class_label=1)
    # for i in range(data1.shape[0]):
    #     print str("0\t") + str(int(data1[i ,0])) + '\t' + str(int(data1[ i ,1]))

    #generate set2
    mean1 = np.array([80,180])
    cov1 = np.array([[10,0],[0,10]])
    data2,class2_para = GENERATE(mean1 , cov1,class_label=2)
    # for i in range(data1.shape[0]):
    #     print str("1\t")+ str(int(data1[i ,0])) + '\t' + str(int(data1[ i ,1]))

    X = np.vstack((data1,data2))
    y = np.ones_like(X)
    y[:100]= 1
    y[100:]= 2
    y = y[:,0]

    LDAS = lda.LDA()
    LDAS.fit(X,y)
    dcf = LDAS.decision_function(X)
    pred = LDAS.predict(X)
    print np.unique(pred) , pred.shape

    # print LDAS.coef_, LDAS.intercept_
    # print dcf[:5]
    ABC = DISCRIMINANT_FUNCTION((class1_para,class2_para))
    class_o =[]
    for vector in X:
        # print vector
        class_o.append(APPLY_DISCRIMINANT_FUNC(ABC, vector))
    class_o = np.array(class_o)
    # print np.unique(class_o) , np.array(class_o).shape
    print(class_o[(class_o != pred)])



    # print LDAS.de


    # data , no_class , no_features , class_parameters = GENERATE_RANDOM_MULTIVARIATE(no_class=3 , no_features =3, no_pts=250)
    # hf_LDA.PLOT_3D( data[ : , 0] ,data[ : , 1] , data[ : , 2] )
    # print data , class_parameters
    # print class1_para
    # print class2_para
    # for i in range(len(ABC)):
    #     for j in range(len(ABC[0])):
    #         print ABC[i][j]



if __name__ == '__main__':
    main()
