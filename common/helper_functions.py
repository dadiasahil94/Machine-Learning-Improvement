import random
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse


def IMAGE_TO_LIST(gimage):
    '''input:
        gimage = grayscale Image

        output:
        image_to_list = COLUMN VECTYOR numpy array of image converted to list
    '''
    image_to_list = gimage.copy().reshape(
        (gimage.shape[0] * gimage.shape[1], 3))  # convert to list for Kmeans processing
    return image_to_list


def LIST_TO_IMAGE(nplist, rows, columns):
    '''input:
        image_to_list = numpy COLUMN VECTOR
        rows = number of rows for image
        columns = number of columns for Image

        output:
        gimage = grayscale Image only
    '''
    gimage = np.zeros((rows, columns))
    # convert to list for Kmeans processing
    gimage = nplist.copy().reshape((rows, columns))
    return gimage


def RESIZE_IMAGE(image1, fx1, fy1):
    '''INPUT:
       imag1 = inpute image1
       fx1 = harozonat stretch
       fy1 = vertical stretch

       OUTPUT:
       im = resized image
    '''
    if(image1 != None):
        im = cv2.resize(image1, (0, 0), fx=fx1, fy=fy1)
        return im
    else:
        print("Image incorrect/Image is NONE")


def SOBEL(gImg):
    '''input:
       gImg = grayscale image

       output:
       GMag = numpy array of Magnitude of resultant X and Y gradient
       GMag_Norm = numpy array of Normalized Magnitude of resultant X and Y gradient
     '''
    scale = 1
    delta = 0
    ddepth = cv2.CV_32F
    # Computing the X- and Y-Gradients, using the Sobel kernel
    grad_x = cv2.Sobel(gImg, ddepth, 1, 0, ksize=3, scale=scale,
                       delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gImg, ddepth, 0, 1, ksize=3, scale=scale,
                       delta=delta, borderType=cv2.BORDER_DEFAULT)

    # Absolute Gradient for Display purposes --- Remove in future, not needed
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    # Gradient magnitude computation  --- Magnitude of the field --- Also for
    # display
    g1 = grad_x * grad_x
    g2 = grad_y * grad_y
    GMag = np.sqrt(g1 + g2)  # Actual magnitude of the gradient

    # Normalized gradient 0-255 scale, and 0-1 scale --- For Display
    mmax = GMag.max()
    GMag_Norm = np.uint8(GMag * 255.0 / mmax)  # Magnitude, for display

    return GMag, GMag_Norm


def INDIVIDUAL_TO_RGB(split_imagesr, split_imagesg, split_imagesb, parts):
    recom = np.dstack((split_imagesr[:, :, parts], split_imagesg[
                      :, :, parts], split_imagesb[:, :, parts]))
    return recom


def IMAGE_DIVIDE_HORIZONTAL(image, parts):
    """input:
       image = one channel of the image
       parts = number of horizonatal parts from bottom

       output:
       end_images : numpy array containing the cropped images
       parts : number of parts
    """
    # paramenters for dividing the image
    row = image.shape[0]
    col = image.shape[1]
    delta = int(row / parts)
    # equal_height = bimage.shape[0]/no_strips
    y = row
    end_images = np.empty((delta, col), dtype=int)
    cropped = image[y - delta: y,  0: col]
    end_images = cropped
    # actually divide the image
    for i in range(1, parts):
        y = y - delta
        cropped = image[y - delta:y, 0:col]
        # end_images[0] = lowest strip`
        end_images = np.dstack((end_images, cropped))
    return end_images


def PLOT_IMAGE(img, figure=1):
    '''input:
       img = any image
       figure *= figure number

       OUTPUT:
       shows image
    '''
    plt.figure(figure)
    plt.imshow(img)
    plt.axis("off")


def PLOT_IMAGE_CV(img,  wait_time=0, name="name"):
    '''input:
       img = any image
       name *= window name
       wait_time *= WAITTIME BEFORE CLOSNG

       OUTPUT:
       shows image in opencv form
    '''
    cv2.imshow(name, img)
    cv2.waitKey(wait_time) and 0xFF


def GINPUT_ROUTINE(image, selected_pointsrow, selected_pointscolumn):
    '''INPUT:
       image =  where points are supposed to be selected

       OUTPUT:
       image image with modified values
    '''
    PLOT_IMAGE(image)
    print ("Please selected 2 points")
    pts = plt.ginput(n=2, timeout=10)
    pts = np.array(pts)
    c = int(pts[0][0])
    r = int(pts[0][1])
    c1 = int(pts[1][0])
    r1 = int(pts[1][1])
    selected_pointsrow.append(r)
    selected_pointsrow.append(r1)
    selected_pointscolumn.append(c)
    selected_pointscolumn.append(c1)
    modified_pixeel_value = image[int(r)][int(c)]
    for row in range(r, r1):
        for col in range(c, c1):
            image[row][col] = modified_pixeel_value
            # plt.show()
    return image, selected_pointsrow, selected_pointscolumn


def COMPUTE_MEAN_RGB1(image):
    '''input:
       image = 2D image with any number of RGB channels

       output:
       mean = column matrix with 3 rows
    '''
    mean_array1 = np.zeros((image[0][0].shape), dtype=int)
    # Computation for mean
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            mean_array1 = mean_array1 + image[row][col]
    mean_array1 = mean_array1 / np.count_nonzero(image)  # change is here
    return mean_array1


def COMPUTE_MEAN_RGB(image):
    '''input:
       image = 2D image with any number of RGB channels

       output:
       mean = column matrix with 3 rows
    '''
    mean_array1 = np.zeros((image[0][0].shape), dtype=int)
    # Computation for mean
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            mean_array1 = mean_array1 + image[row][col]
    mean_array1 = mean_array1 / image.size
    return mean_array1


def TRANSPOSE_1DMATRIX(matrix):
    return (matrix[:, None])


def COMPUTE_VARIANCE_RGB1(image, mean_array_rgb):
    '''input:
       mean = column matrix
       image = 2D image

       output:
       variance = square numpy matrix of size of mean showing varinace
    '''
    variance = np.zeros((image[0][0].shape), dtype=int)
    # computation for variance
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            var = (image[row][col] - mean_array_rgb)
            variance = variance + (var * var[:, None])
    variance = variance / np.count_nonzero(image)   #change is here
    return variance


def COMPUTE_VARIANCE_RGB(image, mean_array_rgb):
    '''input:
       mean = column matrix
       image = 2D image

       output:
       variance = square numpy matrix of size of mean showing varinace
    '''
    variance = np.zeros((image[0][0].shape), dtype=int)
    # computation for variance
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            var = (image[row][col] - mean_array_rgb)
            variance = variance + (var * var[:, None])
    variance = variance / image.size
    return variance


def TRUE_COORDINATES(image, selected_pointsrow, selected_pointscolumn):
    '''INPUT:
       image : size to whihc you want to adjust
       selected_pointsrow = row points to adjust
       selected_pointscolumn = column points to adjust

       OUTPUT:
       adj_row = adjusted row coordinates
       adj_col = adjusted columns coordinates
    '''
    x1 = selected_pointscolumn
    x2 = selected_pointsrow
    partx = len(x2)
    height = image.shape[0] / partx
    c = 0
    for i in range(0, len(x2), 2):
        addition = c * height
        x2[i] = x2[i] + addition
        x2[i + 1] = x2[i + 1] + addition
        col1 = int(x1[i])
        col2 = int(x1[i + 1])
        row1 = int(x2[i])
        row2 = int(x2[i + 1])
        # print selected_pointsrow[i] , selected_pointsrow[i+1] , addition
        cv2.rectangle(image, (col1, row1), (col2, row2), (122, 90, 46), 10)
        c += 1
    return x2, x1  # x1 == adjusted cols , x2 == adjusted rows


def GET_DATA(val_list, image, list_to_image):
    '''INPUT:
       clt.image_ =  Labeled histogram value
       image = colored image
       list_to_image = image with 0,1,2 image

       OUTPUT:
       a , b, c = containing row , column , r ,g ,b values of each label
     '''
    imagez = image
    a = []
    b = []
    c = []
    # a = list containing (label , row , column , red , green , blue)
    for row in range(list_to_image.shape[0]):
        for col in range(list_to_image.shape[1]):
            if(list_to_image[row][col] == val_list[0]):
                a.append((val_list[0], row, col, imagez[row][col][
                         0], imagez[row][col][1], imagez[row][col][2]))
            elif(list_to_image[row][col] == val_list[1]):
                b.append((val_list[1], row, col, imagez[row][col][
                         0], imagez[row][col][1], imagez[row][col][2]))
            else:
                c.append((val_list[2], row, col, imagez[row][col][
                         0], imagez[row][col][1], imagez[row][col][2]))
    return a, b, c


def SEGMENT_STRIPS(list_to_image,  colored_strip, index):  # FOR REPRESENTATION PURPOSE ONLY
    '''INPUT:
       list_to_image = list_to_image image with image
       colored_strip = colored strip where segemtntation is showns
       index = number of nth label

       OUTPUT:
       colored1 = segmented color strip
       a1 = segmented label strip
       val_list[index] = label which is segmented
    '''
    val_list = np.unique(list_to_image)
    colored1 = colored_strip.copy()
    a1 = list_to_image.copy()
    for row in range(colored1.shape[0]):
        for col in range(colored1.shape[1]):
            if(a1[row][col] != val_list[index]):
                colored1[row][col][0] = 0
                colored1[row][col][1] = 0
                colored1[row][col][2] = 0
                a1[row][col] = 0
    return colored1, a1, val_list[index]


def RETURN_POINTS(list_to_image, val_list, current_strip_number, equal_height):
    A10 = []
    A11 = []
    A12 = []
    # RETURN GLOBAL coordinates
    # M10 = np.zeros((3,)) ; M11 = np.zeros((3,)) ; M12 = np.zeros((3,))
    for row in range(list_to_image.shape[0]):
        for col in range(list_to_image.shape[1]):
            if(list_to_image[row][col] == val_list[0]):
                A10.append([row + (equal_height * current_strip_number), col])
                # M10 = M10 + imagez[row][col]
            elif(list_to_image[row][col] == val_list[1]):
                A11.append([row + (equal_height * current_strip_number), col])
                # M11 = M11 + imagez[row][col]
            else:
                A12.append([row + (equal_height * current_strip_number), col])
                # M12 = M12 + imagez[row][col]
    # M10 = M10 / len(A10); M11 = M11 / len(A11); M12 = M12 / len(A12);
    return A10, A11, A12


def GLOBAL_COORDINATES(list_to_image, current_strip_number, no_strips, equal_height, image):
    a0 = []
    a1 = []
    a2 = []
    image_row = image.shape[0]
    image_col = image.shape[1]
    unique_label_list = np.unique(list_to_image)
    for row in range(list_to_image.shape[0]):
        for col in range(list_to_image.shape[1]):
            if(list_to_image[row][col] == unique_label_list[0]):
                absolute_row = (
                    no_strips - 1 - current_strip_number) * equal_height + row
                a0.append([absolute_row, col])
            elif(list_to_image[row][col] == unique_label_list[1]):
                absolute_row = (
                    no_strips - 1 - current_strip_number) * equal_height + row
                a1.append([absolute_row, col])
            else:
                absolute_row = (
                    no_strips - 1 - current_strip_number) * equal_height + row
                a2.append([absolute_row, col])
    return a0, a1, a2


def MEAN_FROM_POINTS(list_of_coordinates, imagez, list_of_coordinates1, imagez1):
    # mean of points in list 1
    mean = np.zeros((3,))
    for point in list_of_coordinates:
        try:
            row = point[0]
            col = point[1]
            # print point , imagez[row][col] , mean
            mean = mean + imagez[row][col]
        except IndexError:
            continue
    # mean = mean/len(list_of_coordinates)
    # mean of points in list 2
    # mean1 = np.zeros((3,))
    for point1 in list_of_coordinates1:
        try:
            row = point1[0]
            col = point1[1]
            mean = mean + imagez1[row][col]
        except IndexError:
            continue
    mean = mean / (len(list_of_coordinates) + len(list_of_coordinates1))

    # resultant mean
    #   M1*N1+M2*N2
    #   -----------
    #      N1+N2
    # numerator = (mean*len(list_of_coordinates)) + (mean1 *len(list_of_coordinates1))
    # denominator = len(list_of_coordinates) + len(list_of_coordinates1)
    # return numerator/denominator
    return mean


def VARIANCE_FROM_POINTS(list_of_coordinates, imagez, list_of_coordinates1, imagez1, mean):
    variance = np.zeros((imagez[0][0].shape), dtype=int)
    # computation for variance
    for point in list_of_coordinates:
        try:
            row = point[0]
            col = point[1]
            var = imagez[row][col] - mean
            variance = variance + (var * var[:, None])
        except IndexError:
            continue
    for point1 in list_of_coordinates1:
        try:
            row = point1[0]
            col = point1[1]
            var = imagez1[row][col] - mean
            variance = variance + (var * var[:, None])
        except IndexError:
            continue
    variance = variance / (len(list_of_coordinates) +
                           len(list_of_coordinates1))
    val, vec = np.linalg.eig(variance)
    return np.linalg.det(variance), np.linalg.eig(variance)


def ADJUST_THE_POINTS_WRT_TO_IMAGE(first_points, second_points, height_of_each_strip, partnumber):
    addition_height = int(height_of_each_strip)
    for i in range(len(first_points)):
        first_points[i][0] = first_points[i][0] + \
            ((partnumber - 1) * addition_height)
    for j in range(len(second_points)):
        second_points[j][0] = second_points[j][
            0] + ((partnumber) * addition_height)
    return first_points, second_points


def HIGHLIGHT_ON_IMAGE(first, second_point, imagez):
    color = np.ones((3,))
    color[0] = 106
    color[2] = 198
    color[1] = 100
    for point in first:
        try:
            row = point[0]
            col = point[1]
            imagez[row][col] = color
        except IndexError:
            continue
    if(first != second_point):
        for point in second_point:
            try:
                row = point[0]
                col = point[1]
                imagez[row][col] = color
            except IndexError:
                continue
    return imagez


def MEAN_FROM_POINT(list_of_coordinates, imagez):
    # mean of points in list 1
    mean = np.zeros((3,))
    for point in list_of_coordinates:
        try:
            row = point[0]
            col = point[1]
            # print point , imagez[row][col] , mean
            mean = mean + imagez[row][col]
        except IndexError:
            continue
    mean = mean / len(list_of_coordinates)
    return mean


def VARIANCE_AND_EIGEN_FROM_POINT(list_of_coordinates, imagez, mean):
    variance = np.zeros((imagez[0][0].shape), dtype=int)
    # computation for variance
    for point in list_of_coordinates:
        try:
            row = point[0]
            col = point[1]
            var = imagez[row][col] - mean
            variance = variance + (var * var[:, None])
        except IndexError:
            continue
    variance = variance / (len(list_of_coordinates) - 1)
    val, vec = np.linalg.eig(variance)
    determinant = np.linalg.det(variance)
    # print val
    return determinant, val, vec


def CO_VALUES(eigenvector, col):
    arr = np.zeros((3, 1))
    for row in range(eigenvector.shape[0]):
        arr[row][1] = eigenvector[row][col]
    return arr


def compute_var(dominant_points, b0, b1, b2, bimage):
    # TRY AND CONBINE BY LEAST VARIANCE
    MA0B0 = MEAN_FROM_POINTS(dominant_points, bimage, b0, bimage)
    MA0B1 = MEAN_FROM_POINTS(dominant_points, bimage, b1, bimage)
    MA0B2 = MEAN_FROM_POINTS(dominant_points, bimage, b2, bimage)
    vara0b0, eig1 = VARIANCE_FROM_POINTS(
        dominant_points, bimage, b0, bimage, MA0B0)
    vara0b1, eig2 = VARIANCE_FROM_POINTS(
        dominant_points, bimage, b1, bimage, MA0B1)
    vara0b2, eig3 = VARIANCE_FROM_POINTS(
        dominant_points, bimage, b2, bimage, MA0B2)
    variance_list1 = [vara0b0, vara0b1, vara0b2]
    min_variance_index = variance_list1.index(min(variance_list1))
    points_list = [b0, b1, b2]
    return variance_list1, min_variance_index, points_list, points_list[min_variance_index], dominant_points


def check_merge(dominant_points, probable_points, bimage):
    # print (dominant_points==probable_points)
    final_list_of_points = []
    mean_of_dominant = MEAN_FROM_POINT(dominant_points, bimage)
    covariance_det, eigenvals, eigenmatrix = VARIANCE_AND_EIGEN_FROM_POINT(
        dominant_points, bimage, mean_of_dominant)
    # print covariance_det
    # print eigenvals
    # print eigenmatrix
    # mean of points from least variance selection
    mean_cx = MEAN_FROM_POINT(probable_points, bimage)
    std_deviation = math.sqrt(eigenvals[0] + eigenvals[1] + eigenvals[2])
    deciding_mean = []
    for time in range(3):
        eigen_vector = TRANSPOSE_1DMATRIX(eigenmatrix[:, time])
        subtracted_mean = TRANSPOSE_1DMATRIX(mean_of_dominant - mean_cx).T
        # print subtracted_mean.shape , eigen_vector.shape
        prdt = np.dot(subtracted_mean, eigen_vector)

        # print a
        deciding_mean.append(float(prdt))
    a1 = deciding_mean[0] * deciding_mean[0]
    b1 = deciding_mean[1] * deciding_mean[1]
    c1 = deciding_mean[2] * deciding_mean[2]
    distance_clust = math.sqrt(a1 + b1 + c1)
    # print distance_clust
    if (distance_clust - (1 * std_deviation) < 0):
        final_list_of_points = final_list_of_points + dominant_points
        dominant_points = probable_points
        return 1, probable_points
    else:
        print("ROAD HAS ENDED")
        # current_failed_points = hf.HIGHLIGHT_ON_IMAGE(probable_points , probable_points , bimage)
        # break
        return 0, dominant_points


def NORMALISE_MATRIX(matrix1):
    return ((matrix1 / 1.0 * np.max(matrix1)))


def HISTOGRAM(image, percent=True):
    numimage = np.arange(0, len(np.unique(image)) + 1)
    (hist, _) = np.histogram(image, bins=numimage)
    hist = hist.astype("float")
    if(percent == True):
        hist /= hist.sum()
        return hist
    else:
        return hist


def HISTOGRAM_ARRAY(array):
    plt.hist(array)
    # plt.show()


def GENERATE_RANDOM_INTS(n_ints = 1 , min_val = -1000000, max_val = 1000000):
    y = []
    for x in range(n_ints):
        y.append(random.randint(min_val,max_val))
    return y

#### THIS FILE CONTAINS ALL THE HELPER FUNTIONS TO DO SMALL TASK#####

def GENERATE_RANDOM_INTS(n_pts=1, n_dim=1, min_val=-100, max_val=100):
    ''' Returns the numpy array of random numbers
        INPUT :
        n_ints = no of rando integers required
        min_val = minium starting value
        max_val = maximum starting value

        OUTPUT:
        numpy array of random numbers
    '''
    if min_val > max_val:
        print ("ERROR! ERROR! in function GENERATE_RANDOM_INTS: \nMin value is greater than Max value. The differece between them should be atleast 1")
        print ("Exiting the program")
        sys.exit(-1)
    elif min_val == max_val:
        print ("ERROR! ERROR! in function GENERATE_RANDOM_INTS: \nMin value is equal to Max value. The differece between them should be atleast 1")
        print ("Exiting the program")
        sys.exit(-1)
    else:
        final_array = np.ones((n_pts, n_dim))
        for m in range(n_dim):
            y = []
            for x in range(n_pts):
                y.append(random.randint(min_val, max_val))
            final_array[:, m] = np.array(y)
        return final_array
