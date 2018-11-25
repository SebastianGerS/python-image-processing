import cv2
import numpy as np
import os, sys

def dummy(value):
    pass
identity_kernal = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
sharpen_kernal = np.array([[0, -1, 0], [-1 ,5, -1], [0, -1, 0]])
gaussian_kernal1 = cv2.getGaussianKernel(3,0)
gaussian_kernal2 = cv2.getGaussianKernel(5,0)
box_kernal = np.array([[1,1,1],[1,1,1],[1,1,1]], np.float32) / 9.0

kernals = [identity_kernal, sharpen_kernal, gaussian_kernal1, gaussian_kernal2, box_kernal]
dirPath = os.path.dirname(os.path.realpath(__file__))
print(dirPath)
original_img = cv2.imread(dirPath +'/dark-clouds-daylight-grass-552501.jpg')
original_img = cv2.resize(original_img, (0,0), fx=0.25, fy=0.25)
grayscale_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
img_matrix = original_img
grayscale_img_matrix = grayscale_img
rows, columns, channels = img_matrix.shape
gray_rows, gray_columns = grayscale_img_matrix.shape
cv2.namedWindow('app')
cv2.createTrackbar('contrast', 'app', 50, 100, dummy)
cv2.createTrackbar('brightness', 'app', 50, 100, dummy)
cv2.createTrackbar('filter', 'app', 0, len(kernals) -1, dummy)
cv2.createTrackbar('grayscale', 'app', 0 ,1, dummy)
value_to_set = ''
mod_img = original_img
count = 1
while True:
    grayscale = cv2.getTrackbarPos('grayscale', 'app')
    brightness = cv2.getTrackbarPos('brightness', 'app')
    contrast = cv2.getTrackbarPos('contrast', 'app')
    kernal_idx = cv2.getTrackbarPos('filter', 'app')

    settings = {'grayscale': grayscale, 'brightness': brightness, 'contrast': contrast, 'filter': kernal_idx}
    color_modified = cv2.filter2D(original_img, -1, kernals[kernal_idx])
    grayscale_modified = cv2.filter2D(grayscale_img, -1, kernals[kernal_idx])

    # for row in range(rows):
    #    for column in range(columns):
    #        img_matrix[row, column] = brightness - 50
    # for gray_row in range(gray_rows):
    #    for gray_column in range(gray_columns):
    #        grayscale_img_matrix[gray_row, gray_column] = brightness - 50

    key = cv2.waitKey(100)
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite('output-{}.jpeg'.format(count), mod_img)
        count += 1
    elif key == ord('g'):
        value_to_set = 'grayscale'
    elif key == ord('c'):
        value_to_set = 'contrast'
    elif key == ord('b'):
        value_to_set = 'brightness'
    elif key == ord('f'):
        value_to_set = 'filter'
    elif key == ord('-'):
        for settings_key in settings.keys():
            if settings_key == 'grayscale' and value_to_set == 'grayscale':
                    cv2.setTrackbarPos('grayscale', 'app', 0)
            if settings_key == 'filter' and value_to_set == 'filter':
                if settings[value_to_set] > 0:
                    cv2.setTrackbarPos('filter', 'app', settings[value_to_set] - 1)
                else:
                    cv2.setTrackbarPos('filter', 'app', 0)
            elif settings_key == value_to_set:
                cv2.setTrackbarPos(settings_key, 'app', settings[value_to_set] - 1)
    elif key == ord('+'):
        for settings_key in settings.keys():
            if settings_key == 'grayscale' and value_to_set == 'grayscale':
                cv2.setTrackbarPos('grayscale', 'app', 1)
            if settings_key == 'filter' and value_to_set == 'filter':
                if settings[value_to_set] < len(kernals) -1:
                    cv2.setTrackbarPos('filter', 'app', settings[value_to_set] + 1)
                else:
                    cv2.setTrackbarPos('filter', 'app', len(kernals) -1)
            elif settings_key == value_to_set:
                cv2.setTrackbarPos(settings_key, 'app', settings[value_to_set] + 1)
    if grayscale == 0:
        mod_img = cv2.addWeighted(color_modified, contrast/50, np.ones_like(color_modified), 0, brightness-50)

    else:
        mod_img = cv2.addWeighted(grayscale_modified, contrast/50, np.ones_like(grayscale_modified), 0, brightness-50)

    cv2.imshow('app',mod_img)
cv2.destroyAllWindows()