import cv2
import numpy as np
import sys
# You should replace these 3 lines with the output in calibration step
# #2560x1440
# DIM=(2560, 1440)
# K=np.array([[1368.1758729459004, 0.0, 1272.962393511171], [0.0, 1370.8905477876365, 552.881250920795], [0.0, 0.0, 1.0]])
# D=np.array([[-0.05419539520472076], [-0.0025170802184101026], [-0.00045405097305216467], [0.0002477061105225029]])

DIM=(848, 480)
K=np.array([[452.878040286532, 0.0, 421.3230658940741], [0.0, 453.9090121024413, 183.60284945447594], [0.0, 0.0, 1.0]])
D=np.array([[-0.05947126444569179], [0.009936290556980098], [-0.010987659982145698], [0.0042154629919721725]])

def undistort_advanced(balance=1.0, dim2=None, dim3=None):#img_path, balance=1.0, dim2=None, dim3=None):
    #img = cv2.imread(img_path)
    print("starting stream..")
    vcap = cv2.VideoCapture("rtsp://192.168.1.254")
    vcap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("stream started")
    dim1 = (848,480)#img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)

    while(1):
        ret, img = vcap.read()
        if img is None:continue
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        #undistorted_img = cv2.resize(undistorted_img, (848, 480), interpolation = cv2.INTER_AREA)
        #cv2.imwrite("undistorted_"+img_path, undistorted_img)
        cv2.imshow("undistorted", undistorted_img)
        cv2.waitKey(1)
    # cv2.destroyAllWindows()


# def undistort():
#     print("starting stream..")
#     vcap = cv2.VideoCapture("rtsp://192.168.1.254")
#     print("stream started")
#     while(1):
#         ret, img = vcap.read()
#         h,w = img.shape[:2]
#         map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
#         undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
#         cv2.imshow("undistorted", undistorted_img)
#         cv2.waitKey(1)

if __name__ == '__main__':
    undistort_advanced()
    #for p in sys.argv[1:]:
    #    undistort(p)