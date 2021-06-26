import cv2, queue, threading, time
import numpy as np

DIM=(848, 480)
K=np.array([[452.878040286532, 0.0, 421.3230658940741], [0.0, 453.9090121024413, 183.60284945447594], [0.0, 0.0, 1.0]])
D=np.array([[-0.05947126444569179], [0.009936290556980098], [-0.010987659982145698], [0.0042154629919721725]])

# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()

cap = VideoCapture("rtsp://192.168.1.254")

dim1 = (848,480)  #dim1 is the dimension of input image to un-distort
assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
dim2 = dim1
dim3 = dim1
scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=1.0)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)

while True:
  #time.sleep(.5)   # simulate time between events
  frame = cap.read()

  undistorted_img = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
  cv2.imshow("undistorted", undistorted_img)
  if chr(cv2.waitKey(1)&255) == 'q':
    break