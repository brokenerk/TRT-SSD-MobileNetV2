import cv2
import os
import sys
import tensorflow as tf

__PATH_SAVED_MODEL = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/../../bin/pretrained_model/saved_model'

model = tf.compat.v2.saved_model.load(__PATH_SAVED_MODEL)
__model = model.signatures['serving_default']

__user = "admin"
__password = "Uno+dos3"
__ip = "169.254.47.87"

__uri = ("rtsp://{}:{}@{}").format(__user, __password, __ip)
__width = 1920
__height = 1080
__latency = 160

gst_str = ('rtspsrc location={} latency={} ! '
			'rtph264depay ! h264parse ! omxh264dec ! '
			'nvvidconv ! '
			'video/x-raw, width=(int){}, height=(int){}, '
			'format=(string)BGRx ! '
			'videoconvert ! appsink').format(__uri, __latency, __width, __height)

#__cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
__cap = cv2.VideoCapture('./../tests/videoSSD.mp4')
WINDOW_NAME = 'CameraDemo'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1920, 1080)	

while True:
	validated, image = __cap.read()

	width=image.shape[1]
	height=image.shape[0]

	# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
	input_tensor = tf.convert_to_tensor(image)
	# The model expects a batch of images, so add an axis with `tf.newaxis`.
	input_tensor = input_tensor[tf.newaxis,...]

	# Run inference
	output_dict = __model(input_tensor)

	output_dict['detection_scores'] = output_dict['detection_scores'][0]
	output_dict['detection_boxes'] = output_dict['detection_boxes'][0]

	count=0#Contador de numero de rostros
	border=(0,0,0,0)
	rostros=[]
	for score in output_dict['detection_scores']:
		"""
		Cuando score>= 0.5 se considera como cara
		"""
		if score >= 0.5:
			"""
			border: lista con coordenadas del rostro en formato (yMin, xMin, yMax, xmax)
			"""

			border = (int(output_dict['detection_boxes'][count][0]*height), int(output_dict['detection_boxes'][count][1]*width),
			int(output_dict['detection_boxes'][count][2]*height), int(output_dict['detection_boxes'][count][3]*width)) # yMin,xMin,yMax,xmax
			cv2.rectangle(image, (border [1], border [0]), (border [3], border [2]),(0,255,0),2)
		else:
			#print('Total de rostros {}'.format(count))
			#print('Rostro NOOO detectado')
			break
	cv2.imshow(WINDOW_NAME, image)

	if cv2.waitKey(1) & 0xFF == ord('q'):#Metodo para salir, oprimir la letra Q del teclado
		break

__cap.release()
