#!/usr/bin/env python

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data, qos_profile_system_default
#from utils import ARUCO_DICT, aruco_display
import argparse
import sys


ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="type of ArUCo tag to detect")
args = vars(ap.parse_args())



class Camera(Node):
	
	def __init__(self):
		
		super().__init__("person_following")
		self.subscription = self.create_subscription(Image, '/color/preview/image', self.listener_callback, 10)
		self.subscription
		self.br = CvBridge()
		
		self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', qos_profile_system_default)
		
	def drive(self, linear_x, angular_z):
		msg = Twist()
		msg.angular.z = angular_z
		msg.linear.x = linear_x
		self.cmd_vel_pub.publish(msg)
		print(angular_z)
		print(linear_x)
		
	def listener_callback(self, data):
		#self.get_logger().info('Receiving video frame')
		current_frame = self.br.imgmsg_to_cv2(data)
		
		#arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])
		#arucoParams = cv2.aruco.DetectorParameters()
		
		arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
		arucoParams = cv2.aruco.DetectorParameters_create()
		gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
		corners, ids, rejected = cv2.aruco.detectMarkers(gray, arucoDict, parameters=arucoParams)
		
		#detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
		#(corners, ids, rejected) = detector.detectMarkers(gray)
		
		if len(corners) > 0:
			#self.drive(0.5, 0.0)
			for (xA, yA, xB, yB) in corners:
				x_c, y_c = (xA + xB/2), (yA + yB/2)
				offset_x = x_c - 125
				theta = offset_x / 250
				angular = - 2 * theta
				self.drive(0.5, angular)
				
		#hog = cv2.HOGDescriptor()
		#hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
		#gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
		#boxes, weights = hog.detectMultiScale(gray, winStride=(8,8))
		#boxes = np.array([[x, y, x+w, y+h] for (x, y, w, h) in boxes])
		#for (xA, yA, xB, yB) in boxes:
		#	self.get_logger().info('Detecting Person')
		#	x_c, y_c = (xA + xB/2), (yA + yB/2)
		#	offset_x = x_c - 125
		#	theta = offset_x / 250
		#	angular = - 2 * theta
		#	self.drive(0.5, angular)
			#cv2.rectangle(imCV, (xA, yA), (xB, yB),
			#	(0, 255, 0), 2)
		#cv2.waitKey(1)

def main(args=None):
	rclpy.init(args=args)
	node = Camera()
	rclpy.spin(node)
	rclpy.shutdown()


if __name__ == '__main__':
	main()


