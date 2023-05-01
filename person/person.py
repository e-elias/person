#!/usr/bin/env python

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data, qos_profile_system_default

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
		hog = cv2.HOGDescriptor()
		hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
		gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
		boxes, weights = hog.detectMultiScale(gray, winStride=(8,8))
		boxes = np.array([[x, y, x+w, y+h] for (x, y, w, h) in boxes])
		for (xA, yA, xB, yB) in boxes:
			self.get_logger().info('Detecting Person')
			x_c, y_c = (xA + xB/2), (yA + yB/2)
			offset_x = x_c - 125
			theta = offset_x / 250
			angular = - 2 * theta
			self.drive(0.5, angular)
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
