#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32
import cv2
import numpy as np

def find_diameter_circle(circle, pixel_to_cm_ratio):
    if circle is not None:
        x, y, radius = circle
        diameter_in_pixels = 2 * radius
        diameter_in_cm = diameter_in_pixels / pixel_to_cm_ratio
        return diameter_in_pixels, diameter_in_cm
    return 0, 0

# ROS Publisher Setup
rospy.init_node('diameter_publisher', anonymous=True)
pub = rospy.Publisher('/diameter', Float32, queue_size=10)
rate = rospy.Rate(1)  

# Load Camera Calibration Data
calibration_data = np.load('/home/mo/catkin_ws/src/first/scripts/camera_calibration.npz')
camera_matrix = calibration_data['camera_matrix']
dist_coeffs = calibration_data['dist_coeffs']

# Aruco Marker Setup
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters()

# Define HSV Color Range for Ripe Oranges
ripe_orange_lower = (10, 150, 150)
ripe_orange_upper = (25, 255, 255)

# Video Capture
cap = cv2.VideoCapture(2)
MARKER_SIZE_CM = 5.25

while not rospy.is_shutdown():
    ret, frame = cap.read()
    if not ret:
        rospy.logerr("Error: Could not read frame.")
        break

    frame = cv2.resize(frame, (640, 480))
    frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, camera_matrix)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Orange Detection
    mask = cv2.inRange(hsv_frame, np.array(ripe_orange_lower), np.array(ripe_orange_upper))
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    orange_position = None
    diameter_cm = 0

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        if radius > 10:
            orange_position = (x, y)

    # Aruco Marker Detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        marker_corners = corners[0][0]
        marker_width_pixels = np.linalg.norm(marker_corners[0] - marker_corners[1])
        pixel_to_cm_ratio = marker_width_pixels / MARKER_SIZE_CM

        if orange_position:
            _, diameter_cm = find_diameter_circle((x, y, radius), pixel_to_cm_ratio)
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)
            cv2.putText(frame, f"Diameter: {diameter_cm:.2f} cm", (int(x) - 50, int(y) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Publish the diameter
    pub.publish(diameter_cm)
    rospy.loginfo(f"Published Diameter: {diameter_cm}")

    # Display the Frame
    cv2.imshow("Detected Orange", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    rate.sleep()

cap.release()
cv2.destroyAllWindows()
