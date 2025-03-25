#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float32
# from gpiozero import Servo
# from gpiozero.pins.pigpio import PiGPIOFactory
import time

# Servo Setup
# factory = PiGPIOFactory()
# servo = Servo(18, min_pulse_width=0.0005, max_pulse_width=0.0025, pin_factory=factory)

# Global Variables
last_update_time = 0

def diameter_callback(msg):
    global last_update_time
    diameter_cm = msg.data

    # Print received diameter for debugging
    rospy.loginfo(f"Received Diameter: {diameter_cm}")

    current_time = time.time()
    if current_time - last_update_time >= 10:  # 10 seconds delay between actions
        if 5 < diameter_cm < 13: 
            t = (-0.136 * diameter_cm) + 2.161
            rospy.loginfo("Orange detected! Rotating clockwise...")
            # servo.value = 0.6  # Full speed clockwise
            time.sleep(t)
        
            rospy.loginfo("Stopping...")
            # servo.value = 0  # Stop
            time.sleep(2)
        
            rospy.loginfo("Rotating counterclockwise...")
            # servo.value = -0.6  # Full speed counterclockwise
            time.sleep(t)
        
            rospy.loginfo("Stopping...")
            # servo.value = 0  # Stop
            time.sleep(2)
        else:
            rospy.loginfo("No valid orange detected. Stopping motor.")
            # servo.value = 0  # Ensure motor is stopped
            time.sleep(1)

        last_update_time = current_time

def diameter_subscriber():
    rospy.init_node('diameter_subscriber', anonymous=True)
    rospy.Subscriber('/diameter', Float32, diameter_callback)
    rospy.loginfo("Diameter subscriber node started.")
    rospy.spin()

if __name__ == '__main__':
    try:
        diameter_subscriber()
    except rospy.ROSInterruptException:
        pass
