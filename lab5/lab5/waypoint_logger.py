#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import atexit
from os.path import expanduser
from time import gmtime, strftime
from numpy import linalg as LA
from tf_transformations import euler_from_quaternion
from nav_msgs.msg import Odometry

import os
file = open(os.path.join(os.getcwd(), strftime('wp-%Y-%m-%d-%H-%M-%S.csv', gmtime())), 'w')

class WaypointsLogger(Node):
    def __init__(self):
        super().__init__('waypoints_logger')
        self.subscription = self.create_subscription(
            Odometry,
            'ego_racecar/odom',
            self.save_waypoint,
            10)
        self.subscription  # prevent unused variable warning
        atexit.register(self.shutdown)
        self.get_logger().info('Saving waypoints...')

    def save_waypoint(self, data):
        quaternion = np.array([data.pose.pose.orientation.x,
                               data.pose.pose.orientation.y,
                               data.pose.pose.orientation.z,
                               data.pose.pose.orientation.w])

        euler = euler_from_quaternion(quaternion)
        speed = LA.norm(np.array([data.twist.twist.linear.x,
                                  data.twist.twist.linear.y,
                                  data.twist.twist.linear.z]), 2)
        
        if data.twist.twist.linear.x > 0.:
            self.get_logger().info(f'Speed X: {data.twist.twist.linear.x}')

        file.write('%f, %f, %f, %f\n' % (data.pose.pose.position.x,
                                         data.pose.pose.position.y,
                                         euler[2],
                                         speed))

    def shutdown(self):
        file.close()
        self.get_logger().info('Goodbye')

def main(args=None):
    rclpy.init(args=args)
    waypoints_logger = WaypointsLogger()

    try:
        rclpy.spin(waypoints_logger)
    except KeyboardInterrupt:
        pass
    finally:
        waypoints_logger.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
