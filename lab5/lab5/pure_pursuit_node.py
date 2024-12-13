import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import math
import csv

LOOKAHEAD_DISTANCE = 2.5
WAYPOINTS_FILENAME = 'waypoints.csv'
WAYPOINTS_INTERVAL = 100

class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')
        
        # Topics
        self.drive_topic = '/drive'
        self.waypoints_marker_topic = '/waypoints_marker'
        self.target_marker_topic = '/target_marker'
        self.odom_topic = '/ego_racecar/odom'

        # Publishers and Subscribers
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        self.waypoints_marker_pub = self.create_publisher(Marker, self.waypoints_marker_topic, 10)
        self.target_marker_pub = self.create_publisher(Marker, self.target_marker_topic, 10)
        self.create_subscription(Odometry, self.odom_topic, self.pose_callback, 10)

        # Load waypoints
        self.load_waypoints(WAYPOINTS_FILENAME, WAYPOINTS_INTERVAL)

        # Initialize current steering angle
        self.current_steering_angle = 0.0  # 초기값 설정
    
    def load_waypoints(self, filename, interval=1):
        waypoints = []
        with open(filename, mode='r') as file:
            reader = csv.reader(file)
            for i, row in enumerate(reader):
                if i % interval == 0:
                    x, y, heading, speed = map(float, row)
                    waypoints.append([x, y])
        
        waypoints = np.array(waypoints)
        self.waypoints_x = waypoints[:, 0]
        self.waypoints_y = waypoints[:, 1]

    def pose_callback(self, msg):
        vehicle_x = msg.pose.pose.position.x
        vehicle_y = msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        _, _, vehicle_heading = self.quaternion_to_euler(orientation_q)

        # 조향 각도에 따라 Lookahead Distance 조정
        goal_x, goal_y = self.pick_goal_point(vehicle_x, vehicle_y, vehicle_heading)

        # 로컬 프레임으로 목표 지점 변환 및 조향 각도 계산
        steering_angle = self.calculate_steering_angle(vehicle_x, vehicle_y, vehicle_heading, goal_x, goal_y)
        
        # 조향 각도에 따라 속도 조절
        max_speed = 10.5
        min_speed = 1.0
        speed = max(min_speed, max_speed * (1 - abs(steering_angle) / 0.4))  # 조향 각도가 클수록 속도 감소

        # 드라이브 명령 퍼블리시
        self.publish_drive(speed, steering_angle)

        # 시각화를 위한 마커 퍼블리시
        self.target_x, self.target_y = goal_x, goal_y
        self.publish_markers()

    def pick_goal_point(self, vehicle_x, vehicle_y, vehicle_heading):
        base_lookahead = LOOKAHEAD_DISTANCE
        adjusted_lookahead = max(1.0, base_lookahead * (1 - abs(self.current_steering_angle) / 1.2))

        distances = np.sqrt((self.waypoints_x - vehicle_x) ** 2 + (self.waypoints_y - vehicle_y) ** 2)
        closest_index = np.argmin(distances)

        for j in range(closest_index, len(distances)):
            distance = distances[j]
            if distance >= adjusted_lookahead:
                if distance == adjusted_lookahead:
                    return self.waypoints_x[j], self.waypoints_y[j]
                else:
                    prev_x, prev_y = self.waypoints_x[j - 1], self.waypoints_y[j - 1]
                    next_x, next_y = self.waypoints_x[j], self.waypoints_y[j]
                    prev_dist = distances[j - 1]
                    ratio = (adjusted_lookahead - prev_dist) / (distance - prev_dist)
                    goal_x = prev_x + ratio * (next_x - prev_x)
                    goal_y = prev_y + ratio * (next_y - prev_y)
                    return goal_x, goal_y

        return self.waypoints_x[-1], self.waypoints_y[-1]

    def calculate_steering_angle(self, vehicle_x, vehicle_y, vehicle_heading, goal_x, goal_y):
        dx = goal_x - vehicle_x
        dy = goal_y - vehicle_y

        local_x = dx * math.cos(-vehicle_heading) - dy * math.sin(-vehicle_heading)
        local_y = dx * math.sin(-vehicle_heading) + dy * math.cos(-vehicle_heading)

        if local_y != 0:
            curvature = (2 * local_y) / (LOOKAHEAD_DISTANCE ** 2)
            steering_angle = curvature
        else:
            steering_angle = 0.0

        self.current_steering_angle = steering_angle  # 저장하여 Lookahead Distance 조정에 사용
        return steering_angle

    def publish_drive(self, speed, steering_angle):
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)

    def publish_markers(self):
        # Waypoints marker
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.color.a = 1.0
        marker.color.b = 1.0
        marker.points = [Point(x=x, y=y, z=0.0) for x, y in zip(self.waypoints_x, self.waypoints_y)]
        self.waypoints_marker_pub.publish(marker)

        # Target marker
        target_marker = Marker()
        target_marker.header.frame_id = "map"
        target_marker.type = Marker.POINTS
        target_marker.action = Marker.ADD
        target_marker.scale.x = 0.2
        target_marker.scale.y = 0.2
        target_marker.color.a = 1.0
        target_marker.color.r = 1.0
        target_marker.points = [Point(x=self.target_x, y=self.target_y, z=0.0)]
        self.target_marker_pub.publish(target_marker)

    def quaternion_to_euler(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return 0.0, 0.0, yaw

def main(args=None):
    rclpy.init(args=args)
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)
    pure_pursuit_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

