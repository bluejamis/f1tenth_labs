import rclpy
from rclpy.node import Node
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import math
import csv
import random

LOOKAHEAD_DISTANCE = 2.5
WAYPOINTS_FILENAME = 'waypoints.csv'
WAYPOINTS_INTERVAL = 100
MAX_ITERATIONS = 3000  # 더 많은 반복으로 목표 지점 탐색 가능성 증가
STEP_SIZE = 1.0
LIDAR_RANGE = 7.0  # 라이다 범위 설정
GRID_RESOLUTION = 0.3 # 점유 그리드 셀 크기
GRID_SIZE = 100  # 점유 그리드 크기 (셀 단위)
GOAL_SAMPLE_PROB = 0.23  # 목표 지점을 직접 샘플링할 확률 감소

class PurePursuitRRT(Node):
    def __init__(self):
        super().__init__('pure_pursuit_rrt_node')
        
        # Topics
        self.lidarscan_topic = '/scan'
        self.drive_topic = '/drive'
        self.waypoints_marker_topic = '/waypoints_marker'
        self.target_marker_topic = '/target_marker'
        self.odom_topic = '/odom'
        self.rrt_marker_topic = '/rrt_marker'

        # Publishers and Subscribers
        self.lidar_sub = self.create_subscription(LaserScan, self.lidarscan_topic, self.lidar_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        self.waypoints_marker_pub = self.create_publisher(Marker, self.waypoints_marker_topic, 10)
        self.target_marker_pub = self.create_publisher(Marker, self.target_marker_topic, 10)
        self.rrt_marker_pub = self.create_publisher(Marker, self.rrt_marker_topic, 10)
        self.create_subscription(Odometry, self.odom_topic, self.pose_callback, 10)

        # Load waypoints
        self.load_waypoints(WAYPOINTS_FILENAME, WAYPOINTS_INTERVAL)

        # Initialize variables
        self.current_steering_angle = 0.0
        self.vehicle_x = 0.0
        self.vehicle_y = 0.0
        self.vehicle_heading = 0.0
        self.scan_data = None
        self.occupancy_grid = np.zeros((int(GRID_SIZE), int(GRID_SIZE)))
        self.target_x = 0.0
        self.target_y = 0.0

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
        self.vehicle_x = msg.pose.pose.position.x
        self.vehicle_y = msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        _, _, self.vehicle_heading = self.quaternion_to_euler(orientation_q)

        # 목표 지점 선택
        self.target_x, self.target_y = self.pick_goal_point(self.vehicle_x, self.vehicle_y, self.vehicle_heading)

        # RRT*로 로컬 플래닝 수행
        path = self.rrt_star(self.vehicle_x, self.vehicle_y, self.target_x, self.target_y)
        if path:
            # 경로 따라가기
            steering_angle = self.calculate_steering_angle(path)
            speed = 2.0  # 기본 속도 설정
            self.publish_drive(speed, steering_angle)
        else:
            # 목표 지점에 도달하지 못한 경우 직진하면서 목표를 다시 탐색
            self.publish_drive(0.5, 0.0)

        # 시각화를 위한 마커 퍼블리시
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

    def lidar_callback(self, msg):
        self.scan_data = msg
        self.update_occupancy_grid()

    def update_occupancy_grid(self):
        if self.scan_data is None:
            return

        # Reset occupancy grid
        self.occupancy_grid.fill(0)

        # Update occupancy grid based on lidar data
        angle_increment = (self.scan_data.angle_max - self.scan_data.angle_min) / len(self.scan_data.ranges)
        for i, distance in enumerate(self.scan_data.ranges):
            if distance < LIDAR_RANGE:
                angle = self.scan_data.angle_min + i * angle_increment + self.vehicle_heading
                x = self.vehicle_x + distance * np.cos(angle)
                y = self.vehicle_y + distance * np.sin(angle)
                grid_x = int((x - self.vehicle_x) / GRID_RESOLUTION + GRID_SIZE / 2)
                grid_y = int((y - self.vehicle_y) / GRID_RESOLUTION + GRID_SIZE / 2)
                if 0 <= grid_x < GRID_SIZE and 0 <= grid_y < GRID_SIZE:
                    self.occupancy_grid[grid_x, grid_y] = 1

    def rrt_star(self, start_x, start_y, goal_x, goal_y):
        nodes = [(start_x, start_y)]
        edges = []

        for i in range(MAX_ITERATIONS):
            # 목표 지점을 직접 샘플링할 확률 적용
            if random.random() < GOAL_SAMPLE_PROB:
                x_rand, y_rand = goal_x, goal_y
            else:
                x_rand, y_rand = self.sample_free()

            if not self.is_in_free_space(x_rand, y_rand):
                continue

            x_nearest, y_nearest = self.nearest(nodes, x_rand, y_rand)
            x_new, y_new = self.steer(x_nearest, y_nearest, x_rand, y_rand)

            if self.is_obstacle_free(x_nearest, y_nearest, x_new, y_new):
                nodes.append((x_new, y_new))
                edges.append(((x_nearest, y_nearest), (x_new, y_new)))

                # 목표 지점에 도달했는지 확인
                if math.sqrt((x_new - goal_x) ** 2 + (y_new - goal_y) ** 2) < STEP_SIZE:
                    self.publish_rrt_markers(nodes, edges)
                    return self.extract_path(nodes, edges, (x_new, y_new))

        self.publish_rrt_markers(nodes, edges)
        return None

    def sample_free(self):
        # -65도에서 65도까지 각도를 랜덤 샘플링 (라디안 단위)
        angle = random.uniform(-math.radians(90), math.radians(90))
        distance = random.uniform(0, LIDAR_RANGE)  # LIDAR_RANGE 이내에서 거리 샘플링

        # 샘플링된 각도와 거리를 차량 기준 좌표계에서 변환
        x_sample = self.vehicle_x + distance * math.cos(self.vehicle_heading + angle)
        y_sample = self.vehicle_y + distance * math.sin(self.vehicle_heading + angle)

        return x_sample, y_sample


    def is_in_free_space(self, x, y):
        grid_x = int((x - self.vehicle_x) / GRID_RESOLUTION + GRID_SIZE / 2)
        grid_y = int((y - self.vehicle_y) / GRID_RESOLUTION + GRID_SIZE / 2)
        if 0 <= grid_x < GRID_SIZE and 0 <= grid_y < GRID_SIZE:
            return self.occupancy_grid[grid_x, grid_y] == 0
        return False

    def nearest(self, nodes, x_rand, y_rand):
        return min(nodes, key=lambda node: (node[0] - x_rand) ** 2 + (node[1] - y_rand) ** 2)

    def steer(self, x_nearest, y_nearest, x_rand, y_rand):
        theta = math.atan2(y_rand - y_nearest, x_rand - x_nearest)
        x_new = x_nearest + STEP_SIZE * math.cos(theta)
        y_new = y_nearest + STEP_SIZE * math.sin(theta)
        return x_new, y_new

    def is_obstacle_free(self, x1, y1, x2, y2):
        # 장애물 충돌 체크 (점유 그리드를 활용)
        num_points = max(1, int(math.hypot(x2 - x1, y2 - y1) / GRID_RESOLUTION))  # num_points가 0이 되는 것을 방지
        for i in range(num_points + 1):
            x = x1 + i * (x2 - x1) / num_points
            y = y1 + i * (y2 - y1) / num_points
            if not self.is_in_free_space(x, y):
                return False
        return True

    def extract_path(self, nodes, edges, goal_node):
        # 경로 추출 (목표 노드부터 시작하여 부모 노드로 거슬러 올라가기)
        path = [goal_node]
        current = goal_node
        while current != (self.vehicle_x, self.vehicle_y):
            for (start, end) in edges:
                if end == current:
                    path.append(start)
                    current = start
                    break
        path.reverse()
        return path

    def calculate_steering_angle(self, path):
        if len(path) < 2:
            return 0.0
        goal_x, goal_y = path[1]
        dx = goal_x - self.vehicle_x
        dy = goal_y - self.vehicle_y
        local_x = dx * math.cos(-self.vehicle_heading) - dy * math.sin(-self.vehicle_heading)
        local_y = dx * math.sin(-self.vehicle_heading) + dy * math.cos(-self.vehicle_heading)
        if local_y != 0:
            curvature = (8 * local_y) / (LOOKAHEAD_DISTANCE ** 2)
            steering_angle = max(min(curvature, 0.5), -0.5)  # 조향 각도 제한 확장
        else:
            steering_angle = 0.0
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

    def publish_rrt_markers(self, nodes, edges):
        # RRT Nodes marker
        node_marker = Marker()
        node_marker.header.frame_id = "map"
        node_marker.type = Marker.POINTS
        node_marker.action = Marker.ADD
        node_marker.scale.x = 0.1
        node_marker.scale.y = 0.1
        node_marker.color.a = 1.0
        node_marker.color.g = 1.0
        node_marker.points = [Point(x=node[0], y=node[1], z=0.0) for node in nodes]
        self.rrt_marker_pub.publish(node_marker)

        # RRT Edges marker
        edge_marker = Marker()
        edge_marker.header.frame_id = "map"
        edge_marker.type = Marker.LINE_LIST
        edge_marker.action = Marker.ADD
        edge_marker.scale.x = 0.05
        edge_marker.color.a = 1.0
        edge_marker.color.b = 1.0
        for start, end in edges:
            edge_marker.points.append(Point(x=start[0], y=start[1], z=0.0))
            edge_marker.points.append(Point(x=end[0], y=end[1], z=0.0))
        self.rrt_marker_pub.publish(edge_marker)

    def quaternion_to_euler(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return 0.0, 0.0, yaw

def main(args=None):
    rclpy.init(args=args)
    pure_pursuit_rrt_node = PurePursuitRRT()
    rclpy.spin(pure_pursuit_rrt_node)
    pure_pursuit_rrt_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
