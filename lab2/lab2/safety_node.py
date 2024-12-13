#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry

class SafetyNode(Node):

    def __init__(self):
        super().__init__('safety_node')
       
        # 퍼블리셔 생성: 브레이크 상태와 차량 제어 명령을 퍼블리시
        self.brake_publisher = self.create_publisher(Bool, '/brake_bool', 10)
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
       
        # 서브스크라이버 생성: LIDAR 데이터와 오도메트리 데이터를 수신
        self.scan_subscriber = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_subscriber = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
       
        # 초기 변수 설정
        self.lidar_data = None  # LIDAR 데이터 저장 변수
        self.speed = 0.0        # 차량 속도 저장 변수
        self.epsilon = 1e-5     # 나눗셈에서 0으로 나누는 것을 방지하기 위한 작은 값
        self.timer = self.create_timer(0.01, self.check_ttc)  # 주기적으로 check_ttc 함수 실행

    def odom_callback(self, odom_msg):
        # 오도메트리 데이터를 수신하여 속도를 업데이트
        self.speed = odom_msg.twist.twist.linear.x
        self.get_logger().info(f"Current speed: {self.speed:.2f} m/s")

    def scan_callback(self, scan_msg):
        # LIDAR 데이터를 수신하여 저장
        self.lidar_data = scan_msg
        self.get_logger().info("LIDAR data received.")

    def check_ttc(self): #여기서부터 ttc 공식 사용
        # LIDAR 데이터가 없으면 함수 종료
        if self.lidar_data is None:
            return
       
        # 속도의 절대값을 사용하여 후진 시에도 동일한 기준으로 계산
        current_speed = abs(self.speed)

        # 현재 속도가 0보다 크면 TTC 임계값을 계산, 아니면 무한대 계산(속도가 0일때를 위해)
        if current_speed > 0:
            ttc_threshold = 0.8 / current_speed + 0.3
        else:
            ttc_threshold = float('inf')
       
        # LIDAR 데이터에서 거리, 각도 정보 추출
        ranges = self.lidar_data.ranges #장애물과의 거리배열
        angle_min = self.lidar_data.angle_min #최소각도,스캔이 시작하는 각도
        angle_increment = self.lidar_data.angle_increment #각 거리 데이터 포인트 사이 각도 증가값
        num_points = len(ranges)  # LIDAR 데이터 포인트 개수

        # 최소 TTC를 무한대로 초기화
        min_ttc = float('inf')
       
        # 각 LIDAR 데이터 포인트에 대해 반복
        for i in range(num_points):
            distance = ranges[i]  # 현재 포인트의 거리 데이터
            if np.isinf(distance) or distance == 0.0:  # 무한대 또는 0인 데이터는 건너뜀
                continue
           
            # 현재 포인트의 각도 계산, i를 곱하는 이유는 i번째 포인트까지의 각도 변화 lecture slide 47참고
            angle = angle_min + i * angle_increment

            # 상대 속도를 계산 (현재 각도에서의 속도 성분)
            relative_speed = self.speed * np.cos(angle)

            # 상대 속도가 0 이하일 경우 무시 (충돌 가능성이 없음)
            if relative_speed <= 0:
                continue

            # TTC(Time to Collision) 계산. epsilon은 0으로 나눠지는걸 방지.
            ttc = distance / (relative_speed + self.epsilon)

            # 최소 TTC 갱신
            if ttc < min_ttc:
                min_ttc = ttc

        # 최소 TTC가 임계값보다 작으면 브레이크 명령 발행
        if min_ttc < ttc_threshold:
            self.get_logger().info(f"TTC is below threshold: {min_ttc:.2f} seconds. Braking!")
            self.publish_stop_drive()
            self.publish_brake(True)
        else:
            # 최소 TTC가 임계값보다 크면 브레이크 해제
            self.get_logger().info(f"TTC is above threshold: {min_ttc:.2f} seconds. No need to brake.")
            self.publish_brake(False)
               
    def publish_brake(self, should_brake):
        # 브레이크 상태를 퍼블리시
        brake_msg = Bool()
        brake_msg.data = should_brake
        self.brake_publisher.publish(brake_msg)
        self.get_logger().info(f"Brake command published: {should_brake}")
       
    def publish_stop_drive(self):
        # 차량 제어 명령을 퍼블리시하여 차량을 정지시킴
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = 0.0
        self.drive_publisher.publish(drive_msg)
        self.get_logger().info("Drive command published: Stop")        

def main(args=None):
    # ROS2 노드 초기화 및 실행
    rclpy.init(args=args)
    safety_node = SafetyNode()
    safety_node.get_logger().info("Run safety_node")
    rclpy.spin(safety_node)
   
    # 노드 종료
    safety_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


