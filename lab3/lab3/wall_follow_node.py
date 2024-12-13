import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
import math

class WallFollow(Node):
    """ 
    차량의 벽 따라가기 알고리즘을 구현한 클래스
    """
    def __init__(self):
        super().__init__('wall_follow_node')

        # 토픽 설정
        lidarscan_topic = '/scan'
        drive_topic = '/drive'

        # 서브스크라이버와 퍼블리셔 생성
        self.lidar_sub = self.create_subscription(LaserScan, lidarscan_topic, self.scan_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)

        # PID 게인 값 설정 (조정된 값)
        self.kp = 2.0
        self.kd = 0.2
        self.ki = 0.01

        self.desired_distance_right = 1.1  # 원하는 오른쪽 벽과의 거리
        self.lookahead_distance = 0.8      # 선행 거리

        # PID 제어를 위한 변수 초기화
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = self.get_clock().now()

    def get_range(self, range_data, angle):
        angle_min = range_data.angle_min
        angle_max = range_data.angle_max
        angle_increment = range_data.angle_increment

        # 각도를 인덱스로 변환
        if angle < angle_min:
            angle = angle_min
        elif angle > angle_max:
            angle = angle_max

        index = int(round((angle - angle_min) / angle_increment))
        index = max(0, min(index, len(range_data.ranges) - 1))

        # 단일 포인트의 거리 값을 가져옴
        distance = range_data.ranges[index]
        if math.isinf(distance) or math.isnan(distance):
            distance = range_data.range_max

        return distance

    def get_error(self, range_data, dist):
        # 각도 설정
        angle_b = math.radians(-90)
        angle_a = math.radians(-50)
        theta = abs(angle_a - angle_b)

        # 거리 측정
        range_b = self.get_range(range_data, angle_b)
        range_a = self.get_range(range_data, angle_a)

        numerator = range_a * math.cos(theta) - range_b
        denominator = range_a * math.sin(theta)
        alpha = math.atan2(numerator, denominator)

        # 현재 차량과 벽 사이의 거리
        Dt = range_b * math.cos(alpha)
        D_t_plus_1 = Dt + self.lookahead_distance * math.sin(alpha)

        # 오차 계산
        error = dist - D_t_plus_1

        # 디버깅 로그 출력
        self.get_logger().info(f'range_90: {range_b:.2f}, range_50: {range_a:.2f}, alpha: {math.degrees(alpha):.2f}')
        self.get_logger().info(f'Error: {error:.2f}')
        
        return error

    def pid_control(self, error, velocity):
        current_time = self.get_clock().now()
        delta_time = (current_time - self.prev_time).nanoseconds / 1e9

        if delta_time == 0:
            delta_time = 1e-6

        # PID 계산
        self.integral += error * delta_time
        derivative = (error - self.prev_error) / delta_time
        steering_angle = self.kp * error + self.ki * self.integral + self.kd * derivative

        # 조향 각도 제한
        max_steering_angle = 0.6109
        steering_angle = max(-max_steering_angle, min(steering_angle, max_steering_angle))

        # AckermannDriveStamped 메시지를 사용하여 제어 명령 발행
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = float(steering_angle)
        drive_msg.drive.speed = velocity
        self.drive_pub.publish(drive_msg)

        # 이전 오차와 시간 업데이트
        self.prev_error = error
        self.prev_time = current_time

        # 디버깅 로그 출력
        self.get_logger().info(f'steering_angle: {steering_angle:.2f}, dedt: {derivative:.2f}')

    def scan_callback(self, msg):
        try:
            error = self.get_error(msg, self.desired_distance_right)
            self.pid_control(error, velocity=1.0)
        except Exception as e:
            self.get_logger().error(f'Error in scan_callback: {e}')

def main(args=None):
    rclpy.init(args=args)
    wall_follow_node = WallFollow()
    rclpy.spin(wall_follow_node)
    wall_follow_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
