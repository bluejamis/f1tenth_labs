import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
import heapq


class GapFollow(Node):
    def __init__(self):
        super().__init__('gap_follow_node')
        
        # 토픽 설정
        self.lidarscan_topic = '/scan'
        self.drive_topic = '/drive'
        self.best_point_marker_topic = '/best_point_marker'
        self.bubble_marker_topic = '/bubble_point_marker'
        
        # 서브스크립션 및 퍼블리셔 초기화
        self.lidar_sub = self.create_subscription(LaserScan, self.lidarscan_topic, self.scan_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.drive_topic, 10)
        self.best_point_marker_pub = self.create_publisher(Marker, self.best_point_marker_topic, 10)
        self.bubble_marker_pub = self.create_publisher(Marker, self.bubble_marker_topic, 10)

        # 파라미터 설정
        self.max_speed = 3.0
        self.min_bubble_radius = 0.5
        self.max_bubble_radius = 0.5
        self.steering_weight = 0.8  # 조향각에 가중치를 부여하지 않고 원본 각도로 사용
        self.bubble_avoidance_distance = 0.8 

    def preprocess_lidar(self, ranges, angle_min, angle_increment):
        """ 정면 ±50도 범위 내의 LIDAR 데이터를 전처리합니다. """
        front_angle_min = -np.radians(50)  
        front_angle_max = np.radians(50)  
        
        # ±50도 범위에 해당하는 인덱스 계산
        min_index = max(0, int((front_angle_min - angle_min) / angle_increment))
        max_index = min(len(ranges) - 1, int((front_angle_max - angle_min) / angle_increment))
        
        # 정면 50도 각도 범위에 해당하는 거리 값만 선택
        proc_ranges = np.array(ranges[min_index:max_index + 1])
        
        # 유효하지 않은 값 처리 (NaN 및 Inf 값 대체)
        max_range = np.nanmax(proc_ranges[np.isfinite(proc_ranges)])
        proc_ranges[np.isnan(proc_ranges)] = max_range
        proc_ranges[np.isinf(proc_ranges)] = max_range
        
        return proc_ranges, min_index

    
    def find_disparities(self, ranges, disparity_threshold=0.7):
        disparities = []
        for i in range(len(ranges) - 1):
            if abs(ranges[i + 1] - ranges[i]) > disparity_threshold:
                disparities.append(i if ranges[i] < ranges[i + 1] else i + 1)
        return disparities
    
    def extend_disparities(self, ranges, disparities):
        extended_ranges = np.copy(ranges)
        visited = set()

        # 가까운 거리 순으로 disparities 정렬
        sorted_disparities = sorted(disparities, key=lambda idx: ranges[idx])

        # 각 disparity 지점 주위에 "버블"을 확장하기 위해 마스킹 적용
        for idx in sorted_disparities:
            if idx in visited:
                continue

            # 현재 disparity 주변의 인덱스 범위를 설정
            start_idx = max(0, idx - 10)
            end_idx = min(len(ranges), idx + 11)  # idx + 10까지 포함

            # 확장할 인덱스 배열 생성
            indices = np.arange(start_idx, end_idx)

            # extended_ranges에서 해당 인덱스들에 disparity 값 적용
            extended_ranges[indices] = ranges[idx]

            # 처리된 인덱스를 방문한 것으로 표시하여 중복 업데이트 방지
            visited.update(indices)

        return extended_ranges


    
    def find_gaps(self, ranges, threshold=1.5, min_consecutive=3):
        gaps, start_idx, consecutive_count = [], None, 0
        for i in range(len(ranges)):
            if ranges[i] > threshold:
                if start_idx is None:
                    start_idx = i
                consecutive_count += 1
            else:
                if consecutive_count >= min_consecutive:
                    gaps.append(list(range(start_idx, i)))
                start_idx, consecutive_count = None, 0
        if consecutive_count >= min_consecutive:
            gaps.append(list(range(start_idx, len(ranges))))
        return gaps

    def find_optimal_gap(self, gap_indices, ranges, min_length=3):
        if not gap_indices:
            return None  # 빈 시퀀스가 주어질 때 None 반환

        long_gaps = [gap for gap in gap_indices if len(gap) >= min_length]
        
        if not long_gaps:
            return max(gap_indices, key=lambda g: len(g)) if gap_indices else None
        
        optimal_gap = max(long_gaps, key=lambda gap: max(ranges[idx] for idx in gap))
        return optimal_gap

    def find_best_point(self, ranges):
        """ 전체 스캔에서 가장 먼 포인트를 찾아 반환 """
        best_point_idx = np.argmax(ranges)  # 가장 큰 거리 값을 가진 인덱스를 찾음
        best_point_distance = ranges[best_point_idx]  # 해당 인덱스의 거리 값
        return best_point_idx, best_point_distance
    
    def scan_callback(self, data):
        # Step 1: LIDAR 데이터 전처리 (정면 ±50도 범위 내 데이터만 처리)
        ranges, min_index = self.preprocess_lidar(data.ranges, data.angle_min, data.angle_increment)
        
        # Step 2: 안전 버블 생성 (가장 가까운 지점 주변을 장애물로 간주)
        disparities = self.find_disparities(ranges)
        extended_ranges = self.extend_disparities(ranges, disparities)
        
        # Step 3: 가장 넓고 깊은 갭 탐색
        gaps = self.find_gaps(extended_ranges, threshold=1.5, min_consecutive=3)
        optimal_gap = self.find_optimal_gap(gaps, extended_ranges)
        
        if optimal_gap is None:
            # 갭을 찾지 못할 경우 가장 먼 포인트를 따라가도록 설정
            best_point_idx, best_point_distance = self.find_best_point(ranges)
            control_point_idx, control_point_distance = best_point_idx, best_point_distance
        else:
            # 갭을 찾은 경우 갭의 중간 지점을 목표로 설정
            gap_center_idx = (optimal_gap[0] + optimal_gap[-1]) // 2
            control_point_idx = gap_center_idx
            control_point_distance = extended_ranges[gap_center_idx]

        # 조향각 계산
        angle_min = data.angle_min
        angle_increment = data.angle_increment
        control_point_angle = angle_min + (min_index + control_point_idx) * angle_increment
        steering_angle = self.steering_weight * control_point_angle

        # 속도 결정
        speed = self.max_speed if control_point_distance > 2.0 else max(0.5, control_point_distance / 2.0 * self.max_speed)
        
        # Step 4: 안전 버블 회피 로직 추가
        if disparities:
            bubble_idx = min(disparities, key=lambda idx: ranges[idx])
            bubble_distance = ranges[bubble_idx]
            bubble_radius = self.min_bubble_radius if bubble_distance < 1.0 else self.max_bubble_radius
            
            # 1.5미터 이내로 근접한 경우 회피 동작 수행
            if bubble_distance < self.bubble_avoidance_distance:
                bubble_angle = angle_min + (min_index + bubble_idx) * angle_increment
                angle_difference = bubble_angle - control_point_angle
                
                # 회피 동작: 안전 버블이 우측에 있으면 좌회전, 좌측에 있으면 우회전
                if angle_difference > 0:  # 우측 회피
                    steering_angle -= 0.4  # 좌회전 각도 추가
                else:  # 좌측 회피
                    steering_angle += 0.4  # 우회전 각도 추가

                # 속도를 줄여 회피 동작을 더 안전하게 수행
                speed = min(speed, 0.5)  # 회피 시 속도 감소
        
        # 주행 명령 퍼블리시
        self.reactive_control(steering_angle=steering_angle, speed=speed)
        
        # 시각화 마커 퍼블리시
        self.publish_best_point_marker(control_point_idx + min_index, control_point_distance, angle_min, angle_increment)
        
        if disparities:
            closest_bubble_idx = min(disparities, key=lambda idx: ranges[idx])
            closest_bubble_distance = ranges[closest_bubble_idx]
            bubble_radius = np.clip(
                self.min_bubble_radius + 
                (closest_bubble_distance - 1.0) * (self.max_bubble_radius - self.min_bubble_radius) / (5.0 - 1.0), 
                self.min_bubble_radius, 
                self.max_bubble_radius
            )
            self.publish_closest_bubble_marker(
                closest_bubble_idx + min_index, 
                closest_bubble_distance, 
                bubble_radius, 
                angle_min, 
                angle_increment
            )



    def reactive_control(self, steering_angle, speed):
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)

    def publish_best_point_marker(self, best_point_idx, best_point_distance, angle_min, angle_increment):
        """ Publish a visualization marker for the best point in Rviz """
        # Calculate angle of best point
        best_point_angle = angle_min + best_point_idx * angle_increment

        # Convert polar coordinates to Cartesian for visualization
        x = best_point_distance * np.cos(best_point_angle)
        y = best_point_distance * np.sin(best_point_angle)

        # Create the Marker message
        marker = Marker()
        marker.header.frame_id = "ego_racecar/laser"  # Set the reference frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "best_point"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0  # LiDAR is in 2D plane, so z = 0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # Scale the marker (make it a small sphere)
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3

        # Set the color of the marker
        marker.color.a = 1.0  # Alpha channel
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        # Publish the Marker
        self.best_point_marker_pub.publish(marker)
        
    def publish_closest_bubble_marker(self, bubble_idx, bubble_distance, bubble_radius, angle_min, angle_increment):
        """ Publish a visualization marker for the closest bubble point in Rviz """
        # Calculate angle of best point
        bubble_point_angle = angle_min + bubble_idx * angle_increment

        # Convert polar coordinates to Cartesian for visualization
        x = bubble_distance * np.cos(bubble_point_angle)
        y = bubble_distance * np.sin(bubble_point_angle)

        # Create the Marker message
        marker = Marker()
        marker.header.frame_id = "ego_racecar/laser"  # Set the reference frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "bubble_point"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0  # LiDAR is in 2D plane, so z = 0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # Scale the marker (make it a small sphere)
        marker.scale.x = bubble_radius 
        marker.scale.y = bubble_radius 
        marker.scale.z = bubble_radius 

        # Set the color of the marker
        marker.color.a = 0.5  # Alpha channel
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0

        # Publish the Marker
        self.bubble_marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    reactive_node = GapFollow()
    rclpy.spin(reactive_node)

    reactive_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
