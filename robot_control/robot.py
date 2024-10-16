import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped
import numpy as np
import math

class Robot_controller(Node):
    def __init__(self):
        super().__init__('robot_controller')
        
        # Topicos a suscribirse
        self.lidar_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.lidar_callback,
            10)
        self.lidar_sub
        
        self.pose_sub = self.create_subscription(
            PoseStamped,
            'goal_pose',
            self.goal_callback,
            10)
        self.pose_sub
        
        # Topicos a publicar
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Almacenamiento de datos
        self.current_scan = None
        self.goal_distance = None
        self.angle_to_goal = None
        self.min_distance = None
        self.min_angle = None
        self.cmd_vel = Twist()  
        
        self.timer = self.create_timer(0.1, self.control_loop)

    def goal_callback(self, msg):
        # Calcula la distancia y el ángulo hacia el objetivo
        goal_x = msg.pose.position.x
        goal_y = msg.pose.position.y

        # Calcular la distancia al objetivo
        self.goal_distance = math.sqrt(goal_x**2 + goal_y**2)
        # Calcular el ángulo hacia el objetivo
        self.angle_to_goal = math.atan2(goal_y, goal_x)

    def lidar_callback(self, msg):
        self.current_scan = msg
        if self.current_scan is None or self.goal_distance is None:
            return
        
        # Procesar datos del LiDAR
        ranges = np.array(self.current_scan.ranges)
        angle_min = self.current_scan.angle_min
        angle_increment = self.current_scan.angle_increment
        
        # Encontrar el punto más cercano 
        self.min_distance = np.min(ranges)
        self.min_angle = angle_min + np.argmin(ranges) * angle_increment
        
    def control_loop(self):
        # Controla el robot basándose en la información del Lidar y la navegación al objetivo
        if self.current_scan and self.goal_distance and self.angle_to_goal:
            
            # Comportamiento de evasión de obstáculos
            if self.min_distance < 0.1:  # Si hay un obstáculo muy cerca
                self.cmd_vel.linear.x = 0.0
                self.cmd_vel.angular.z = 0.5  # Girar para evitar el obstáculo
            else:
                # Si no hay obstáculos cercanos, moverse hacia el objetivo
                self.cmd_vel.linear.x = 1.0  # Avanzar hacia el objetivo

                # Corregir el ángulo para orientarse hacia el objetivo
                if abs(self.angle_to_goal) > 0.1:
                    self.cmd_vel.angular.z = self.angle_to_goal
                else:
                    self.cmd_vel.angular.z = 0.0  # Si está bien alineado, no girar

        # Publicar velocidades
        self.cmd_vel_pub.publish(self.cmd_vel)
 

def main(args=None):            # Función main
    rclpy.init(args=args)       # Iniciamos la comunicacion con ROS2
    node = Robot_controller()   # Creamos el objeto nodo
    rclpy.spin(node)            # Ejectutamos el nodo en un loop
    node.destroy_node()         # Cuando salimos del nodo lo destruimos
    rclpy.shutdown()            # Cerrramos la comunicacion con ROS2

if __name__ == '__main__':      # Si ejecutamos el archivo de python
    main()
