import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped
import numpy as np
 
class Nodo(Node):
    def __init__(self):
        super().__init__('robot')
        # Suscriptores
        self.lidar_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.lidar_callback,
            10)
        self.lidar_sub
        self.pose_sub = self.create_subscription(
            PoseStamped,
            'goal_pose',
            self.pose_callback,
            10)
        self.pose_sub
        # Publicador
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        # Almacenamiento de datos
        self.current_scan = None
        self.current_pose = None

    def pose_callback(self, msg):
        self.current_pose = msg

    def lidar_callback(self,msg):
        self.current_scan = msg
        if self.current_scan is None or self.current_pose is None:
            return
        # Procesar datos del LiDAR
        ranges = np.array(self.current_scan.ranges)
        angle_min = self.current_scan.angle_min
        angle_increment = self.current_scan.angle_increment
        # Encontrar el punto más cercano
        min_distance = np.min(ranges)
        min_angle = angle_min + np.argmin(ranges) * angle_increment
        # Lógica simple de navegación
        cmd_vel = Twist()
        if min_distance < 0.1:  # Si hay un obstáculo cerca
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.5  # Girar
        else:
            cmd_vel.linear.x = 1.0  # Avanzar
            cmd_vel.angular.z = -0.1 * min_angle  # Corregir dirección
        # Publicar cmd_vel
        self.cmd_vel_pub.publish(cmd_vel)
 

def main(args=None):            # Función main
    rclpy.init(args=args)       # Iniciamos la comunicacion con ROS2
    node = Nodo()               # Creamos el objeto nodo
    rclpy.spin(node)            # Ejectutamos el nodo en un loop
    node.destroy_node()         # Cuando salimos del nodo lo destruimos
    rclpy.shutdown()            # Cerrramos la comunicacion con ROS2

if __name__ == '__main__':        # Si ejecutamos el archivo de python
    main()     