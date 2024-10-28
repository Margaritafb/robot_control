import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
import numpy as np
import math

class PIDController:
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp = Kp  
        self.Ki = Ki 
        self.Kd = Kd  
        self.dt = dt  
        self.prev_error = 0.0
        self.integral_error = 0.0

    def compute(self, error):
        proportional = self.Kp * error
        
        self.integral_error += error * self.dt
        integral = self.Ki * self.integral_error
        
        derivative = self.Kd * (error - self.prev_error) / self.dt
        
        self.prev_error = error
        
        output = proportional + integral + derivative
        return output

    def reset(self):
        self.prev_error = 0.0
        self.integral_error = 0.0


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
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        self.odom_sub
        
        # Topicos a publicar
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Almacenamiento de datos
        self.current_scan = None
        self.goal_distance = None
        self.angle_to_goal = None
        self.robot_x = None
        self.robot_y = None
        self.robot_yaw = None
        self.min_distance_front = None
        self.min_angle = None
        self.cmd_vel = Twist()
        
        # Estado del sistema
        self.state = 'GTG'  # Estados posibles: GTG, AO, FW, STOP
        self.dt = None      # Distancia al objetivo al entrar en FW
        
        # Parámetros
        self.delta = 0.4  # Distancia crítica para evitar obstáculos
        self.epsilon = 0.4  # Distancia de parada al objetivo
        
        # Instanciar el controlador PID para la velocidad angular en ir hacia la meta
        self.pid_controller = PIDController(Kp=3.0, Ki=0.0, Kd=0.00, dt=0.1)
        
        self.timer = self.create_timer(0.1, self.control_loop)

    def odom_callback(self, msg):
        # Obtener la posición actual del robot
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        
        # Obtener la orientación en el plano (yaw) pues el angulo de orientación del topico se encuentra en cuarteniones y hay que pasarlo a un angulo Euler - plano 2d
        orientation_q = msg.pose.pose.orientation
        siny_cosp = 2 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y)
        cosy_cosp = 1 - 2 * (orientation_q.y * orientation_q.y + orientation_q.z * orientation_q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)

    def goal_callback(self, msg):
        # Posición del objetivo
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        
        if self.robot_x is not None and self.robot_y is not None:
            # Calcular la distancia al objetivo
            dx = self.goal_x - self.robot_x
            dy = self.goal_y - self.robot_y
            self.goal_distance = math.sqrt(dx**2 + dy**2)
            # Calcular el ángulo hacia el objetivo relativo a la orientación del robot
            self.angle_to_goal = math.atan2(dy, dx)
            self.get_logger().info(f'Alcanzado el objetivo, distancia: {self.goal_distance:.2f}')
            
    def lidar_callback(self, msg):
        self.current_scan = msg
        if self.current_scan is None or self.goal_distance is None:
            return
        
        # Procesar datos del LiDAR
        ranges = np.array(self.current_scan.ranges)
        angle_min = self.current_scan.angle_min
        angle_max = self.current_scan.angle_max
        angle_increment = self.current_scan.angle_increment
        
        # Calcular el índice correspondiente al frente del robot)
        measures = len(ranges)
        angulo_medio = (angle_max + angle_min) / 2.0
        index = int((angulo_medio - angle_min) / angle_increment)
        
        # Definir un rango de +/- 20 grados alrededor del frente
        range_front = int(np.radians(20) / angle_increment)

        # Extraer las distancias en el rango de interés
        front_angle_indices = range(max(0, index - range_front), 
                                min(measures, index + range_front))
        
        # Filtrar las distancias válidas
        valid_distances = [(ranges[i], angle_min + i * angle_increment) 
                       for i in front_angle_indices if ranges[i] > 0]

        # Calcular la distancia mínima en el rango seleccionado
        if valid_distances:
            self.min_distance_front, self.min_angle = min(valid_distances, key=lambda x: x[0])
        else:
            self.min_distance_front = float('inf')
            self.min_angle = None            
            
    def goal_control(self):
 
        # Definir la velocidad lineal constante
        vx = 0.2  

        if self.angle_to_goal is not None and self.robot_yaw is not None:

            error = self.angle_to_goal - self.robot_yaw
            error = math.atan2(math.sin(error), math.cos(error))

            # Calcular la salida del controlador PID
            angular_velocity = float(self.pid_controller.compute(error))

            # Publicar las velocidades en el tópico cmd_vel
            self.cmd_vel.linear.x = float(vx)
            self.cmd_vel.angular.z = float(angular_velocity)
            self.cmd_vel_pub.publish(self.cmd_vel)
    
    def obstacle_control(self):
        
        if self.min_angle is None or self.robot_yaw is None:
            return

        # Calcular vectores unitarios
        u_AO_x = math.cos(self.min_angle)
        u_AO_y = math.sin(self.min_angle)
        
        # Calcular ángulo de evitación perpendicular al obstáculo
        avoid_angle = math.atan2(u_AO_y, u_AO_x) + math.pi  # Girar 180° para alejarse
        
        # Calcular error de orientación
        error = avoid_angle - self.robot_yaw
        error = math.atan2(math.sin(error), math.cos(error))
        
        # Control de velocidad
        angular_velocity = float(self.pid_controller.compute(error))
        
        # Si estamos bien orientados, avanzar
        if abs(error) < 0.1:
            self.cmd_vel.linear.x = 0.2
        else:
            self.cmd_vel.linear.x = 0.0
            
        self.cmd_vel.angular.z = angular_velocity
        self.cmd_vel_pub.publish(self.cmd_vel)
        self.get_logger().info(f'[Obstacle Control] Girando para alinearse: error = {error:.2f}, angular.z = {angular_velocity:.2f}')
 
    def follow_wall_clockwise(self):
        if self.min_angle is None or self.robot_yaw is None:
            return
        
        # Vector perpendicular al obstáculo en sentido horario
        u_FW_x = math.sin(self.min_angle)  # Rotación 90° en sentido horario
        u_FW_y = -math.cos(self.min_angle)
        
        follow_angle = math.atan2(u_FW_y, u_FW_x)
        
        error = follow_angle - self.robot_yaw
        error = math.atan2(math.sin(error), math.cos(error))
        
        angular_velocity = float(self.pid_controller.compute(error))
        
        if abs(error) < 0.1 and abs(self.min_distance_front - self.delta) < 0.3:
            self.cmd_vel.linear.x = 0.2
        else:
            self.cmd_vel.linear.x = 0.0
            
        self.cmd_vel.angular.z = angular_velocity
        self.cmd_vel_pub.publish(self.cmd_vel)
        
    def follow_wall_counterclockwise(self):
        if self.min_angle is None or self.robot_yaw is None:
            return
        
        # Vector perpendicular al obstáculo en sentido antihorario
        u_FW_x = -math.sin(self.min_angle)  # Rotación 90° en sentido antihorario
        u_FW_y = math.cos(self.min_angle)
        
        follow_angle = math.atan2(u_FW_y, u_FW_x)
        
        error = follow_angle - self.robot_yaw
        error = math.atan2(math.sin(error), math.cos(error))
        
        angular_velocity = float(self.pid_controller.compute(error))
        
        if abs(error) < 0.1 and abs(self.min_distance_front - self.delta) < 0.3:
            self.cmd_vel.linear.x = 0.2
        else:
            self.cmd_vel.linear.x = 0.0
            
        self.cmd_vel.angular.z = angular_velocity
        self.cmd_vel_pub.publish(self.cmd_vel)       

    def control_loop(self):
        if not all([self.robot_x, self.robot_y, self.goal_distance, self.min_distance_front]):
            return
            
        # Actualizar distancia al objetivo
        dx = self.goal_x - self.robot_x
        dy = self.goal_y - self.robot_y
        self.goal_distance = math.sqrt(dx**2 + dy**2)
        
        # Calcular productos punto necesarios
        dot_GTG_FW_C = self.compute_dot_product_u_GTG_u_FW_clockwise()
        dot_GTG_FW_CC = self.compute_dot_product_u_GTG_u_FW_counterclockwise()
        dot_AO_GTG = self.compute_dot_product_u_AO_u_GTG()
        
        # Transiciones de estados
        if self.state == 'GTG':
            if self.goal_distance <= self.epsilon:
                self.state = 'STOP'
            elif self.min_distance_front <= self.delta:
                if dot_GTG_FW_C > 0:
                    self.state = 'FW_C'
                    self.dt = self.goal_distance
                elif dot_GTG_FW_CC > 0:
                    self.state = 'FW_CC'
                    self.dt = self.goal_distance
                
        elif self.state in ['FW_C', 'FW_CC']:
            if self.min_distance_front < self.delta:
                self.state = 'AO'
            elif self.goal_distance < self.dt and dot_AO_GTG > 0:
                self.state = 'GTG'
                
        elif self.state == 'AO':
            if self.min_distance_front >= self.delta:
                if dot_GTG_FW_C > 0:
                    self.state = 'FW_C'
                    self.dt = self.goal_distance
                elif dot_GTG_FW_CC > 0:
                    self.state = 'FW_CC'
                    self.dt = self.goal_distance
                
        # Ejecutar comportamiento según estado
        if self.state == 'GTG':
            self.goal_control()
        elif self.state == 'AO':
            self.obstacle_control()
        elif self.state == 'FW':
            self.follow_wall_control()
        elif self.state == 'STOP':
            self.cmd_vel.linear.x = 0.0
            self.cmd_vel.angular.z = 0.0
            self.cmd_vel_pub.publish(self.cmd_vel)
            
        self.get_logger().info(f'Estado: {self.state}, Distancia objetivo: {self.goal_distance:.2f}, '
                              f'Distancia obstáculo: {self.min_distance_front:.2f}')
            
    def compute_dot_product_u_AO_u_GTG(self):
        if self.angle_to_goal is None or self.min_angle is None:
            return 0.0
        # Calcular el producto interno entre u_AO y u_GTG
        u_AO_x = math.cos(self.min_angle)
        u_AO_y = math.sin(self.min_angle)
        u_GTG_x = math.cos(self.angle_to_goal)
        u_GTG_y = math.sin(self.angle_to_goal)
        return u_AO_x * u_GTG_x + u_AO_y * u_GTG_y

    def compute_dot_product_u_GTG_u_FW_clockwise(self):
        if self.angle_to_goal is None or self.min_angle is None:
            return 0.0
        
        u_GTG_x = math.cos(self.angle_to_goal)
        u_GTG_y = math.sin(self.angle_to_goal)
        
        u_FW_x = math.sin(self.min_angle)
        u_FW_y = -math.cos(self.min_angle)
        
        return u_GTG_x * u_FW_x + u_GTG_y * u_FW_y
    
    def compute_dot_product_u_GTG_u_FW_counterclockwise(self):
        if self.angle_to_goal is None or self.min_angle is None:
            return 0.0
        
        u_GTG_x = math.cos(self.angle_to_goal)
        u_GTG_y = math.sin(self.angle_to_goal)
        
        u_FW_x = -math.sin(self.min_angle)
        u_FW_y = math.cos(self.min_angle)
        
        return u_GTG_x * u_FW_x + u_GTG_y * u_FW_y
    

def main(args=None):            
    rclpy.init(args=args)       
    node = Robot_controller()   
    rclpy.spin(node)            
    node.destroy_node()         
    rclpy.shutdown()            

if __name__ == '__main__':      
    main()
