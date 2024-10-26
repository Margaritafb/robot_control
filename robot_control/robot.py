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
        self.delta = 1.0  # Distancia crítica para evitar obstáculos
        self.epsilon = 0.2  # Distancia de parada al objetivo
        
        # Instanciar el controlador PID para la velocidad angular en ir hacia la meta
        self.pid_controller = PIDController(Kp=1.0, Ki=0.1, Kd=0.05, dt=0.1)
        
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
        goal_x = msg.pose.position.x
        goal_y = msg.pose.position.y
        
        if self.robot_x is not None and self.robot_y is not None:
            # Calcular la distancia al objetivo
            dx = goal_x - self.robot_x
            dy = goal_y - self.robot_y
            self.goal_distance = math.sqrt(dx**2 + dy**2)
            # Calcular el ángulo hacia el objetivo relativo a la orientación del robot
            self.angle_to_goal = math.atan2(dy, dx)  
            
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
        
        # Definir un rango de +/- 10 grados alrededor del frente
        range_front = int(np.radians(10) / angle_increment)

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
        vx = 0.5  

        if self.angle_to_goal is not None and self.robot_yaw is not None:

            error = self.angle_to_goal - self.robot_yaw
            error = math.atan2(math.sin(error), math.cos(error))

            # Calcular la salida del controlador PID
            angular_velocity = self.pid_controller.compute(error)

            # Publicar las velocidades en el tópico cmd_vel
            self.cmd_vel.linear.x = vx
            self.cmd_vel.angular.z = angular_velocity
            self.cmd_vel_pub.publish(self.cmd_vel)

    
    def obstacle_control(self):
        
        #el robot se detiene y deja de moverse de forma lineal
        vx = 0 

        if self.min_angle is not None and self.robot_yaw is not None:
            # Calcular el ángulo de evitación sumando 90 grados (π/2 radianes) al ángulo mínimo
            avoid_angle = self.min_angle + math.radians(90)
            avoid_angle = math.atan2(math.sin(avoid_angle), math.cos(avoid_angle))

            # Calcular el error de orientación (avoid_angle - orientación actual)
            error = avoid_angle - self.robot_yaw
            error = math.atan2(math.sin(error), math.cos(error))

            # Calcular la salida del controlador PID para la velocidad angular
            angular_velocity = self.pid_controller.compute(error)

            # Publicar las velocidades en el tópico cmd_vel
            self.cmd_vel.linear.x = vx
            self.cmd_vel.angular.z = angular_velocity
            self.cmd_vel_pub.publish(self.cmd_vel) 
 
    def follow_wall_control(self):
        # El robot se mueve siguiendo el contorno del obstáculo como modo inducido
        
        if self.min_angle is not None and self.robot_yaw is not None:
            # Calcular el vector de evitación del obstáculo (u_AO)
            u_AO_x = math.cos(self.min_angle)
            u_AO_y = math.sin(self.min_angle)

            # Rotar 90 grados para obtener el vector de seguir el muro (u_FW)
            u_FW_x = -u_AO_y
            u_FW_y = u_AO_x

            # Decidir si seguir en sentido horario o antihorario
            # Calcular el ángulo hacia la meta (u_GTG)
            if self.angle_to_goal is not None:
                u_GTG_x = math.cos(self.angle_to_goal)
                u_GTG_y = math.sin(self.angle_to_goal)

                # Producto interno entre u_GTG y u_FW para decidir la dirección de seguimiento
                dot_product = u_GTG_x * u_FW_x + u_GTG_y * u_FW_y

                # Si el producto es positivo, seguimos en sentido horario, si es negativo, antihorario
                if dot_product > 0:
                    follow_angle = math.atan2(u_FW_y, u_FW_x)  # Sentido horario
                else:
                    follow_angle = math.atan2(-u_FW_y, -u_FW_x)  # Sentido antihorario

                # Calcular el error de orientación (follow_angle - orientación actual)
                error = follow_angle - self.robot_yaw
                error = math.atan2(math.sin(error), math.cos(error))

                # Calcular la salida del controlador PID para la velocidad angular
                angular_velocity = self.pid_controller.compute(error)

                # Publicar las velocidades en el tópico cmd_vel
                self.cmd_vel.linear.x = 0.5
                self.cmd_vel.angular.z = angular_velocity
                self.cmd_vel_pub.publish(self.cmd_vel)

    def control_loop(self):
        # Lógica de transición entre estados
        tolerance = 0.2  # Error para comparaciones de distancias

        if self.state == 'GTG':
            if self.goal_distance is not None and self.goal_distance <= self.epsilon:
                self.state = 'STOP'
                self.pid_controller.reset()
            elif self.min_distance_front is not None and abs(self.min_distance_front - self.delta) < tolerance:
                self.state = 'FW'
                self.dt = self.goal_distance  # Calcular y almacenar la distancia al objetivo al entrar en FW
                self.pid_controller.reset()

        elif self.state == 'AO':
            if self.min_distance_front is not None and abs(self.min_distance_front - self.delta) < tolerance and self.compute_dot_product_u_GTG_u_FW() > 0:
                self.state = 'FW'
                self.dt = self.goal_distance  # Recalcular dt al volver a FW
                self.pid_controller.reset()

        elif self.state == 'FW':
            if self.min_distance_front is not None and self.min_distance_front < self.delta:
                self.state = 'AO'
                self.pid_controller.reset()
            elif self.goal_distance is not None and self.goal_distance < self.delta and self.compute_dot_product_u_AO_u_GTG() > 0:
                self.state = 'GTG'
                self.dt = None  # Reiniciar la distancia al objetivo para futuras transiciones
                self.pid_controller.reset()

        elif self.state == 'STOP':
            self.cmd_vel.linear.x = 0.0
            self.cmd_vel.angular.z = 0.0
            self.cmd_vel_pub.publish(self.cmd_vel)
            self.pid_controller.reset()
            return

        # Ejecutar el comportamiento basado en el estado
        if self.state == 'GTG':
            self.goal_control()
        elif self.state == 'AO':
            self.obstacle_control()
        elif self.state == 'FW':
            self.follow_wall_control()
            
    def compute_dot_product_u_AO_u_GTG(self):
            # Calcular el producto interno entre u_AO y u_GTG
            u_AO_x = math.cos(self.min_angle)
            u_AO_y = math.sin(self.min_angle)
            u_GTG_x = math.cos(self.angle_to_goal)
            u_GTG_y = math.sin(self.angle_to_goal)
            return u_AO_x * u_GTG_x + u_AO_y * u_GTG_y

    def compute_dot_product_u_GTG_u_FW(self):
            # Calcular el producto interno entre u_GTG y u_FW
            u_FW_x = -math.sin(self.min_angle)
            u_FW_y = math.cos(self.min_angle)
            u_GTG_x = math.cos(self.angle_to_goal)
            u_GTG_y = math.sin(self.angle_to_goal)
            return u_FW_x * u_GTG_x + u_FW_y * u_GTG_y

def main(args=None):            
    rclpy.init(args=args)       
    node = Robot_controller()   
    rclpy.spin(node)            
    node.destroy_node()         
    rclpy.shutdown()            

if __name__ == '__main__':      
    main()
