import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped, PointStamped, Twist
from nav_msgs.msg import Odometry
import numpy as np
import math
import tf2_ros
from geometry_msgs.msg import TransformStamped
import tf2_geometry_msgs

class PIDController:
    def __init__(self, Kp=1.0, Ki=0.0, Kd=0.1, dt=0.1):
        self.default_Kp = Kp  
        self.default_Ki = Ki 
        self.default_Kd = Kd  
        self.dt = dt  
        self.prev_error = 0.0
        self.integral_error = 0.0

    def compute(self, error, Kp=None, Ki=None, Kd=None):
        # Usa los valores proporcionados o los predeterminados
        Kp = Kp if Kp is not None else self.default_Kp
        Ki = Ki if Ki is not None else self.default_Ki
        Kd = Kd if Kd is not None else self.default_Kd

        proportional = Kp * error
        self.integral_error += error * self.dt
        integral = Ki * self.integral_error
        derivative = Kd * (error - self.prev_error) / self.dt
        self.prev_error = error
        
        output = proportional + integral + derivative
        return output

    def reset(self):
        self.prev_error = 0.0
        self.integral_error = 0.0



class Robot_controller(Node):
    def __init__(self):
        super().__init__('robot_controller')
        
        # Suscripción y publicancion en topicos requeridos
        self.pose_sub = self.create_subscription(PoseStamped, 'goal_pose', self.goal_callback, 10)
        self.lidar_sub = self.create_subscription(LaserScan, 'scan', self.lidar_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Buffer para las transformaciones
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Variables para almacenar la posición de la meta en `odom`
        self.goal_x_odom = None
        self.goal_y_odom = None
        self.goal_x_lidar = None

        # Variables para el cálculo en `lidar_link`
        self.angulo_meta = None
        self.distancia_meta = None
       
        #Variables para el LiDAR
        self.distancia_obstaculo_izq = None
        self.angulo_obstaculo_izq = None
        self.distancia_obstaculo_frontal = None
        self.angulo_obstaculo_frontal = None
        self.distancia_obstaculo_der = None
        self.angulo_obstaculo_der = None
        
        #variables para FW
        self.initial_angle_to_obstacle = None
        self.turning = False
        
        #Variables para transiciones de estados
        self.delta = 0.30  # Distancia crítica para activar el seguimiento de pared
        self.epsilon = 0.05  # Distancia de parada al objetivo
        self.state = 'GTG'  # Estado inicial de la máquina de estados
        self.dt = None
        
        #Variable para publicar en velocidad
        self.cmd_vel = Twist()
        
        # Instanciar el controlador PID para la velocidad angular en ir hacia la meta
        self.pid_controller = PIDController(Kp=1.0, Ki=0.0, Kd=0.1, dt=0.1)
        
        self.timer = self.create_timer(0.1, self.control_loop)

    def goal_callback(self, msg):
        # Guardar la posición de la meta en el marco `odom`
        self.goal_x_odom = msg.pose.position.x
        self.goal_y_odom = msg.pose.position.y
        self.get_logger().info(f"Objetivo registrado en odom: x={self.goal_x_odom}, y={self.goal_y_odom}")
        
    def update_goal(self):
        if self.goal_x_odom is None or self.goal_y_odom is None:
            return

        # Crear un punto en el marco `odom` para representar la posición del objetivo
        goal_point_in_odom = PointStamped()
        goal_point_in_odom.header.frame_id = "odom"
        goal_point_in_odom.point.x = self.goal_x_odom
        goal_point_in_odom.point.y = self.goal_y_odom
        goal_point_in_odom.point.z = 0.0

        try:
            # Transformar la posición del objetivo desde `odom` a `lidar_link`
            goal_point_in_lidar_link = self.tf_buffer.transform(goal_point_in_odom, "lidar_link")

            # Guardar las coordenadas transformadas del objetivo en `lidar_link`
            self.goal_x_lidar = goal_point_in_lidar_link.point.x
            goal_y_lidar = goal_point_in_lidar_link.point.y

            # Calcular el ángulo hacia la meta en `lidar_link`
            self.angulo_meta = math.atan2(goal_y_lidar, self.goal_x_lidar)

            # Calcular la distancia hacia la meta en `lidar_link`
            self.distancia_meta = math.sqrt(self.goal_x_lidar**2 + goal_y_lidar**2)

            self.get_logger().info(f"Goal en lidar_link: x={self.goal_x_lidar:.2f}, y={goal_y_lidar:.2f}), "
                                   f"angle_to_goal={self.angulo_meta:.2f}, distance_to_goal={self.distancia_meta:.2f}")

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            self.get_logger().info("No se pudo transformar el objetivo de 'odom' a 'lidar_link'.")

    def lidar_callback(self, msg):
        # Extraer datos del LiDAR
        self.angle_min, self.angle_increment = msg.angle_min, msg.angle_increment
        self.ranges = np.array(msg.ranges)
    
        # Calcular los índices correspondientes a -84, 0, y +84 grados
        index_84_pos = int((math.radians(84) - self.angle_min) / self.angle_increment)
        index_0 = int((0.0 - self.angle_min) / self.angle_increment)
        index_84_neg = int((math.radians(-84) - self.angle_min) / self.angle_increment)
    
        # Obtener distancias y ángulos en ±84 y 0 grados si están en rango
        distance_84_pos = self.ranges[index_84_pos] if 0 <= index_84_pos < len(self.ranges) else float('inf')
        distance_0 = self.ranges[index_0] if 0 <= index_0 < len(self.ranges) else float('inf')
        distance_84_neg = self.ranges[index_84_neg] if 0 <= index_84_neg < len(self.ranges) else float('inf')
    
        # Almacenar y mostrar los resultados
        self.distancia_obstaculo_izq = math.cos(math.radians(6)) * distance_84_pos
        #self.angulo_obstaculo_izq = angle_84_pos
        self.distancia_obstaculo_frontal = distance_0
        #self.angulo_obstaculo_frontal = angle_0
        self.distancia_obstaculo_der = math.cos(math.radians(6)) * distance_84_neg
        #self.angulo_obstaculo_der = angle_84_neg
        
        angle_0 = 0.0
        self.angulo_obstaculo_frontal = angle_0
    
        # Log de las distancias y ángulos detectados
        #self.get_logger().info(f"Obstáculo a derecha: distancia = {self.distancia_obstaculo_der:.2f}, dist a 84 = {distance_84_neg}")
        #self.get_logger().info(f"Obstáculo al frente (0°): distancia = {self.distancia_obstaculo_frontal:.2f}")
        #self.get_logger().info(f"Obstáculo a la izquierda: distancia = {self.distancia_obstaculo_izq:.2f}, dist a 84 = {distance_84_pos}")
    

    def goal_control(self):
        # Asegurarse de que el ángulo y la distancia hacia la meta están calculados
        if self.angulo_meta is None or self.distancia_meta is None:
            return

        if self.distancia_meta <= self.epsilon:
                linear_speed = 0.0
                self.cmd_vel.angular.z = 0.0
                self.cmd_vel_pub.publish(self.cmd_vel) 
        else:
            linear_speed = 0.2
        angular_velocity = float(self.pid_controller.compute(self.angulo_meta))

        # Establece la velocidad angular basada en el error de orientación
        self.cmd_vel.angular.z = angular_velocity
        self.cmd_vel.linear.x = linear_speed
        self.cmd_vel_pub.publish(self.cmd_vel)
        
        self.get_logger().info("Avanzando hacia la meta")
        self.get_logger().info(f"Obstáculo a derecha: distancia = {self.distancia_obstaculo_der:.2f}")
        self.get_logger().info(f"Obstáculo al frente (0°): distancia = {self.distancia_obstaculo_frontal:.2f}")
        self.get_logger().info(f"Obstáculo a la izquierda: distancia = {self.distancia_obstaculo_izq:.2f}")
           
    def follow_wall_decision(self):
        
        # Vector hacia la meta (`u_GTG`)
        u_GTG_x = math.cos(self.angulo_meta)
        u_GTG_y = math.sin(self.angulo_meta)

        # Vector hacia el obstáculo (`u_AO`)
        u_AO_x = math.cos(self.angulo_obstaculo_frontal)
        u_AO_y = math.sin(self.angulo_obstaculo_frontal)

        # Calcular `u_FW_C` para el seguimiento horario (rotar `u_AO` -90°)
        u_FW_C_x = u_AO_y
        u_FW_C_y = -u_AO_x

        # Calcular `u_FW_CC` para el seguimiento antihorario (rotar `u_AO` +90°)
        u_FW_CC_x = -u_AO_y
        u_FW_CC_y = u_AO_x

        # Producto punto con `u_GTG`
        dot_GTG_FW_C = u_GTG_x * u_FW_C_x + u_GTG_y * u_FW_C_y
        dot_GTG_FW_CC = u_GTG_x * u_FW_CC_x + u_GTG_y * u_FW_CC_y

        # Decidir dirección de seguimiento de pared
        if dot_GTG_FW_C > dot_GTG_FW_CC:
            self.state = 'R_CC'
            self.get_logger().info("Cambio de estado a R_CC (rotando sentido horario)")
        else:
            self.state = 'R_C'
            self.get_logger().info("Cambio de estado a R_C (rotando en sentido antihorario)")
    
    def rotar_horario(self):
        
        if self.distancia_meta <= (self.dt - 2*self.delta) and self.distancia_obstaculo_frontal > 0.2: # and self.goal_x_lidar > 0:
            self.state = 'GTG'
            self.dt = None
        
        #Empezar a girar en sentido antihorario para alinearse
        self.cmd_vel.angular.z = -1.2
        self.cmd_vel.linear.x = 0.05
        
        if self.distancia_obstaculo_frontal > self.delta and abs(self.distancia_obstaculo_izq - self.delta) <= 0.02:
            self.state = 'FW_C'

        # Publicar el comando de velocidad
        self.cmd_vel_pub.publish(self.cmd_vel)
        
        self.get_logger().info("rotar horario")
        self.get_logger().info(f"vel angular = {self.cmd_vel.angular.z:.2f}, vel linear = {self.cmd_vel.linear.x:.2f}")
        
    def rotar_antihorario(self):
        
        if self.distancia_meta <= (self.dt - 2*self.delta) and self.distancia_obstaculo_frontal >= 0.2: # and self.goal_x_lidar > 0:
            self.state = 'GTG'
            self.dt = None
        
        #Empezar a girar en sentido horario para alinearse
        self.cmd_vel.angular.z = 1.2
        self.cmd_vel.linear.x = 0.05
        
        if self.distancia_obstaculo_frontal > self.delta and self.distancia_obstaculo_der <= self.delta:
            self.state = 'FW_CC'

        # Publicar el comando de velocidad
        self.cmd_vel_pub.publish(self.cmd_vel)
        
        self.get_logger().info("rotar antihorario")
        self.get_logger().info(f"Obstáculo a derecha: distancia = {self.distancia_obstaculo_der:.2f}")
        self.get_logger().info(f"Obstáculo al frente (0°): distancia = {self.distancia_obstaculo_frontal:.2f}")

    def pared_horario(self):
        
        if self.distancia_meta <= (self.dt - 2*self.delta) and self.distancia_obstaculo_frontal >= 0.2: # and self.goal_x_lidar > 0:
            self.state = 'GTG'
            self.dt = None
        
        elif abs(self.distancia_obstaculo_frontal - self.delta) <= 0.05:
            self.state = 'R_C'
        
        elif self.distancia_obstaculo_izq > self.delta: 
            self.cmd_vel.linear.x = 0.1
            self.cmd_vel.angular.z = 0.4
        elif self.distancia_obstaculo_izq < self.delta:
            self.cmd_vel.linear.x = 0.1
            self.cmd_vel.angular.z = -0.2
        else: 
            self.cmd_vel.linear.x = 0.2
            self.cmd_vel.angular.z = 0.0
            
        # Publicar el comando de velocidad
        self.cmd_vel_pub.publish(self.cmd_vel)
        self.get_logger().info("pared horario")
        self.get_logger().info(f"Obstáculo a izquierda: distancia = {self.distancia_obstaculo_izq:.2f}")
        self.get_logger().info(f"Obstáculo al frente (0°): distancia = {self.distancia_obstaculo_frontal:.2f}")   
        self.get_logger().info(f"distancia a meta = {self.distancia_meta}")
        self.get_logger().info(f"dt = {self.dt}") 
    
    def pared_antihorario(self):
        
        if self.distancia_meta <= (self.dt - 2*self.delta) and self.distancia_obstaculo_frontal >= 0.2: #and self.goal_x_lidar > 0:
            self.state = 'GTG'
            self.dt = None
        
        elif abs(self.distancia_obstaculo_frontal - self.delta) <= 0.05:
            self.state = 'R_CC'
        
        elif self.distancia_obstaculo_der > self.delta: 
            self.cmd_vel.linear.x = 0.1
            self.cmd_vel.angular.z = -0.4
        elif self.distancia_obstaculo_der < self.delta:
            self.cmd_vel.linear.x = 0.1
            self.cmd_vel.angular.z = 0.2
        else: 
            self.cmd_vel.linear.x = 0.2
            self.cmd_vel.angular.z = 0.0
            
        # Publicar el comando de velocidad
        self.cmd_vel_pub.publish(self.cmd_vel)
        self.get_logger().info("pared antihorario")
        self.get_logger().info(f"Obstáculo a derecha: distancia = {self.distancia_obstaculo_der:.2f}")
        self.get_logger().info(f"Obstáculo al frente (0°): distancia = {self.distancia_obstaculo_frontal:.2f}")   
        self.get_logger().info(f"distancia a meta = {self.distancia_meta}")
        self.get_logger().info(f"dt = {self.dt}") 
        

    def control_loop(self):

        self.update_goal()
        
        # Máquina de estados finitos
        if self.state == 'GTG':
            
            if abs(self.distancia_obstaculo_frontal - self.delta) <= 0.05:
                self.dt = self.distancia_meta
                # Hace decision para rotacion
                self.follow_wall_decision()
            else:
                # Continuar yendo hacia la meta
                self.goal_control()
        
        elif self.state == 'R_C':
            self.rotar_horario()
        elif self.state == 'R_CC':
            self.rotar_antihorario()
        elif self.state == 'FW_C':
            self.pared_horario()
        elif self.state == 'FW_CC':
            self.pared_antihorario()

        
def main(args=None):            
    rclpy.init(args=args)       
    node = Robot_controller()   
    rclpy.spin(node)            
    node.destroy_node()         
    rclpy.shutdown()            

if __name__ == '__main__':      
    main()
