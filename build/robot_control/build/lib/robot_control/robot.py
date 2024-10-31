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

        # Variables para el cálculo en `lidar_link`
        self.angle_to_goal = None
        self.distance_to_goal = None
        
        #Variables para el LiDAR
        self.obstacle_distance_front = None
        self.obstacle_angle_front = None
        
        #variables para FW
        self.initial_angle_to_obstacle = None
        self.turning = False
        
        #Variables para transiciones de estados
        self.delta = 0.4  # Distancia crítica para activar AO
        self.epsilon = 0.2  # Distancia de parada al objetivo
        self.state = 'GTG'  # Estado inicial de la máquina de estados
        self.dt = None
        
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
            goal_x_lidar = goal_point_in_lidar_link.point.x
            goal_y_lidar = goal_point_in_lidar_link.point.y

            # Calcular el ángulo hacia la meta en `lidar_link`
            self.angle_to_goal = math.atan2(goal_y_lidar, goal_x_lidar)

            # Calcular la distancia hacia la meta en `lidar_link`
            self.distance_to_goal = math.sqrt(goal_x_lidar**2 + goal_y_lidar**2)

            #self.get_logger().info(f"Goal en lidar_link: x={goal_x_lidar:.2f}, y={goal_y_lidar:.2f}, "
            #                       f"angle_to_goal={self.angle_to_goal:.2f}, distance_to_goal={self.distance_to_goal:.2f}")

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
    
        angle_84_pos = math.radians(84)
        angle_0 = 0.0
        angle_84_neg = math.radians(-84)
    
        # Almacenar y mostrar los resultados
        self.distancia_obstaculo_izq = distance_84_pos
        self.angulo_obstaculo_izq = angle_84_pos
        self.distancia_obstaculo_frontal = distance_0
        self.angulo_obstaculo_frontal = angle_0
        self.distancia_obstaculo_der = distance_84_neg
        self.angulo_obstaculo_der = angle_84_neg
    
        # Log de las distancias y ángulos detectados
        self.get_logger().info(f"Obstáculo a +84°: distancia = {distance_84_pos:.2f}, ángulo = {angle_84_pos:.2f} rad")
        self.get_logger().info(f"Obstáculo al frente (0°): distancia = {distance_0:.2f}, ángulo = {angle_0:.2f} rad")
        self.get_logger().info(f"Obstáculo a -84°: distancia = {distance_84_neg:.2f}, ángulo = {angle_84_neg:.2f} rad")
    

    def goal_control(self):
        # Asegurarse de que el ángulo y la distancia hacia la meta están calculados
        if self.angle_to_goal is None or self.distance_to_goal is None:
            return

        # Define la velocidad lineal en función de la distancia a la meta
        linear_speed = 0.2 if self.distance_to_goal > 0.1 else 0.0  # Detenerse si está muy cerca
        angular_velocity = float(self.pid_controller.compute(self.angle_to_goal))

        # Establece la velocidad angular basada en el error de orientación
        self.cmd_vel.angular.z = angular_velocity
        self.cmd_vel.linear.x = linear_speed
        self.cmd_vel_pub.publish(self.cmd_vel)

    def obstacle_control(self):
        if self.obstacle_angle_front is None:
            return

        # Calcular el ángulo de evasión sumando 90 grados
        avoid_angle = self.obstacle_angle_front + math.radians(90)
        avoid_angle = math.atan2(math.sin(avoid_angle), math.cos(avoid_angle))  # Normalizar entre [-pi, pi]
        
        # Log de información sobre el cálculo del ángulo de evasión
        self.get_logger().info(f"[OBSTACLE CONTROL] Calculando ángulo de evasión: obstacle_angle={self.obstacle_angle_front:.2f} rad, "
                               f"avoid_angle={avoid_angle:.2f} rad")

        # Calcular la salida del controlador PID basada en el ángulo de evasión
        angular_velocity = float(self.pid_controller.compute(avoid_angle))
        
        # Log de información sobre la velocidad angular calculada
        self.get_logger().info(f"[OBSTACLE CONTROL] Angular velocity calculada para evasión: angular_velocity={angular_velocity:.2f}")
        
        self.cmd_vel.angular.z = angular_velocity
        self.cmd_vel.linear.x = 0.0  # No avanzar mientras evita el obstáculo
        self.cmd_vel_pub.publish(self.cmd_vel)
    
    def register_objective(self, angle, distance):
        """
        Registra la posición inicial del objetivo en el marco `lidar_link` usando ángulo y distancia.
        :param angle: Ángulo inicial hacia el objetivo en `lidar_link`.
        :param distance: Distancia inicial hacia el objetivo en `lidar_link`.
        """
        # Convertir ángulo y distancia en coordenadas x, y en `lidar_link`
        x = distance * math.cos(angle)
        y = distance * math.sin(angle)
        self.objective_position = (x, y)
        self.get_logger().info(f"Objetivo registrado en lidar_link: x={x}, y={y}")
        
    def calculate_angle_to_objective(self):
        """
        Calcula el ángulo actual del robot hacia el objetivo registrado en el marco `lidar_link`.
        :return: Ángulo en radianes desde el robot hacia el objetivo registrado, o None si no se puede calcular.
        """
        if self.objective_position is None:
            self.get_logger().info("No se ha registrado ningún objetivo.")
            return None

        # Obtener la posición actual del robot en `lidar_link`
        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                "lidar_link", "base_link", rclpy.time.Time()
            )
            robot_x = transform.transform.translation.x
            robot_y = transform.transform.translation.y

            # Calcular el ángulo hacia el objetivo usando la posición registrada
            target_x, target_y = self.objective_position
            angle_to_objective = math.atan2(target_y - robot_y, target_x - robot_x)
            self.get_logger().info(f"Ángulo actual hacia el objetivo: {angle_to_objective:.2f} rad")
            return angle_to_objective
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            self.get_logger().info("No se pudo obtener la transformación entre `base_link` y `lidar_link`.")
            return None
           
    def follow_wall_decision(self):
        
        # Vector hacia la meta (`u_GTG`)
        u_GTG_x = math.cos(self.angle_to_goal)
        u_GTG_y = math.sin(self.angle_to_goal)

        # Vector hacia el obstáculo (`u_AO`)
        u_AO_x = math.cos(self.obstacle_angle_front)
        u_AO_y = math.sin(self.obstacle_angle_front)

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
            self.state = 'FW_C'
            self.follow_wall_angle = math.atan2(u_FW_C_y, u_FW_C_x)
            self.get_logger().info("Cambio de estado a FW_C (siguiendo pared en sentido horario)")
        else:
            self.state = 'FW_CC'
            self.follow_wall_angle = math.atan2(u_FW_CC_y, u_FW_CC_x)
            self.get_logger().info("Cambio de estado a FW_CC (siguiendo pared en sentido antihorario)")
    
    def rotate_to_follow_wall(self):
        # Detener el movimiento lineal del robot mientras se alinea
        self.cmd_vel.linear.x = 0.0

        if self.initial_angle_to_obstacle is None:
            self.get_logger().info("No hay obstáculo detectado para seguir la pared.")
            return  # Salir si no hay un ángulo de obstáculo detectado

        # Configuración de los límites de alineación con la pared
        target_alignment_angle = math.radians(84)  # Límite del LiDAR
        alignment_threshold = math.radians(4)  # Umbral de alineación de 4 grados

        # Control de dirección según el estado
        if self.state == 'FW_C':
            # Alinear a +84 grados (sentido horario)
            error_angle = target_alignment_angle - self.initial_angle_to_obstacle
            angular_velocity = -abs(self.pid_controller.compute(error_angle))
            direction = "horario"
        elif self.state == 'FW_CC':
            # Alinear a -84 grados (sentido antihorario)
            error_angle = -target_alignment_angle - self.initial_angle_to_obstacle
            angular_velocity = abs(self.pid_controller.compute(error_angle))
            direction = "antihorario"
        else:
            return  # Salir si no está en un estado de giro

        # Log de la dirección de rotación y la velocidad angular calculada
        self.get_logger().info(f"Girando en sentido {direction}: angular_velocity={angular_velocity:.2f}, initial angle to obstacle={self.initial_angle_to_obstacle:.2f}, error={error_angle:.2f}")

        # Aplicar el comando de velocidad angular
        if abs(error_angle) > alignment_threshold:  # Si el error angular es significativo
            self.cmd_vel.angular.z = angular_velocity
        else:
            # Si ya está alineado, detener la rotación y cambiar de estado a seguimiento de pared
            self.cmd_vel.angular.z = 0.0
            self.state = 'FW_FOLLOWING'
            self.get_logger().info("Alineado con la pared, cambiando a estado de seguimiento de pared.")

        # Publicar el comando de velocidad
        self.cmd_vel_pub.publish(self.cmd_vel)


    
    def control_loop(self):

        self.update_goal()
        
        # Máquina de estados finitos
        if self.state == 'GTG':
            # Transición a STOP si está cerca de la meta
            if self.distance_to_goal is not None and self.distance_to_goal <= self.epsilon:
                self.state = 'STOP'
                self.cmd_vel.linear.x = 0.0
                self.cmd_vel.angular.z = 0.0
                self.cmd_vel_pub.publish(self.cmd_vel)
                self.get_logger().info("Estado cambiado a STOP, objetivo alcanzado.")
            elif abs(self.distancia_obstaculo_frontal - self.delta) <= 0.1:
                self.dt = self.distance_to_goal
                self.fw_objective = self.register_objective(self.obstacle_angle_front,self.obstacle_distance_front)
                self.initial_angle_to_obstacle = self.obstacle_angle_front
                # Cambiar a seguimiento de pared
                self.follow_wall_decision()
            else:
                # Continuar yendo hacia la meta
                self.goal_control()
        
        elif self.state == 'FW_C':
            #if self.obstacle_distance_front < self.delta:
            #    self.state = 'AO'
            #    self.dt = None
            #    self.follow_wall_angle = None
            #else: 
                self.rotate_to_follow_wall()

        elif self.state == 'FW_CC':
            #if self.obstacle_distance_front < self.delta:
            #    self.state = 'AO'
            #    self.dt = None
            #    self.follow_wall_angle = None
            #else: 
                self.rotate_to_follow_wall()

        elif self.state == 'STOP':
            # El robot se detiene
            self.cmd_vel.linear.x = 0.0
            self.cmd_vel.angular.z = 0.0
            self.cmd_vel_pub.publish(self.cmd_vel)
        
        #elif self.state == 'AO':
        #    # Regresar a GTG si el obstáculo está fuera de rango
        #    if self.obstacle_distance_front is not None and self.obstacle_distance_front >= self.delta:
        #        self.state = 'GTG'
        #        self.get_logger().info("Regresando a GTG (ir hacia la meta)")
        #    else:
        #        # Realizar el control de evasión de obstáculos
        #        self.obstacle_control()
        

def main(args=None):            
    rclpy.init(args=args)       
    node = Robot_controller()   
    rclpy.spin(node)            
    node.destroy_node()         
    rclpy.shutdown()            

if __name__ == '__main__':      
    main()
