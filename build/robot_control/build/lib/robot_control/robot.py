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
        self.min_distance_to_obstacle = None
        self.min_angle_to_obstacle = None
        self.follow_wall_angle = None
        self.obstacle_distance = None
        self.ao_obstacle_angle = None
        self.cmd_vel = Twist()
        
        # Estado del sistema
        self.state = 'GTG'  # Estados posibles: GTG, AO, FW, STOP
        self.dt = None      # Distancia al objetivo al entrar en FW
        
        # Parámetros
        self.delta = 0.3  # Distancia crítica para evitar obstáculos
        self.epsilon = 0.3  # Distancia de parada al objetivo
        self.gamma = 0.15  # Holgura para las condiciones de transición
        self.fw_initialized = False  # Bandera para inicializar u_FW solo una vez al entrar en FW
        
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
            
    def lidar_callback(self, msg):
        self.current_scan = msg
        if self.current_scan is None or self.goal_distance is None:
            return

        # Procesar datos del LiDAR
        ranges = np.array(self.current_scan.ranges)
        angle_min = self.current_scan.angle_min
        angle_max = self.current_scan.angle_max
        angle_increment = self.current_scan.angle_increment

        # Calcular el índice correspondiente al frente del robot
        measures = len(ranges)
        angulo_medio = (angle_max + angle_min) / 2.0
        index = int((angulo_medio - angle_min) / angle_increment)

        # Definir un rango de +/- 5 grados alrededor del frente
        range_front = int(np.radians(5) / angle_increment)

        # Extraer las distancias en el rango de interés alrededor del frente
        front_angle_indices = range(max(0, index - range_front), 
                                    min(measures, index + range_front))

        # Filtrar las distancias válidas con sus ángulos relativos
        valid_distances = [(ranges[i], angle_min + i * angle_increment) 
                           for i in front_angle_indices if ranges[i] > 0]

        # Calcular la distancia mínima y el ángulo al obstáculo más cercano en el rango
        if valid_distances:
            self.min_distance_to_obstacle, self.min_angle_to_obstacle = min(valid_distances, key=lambda x: x[0])
        else:
            self.min_distance_to_obstacle = float('inf')
            self.min_angle_to_obstacle = None  # Indicar que no hay obstáculos detectados en el rango

        # Asegurarse de que el ángulo esté en el rango [-pi, pi] 
        if self.min_angle_to_obstacle is not None:
            self.min_angle_to_obstacle = math.atan2(math.sin(self.min_angle_to_obstacle), math.cos(self.min_angle_to_obstacle))

        
    def get_distance_at_angle(self, target_angle):
        """
        Devuelve la distancia mínima al obstáculo en un rango de ±5 grados alrededor de un ángulo específico relativo al robot.
        :param target_angle: Ángulo en radianes (relativo al frente del robot).
        :return: Distancia mínima al obstáculo en el rango de ±5 grados alrededor del ángulo dado.
        """
        if self.current_scan is None:
            return float('inf')  # Si no hay datos de LiDAR, devuelve infinito

        # Obtener los parámetros del LiDAR
        ranges = np.array(self.current_scan.ranges)
        angle_min = self.current_scan.angle_min
        angle_increment = self.current_scan.angle_increment

        # Calcular el índice correspondiente al ángulo deseado
        angle_offset = target_angle - angle_min
        center_index = int(angle_offset / angle_increment)

        # Definir un rango de ±5 grados alrededor del ángulo deseado
        range_offset = int(np.radians(5) / angle_increment)
        start_index = max(0, center_index - range_offset)
        end_index = min(len(ranges) - 1, center_index + range_offset)

        # Extraer las distancias en el rango de interés y filtrar las distancias válidas (mayores que 0)
        valid_distances = [ranges[i] for i in range(start_index, end_index + 1) if ranges[i] > 0]

        # Devolver la distancia mínima en el rango o infinito si no hay mediciones válidas
        return min(valid_distances) if valid_distances else float('inf')
     
            
    def goal_control(self):
 
        # Definir la velocidad lineal constante
        vx = 0.3  

        if self.angle_to_goal is not None and self.robot_yaw is not None:

            error = self.angle_to_goal - self.robot_yaw
            error = math.atan2(math.sin(error), math.cos(error))

            # Calcular la salida del controlador PID
            angular_velocity = float(self.pid_controller.compute(error))

            # Publicar las velocidades en el tópico cmd_vel
            self.cmd_vel.linear.x = float(vx)
            self.cmd_vel.angular.z = float(angular_velocity)
            self.cmd_vel_pub.publish(self.cmd_vel)
    
    def avoid_obstacle_control(self):
        
        if self.min_angle_to_obstacle is None or self.robot_yaw is None:
            return
        
        avoid_angle = self.ao_obstacle_angle + math.pi / 2
        avoid_angle = math.atan2(math.sin(avoid_angle), math.cos(avoid_angle))
        
        error = avoid_angle - self.robot_yaw
        error = math.atan2(math.sin(error), math.cos(error))
        
        angular_velocity = float(self.pid_controller.compute(error))
        
        if abs(error) > 0.5:
            self.cmd_vel.linear.x = 0.0
        else:
            self.cmd_vel.linear.x = 0.05
            
        self.cmd_vel.angular.z = angular_velocity
        self.cmd_vel_pub.publish(self.cmd_vel)
        
        self.get_logger().info(f"[AO] Ángulo obstáculo = {self.ao_obstacle_angle:.2f}, "
                               f"Ángulo evitación = {avoid_angle:.2f}, "
                               f"Robot yaw = {self.robot_yaw:.2f}, "
                               f"Error = {error:.2f}, "
                               f"Velocidad angular = {angular_velocity:.2f}")

        
    def follow_wall_direction_decision(self, angle_to_goal, angle_to_obstacle):
        """
        Decide si el robot debe seguir la pared en sentido horario (FW_C) o antihorario (FW_CC).
        Utiliza el `angle_to_goal` y `angle_to_obstacle` para determinar la dirección del seguimiento.
        """
        self.dt = self.goal_distance  # Guardar la distancia actual al objetivo para referencia

        # Obtener las componentes del vector unitario hacia la meta
        u_GTG_x = math.cos(angle_to_goal)
        u_GTG_y = math.sin(angle_to_goal)

        # Obtener las componentes del vector unitario hacia el obstáculo
        u_AO_x = math.cos(angle_to_obstacle)
        u_AO_y = math.sin(angle_to_obstacle)

        # Calcular el vector perpendicular para el seguimiento horario (Relativo al frente del robot cuando entro a FW)
        u_FW_C_x = u_AO_y  # Rotación 90° en sentido horario
        u_FW_C_y = -u_AO_x

        # Calcular el vector perpendicular para el seguimiento antihorario
        u_FW_CC_x = -u_AO_y  # Rotación 90° en sentido antihorario
        u_FW_CC_y = u_AO_x

        # Calcular productos punto para decidir la dirección
        dot_GTG_FW_C = u_GTG_x * u_FW_C_x + u_GTG_y * u_FW_C_y
        dot_GTG_FW_CC = u_GTG_x * u_FW_CC_x + u_GTG_y * u_FW_CC_y

        # Decidir la dirección de seguimiento de pared
        if dot_GTG_FW_C > dot_GTG_FW_CC:
            self.state = 'FW_C'
            self.follow_wall_angle = math.atan2(u_FW_C_y, u_FW_C_x) + math.radians(90)
            # Normalización adicional (opcional)
            self.follow_wall_angle = math.atan2(math.sin(self.follow_wall_angle), math.cos(self.follow_wall_angle))
            self.get_logger().info(f"Cambiando a estado FW_C (seguir pared en sentido horario), d_t = {self.dt:.2f}")
        elif dot_GTG_FW_CC > dot_GTG_FW_C:
            self.state = 'FW_CC'
            self.follow_wall_angle = math.atan2(u_FW_CC_y, u_FW_CC_x) + math.radians(90)
            # Normalización adicional (opcional)
            self.follow_wall_angle = math.atan2(math.sin(self.follow_wall_angle), math.cos(self.follow_wall_angle))
            self.get_logger().info(f"Cambiando a estado FW_CC (seguir pared en sentido antihorario), d_t = {self.dt:.2f}")

    def follow_wall_clockwise(self):

        # Constantes de alineación
        alineacion_epsilon = 0.05  # Umbral para considerar que el robot está alineado con la pared
        velocidad_lineal_fw = 0.1  # Velocidad lineal constante al seguir la pared

        # Calcular el error de orientación entre la dirección actual y el follow_wall_angle
        error = self.follow_wall_angle - self.robot_yaw
        error = math.atan2(math.sin(error), math.cos(error))  # Normalizar error entre [-pi, pi]

        # Si el error de alineación es mayor que alineacion_epsilon, solo gira (no avanza)
        if abs(error) > alineacion_epsilon:
            self.cmd_vel.linear.x = 0.0  # No avanzar hasta estar alineado
            angular_velocity = self.pid_controller.compute(error)  # Ajustar velocidad angular con PID
            self.cmd_vel.angular.z = angular_velocity
        else:
            # Si el robot está alineado, avanza en línea recta y ajusta su orientación para mantener la alineación
            self.cmd_vel.linear.x = velocidad_lineal_fw
            angular_velocity = self.pid_controller.compute(error)
            self.cmd_vel.angular.z = angular_velocity

        # Publicar los comandos de velocidad
        self.cmd_vel_pub.publish(self.cmd_vel)
        self.get_logger().info(f'[FW_C] Error de alineación = {error:.2f}, angular.z = {angular_velocity:.2f}, linear.x = {self.cmd_vel.linear.x:.2f}')

    def follow_wall_counterclockwise(self):
        
        # Constantes de alineación
        alineacion_epsilon = 0.05  # Umbral para considerar que el robot está alineado con la pared
        velocidad_lineal_fw = 0.1  # Velocidad lineal constante al seguir la pared

        # Calcular el error de orientación entre la dirección actual y el follow_wall_angle
        error = self.follow_wall_angle - self.robot_yaw
        error = math.atan2(math.sin(error), math.cos(error))  # Normalizar error entre [-pi, pi]

        # Si el error de alineación es mayor que alineacion_epsilon, solo gira (no avanza)
        if abs(error) > alineacion_epsilon:
            self.cmd_vel.linear.x = 0.0  # No avanzar hasta estar alineado
            angular_velocity = self.pid_controller.compute(error)  # Ajustar velocidad angular con PID
            self.cmd_vel.angular.z = angular_velocity
        else:
            # Si el robot está alineado, avanza en línea recta y ajusta su orientación para mantener la alineación
            self.cmd_vel.linear.x = velocidad_lineal_fw
            angular_velocity = self.pid_controller.compute(error)
            self.cmd_vel.angular.z = angular_velocity

        # Publicar los comandos de velocidad
        self.cmd_vel_pub.publish(self.cmd_vel)
        self.get_logger().info(f'[FW_CC] Error de alineación = {error:.2f}, angular.z = {angular_velocity:.2f}, linear.x = {self.cmd_vel.linear.x:.2f}')

    def control_loop(self):
        if not all([self.robot_x, self.robot_y, self.goal_distance, self.min_distance_to_obstacle]):
            return
            
        # Actualizar distancia al objetivo
        dx = self.goal_x - self.robot_x
        dy = self.goal_y - self.robot_y
        self.goal_distance = math.sqrt(dx**2 + dy**2)
        self.angle_to_goal = math.atan2(dy, dx)
        
        # Transiciones de estados
        if self.state == 'GTG':
            if self.goal_distance <= self.epsilon:
                self.state = 'STOP'
                self.get_logger().info("Estado cambiado a STOP, objetivo alcanzado.")
            elif abs(self.min_distance_to_obstacle - self.delta) < self.gamma:
                # Si hay un obstáculo cerca, decidir dirección de seguimiento de pared
                PIDController.reset
                self.follow_wall_direction_decision(self.angle_to_goal, self.min_angle_to_obstacle)
                
        if self.state in ['FW_C', 'FW_CC']:
            # Verificar la distancia al frente del robot para posible transición a AO
            front_distance = self.get_distance_at_angle(0.0)  # 0.0 rad es el ángulo al frente
            self.get_logger().info(f'Distancia frontal en FW: {front_distance:.2f}')
        #    # Verificar condiciones para volver al estado GTG
        #    if (self.obstacle_distance > self.delta and
        #        self.goal_distance < self.dt and
        #        self.compute_dot_product_u_FW_u_GTG() > 0):
        #        self.state = 'GTG'
        #        self.dt = None
        #        self.follow_wall_angle = None
        #        self.obstacle_distance = None
        #        self.get_logger().info("Transición de vuelta a GTG (ir hacia la meta)")
        #        
        #   # Verificar condición para cambiar al estado AO solo si la distancia es menor a delta - gamma
            if front_distance < self.delta:
                self.state = 'AO'
                self.ao_obstacle_angle = self.min_angle_to_obstacle
                self.dt = None
                self.follow_wall_angle = None
                self.obstacle_distance = None
                PIDController.reset
                self.get_logger().info("Transición a AO (evitar obstáculo), distancia al obstáculo muy cercana")
                
        elif self.state == 'AO':
            # Verificar si el obstáculo en la dirección de `ao_obstacle_angle` está suficientemente lejos
            avoid_angle = self.ao_obstacle_angle
            obstacle_distance_in_direction = self.get_distance_at_angle(avoid_angle)
            
            self.get_logger().info(f'Ángulo AO: {self.ao_obstacle_angle:.2f}, '
                                 f'Distancia en dirección evitación: {obstacle_distance_in_direction:.2f}')

            if obstacle_distance_in_direction >= self.delta:
                # Determinar las rotaciones para alinearse a la pared
                follow_wall_angle_clockwise = self.ao_obstacle_angle + math.pi/2
                follow_wall_angle_counterclockwise = self.ao_obstacle_angle - math.pi/2
                
                # Normalizar ángulos
                follow_wall_angle_clockwise = math.atan2(math.sin(follow_wall_angle_clockwise),
                                                       math.cos(follow_wall_angle_clockwise))
                follow_wall_angle_counterclockwise = math.atan2(math.sin(follow_wall_angle_counterclockwise),
                                                              math.cos(follow_wall_angle_counterclockwise))
                
                # Calcular rotaciones necesarias
                rotation_clockwise = abs(math.atan2(math.sin(follow_wall_angle_clockwise - self.robot_yaw),
                                                  math.cos(follow_wall_angle_clockwise - self.robot_yaw)))
                rotation_counterclockwise = abs(math.atan2(math.sin(follow_wall_angle_counterclockwise - self.robot_yaw),
                                                         math.cos(follow_wall_angle_counterclockwise - self.robot_yaw)))

                # Elegir la dirección de menor rotación
                if rotation_clockwise < rotation_counterclockwise:
                    self.state = 'FW_C'
                    self.follow_wall_angle = follow_wall_angle_clockwise
                else:
                    self.state = 'FW_CC'
                    self.follow_wall_angle = follow_wall_angle_counterclockwise
                
                self.dt = self.goal_distance
                self.ao_obstacle_angle = None
                self.pid_controller.reset()
                self.get_logger().info(f"Transición desde AO a {self.state}, "
                                     f"ángulo seguimiento: {self.follow_wall_angle:.2f}")

        if self.state != self.previous_state:
            self.pid_controller.reset()  # Resetear el controlador al cambiar de estado
            if self.state == 'AO':
                self.ao_obstacle_angle = self.min_angle_to_obstacle  # Recalcular ángulo de evitación basado en obstáculo actual
        self.previous_state = self.state
    
    # Ejecución de comportamiento según el estado
        if self.state == 'GTG':
            self.goal_control()
        elif self.state == 'AO':
            self.avoid_obstacle_control()
        elif self.state == 'FW_C':
            self.follow_wall_clockwise()
        elif self.state == 'FW_CC':
            self.follow_wall_counterclockwise()
        elif self.state == 'STOP':
            self.cmd_vel.linear.x = 0.0
            self.cmd_vel.angular.z = 0.0
            self.cmd_vel_pub.publish(self.cmd_vel)     
    
    def compute_dot_product_u_FW_u_GTG(self):
        u_GTG_x = math.cos(self.angle_to_goal)
        u_GTG_y = math.sin(self.angle_to_goal)

        u_FW_x = math.cos(self.follow_wall_angle)
        u_FW_y = math.sin(self.follow_wall_angle)

        return u_FW_x * u_GTG_x + u_FW_y * u_GTG_y

def main(args=None):            
    rclpy.init(args=args)       
    node = Robot_controller()   
    rclpy.spin(node)            
    node.destroy_node()         
    rclpy.shutdown()            

if __name__ == '__main__':      
    main()
