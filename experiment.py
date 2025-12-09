import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from std_srvs.srv import Empty
from geometry_msgs.msg import Pose
import time
import os
import xml.etree.ElementTree as ET
from geometry_msgs.msg import Pose, Point, Quaternion
import math

# Definindo a pose (x, y, z, roll, pitch, yaw)
def create_pose(x, y, z, roll, pitch, yaw):
    pose = Pose()
    pose.position = Point(x=x, y=y, z=z)
    
    # Convertendo a orientação de Roll, Pitch, Yaw para Quaternion
    q = quaternion_from_euler(roll, pitch, yaw)
    pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

    return pose

# Função para converter Euler para Quaternion
def quaternion_from_euler(roll, pitch, yaw):
    qx = math.sin(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) - math.cos(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)
    qy = math.cos(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2) + math.sin(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2)
    qz = math.cos(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2) - math.sin(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2)
    qw = math.cos(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) + math.sin(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)
    return [qx, qy, qz, qw]

class GazeboWorldManager(Node):

    def __init__(self):
        super().__init__('gazebo_world_manager')

        # Cliente para resetar o mundo
        self.reset_world_client = self.create_client(Empty, '/reset_world')
        # Cliente para deletar entidades
        self.delete_client = self.create_client(DeleteEntity, '/delete_entity')
        # Cliente para spawnar novos modelos
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')

        # Espera até que os serviços estejam disponíveis
        while not self.reset_world_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Aguardando serviço /reset_world...')
        while not self.delete_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Aguardando serviço /delete_entity...')
        while not self.spawn_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Aguardando serviço /spawn_entity...')

    def reset_world(self):
        """Função para resetar o mundo via o serviço ROS2."""
        req = Empty.Request()
        self.reset_world_client.call_async(req)

    def delete_model(self, model_name):
        """Deletar um modelo específico no Gazebo."""
        req = DeleteEntity.Request()
        req.name = model_name
        self.delete_client.call_async(req)

    def spawn_model(self, model_name, sdf_path, pose):
        """Função para spawnar um novo modelo no Gazebo."""
        req = SpawnEntity.Request()
        req.name = model_name

        # Carregar o SDF/URDF do modelo
        with open(sdf_path, 'r') as sdf_file:
            req.xml = sdf_file.read()

        req.initial_pose = pose
        self.spawn_client.call_async(req)

    def parse_model_pose(self, model_file):
        """Parseia o arquivo .sdf ou .model e extrai a pose."""
        tree = ET.parse(model_file)
        root = tree.getroot()

        # Extrai a pose do arquivo SDF
        pose_element = root.find('.//pose')
        if pose_element is not None:
            pose_values = list(map(float, pose_element.text.split()))
            pose = Pose()
            pose.position.x = pose_values[0]
            pose.position.y = pose_values[1]
            pose.position.z = pose_values[2]
            pose.orientation.x = pose_values[3]
            pose.orientation.y = pose_values[4]
            pose.orientation.z = pose_values[5]
            pose.orientation.w = 1.0  # Assumindo w como 1.0
            return pose
        return None

def main(args=None):
    rclpy.init(args=args)

    node = GazeboWorldManager()

    # Caminhos para os arquivos .sdf ou .model
    base_path = '/home/fabio/ros2_humble/src/turtlebot3_simulations/turtlebot3_gazebo/models/'

    # stage 1
    model_paths = {
        'burger': os.path.join(base_path, 'turtlebot3_burger/model-1_4.sdf'),
        'world_model': os.path.join(base_path, 'turtlebot3_dqn_world/model_stage_1and2/model.sdf'),
    }
    # stage 3
    # model_paths = {
    #     'burger': os.path.join(base_path, 'turtlebot3_burger/model.sdf'),
    #     'world_model': os.path.join(base_path, 'turtlebot3_dqn_world/model_stage_3/model.sdf'),
    # }

    # Exemplo de como deletar os modelos antigos
    node.get_logger().info('Deletando modelos antigos...')
    node.delete_model('turtlebot3_burger')
    node.delete_model('turtlebot3_dqn_world')
    node.delete_model('turtlebot3_dqn_obstacles')

    time.sleep(2)  # Aguarda um tempo para garantir que os modelos sejam deletados

    # Resetando o mundo
    node.get_logger().info('Resetando o mundo...')
    node.reset_world()

    time.sleep(2)  # Dá tempo para o mundo resetar

    model_paths = {
        'burger': os.path.join(base_path, 'turtlebot3_burger/model.sdf'),
        'world_model': os.path.join(base_path, 'turtlebot3_dqn_world/model_stage_3/model.sdf'),
        'obstacles': os.path.join(base_path, 'turtlebot3_dqn_world/obstacles_stage3/model.sdf'),
    }

    # Carregando as poses dinamicamente a partir dos arquivos
    world_model_pose = node.parse_model_pose(model_paths['world_model'])
    obstacles_pose = node.parse_model_pose(model_paths['obstacles'])
    # burger_pose = node.parse_model_pose(model_paths['burger'])
    burger_pose = create_pose(1.6, 1.6, 0.0, 0.0, 0.0, 4.0)
    node.get_logger().info(f'Pose do Burger: posição=({burger_pose.position.x}, {burger_pose.position.y}, {burger_pose.position.z}), '
                       f'orientação=({burger_pose.orientation.x}, {burger_pose.orientation.y}, '
                       f'{burger_pose.orientation.z}, {burger_pose.orientation.w})')


    if burger_pose is None or world_model_pose is None:
        node.get_logger().error('Falha ao carregar as poses dos arquivos SDF!')
        return
    
    # Spawn dos modelos com poses dinâmicas
    node.get_logger().info('Spawnando novos modelos...')
    node.spawn_model('turtlebot3_burger', model_paths['burger'], burger_pose)
    node.spawn_model('turtlebot3_dqn_world', model_paths['world_model'], world_model_pose)
    time.sleep(5)
    node.spawn_model('turtlebot3_dqn_obstacles', model_paths['obstacles'], obstacles_pose)

    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
