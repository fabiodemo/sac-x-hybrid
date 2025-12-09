import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from std_srvs.srv import Empty
from geometry_msgs.msg import Pose, Point, Quaternion
import os
import math
import time

def create_pose(x, y, z, roll, pitch, yaw):
    """Cria uma Pose."""
    pose = Pose()
    pose.position = Point(x=x, y=y, z=z)
    q = quaternion_from_euler(roll, pitch, yaw)
    pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
    return pose

def quaternion_from_euler(roll, pitch, yaw):
    """Converte ângulos de Euler para Quaternion."""
    qx = math.sin(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) - math.cos(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)
    qy = math.cos(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2) + math.sin(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2)
    qz = math.cos(roll / 2) * math.cos(pitch / 2) * math.sin(yaw / 2) - math.sin(roll / 2) * math.sin(pitch / 2) * math.cos(yaw / 2)
    qw = math.cos(roll / 2) * math.cos(pitch / 2) * math.cos(yaw / 2) + math.sin(roll / 2) * math.sin(pitch / 2) * math.sin(yaw / 2)
    return [qx, qy, qz, qw]

class GazeboWorldManager(Node):
    def __init__(self):
        super().__init__('gazebo_world_manager')

        # Inicializando os clientes para os serviços
        self.reset_client = self.create_client(Empty, '/reset_world')
        self.delete_client = self.create_client(DeleteEntity, '/delete_entity')
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')

    def reset_world(self):
        """Reseta o mundo."""
        req = Empty.Request()
        future = self.reset_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

    def delete_model(self, model_name):
        """Deleta um modelo."""
        req = DeleteEntity.Request()
        req.name = model_name
        future = self.delete_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

    def spawn_model(self, model_name, sdf_path, pose):
        """Spawn de novo modelo."""
        req = SpawnEntity.Request()
        req.name = model_name
        with open(sdf_path, 'r') as sdf_file:
            req.xml = sdf_file.read()
        req.initial_pose = pose
        future = self.spawn_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

def main(args=None):
    rclpy.init(args=args)
    node = GazeboWorldManager()

    # Caminho dos modelos
    base_path = '/home/fabio/ros2_humble/src/turtlebot3_simulations/turtlebot3_gazebo/models/'
    model_paths = {
        'burger': os.path.join(base_path, 'turtlebot3_burger/model.sdf'),
        'world': os.path.join(base_path, 'turtlebot3_dqn_world/model_stage_3/model.sdf'),
        'obstacles': os.path.join(base_path, 'turtlebot3_dqn_world/obstacles_stage3/model.sdf')
    }

    # Deletando modelos antigos
    node.get_logger().info('Deletando modelos antigos...')
    node.delete_model('turtlebot3_burger')
    node.delete_model('turtlebot3_dqn_world')
    node.delete_model('turtlebot3_dqn_obstacles')

    # Resetando o mundo
    node.get_logger().info('Resetando o mundo...')
    node.reset_world()
    time.sleep(2)  # Garantir que o reset tenha tempo para ser processado

    # Criando poses
    # Coordenadas do turtlebot3_burger replicadas da interface Gazebo
    # burger_pose = create_pose(
    #     1.599955,  # x
    #     1.599950,  # y
    #     0.008535,  # z
    #     0.000100,  # roll
    #     0.005735,  # pitch
    #    -2.283181   # yaw (em radianos)
    # )
    burger_pose = create_pose(
        10.0,  # x
        10.0,  # y
        10.0,  # z
        10.0,  # roll
        10.0,  # pitch
        10.0   # yaw (em radianos)
    )

    world_pose = create_pose(3.0, 3.0, 0.0, 0.0, 0.0, 0.0)
    obstacles_pose = create_pose(1.0, 1.0, 0.0, 0.0, 0.0, 0.0)

    # Spawnando novos modelos
    node.get_logger().info('Spawnando novos modelos...')
    node.spawn_model('turtlebot3_burger', model_paths['burger'], burger_pose)
    node.spawn_model('turtlebot3_dqn_world', model_paths['world'], world_pose)
    node.spawn_model('turtlebot3_dqn_obstacles', model_paths['obstacles'], obstacles_pose)

    node.get_logger().info('Ambiente resetado e modelos spawnados com sucesso.')
    rclpy.shutdown()

if __name__ == '__main__':
    main()
