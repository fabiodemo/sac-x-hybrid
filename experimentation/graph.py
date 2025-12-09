import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from visualization_msgs.msg import Marker, MarkerArray

class LidarGraphNode(Node):
    def __init__(self):
        super().__init__('lidar_graph_node')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )
        self.graph = nx.Graph()
        self.marker_pub = self.create_publisher(MarkerArray, 'lidar_graph_markers', 10)
        
        # Diretório para salvar os arquivos de grafo
        self.save_dir = "graphs/"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Configuração para Matplotlib em modo interativo
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

    def lidar_callback(self, msg):
        # Limpa o grafo a cada varredura para atualização
        self.graph.clear()  
        turtlebot_node = (0, 0)
        self.graph.add_node(turtlebot_node)  # Nó central (Turtlebot)

        # Convertendo dados do LIDAR para coordenadas cartesianas
        for i, distance in enumerate(msg.ranges):
            if 0 < distance < msg.range_max:  # Filtra valores inválidos
                angle = msg.angle_min + i * msg.angle_increment
                x = distance * np.cos(angle)
                y = distance * np.sin(angle)
                obstacle_node = (x, y)
                self.graph.add_node(obstacle_node)
                self.graph.add_edge(turtlebot_node, obstacle_node)  # Conectando Turtlebot aos obstáculos próximos

        # Salvar o grafo em formato GEXF
        nx.write_gexf(self.graph, os.path.join(self.save_dir, "grafo_ambiente.gexf"))

        # Salvar o grafo em formato JSON
        with open(os.path.join(self.save_dir, "grafo_ambiente.json"), "w") as f:
            json.dump(nx.node_link_data(self.graph), f)

        # Visualização em tempo real com Matplotlib
        self.ax.clear()  # Limpa a figura antes de redesenhar
        nx.draw(self.graph, pos={n: n for n in self.graph.nodes()}, ax=self.ax, with_labels=False, node_size=10)
        plt.draw()
        plt.pause(0.01)  # Pausa para atualizar o gráfico

        # Publicação dos nós como Markers para visualização no RViz
        markers = MarkerArray()
        for i, node in enumerate(self.graph.nodes()):
            # Verifica se o nó é uma tupla com coordenadas (x, y)
            if isinstance(node, tuple) and len(node) == 2:
                marker = Marker()
                marker.header.frame_id = "base_link"
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = float(node[0])
                marker.pose.position.y = float(node[1])
                marker.pose.position.z = 0.0
                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.1
                marker.color.a = 1.0
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.id = i
                markers.markers.append(marker)
            else:
                self.get_logger().info(f"Nó inválido detectado: {node}")
                
        self.marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    lidar_graph_node = LidarGraphNode()
    rclpy.spin(lidar_graph_node)
    lidar_graph_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
