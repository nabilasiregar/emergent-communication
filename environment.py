import math
import random
import networkx as nx
import matplotlib.pyplot as plt

class Environment:
    def __init__(self, num_nodes=6, edge_connection_prob=0.3, max_tries=100, plot=False):
        """
        Initializes the environment.

        :param num_nodes: Total number of nodes in the graph
        :param edge_connection_prob: Probability for each ordered pair of nodes to have an edge
        :param max_tries: Maximum attempts to generate a valid graph
        :param plot: If True, visualize the environment after creation
        """
        self.num_nodes = num_nodes
        self.edge_connection_prob = edge_connection_prob
        self.max_tries = max_tries
        self.graph = nx.DiGraph()
        self.nest = None
        self.food = None

        self._setup_environment()
        if plot:
            self.visualize_environment()

    def _setup_environment(self):
        attempt = 0

        while attempt < self.max_tries:
            self.graph.clear()
            # create nodes with random positions
            for node in range(self.num_nodes):
                pos = self._random_position()
                self.graph.add_node(node, position=pos, type='Distractor')

            # randomly add directed edges
            nodes = list(self.graph.nodes())
            for i in nodes:
                for j in nodes:
                    if i == j:
                        continue
                    if random.random() < self.edge_connection_prob:
                        self.graph.add_edge(i, j)

            # randomly choose nest and food nodes from the graph.
            self.nest, self.food = random.sample(nodes, 2)
            self.graph.nodes[self.nest]['type'] = 'Nest'
            self.graph.nodes[self.food]['type'] = 'Food'

            # remove any direct connection between nest and food (vice versa)
            if self.graph.has_edge(self.nest, self.food):
                self.graph.remove_edge(self.nest, self.food)
            if self.graph.has_edge(self.food, self.nest):
                self.graph.remove_edge(self.food, self.nest)

            # every node must have at least one connection
            all_have_edge = all(self.graph.degree(n) > 0 for n in nodes)
            undirected = self.graph.to_undirected()
            connected = nx.is_connected(undirected)
            # there must be a directed path from nest to food
            path_exists = nx.has_path(self.graph, self.nest, self.food)

            if all_have_edge and connected and path_exists:
                self._assign_edge_attributes()
                return

            attempt += 1

        raise RuntimeError(
            f"Failed to generate a valid graph after {self.max_tries} trials"
        )

    def _assign_edge_attributes(self):
        """Assigns distance and direction to each edge based on node positions"""
        positions = nx.get_node_attributes(self.graph, 'position')
        for u, v in self.graph.edges():
            pos_u = positions[u]
            pos_v = positions[v]
            distance = self.calculate_distance(pos_u, pos_v)
            direction = self.calculate_direction(pos_u, pos_v)
            self.graph[u][v]['distance'] = distance
            self.graph[u][v]['direction'] = direction

    def _random_position(self):
        """Generates a random (x, y) position"""
        return (random.randint(-100, 100), random.randint(-100, 100))

    def calculate_distance(self, pos1, pos2):
        """Calculates the Euclidean distance between two points"""
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        return math.hypot(dx, dy)

    def calculate_direction(self, pos1, pos2):
        """
        Computes the direction from pos1 to pos2 to one of the 8 quadrants
        """
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0:
            angle += 360

        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        idx = int((angle + 22.5) // 45) % 8
        return directions[idx]

    def visualize_environment(self):
        """Visualizes the directed graph with node types and edge attributes"""
        pos = nx.get_node_attributes(self.graph, 'position')
        types = nx.get_node_attributes(self.graph, 'type')
        colors = {'Nest': 'gold', 'Food': 'red', 'Distractor': 'skyblue'}

        node_colors = [colors.get(types[n], 'gray') for n in self.graph.nodes()]
        labels = {n: f"{types[n]}" for n in self.graph.nodes()}

        plt.figure(figsize=(8, 6))
        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=700)
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=10)
        nx.draw_networkx_edges(self.graph, pos, arrowstyle='->', arrowsize=15, edge_color='gray')

        edge_labels = {(u, v): f"{self.graph[u][v]['distance']:.1f}m\n{self.graph[u][v]['direction']}"
                       for u, v in self.graph.edges()}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=8)

        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    env = Environment(num_nodes=6, edge_connection_prob=0.3, plot=True)
    if nx.has_path(env.graph, env.nest, env.food):
        path = nx.shortest_path(env.graph, source=env.nest, target=env.food, weight="distance")
        
        print(f"Path from Nest ({env.nest}) to Food ({env.food}):")
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_data = env.graph[u][v]
            print(f"  {u} -> {v} | Distance: {edge_data['distance']:.2f}m, Direction: {edge_data['direction']}")
    else:
        print("No valid path from Nest to Food!")
