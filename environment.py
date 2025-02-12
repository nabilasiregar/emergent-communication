import math
import random
import networkx as nx
import matplotlib.pyplot as plt

class Environment:
    def __init__(self, num_distractors=4):
        """
        Initializes the game environment.
        :param num_distractors: Number of distractor nodes
        """
        self.graph = nx.Graph()
        self.nest_position = (0, 0) # the nest is always at the center
        self.setup_environment(num_distractors)

    def setup_environment(self, num_distractors):
        # create the nest node
        self.graph.add_node(0,
                            position=self.nest_position,
                            type='Nest',
                            direction='Center')

        # create the food node
        food_position = self.random_position()
        food_direction = self.calculate_direction(self.nest_position, food_position)
        self.graph.add_node(1,
                            position=food_position,
                            type='Food',
                            direction=food_direction)
        self.graph.add_edge(0, 1)
        distance = self.calculate_distance(self.nest_position, food_position)
        self.graph[0][1]['distance'] = distance

        # create distractor nodes
        for node_index in range(2, num_distractors + 2):
            pos = self.random_position()
            direction = self.calculate_direction(self.nest_position, pos)
            self.graph.add_node(node_index,
                                position=pos,
                                type='Distractor',
                                direction=direction)
            self.graph.add_edge(0, node_index)
            distance = self.calculate_distance(self.nest_position, pos)
            self.graph[0][node_index]['distance'] = distance

    def random_position(self):
        """Generates a random (x, y) position"""
        return (random.randint(-100, 100), random.randint(-100, 100))

    def calculate_distance(self, pos1, pos2):
        """Calculates Euclidean distance between two points"""
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        return math.hypot(dx, dy)

    def calculate_direction(self, pos1, pos2):
        """
        Computes a cardinal/intercardinal direction into 8 quadrants
        from pos1 to pos2
        """
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        angle = math.degrees(math.atan2(dy, dx))
        if angle < 0:
            angle += 360

        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        index = int((angle + 22.5) / 45) % 8
        return directions[index]

    def visualize_environment(self):
        """Visualizes the environment using matplotlib and networkx."""
        pos = {node: self.graph.nodes[node]['position'] for node in self.graph.nodes()}
        types = {node: self.graph.nodes[node]['type'] for node in self.graph.nodes()}
        directions = {node: self.graph.nodes[node]['direction'] for node in self.graph.nodes()}
        colors = {'Nest': 'gold', 'Food': 'red', 'Distractor': 'skyblue'}

        # Label nodes with type and direction.
        labels = {node: f"{types[node]} ({directions[node]})" for node in self.graph.nodes()}

        # Label edges with their distance.
        edge_labels = {(u, v): f"{self.graph[u][v]['distance']:.1f}m" for u, v in self.graph.edges()}

        nx.draw(self.graph,
                pos,
                labels=labels,
                with_labels=True,
                node_color=[colors[types[node]] for node in self.graph.nodes()],
                edge_color='gray',
                node_size=700,
                font_size=9)
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_color='red')
        plt.show()

if __name__ == '__main__':
    env = Environment(num_distractors=4)
    env.visualize_environment()
