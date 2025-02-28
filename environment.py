import math
import random
import networkx as nx
import matplotlib.pyplot as plt

class Environment:
    def __init__(self, num_distractors=4, connect_threshold=50.0, max_tries=1000, plot=False):
        """
        Initializes the environment/network.
        
        :param num_distractors: Number of distractor nodes
        :param connect_threshold: Controls the maximum Euclidean distance between two nodes 
                                  for them to be connected by an edge in the graph
        :param max_tries: Max number of attempts to generate a single connected graph
                          from nest (node 0) to food (node 1), skipping direct edge
        :param plot: If True, visualize the environment after creation
        """
        self.num_distractors = num_distractors
        self.connect_threshold = connect_threshold
        self.max_tries = max_tries
        self.nest_position = (0, 0)
        self.graph = nx.Graph()

        self._setup_environment()
        if plot:
            self.visualize_environment()

    def _setup_environment(self):
        """
        Generate the graph:
          1. Create nest (node 0) at (0,0)
          2. Create food (node 1) at a random position
          3. Create distractors (2..N)
          4. Form edges based on distance threshold, but skip nest->food direct edge
          5. Ensure no isolated node (only 1 cluster in the graph)
        """
        attempt = 0
        while attempt < self.max_tries:
            self.graph.clear()

            # create nest node
            self.graph.add_node(0,
                                position=self.nest_position,
                                type='Nest')
            # create food node
            food_position = self._random_position()
            self.graph.add_node(1,
                                position=food_position,
                                type='Food')

            # create distractors
            for node_index in range(2, self.num_distractors + 2):
                pos = self._random_position()
                self.graph.add_node(node_index,
                                    position=pos,
                                    type='Distractor')

            # connect nodes (except direct 0->1)
            self._connect_nodes()

            # check if the entire graph is a single connected component
            if nx.is_connected(self.graph):
                return

            attempt += 1

        raise RuntimeError(
            f"Failed to generate a single-cluster graph (no disjoint nodes)"
            f"within {self.max_tries} attempts"
        )

    def _connect_nodes(self):
        """
        Connect nodes pairwise if their distance is below 'connect_threshold',
        skipping the direct edge (0->1) between nest and food.
        """
        all_nodes = list(self.graph.nodes())
        positions = nx.get_node_attributes(self.graph, 'position')

        for i in all_nodes:
            for j in all_nodes:
                if i < j:
                    # skip direct edge (0->1)
                    if (i == 0 and j == 1) or (i == 1 and j == 0):
                        continue

                    dist_ij = self.calculate_distance(positions[i], positions[j])
                    if dist_ij <= self.connect_threshold:
                        self.graph.add_edge(i, j)

    def _random_position(self):
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
        from pos1 to pos2.
        
        Use this in the agent if/when you want to find the direction
        from the nest to another node.
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
        pos = nx.get_node_attributes(self.graph, 'position')
        types = nx.get_node_attributes(self.graph, 'type')
        colors = {'Nest': 'gold', 'Food': 'red', 'Distractor': 'skyblue'}

        labels = {node: f"{types[node]}" for node in self.graph.nodes()}

        nx.draw(
            self.graph,
            pos,
            labels=labels,
            with_labels=True,
            node_color=[colors[types[n]] for n in self.graph.nodes()],
            edge_color='gray',
            node_size=700,
            font_size=9
        )
        plt.show()

if __name__ == '__main__':
    env = Environment(num_distractors=4, connect_threshold=60.0, plot=True)
