from __future__ import annotations

import itertools
import math
import random
from typing import Tuple

import networkx as nx
from graph.draw import visualize_network
class Environment:
    def __init__(
        self,
        num_nodes: int = 20,
        extra_degree: float = 0.5,
        max_tries: int = 100,
        world_size: Tuple[int, int] = (100, 100),
        seed: int | None = None,
        allow_adjacent_nest_food: bool = False
    ):
        if num_nodes < 2:
            raise ValueError("Need at least 2 nodes: one nest and one food")

        self.num_nodes = num_nodes
        self.extra_degree = extra_degree
        self.max_tries = max_tries
        self.world_size = world_size
        self.seed = seed
        self.rng = random.Random(self.seed)
        self.allow_adjacent_nest_food = allow_adjacent_nest_food
        self.directed_graph: nx.DiGraph = nx.DiGraph()

        self._create_random_graph()
        self._assign_node_attributes()
        self._assign_edge_attributes()

    @staticmethod
    def _uniform_random_tree(n: int, seed: int | None = None) -> nx.Graph:
        return nx.random_labeled_tree(n, seed=seed)

    def _create_random_graph(self) -> None:
        """
        1) If num_nodes == 2, create two nodes and a bidirectional edge.
        2) Else: sample a uniform random labeled tree
        3) Add O(n) shortcuts so avg-degree approx. 2 + extra_degree
        4) Convert to a bidirectional graph
        """
        self.directed_graph.clear()

        if self.num_nodes == 2:
            u, v = 0, 1
            self.directed_graph.add_nodes_from([u, v])
            self.directed_graph.add_edge(u, v)
            self.directed_graph.add_edge(v, u)
            return

        undirected: nx.Graph = self._uniform_random_tree(self.num_nodes, seed=self.seed)

        # randomly add a few extra edges on top of the tree
        p = self.extra_degree / self.num_nodes
        for u, v in itertools.combinations(undirected.nodes(), 2):
            if undirected.has_edge(u, v):
                continue
            if self.rng.random() < p:
                undirected.add_edge(u, v)

        # add bidirectional edges
        self.directed_graph.add_nodes_from(undirected.nodes())
        self.directed_graph.add_edges_from(undirected.edges())
        self.directed_graph.add_edges_from((v, u) for u, v in undirected.edges())

    def _assign_node_attributes(self) -> None:
        nodes = list(self.directed_graph.nodes())

        if self.num_nodes == 2:
            nest, food = nodes
            for n in nodes:
                self.directed_graph.nodes[n]["node_type"] = (
                    "nest" if n == nest else "food"
                )
            w, h = self.world_size
            for n in nodes:
                self.directed_graph.nodes[n]["position"] = (
                    self.rng.uniform(0, w),
                    self.rng.uniform(0, h),
                )
            return

        graph = self.directed_graph.to_undirected()
        for _ in range(self.max_tries):
            nest = self.rng.choice(nodes)

            # this is to generate dataset that allows 1 hop
            if self.allow_adjacent_nest_food:
                non_neighbours = [n for n in nodes if n != nest]
            else:
                non_neighbours = [n for n in nodes if n != nest and not graph.has_edge(nest, n)]

            if not non_neighbours:
                self._create_random_graph()
                graph = self.directed_graph.to_undirected()
                nodes = list(self.directed_graph.nodes())
                continue

            food = self.rng.choice(non_neighbours)
            for n in nodes:
                self.directed_graph.nodes[n]["node_type"] = (
                    "nest" if n == nest else "food" if n == food else "distractor"
                )
            w, h = self.world_size
            for n in nodes:
                self.directed_graph.nodes[n]["position"] = (
                    self.rng.uniform(0, w),
                    self.rng.uniform(0, h),
                )
            return
        raise RuntimeError("Could not choose non-adjacent nest/food pair")

    def _assign_edge_attributes(self) -> None:
        for u, v in self.directed_graph.edges():
            x1, y1 = self.directed_graph.nodes[u]["position"]
            x2, y2 = self.directed_graph.nodes[v]["position"]
            dx, dy = x2 - x1, y2 - y1

            distance = math.hypot(dx, dy)
            angle = math.degrees(math.atan2(dy, dx)) % 360
            dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
            direction = dirs[int((angle + 22.5) // 45) % 8]

            self.directed_graph.edges[u, v]["distance"] = distance
            self.directed_graph.edges[u, v]["direction"] = direction


def path_stats(env, use_directed: bool = True, max_paths: int = 10_000) -> None:
    G = env.directed_graph if use_directed else env.directed_graph.to_undirected()
    nest = next(n for n, d in G.nodes(data=True) if d["node_type"] == "nest")
    food = next(n for n, d in G.nodes(data=True) if d["node_type"] == "food")

    paths_iter = nx.all_simple_paths(G, source=nest, target=food)
    paths = list(itertools.islice(paths_iter, max_paths + 1))

    print(f"total simple paths = {len(paths) if len(paths) <= max_paths else '> ' + str(max_paths)}")
    shortest = min(paths, key=len)
    print("shortest path ({} hops):".format(len(shortest) - 1), " → ".join(map(str, shortest)))

if __name__ == "__main__":
    import statistics

    def hop_stats(runs=1000, **cfg):
        hops = []
        for _ in range(runs):
            env = Environment(**cfg)
            g = env.directed_graph.to_undirected()
            nest = next(n for n, d in g.nodes(data=True) if d["node_type"] == "nest")
            food = next(n for n, d in g.nodes(data=True) if d["node_type"] == "food")
            hops.append(nx.shortest_path_length(g, nest, food) - 1)
        return statistics.mean(hops), statistics.stdev(hops)

    avg, sd = hop_stats(num_nodes=30, extra_degree=0.05)
    print(f"avg intermediate nodes = {avg:.2f}  (σ = {sd:.2f})")
    env = Environment(num_nodes=30, extra_degree=0.05)
    path_stats(env, use_directed=False)
    visualize_network(env)