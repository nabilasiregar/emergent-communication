import networkx as nx
import matplotlib.pyplot as plt

def visualize_environment(env):
    pos = nx.get_node_attributes(env.graph, 'pos')
    types = nx.get_node_attributes(env.graph, 'type')

    color_map = {'Nest': 'gold', 'Food': 'red', 'Distractor': 'skyblue'}
    node_colors = [color_map.get(types[n], 'gray') for n in env.graph.nodes()]
    labels = {n: f"{types[n]}" for n in env.graph.nodes()}

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(env.graph, pos, node_color=node_colors, node_size=700)
    nx.draw_networkx_labels(env.graph, pos, labels, font_size=10)
    
    for (u, v, d) in env.graph.edges(data=True):
        nx.draw_networkx_edges(env.graph, pos, edgelist=[(u, v)], arrowstyle='-|>', arrowsize=15,
                            edge_color='gray', connectionstyle='arc3,rad=0.1')
        edge_label = { (u, v): f"{d['distance']:.1f}m ({d['direction']})" }
        nx.draw_networkx_edge_labels(env.graph, pos, edge_labels=edge_label, font_color='black', label_pos=0.4, font_size=8, connectionstyle='arc3,rad=0.1')
        
    plt.axis('off')
    plt.show()