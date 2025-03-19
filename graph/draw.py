import networkx as nx
import matplotlib.pyplot as plt

def visualize_network(env):
    pos = nx.get_node_attributes(env.directed_graph, 'position')
    types = nx.get_node_attributes(env.directed_graph, 'node_type')
    
    color_map = {'nest': 'lightblue', 'food': 'red', 'distractor': 'grey'}
    node_colors = [color_map.get(types[n], 'gray') for n in env.directed_graph.nodes()]
    labels = {n: f"{types[n]}" for n in env.directed_graph.nodes()}

    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(env.directed_graph, pos, node_color=node_colors, node_size=700)
    nx.draw_networkx_labels(env.directed_graph, pos, labels, font_size=8)
    
    for (u, v, d) in env.directed_graph.edges(data=True):
        nx.draw_networkx_edges(
            env.directed_graph,
            pos,
            edgelist=[(u, v)],
            arrowstyle='-|>',
            arrowsize=15,
            edge_color='gray',
            connectionstyle='arc3,rad=0.1'
        )
        edge_label = {(u, v): f"{d['distance']:.1f}m ({d['direction']})"}
        nx.draw_networkx_edge_labels(
            env.directed_graph,
            pos,
            edge_labels=edge_label,
            font_color='black',
            label_pos=0.4,
            font_size=8,
            connectionstyle='arc3,rad=0.1'
        )
        
    plt.axis('off')
    plt.show()