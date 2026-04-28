import networkx as nx
import matplotlib.pyplot as plt

G = nx.read_graphml("./cascadewatch_graph.graphml")

print("Nodes:", len(G.nodes))
print("Edges:", len(G.edges))

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)

nx.draw(
    G, pos,
    node_size=10,
    edge_color='gray',
    with_labels=False
)

plt.title("CascadeWatch Graph")
plt.show()