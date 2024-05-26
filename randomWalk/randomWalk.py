import numpy as np
import scipy.sparse as sp
from .kgEmbedding import kgEmbedding
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pyvis.network import Network

class randomWalk:
    def __init__(self, kg_embedding, alpha=0.85):
        self.kg_embedding = kg_embedding
        self.alpha = alpha  
    
    def build_adjacency_matrix(self):
        num_entities = self.kg_embedding.triples_factory.num_entities
        rows, cols = [], []

        for (head_id, _, tail_id) in self.kg_embedding.triples_factory.mapped_triples:
            rows.append(head_id)
            cols.append(tail_id)

        data = np.ones(len(rows), dtype=int)  
        adjacency_matrix = sp.coo_matrix((data, (rows, cols)), shape=(num_entities, num_entities))

        return adjacency_matrix.tocsr()

    def build_transition_matrix(self):
        entity_embeddings = self.kg_embedding.pipeline_result.model.entity_representations[0]._embeddings.weight.detach().cpu().numpy()

        # FOR DEBUGGING PURPOSES
        print(entity_embeddings)

        adjacency_matrix = self.build_adjacency_matrix()

        # FOR DEBUGGING PURPOSES
        print(adjacency_matrix)

        similarity_matrix = np.dot(entity_embeddings, entity_embeddings.T)

        # FOR DEBUGGING PURPOSES
        print("Similarity Matrix")
        print(similarity_matrix)

        weighted_adjacency = adjacency_matrix.multiply(similarity_matrix).tocsr()

        # FOR DEBUGGING PURPOSES
        print("Weighted Adjacency")
        print(weighted_adjacency)

        row_sums = np.array(weighted_adjacency.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1
        normalization = sp.diags(1 / row_sums)

        transition_probabilities = normalization.dot(weighted_adjacency)

        n = transition_probabilities.shape[0]
        self.transition_matrix = self.alpha * transition_probabilities + (1 - self.alpha) * sp.eye(n)

    def random_walk(self, num_steps=100):
        n = self.transition_matrix.shape[0]
        v = np.random.rand(n)
        v /= v.sum()

        for _ in range(num_steps):
            v = v @ self.transition_matrix
        
        return v

def visualize_graph_with_weights_2(kg_embedding, transition_matrix):
    G = nx.DiGraph()

    entity_to_id = kg_embedding.triples_factory.entity_to_id
    for entity in entity_to_id:
        G.add_node(entity)

    rows, cols = transition_matrix.nonzero()
    weights = [transition_matrix[row, col] for row, col in zip(rows, cols)]

    for row, col, weight in zip(rows, cols, weights):
        if weight > 0.01:  # Filter out insignificant weights
            source = list(entity_to_id.keys())[row]
            target = list(entity_to_id.keys())[col]
            # Skip self-looping edges
            if source != target:
                G.add_edge(source, target, weight=np.round(weight, 3))

    pos = nx.circular_layout(G, scale=2)
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, font_size=12)

    edge_labels = {(u, v): f'{d["weight"]:.2f}' for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrowstyle='->', arrowsize=5, edge_color='gray')

    plt.title('Knowledge Graph Transition Probabilities')
    plt.axis('off')  # Hide the axes
    plt.show()

def visualize_graph_with_weights(kg_embedding, transition_matrix):
    G = nx.DiGraph()

    entity_to_id = kg_embedding.triples_factory.entity_to_id
    for entity in entity_to_id:
        G.add_node(entity)

    rows, cols = transition_matrix.nonzero()
    weights = [transition_matrix[row, col] for row, col in zip(rows, cols)]

    for row, col, weight in zip(rows, cols, weights):
        if weight > 0.01:
            source = list(entity_to_id.keys())[row]
            target = list(entity_to_id.keys())[col]
            G.add_edge(source, target, weight=np.round(weight, 3))

    pos = nx.circular_layout(G, scale=2)

    edge_x = []
    edge_y = []
    edge_text = []
    edge_text_x = []
    edge_text_y = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        edge_text.append(f'{edge[2]["weight"]:.2f}')
        edge_text_x.append(mid_x)
        edge_text_y.append(mid_y)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines')

    edge_text_trace = go.Scatter(
        x=edge_text_x, y=edge_text_y,
        mode='text',
        text=edge_text,
        textfont=dict(
            family="sans serif",
            size=16,
            color="LightSeaGreen"),
        hoverinfo='none')

    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            color=[],
            line_width=2),
        text=node_text,
        textposition="top center")

    fig = go.Figure(data=[edge_trace, edge_text_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    fig.show()

# FOR TESTING PURPOSES
if __name__ == "__main__":
    kg = kgEmbedding()
    rw = randomWalk(kg)
    rw.build_transition_matrix()
    visualize_graph_with_weights_2(rw.kg_embedding, rw.transition_matrix)