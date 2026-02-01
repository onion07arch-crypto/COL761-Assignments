import sys
import numpy as np
import networkx as nx
from collections import defaultdict

def read_graphs_fast(filename):
    graphs = []
    current_graph = None
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('#'):
                if current_graph is not None:
                    graphs.append(current_graph)
                
                current_graph = {'vertices': [], 'edges': []}
            elif line.startswith('v'):
                parts = line.split()
                current_graph['vertices'].append((int(parts[1]), int(parts[2])))
            elif line.startswith('e'):
                parts = line.split()
                current_graph['edges'].append((int(parts[1]), int(parts[2]), int(parts[3])))
        
        if current_graph is not None:
            graphs.append(current_graph)
    
    return graphs

def read_fragments(filename):
    fragments = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                continue
            
            if '|' in line:
                parts = line.split('|')
                if len(parts) >= 4:
                    node_part = parts[0].strip('()')
                    edge_part = parts[1].strip('()')
                    
                    node_labels = tuple(map(int, node_part.split(','))) if node_part else ()
                    edge_labels = tuple(map(int, edge_part.split(','))) if edge_part else ()
                    num_nodes = int(parts[2])
                    num_edges = int(parts[3])
                    
                    fragments.append({
                        'node_labels': node_labels,
                        'edge_labels': edge_labels,
                        'num_nodes': num_nodes,
                        'num_edges': num_edges
                    })
    
    return fragments

def check_fragment_presence(graph_data, fragment):
    graph_nodes = [label for _, label in graph_data['vertices']]
    graph_edges = [label for _, _, label in graph_data['edges']]
    
    fragment_nodes = sorted(fragment['node_labels'])
    fragment_edges = sorted(fragment['edge_labels'])
    
    if len(fragment_nodes) > len(graph_nodes):
        return False
    
    if len(fragment_edges) > len(graph_edges):
        return False
    
    graph_node_counter = Counter(graph_nodes)
    graph_edge_counter = Counter(graph_edges)
    
    frag_node_counter = Counter(fragment_nodes)
    frag_edge_counter = Counter(fragment_edges)
    
    for label, count in frag_node_counter.items():
        if graph_node_counter[label] < count:
            return False
    
    for label, count in frag_edge_counter.items():
        if graph_edge_counter[label] < count:
            return False
    
    return True

def convert_to_features_fast(graphs_file, fragments_file, output_file):
    print(f"Reading graphs from {graphs_file}")
    graphs = read_graphs_fast(graphs_file)
    print(f"Read {len(graphs)} graphs")
    
    print(f"Reading fragments from {fragments_file}")
    fragments = read_fragments(fragments_file)
    print(f"Read {len(fragments)} fragments")
    
    features = np.zeros((len(graphs), len(fragments)), dtype=int)
    
    for i, graph_data in enumerate(graphs):
        if i % 100 == 0:
            print(f"Processing graph {i}/{len(graphs)}")
        
        for j, fragment in enumerate(fragments):
            if check_fragment_presence(graph_data, fragment):
                features[i, j] = 1
    
    np.save(output_file, features)
    print(f"Saved feature vectors to {output_file}")
    print(f"Feature matrix shape: {features.shape}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 fast_convert.py <graphs_file> <fragments_file> <output_file>")
        sys.exit(1)
    
    convert_to_features_fast(sys.argv[1], sys.argv[2], sys.argv[3])