import sys
import numpy as np
from collections import Counter

def read_graphs_fast(filename):
    graphs = []
    current_graph = {'vertices': [], 'edges': []}
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('#'):
                if current_graph['vertices'] or current_graph['edges']:
                    graphs.append(current_graph)
                current_graph = {'vertices': [], 'edges': []}
            elif line.startswith('v'):
                parts = line.split()
                current_graph['vertices'].append(int(parts[2]))
            elif line.startswith('e'):
                parts = line.split()
                current_graph['edges'].append(int(parts[3]))
    
    if current_graph['vertices'] or current_graph['edges']:
        graphs.append(current_graph)
    
    return graphs

def read_patterns_simple(filename):
    patterns = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            patterns.append(line.split(','))
    
    return patterns

def check_pattern(graph_data, pattern):
    vertices = graph_data['vertices']
    edges = graph_data['edges']
    
    vertex_counter = Counter(vertices)
    edge_counter = Counter(edges)
    
    for p in pattern:
        if p.startswith('v'):
            if '_pair' in p:
                label = int(p[1:].split('_')[0])
                if vertex_counter.get(label, 0) >= 2:
                    continue
                else:
                    return False
            else:
                label = int(p[1:])
                if vertex_counter.get(label, 0) >= 1:
                    continue
                else:
                    return False
        elif p.startswith('e'):
            label = int(p[1:])
            if edge_counter.get(label, 0) >= 1:
                continue
            else:
                return False
        elif p == 'has_edge':
            if len(edges) >= 1:
                continue
            else:
                return False
    
    return True

def convert_to_features_simple(graphs_file, patterns_file, output_file):
    print(f"Reading graphs from {graphs_file}")
    graphs = read_graphs_fast(graphs_file)
    print(f"Read {len(graphs)} graphs")
    
    print(f"Reading patterns from {patterns_file}")
    patterns = read_patterns_simple(patterns_file)
    print(f"Read {len(patterns)} patterns")
    
    features = np.zeros((len(graphs), len(patterns)), dtype=int)
    
    for i, graph_data in enumerate(graphs):
        if i % 100 == 0:
            print(f"Processing graph {i}/{len(graphs)}")
        
        for j, pattern in enumerate(patterns):
            if check_pattern(graph_data, pattern):
                features[i, j] = 1
    
    np.save(output_file, features, allow_pickle=False)
    print(f"Saved features to {output_file}")
    print(f"Feature matrix shape: {features.shape}")
    
    return features

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 convert_to_features.py <graphs_file> <patterns_file> <output_file>")
        sys.exit(1)
    
    convert_to_features_simple(sys.argv[1], sys.argv[2], sys.argv[3])