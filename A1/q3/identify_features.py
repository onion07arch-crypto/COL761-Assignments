import sys
import numpy as np
from collections import Counter

def read_graphs_simple(filename):
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

def extract_fragment_signature(graph_data):
    vertices = sorted(graph_data['vertices'])
    edges = sorted(graph_data['edges'])
    return (tuple(vertices), tuple(edges), len(vertices), len(edges))

def identify_features_simple(database_file, output_file):
    print(f"Reading database from {database_file}")
    graphs = read_graphs_simple(database_file)
    print(f"Read {len(graphs)} graphs")
    
    if len(graphs) > 1000:
        print(f"Using first 1000 graphs for speed")
        graphs = graphs[:1000]
    
    fragment_counter = Counter()
    
    for idx, graph_data in enumerate(graphs):
        if idx % 100 == 0:
            print(f"Processing graph {idx}/{len(graphs)}")
        
        signature = extract_fragment_signature(graph_data)
        fragment_counter[signature] += 1
    
    print(f"Total unique fragments: {len(fragment_counter)}")
    
    k = 50
    if len(fragment_counter) < k:
        k = len(fragment_counter)
    
    top_fragments = fragment_counter.most_common(k)
    
    print(f"Selected {len(top_fragments)} fragments")
    
    with open(output_file, 'w') as f:
        for frag, count in top_fragments:
            node_labels, edge_labels, num_nodes, num_edges = frag
            
            node_str = ','.join(map(str, node_labels))
            edge_str = ','.join(map(str, edge_labels))
            f.write(f"{node_str}|{edge_str}|{num_nodes}|{num_edges}\n")
    
    print(f"Saved fragments to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 identify_features.py <database_file> <output_file>")
        sys.exit(1)
    
    identify_features_simple(sys.argv[1], sys.argv[2])