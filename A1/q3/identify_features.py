import sys
import numpy as np
from collections import Counter
import random
import itertools

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

def extract_small_fragments(graph_data, max_nodes=5):
    fragments = []
    vertices = graph_data['vertices']
    edges = graph_data['edges']
    
    if len(vertices) <= max_nodes:
        fragments.append((tuple(sorted(vertices)), tuple(sorted(edges)), len(vertices), len(edges)))
    else:
        for size in range(1, max_nodes + 1):
            vertex_counts = Counter(vertices)
            
            for label, count in vertex_counts.items():
                if count >= size:
                    node_fragment = tuple([label] * size)
                    
                    edge_count = edges.count(0)
                    edge_fragment = tuple([0] * min(edge_count, size-1))
                    
                    fragments.append((node_fragment, edge_fragment, size, len(edge_fragment)))
    
    return list(set(fragments))

def identify_discriminative_features(database_file, output_file, k=50, max_nodes=4):
    print(f"Reading database from {database_file}")
    graphs = read_graphs_simple(database_file)
    print(f"Read {len(graphs)} graphs")
    
    if len(graphs) > 1000:
        sample_size = 1000
        print(f"Using {sample_size} random graphs for feature selection")
        graphs = random.sample(graphs, sample_size)
    
    fragment_counter = Counter()
    
    for idx, graph_data in enumerate(graphs):
        if idx % 100 == 0:
            print(f"Processing graph {idx}/{len(graphs)}")
        
        fragments = extract_small_fragments(graph_data, max_nodes=max_nodes)
        
        for frag in fragments:
            fragment_counter[frag] += 1
    
    print(f"Total unique fragments: {len(fragment_counter)}")
    
    total_graphs = len(graphs)
    
    fragments_with_scores = []
    for frag, count in fragment_counter.items():
        frequency = count / total_graphs
        node_labels, edge_labels, num_nodes, num_edges = frag
        
        discriminative_score = abs(frequency - 0.3)
        
        fragments_with_scores.append({
            'fragment': frag,
            'frequency': frequency,
            'score': discriminative_score,
            'count': count,
            'node_labels': node_labels,
            'edge_labels': edge_labels,
            'num_nodes': num_nodes,
            'num_edges': num_edges
        })
    
    fragments_with_scores.sort(key=lambda x: (-x['score'], -x['count']))
    
    selected_count = min(k, len(fragments_with_scores))
    selected_fragments = fragments_with_scores[:selected_count]
    
    print(f"Selected {len(selected_fragments)} discriminative fragments")
    print(f"Fragment size range: {min(f['num_nodes'] for f in selected_fragments)} to {max(f['num_nodes'] for f in selected_fragments)} nodes")
    print(f"Frequency range: {selected_fragments[0]['frequency']:.3f} to {selected_fragments[-1]['frequency']:.3f}")
    
    with open(output_file, 'w') as f:
        for frag_info in selected_fragments:
            node_str = ','.join(map(str, frag_info['node_labels']))
            edge_str = ','.join(map(str, frag_info['edge_labels']))
            num_nodes = frag_info['num_nodes']
            num_edges = frag_info['num_edges']
            f.write(f"{node_str}|{edge_str}|{num_nodes}|{num_edges}\n")
    
    print(f"Saved fragments to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 identify_features.py <database_file> <output_file>")
        sys.exit(1)
    
    identify_discriminative_features(sys.argv[1], sys.argv[2])