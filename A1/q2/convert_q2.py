import sys
mappers = ['Br', 'C', 'Cl', 'F', 'H', 'I', 'N', 'O', 'P', 'S', 'Si']

def convert_yeast(input_file):
    with open(input_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    f_gspan = open("yeast_gspan.txt", "w")
    f_fsg = open("yeast_fsg.txt", "w")
    f_gaston = open("yeast_gaston.txt", "w")

    idx = 0
    graph_count = 0
    
    while idx < len(lines):
        if idx >= len(lines):
            break
            
        if not lines[idx].startswith('#'):
            idx += 1
            continue
            
        idx += 1
        
        if idx >= len(lines):
            break
        
        try:
            num_v = int(lines[idx])
        except:
            idx += 1
            continue
        idx += 1
        
        vertices = []
        v_count = 0
        while v_count < num_v and idx < len(lines) and not lines[idx].startswith('#'):
            vertices.append(lines[idx])
            idx += 1
            v_count += 1
        
        if len(vertices) != num_v:
            while idx < len(lines) and not lines[idx].startswith('#'):
                idx += 1
            continue
        
        f_gspan.write(f"t # {graph_count}\n")
        f_fsg.write(f"t # {graph_count}\n")
        f_gaston.write(f"t # {graph_count}\n")
        
        for v_id in range(num_v):
            label = vertices[v_id]
            f_gspan.write(f"v {v_id} {label}\n")
            f_fsg.write(f"v {v_id} {label}\n")
            if label in mappers:
                gaston_label = str(mappers.index(label))
            else:
                gaston_label = "0"
            f_gaston.write(f"v {v_id} {gaston_label}\n")
        
        while idx < len(lines) and lines[idx].startswith('#'):
            idx += 1
        
        if idx >= len(lines):
            f_gspan.write("\n")
            f_fsg.write("\n")
            f_gaston.write("\n")
            graph_count += 1
            continue
        
        try:
            num_e = int(lines[idx])
        except:
            idx += 1
            f_gspan.write("\n")
            f_fsg.write("\n")
            f_gaston.write("\n")
            graph_count += 1
            continue
        idx += 1
        
        e_count = 0
        edges_added = 0
        while e_count < num_e and idx < len(lines) and not lines[idx].startswith('#'):
            line = lines[idx]
            parts = line.replace(',', ' ').split()
            
            if len(parts) >= 3:
                try:
                    u = int(parts[0])
                    v = int(parts[1])
                    lbl = parts[2]
                    
                    if 0 <= u < num_v and 0 <= v < num_v:
                        f_gspan.write(f"e {u} {v} {lbl}\n")
                        f_fsg.write(f"u {u} {v} {lbl}\n")
                        if lbl in mappers:
                            gaston_lbl = str(mappers.index(lbl))
                        else:
                            gaston_lbl = "0"
                        f_gaston.write(f"e {u} {v} {gaston_lbl}\n")
                        edges_added += 1
                except:
                    pass
            idx += 1
            e_count += 1
        
        if edges_added == 0:
            print(f"Warning: Graph {graph_count} has no valid edges")
        
        graph_count += 1
        
        if graph_count < 1000:
            f_gspan.write("\n")
            f_fsg.write("\n")
            f_gaston.write("\n")
    
    f_gspan.close()
    f_fsg.close()
    f_gaston.close()
    
    print(f"Converted {graph_count} graphs")
    return graph_count

if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = convert_yeast(sys.argv[1])
        print(result)