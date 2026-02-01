import sys
import numpy as np

def generate_candidates_simple(db_features_file, query_features_file, output_file):
    db_features = np.load(db_features_file)
    query_features = np.load(query_features_file)
    
    
    num_queries = query_features.shape[0]
    num_db = db_features.shape[0]
    
    with open(output_file, 'w') as f:
        for q_idx in range(num_queries):
            q_vec = query_features[q_idx]
            
            candidates = []
            for db_idx in range(num_db):
                db_vec = db_features[db_idx]
                
                if np.all(q_vec <= db_vec):
                    candidates.append(db_idx + 1)
            
            f.write(f"q # {q_idx + 1}\n")
            if candidates:
                f.write(f"c # {' '.join(map(str, candidates))}\n")
            else:
                f.write(f"c #\n")
            
            if (q_idx + 1) % 10 == 0:
                print(f"Processed {q_idx + 1}/{num_queries} queries")
    

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(1)
    
    generate_candidates_simple(sys.argv[1], sys.argv[2], sys.argv[3])