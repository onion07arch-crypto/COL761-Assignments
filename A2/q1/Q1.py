import sys
import urllib.request
import json
import numpy as np
import matplotlib.pyplot as plt

MY_ID = "siy257621"

def load_data(source):
    if source.endswith('.npy'):
        return np.load(source)
    
    url = f"http://10.208.23.248:3000/dataset?student_id={MY_ID}&dataset_num={source}"
    try:
        with urllib.request.urlopen(url) as r:
            raw = r.read().decode('utf-8')
            content = json.loads(raw)
            return np.array(content["X"])
    except:
        sys.exit(1)

def run_kmeans(data, k):
    idx = np.random.choice(len(data), k, replace=False)
    cent = data[idx]
    
    for _ in range(100):
        diff = data[:, np.newaxis] - cent
        dist = np.sum(diff**2, axis=2)
        labels = np.argmin(dist, axis=1)
   
        new_cent = []
        for i in range(k):
            cluster_pts = data[labels == i]
            if len(cluster_pts) > 0:
                new_cent.append(cluster_pts.mean(axis=0))
            else:
                new_cent.append(cent[i])
        
        new_cent = np.array(new_cent)
        if np.all(cent == new_cent):
            break
        cent = new_cent

    wcss = 0
    for i in range(k):
        pts = data[labels == i]
        if len(pts) > 0:
            wcss += np.sum((pts - cent[i])**2)
    return wcss

def main():
    if len(sys.argv) != 2:
        return
        
    X = load_data(sys.argv[1])
    print({X.shape})
    # print(X[:5])
    
    ks = range(1, 16) 
    costs = []
    
    for k in ks:
        best_for_k = min([run_kmeans(X, k) for _ in range(3)])
        costs.append(best_for_k)

    plt.figure()
    plt.plot(ks, costs, 'b-o')
    plt.xlabel('k')
    plt.ylabel('WCSS(Within cluster sum of squares)')
    plt.title('Elbow Plot')
    plt.savefig('plot.png')

    if len(costs) > 2:
        slopes = np.diff(costs)
        acceleration = np.diff(slopes)
        best_k = ks[np.argmax(acceleration) + 1]
        print(best_k)
    else:
        print(1)

if __name__ == "__main__":
    main()