import matplotlib.pyplot as plt
import sys
import csv

def make_plot(csv_file, out_png):
    data = {"gspan": [], "fsg": [], "gaston": []}
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = int(row['sup'])
            data["gspan"].append((s, float(row['gspan'])))
            data["fsg"].append((s, float(row['fsg'])))
            data["gaston"].append((s, float(row['gaston'])))
    
    plt.figure(figsize=(10, 6))
    
    sorted_data = sorted(data["gspan"])
    x = [v[0] for v in sorted_data]
    y = [v[1] for v in sorted_data]
    plt.plot(x, y, marker='o', label='gSpan')
    
    sorted_data = sorted(data["fsg"])
    x = [v[0] for v in sorted_data]
    y = [v[1] for v in sorted_data]
    plt.plot(x, y, marker='s', label='FSG')
    
    sorted_data = sorted(data["gaston"])
    x = [v[0] for v in sorted_data]
    y = [v[1] for v in sorted_data]
    plt.plot(x, y, marker='^', label='Gaston')
    
    plt.xlabel('Minimum Support (%)')
    plt.ylabel('Runtime (seconds)')
    plt.title('Frequent Subgraph Mining Algorithm Performance')
    plt.legend()
    plt.grid(True)
    plt.savefig(out_png)

if __name__ == "__main__":
    make_plot(sys.argv[1], sys.argv[2])