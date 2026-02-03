import sys
import subprocess
import os
import time
import csv
import re

def run_algorithm(algo_name, executable, input_file, support_percent, output_dir, total_graphs):
    output_file = f"{output_dir}/{algo_name}{support_percent}"
    
    support_absolute = int(total_graphs * support_percent / 100)
    if support_absolute < 1:
        support_absolute = 1
    
    print(f"  Support: {support_percent}% = {support_absolute}/{total_graphs} graphs", flush=True)
    
    start_time = time.time()
    
    if algo_name == "gspan":
        cmd = [executable, "-f", input_file, "-s", str(support_absolute), "-o", "-i"]
        print(f"  Command: {' '.join(cmd)}", flush=True)
        
        try:
            with open(output_file, 'w') as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, timeout=300)
        except subprocess.TimeoutExpired:
            print(f"    TIMEOUT", flush=True)
            return float('inf')
            
    elif algo_name == "fsg":
        cmd = [executable, "-s", str(support_percent), input_file, output_file]
        print(f"  Command: {' '.join(cmd)}", flush=True)
        
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        except subprocess.TimeoutExpired:
            print(f"    TIMEOUT", flush=True)
            return float('inf')

    # elif algo_name == "gaston":
    #     cmd = [executable, str(support_percent), input_file, output_file]
    #     print(f"  Command: {' '.join(cmd)}", flush=True)
    
    #     try:
    #         result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
    #         with open(output_file, 'wb') as f:
    #             f.write(result.stdout)
    #     except subprocess.TimeoutExpired:
    #         print(f"    TIMEOUT: Gaston exceeded 5 minutes", flush=True)
    #         return float('inf')

    elif algo_name == "gaston":
        cmd = [executable, str(support_absolute), input_file]
        print(f"  Command: {' '.join(cmd)}", flush=True)
        
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                timeout=300, universal_newlines=True)
            
            with open(output_file, 'w') as f:
                f.write(result.stdout)
                
        except subprocess.TimeoutExpired:
            print(f"    TIMEOUT", flush=True)
            return float('inf')
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    if os.path.exists(output_file):
        size = os.path.getsize(output_file)
        print(f"    Output file: {size} bytes", flush=True)
    
    return elapsed

def main():

    if len(sys.argv) != 6:
        print("Usage command: python3 run_q2.py <gspan_exe> <fsg_exe> <gaston_exe> <dataset> <output_dir>", flush=True)
        sys.exit(1)
    
    gspan_exe = sys.argv[1]
    fsg_exe = sys.argv[2]
    gaston_exe = sys.argv[3]
    dataset = sys.argv[4]
    output_dir = sys.argv[5]

    
    if not os.path.exists(dataset):
        print(f"\n Dataset file '{dataset}' not found!", flush=True)
        sys.exit(1)
    
    os.makedirs(output_dir, exist_ok=True)

    conv_result = subprocess.run(
        ["python3", "convert_q2.py", dataset],
        capture_output=True,
        text=True
    )
    
    output_lines = conv_result.stdout.strip().split('\n')
    graph_count = 0
    
    for line in reversed(output_lines):
        line = line.strip()
        if line.isdigit():
            graph_count = int(line)
            break
        elif "Converted" in line:
            numbers = re.findall(r'\d+', line)
            if numbers:
                graph_count = int(numbers[0])
                break
    
    if graph_count == 0:
        try:
            with open("yeast_gspan.txt", 'r') as f:
                graph_count = f.read().count("t #")
        except:
            sys.exit(1)

    print(f"\nGenerated files:", flush=True)
    for fname in ["yeast_gspan.txt", "yeast_fsg.txt", "yeast_gaston.txt"]:
        if os.path.exists(fname):
            size = os.path.getsize(fname)
            lines = sum(1 for _ in open(fname))
            print(f"  {fname}: {size:,} bytes, {lines:,} lines", flush=True)
        else:
            print(f"  {fname}: NOT FOUND!", flush=True)
    
    supports = [95, 50, 25, 10, 5]
    results = []
    
    for i, sup in enumerate(supports):
        
        row = {"sup": sup}
        
        gspan_time = run_algorithm("gspan", gspan_exe, "yeast_gspan.txt", sup, output_dir, graph_count)
        row["gspan"] = gspan_time
        
        fsg_time = run_algorithm("fsg", fsg_exe, "yeast_fsg.txt", sup, output_dir, graph_count)
        row["fsg"] = fsg_time
  
        gaston_time = run_algorithm("gaston", gaston_exe, "yeast_gaston.txt", sup, output_dir, graph_count)
        row["gaston"] = gaston_time
        
        results.append(row)
        
        print(f"\n Completed support level {sup}%", flush=True)
        gspan_str = "TIMEOUT" if gspan_time == float('inf') else f"{gspan_time:.2f}s"
        fsg_str = "TIMEOUT" if fsg_time == float('inf') else f"{fsg_time:.2f}s"
        gaston_str = "TIMEOUT" if gaston_time == float('inf') else f"{gaston_time:.2f}s"
        
    
    
    csv_file = os.path.join(output_dir, "results.csv")
    
    with open(csv_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=["sup", "gspan", "fsg", "gaston"])
        writer.writeheader()
        writer.writerows(results)
    
    print("CSV file contents:", flush=True)
    with open(csv_file, 'r') as f:
        print(f.read(), flush=True)
    
    print(f"\nGenerating plot...", flush=True)
    plot_result = subprocess.run(
        ["python3", "plot_q2.py", csv_file, f"{output_dir}/plot.png"],
        capture_output=True,
        text=True
    )
    
    if plot_result.returncode == 0:
        print(f" Plot saved to: {output_dir}/plot.png", flush=True)
    else:
        print(f" Plot generation failed!", flush=True)

    
    print(f"\nOutput files in '{output_dir}':", flush=True)
    files = os.listdir(output_dir)
    for f in sorted(files):
        if os.path.isfile(os.path.join(output_dir, f)):
            size = os.path.getsize(os.path.join(output_dir, f))
            print(f"  {f}: {size:,} bytes", flush=True)

if __name__ == "__main__":
    main()