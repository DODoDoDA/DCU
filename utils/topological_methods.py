import torch
import os

def generate_dynamic_topology(n,density=0.5):
    assert 0 <= density <= 1
    upper_tri = torch.rand(n, n)
    upper_tri = torch.triu(upper_tri, 1) 
    threshold = 1 - density
    upper_tri = (upper_tri > threshold).int()
    topology = upper_tri + upper_tri.T
    topology.fill_diagonal_(1)  
    return topology


def compute_weight_matrix(topology):
    if isinstance(topology, list):
        topology = torch.tensor(topology, dtype=torch.int32)

    N = topology.size(0)
    degrees = topology.sum(dim=1)  
    W = torch.zeros(N, N)

    for i in range(N):
        for j in range(N):
            if i != j and topology[i, j] == 1:  
                W[i, j] = 1 / max(degrees[i], degrees[j])
        W[i, i] = 1 - W[i].sum()

    return W

def gen_round_topologies(round_num, density, client_num, file_path):

    folder = os.path.dirname(file_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    with open(file_path, "w") as f:
        for round_idx in range(round_num):
            topology = generate_dynamic_topology(client_num, density)
            f.write(f"Round {round_idx + 1}:\n")
            for row in topology:
                f.write(" ".join(map(str, row.tolist())) + "\n")
            f.write("\n") 

    print(f"Topology for {round_num} rounds saved to {file_path}")

def load_topologies(file_path):
    topologies = []
    with open(file_path, "r") as f:
        current_topology = []
        for line in f:
            line = line.strip()
            if not line:
                if current_topology:
                    topologies.append(current_topology)
                    current_topology = []

            elif "Round" not in line:
                row = list(map(int, line.split()))
                current_topology.append(row)

        if current_topology:
            topology_tensor = torch.tensor(current_topology, dtype=torch.int32)
            topologies.append(topology_tensor)

    return topologies

# gen_round_topologies(5, 0.6, 5, "./topology/topologies.txt")
# topologies = load_topologies("./topology/topologies.txt")
