import random
import numpy as np
import scipy.sparse as sp

def read_fpga_matrix(path):
    with open(path, 'r') as file:
        a, b = map(int, file.readline().split())
        row_ind = []
        col_ind = []
        data = []
        for _ in range(b):
            u, v = map(int, file.readline().split())
            row_ind.append(u)
            col_ind.append(v)
            data.append(1)
            row_ind.append(v)
            col_ind.append(u)
            data.append(1)
    fpga_matrix = sp.csr_matrix((data, (row_ind, col_ind)), shape=(a, a))
    return a, fpga_matrix


def assign_points_to_fpgas(n, a):

    points = list(range(n))
    random.shuffle(points)


    points_per_fpga = [[] for _ in range(a)]


    base = n // a
    extra = n % a
    max_points = int(3 * n / a)


    current = 0
    for i in range(a):
        num_points = base + (1 if i < extra else 0)
        if num_points > max_points:
            num_points = max_points
        points_per_fpga[i] = points[current:current + num_points]
        current += num_points

    if current != n:
        raise ValueError("Not all points were assigned due to max_points constraint.")


    fpga_assignment = [-1] * n
    for i in range(a):
        for p in points_per_fpga[i]:
            fpga_assignment[p] = i

    return points_per_fpga, fpga_assignment


def find(parent, x):
    if parent[x] != x:
        parent[x] = find(parent, parent[x])
    return parent[x]

def union(parent, x, y):
    root_x = find(parent, x)
    root_y = find(parent, y)
    if root_x != root_y:
        parent[root_y] = root_x


def generate_netlist(n, a, alpha, output_path, fpga_matrix, points_per_fpga, fpga_assignment):
    netlist = []
    total_nets = 0
    k_min = 2
    k_max = min(n, 1000)
    parent = list(range(n)) 


    S_fpga = [[] for _ in range(a)]
    for f0 in range(a):
        adj_f = fpga_matrix.indices[fpga_matrix.indptr[f0]:fpga_matrix.indptr[f0+1]]
        S = []
        S.extend(points_per_fpga[f0])
        for f in adj_f:
            S.extend(points_per_fpga[f]) 
        S_fpga[f0] = S


    for k in range(k_min, k_max + 1):
        num_nets = int(0.3 * n * (k ** -alpha))
        total_nets += num_nets
        for _ in range(num_nets):
            p0 = random.randint(0, n - 1)
            f0 = fpga_assignment[p0]


            if len(S_fpga[f0]) < k:
                continue
            net_rest = random.sample(S_fpga[f0], k - 1)
            net = [p0] + net_rest
            netlist.append(net)

            for p in net[1:]:
                union(parent, net[0], p)



    for f in range(a):
        points_f = points_per_fpga[f]
        if len(points_f) < 2:
            continue
        for i in range(len(points_f) - 1):
            p1 = points_f[i]
            p2 = points_f[i+1]
            if find(parent, p1) != find(parent, p2):
                netlist.append([p1, p2])
                total_nets += 1
                union(parent, p1, p2) 


    for i in range(a):
        start = fpga_matrix.indptr[i]
        end = fpga_matrix.indptr[i + 1]
        for idx in range(start, end):
            j = fpga_matrix.indices[idx]
            if i < j: 
                if not points_per_fpga[i] or not points_per_fpga[j]:
                    continue
                p1 = random.choice(points_per_fpga[i])
                p2 = random.choice(points_per_fpga[j]) 
                if find(parent, p1) != find(parent, p2):
                    netlist.append([p1, p2])
                    total_nets += 1
                    union(parent, p1, p2)


    root = find(parent, 0)
    is_connected = all(find(parent, i) == root for i in range(n))
    if is_connected:
        print("successful!")
    else:
        print("Warning.")



    with open(output_path, 'w') as file:
        file.write(f"{n}\n")
        file.write(f"{total_nets}\n")
        for net in netlist:
            file.write(" ".join(map(str, net)) + "\n")

        for f in range(a):
            if points_per_fpga[f]:
                fixed_node = points_per_fpga[f][0]
                file.write(f"{fixed_node}\n")


def main():
    alpha = 1.79
    fpga_path = "./ICCAD2021-TopoPart-Benchmarks/FPGA_Graph/FPGA_Graph/MFS1"
    a, fpga_matrix = read_fpga_matrix(fpga_path)
    n_values = [int(2e7)]

    for n in n_values:
        for j in [1,2]:
            output_path = f"./ICCAD2021-TopoPart-Benchmarks/generated_large_netlists/MFS1/fpga_based_netlist_{n}_{j}.txt"
            points_per_fpga, fpga_assignment = assign_points_to_fpgas(n, a)
            generate_netlist(n, a, alpha, output_path, fpga_matrix, points_per_fpga, fpga_assignment)

if __name__ == "__main__":
    main()