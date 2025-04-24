//should work for DATE initial/refinement, something is wrong with cutsize++
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <algorithm>
#include <sstream>
#include <queue>
#include <set>
#include <cmath>
#include <ctime>
#include <unordered_set>
#include <unordered_map>
#include <Eigen/Sparse>
#include <boost/heap/fibonacci_heap.hpp>
#include <boost/functional/hash.hpp>
#include <stack>

using Eigen::SparseMatrix;
using Eigen::Triplet;

// Equivalent of Python's csr_matrix
typedef SparseMatrix<int, Eigen::RowMajor> CSRMatrix;

CSRMatrix create_FPGA_matrix(const std::string& file_path) {
    std::ifstream f(file_path);
    int n, m;
    f >> n >> m;
  
    std::vector<Triplet<int>> triplets;
  
    for (int i = 0; i < m; ++i) {
        int u, v;
        f >> u >> v;
        if (u != v) {
          triplets.push_back(Triplet<int>(u, v, 1));
          triplets.push_back(Triplet<int>(v, u, 1)); // Add for transpose
        }
    }
  
    CSRMatrix A(n, n);
    A.setFromTriplets(triplets.begin(), triplets.end());
  
    return A;
}

std::tuple<CSRMatrix, std::vector<std::tuple<int, int>>> create_netlist_matrix(const std::string& file_path) {
    std::ifstream f(file_path);
    int n, m, u, v;
    std::string line;
    std::getline(f, line);
    n = std::stoi(line);
    std::getline(f, line);
    m = std::stoi(line);
  
    std::vector<Triplet<int>> triplets;
    for (int i = 0; i < m; ++i) {
        std::getline(f, line);
        std::stringstream ss(line);
        
        ss >> u;
        while (ss >> v) {
            if (u != v) {
                triplets.push_back(Triplet<int>(u, v, 1));
                triplets.push_back(Triplet<int>(v, u, 1)); // Add for symmetry
            }
        }
    }
  
    CSRMatrix A(n, n);
    A.setFromTriplets(triplets.begin(), triplets.end());
  
    std::vector<std::tuple<int, int>> fixed_nodes;
    int fpga_id = 0;
    while (std::getline(f, line)) {
        std::stringstream ss(line);
        int node_id;
        while (ss >> node_id) {
            fixed_nodes.push_back(std::make_tuple(node_id, fpga_id));
        }
        fpga_id++;
    }
  
    return std::make_tuple(A, fixed_nodes);
}


// New function to read fixed nodes
std::vector<std::tuple<int, int>> read_fixed_nodes(const std::string& file_path) {
    std::ifstream f(file_path);
    std::string line;
    std::vector<std::tuple<int, int>> fixed_nodes;
    int fpga_id = 0;
    while (std::getline(f, line)) {
        std::stringstream ss(line);
        int node_id;
        while (ss >> node_id) {
            fixed_nodes.push_back(std::make_tuple(node_id, fpga_id));
        }
        fpga_id++;
    }
    return fixed_nodes;
}

void DFS(int node, const Eigen::SparseMatrix<int, Eigen::RowMajor> &mat, 
         std::vector<int> &component, int label, int &currentComponentSize) {
    std::stack<int> stack;
    stack.push(node);

    while (!stack.empty()) {
        int current = stack.top();
        stack.pop();

        if (component[current] != -1) 
            continue;

        component[current] = label;

        ++currentComponentSize;

        for (Eigen::SparseMatrix<int, Eigen::RowMajor>::InnerIterator it(mat, current); it; ++it) {
            int neighbor = it.index();
            if (component[neighbor] == -1) {
                stack.push(neighbor);
            }
        }
    }
}

std::unordered_set<int> findLargestConnectedComponent(const Eigen::SparseMatrix<int, Eigen::RowMajor> &mat) {
    int n = mat.rows();
    std::vector<int> component(n, -1);
    int label = 0; 
    int maxSize = 0; 
    int maxLabel = -1;

    for (int i = 0; i < n; ++i) {
        if (component[i] == -1) {
            int currentComponentSize = 0;
            DFS(i, mat, component, label, currentComponentSize);

            if (currentComponentSize > maxSize) {
                maxSize = currentComponentSize;
                maxLabel = label;
            }

            ++label;
        }
    }
    
    std::unordered_set<int> largestComponent;
    for (int i = 0; i < n; ++i) {
        if (component[i] == maxLabel) {
            largestComponent.insert(i);
        }
    }
    
    return largestComponent;
}

std::vector<double> dijkstra_shortest_paths(const Eigen::SparseMatrix<int, Eigen::RowMajor>& graph, int start_node) {
    int n = graph.rows();
    std::vector<double> distances(n, std::numeric_limits<double>::infinity());
    distances[start_node] = 0;

    using P = std::pair<double, int>;  // pair (distance, node)
    std::priority_queue<P, std::vector<P>, std::greater<P>> pq;
    pq.push({0.0, start_node});

    while (!pq.empty()) {
        double dist = pq.top().first;
        int node = pq.top().second;
        pq.pop();

        if (dist > distances[node])
            continue;

        for (Eigen::SparseMatrix<int, Eigen::RowMajor>::InnerIterator it(graph, node); it; ++it) {
            int neighbor = it.index();
            double newDist = dist + 1.0/ it.value();
            if (newDist < distances[neighbor]) {
                distances[neighbor] = newDist;
                pq.push({newDist, neighbor});
            }
        }
    }

    return distances;
}


std::vector<std::tuple<int, int>> find_fixed_nodes(const CSRMatrix&netlist_matrix, int num_fpgas){

    int num_nodes = netlist_matrix.rows();

    std::unordered_set<int> largestComponent = findLargestConnectedComponent(netlist_matrix);

    // Create a set to store nodes outside the largest component
    std::unordered_set<int> exclude_nodes;

    // Add nodes from all components except the largest one
    for (int node_id = 0; node_id < num_nodes; ++node_id) {
        if (largestComponent.count(node_id)==0) {
            exclude_nodes.insert(node_id);
        }
    }

    // Choose the source node (node 1 in this case)
    int start_node = 1;
    while (exclude_nodes.count(start_node)) {
        start_node++;
    }

    std::unordered_set<int> fixed_nodes;
    int k = num_fpgas;
    // Calculate distances from the source node to all other nodes
    std::vector<double> distances = dijkstra_shortest_paths(netlist_matrix, start_node);

    double max_distance = 0;
    int max_distance_node = -1;
    for (int i = 0; i < num_nodes; ++i) {
        if (exclude_nodes.count(i)) {
            continue;
        }
        if (distances[i] > max_distance) {
            max_distance = distances[i];
            max_distance_node = i;
        }
    }
    fixed_nodes.insert(max_distance_node);

    distances = dijkstra_shortest_paths(netlist_matrix, max_distance_node);

    // Initialize a dictionary to store distances for each fixed node
    std::unordered_map<int, std::vector<double>> distances_dict;
    distances_dict[max_distance_node] = distances;

    for (int i = 1; i < k; ++i) {
        max_distance = 0;
        max_distance_node = -1;
        for (int j = 0; j < num_nodes; ++j) {
            if (fixed_nodes.count(j) || exclude_nodes.count(j)) {
                continue;
            }
            if (distances[j] > max_distance) {
                max_distance = distances[j];
                max_distance_node = j;
            }
        }
        if (max_distance_node == -1) {
            std::cout << "error in fixed node searching." << std::endl;
            std::exit(0);
        }
        fixed_nodes.insert(max_distance_node);
        distances_dict[max_distance_node] = dijkstra_shortest_paths(netlist_matrix, max_distance_node);

        for(int j=0; j<num_nodes; ++j){
            distances[j] = std::min(distances[j], distances_dict[max_distance_node][j]);
        }
    }
    
    std::vector<std::tuple<int, int>>fixed_nodes_in_fpga;
    int fpga_id=0;
    
    bool more_fixed_nodes = false;
    std::unordered_set<int> fixed_nodes_copy=fixed_nodes;
    for (int node : fixed_nodes_copy) {
        fixed_nodes_in_fpga.emplace_back(node, fpga_id);
        if(more_fixed_nodes){
            int num=0;
            int threshold=2;
            for (Eigen::SparseMatrix<int, Eigen::RowMajor>::InnerIterator it(netlist_matrix, node); it; ++it) {
                int neighbor = it.col();
                if (num >= threshold)
                    break;
                if (fixed_nodes.count(neighbor))
                    continue;
                fixed_nodes.insert(neighbor);
                fixed_nodes_in_fpga.emplace_back(neighbor, fpga_id);
            }
        }
        fpga_id++;
    }
    
    
    return fixed_nodes_in_fpga;
}



std::vector<std::tuple<int, int>> find_reachable_nodes(const CSRMatrix& A, 
                                                      int start_node, 
                                                      int max_distance, 
                                                      const std::unordered_set<int>& ignored_nodes_set) {
    std::vector<std::tuple<int, int>> reachable_nodes;
    std::unordered_set<int> visited;
    std::queue<std::pair<int, int>> queue;
    queue.push({start_node, 0});

    while (!queue.empty()) {
        auto [node_id, distance] = queue.front();
        queue.pop();

        if (visited.count(node_id)) {
            continue;
        }

        visited.insert(node_id);
        reachable_nodes.push_back({node_id, distance});

        if (distance < max_distance) {
            for (CSRMatrix::InnerIterator it(A, node_id); it; ++it) {
                int neighbor = it.index();
                if (visited.count(neighbor) == 0 && ignored_nodes_set.count(neighbor) == 0) {
                    queue.push({neighbor, distance + 1});
                }
            }
        }
    }

    return reachable_nodes;
}

std::tuple<CSRMatrix, std::vector<int>, std::vector<std::vector<int>>, std::vector<int>, std::vector<std::unordered_set<int>>, std::unordered_set<int>> coarsen_graph(
    CSRMatrix& graph, 
    std::vector<std::unordered_set<int>>& CP, 
    std::vector<int>& node_weight,
    int capacity,
    std::unordered_set<int>fixed_nodes_set,
    int num_fpgas) 
{   
    int n = graph.rows();

    std::vector<int> mapping(n,0); 
    std::vector<std::vector<int>> remapping;
    std::vector<int> new_node_weight;
    std::unordered_set<int> merged_nodes;
    std::vector<std::unordered_set<int>> new_CP;
    std::unordered_set<int>new_fixed_nodes_set;
    std::vector<int> degree(n, 0);
    
    for (int node_id = 0; node_id < n; ++node_id) {
        for (CSRMatrix::InnerIterator it(graph, node_id); it; ++it) {
            degree[node_id] += it.value();
        }
    }
    
    int new_node_id = 0;
    std::vector<std::tuple<double, int, int>> neighbor_priority;
    int threshold = 3; //std::round(2*num_fpgas/3) or 3;
    
    for (const int& node_id : fixed_nodes_set) {
        merged_nodes.insert(node_id);
        new_fixed_nodes_set.insert(new_node_id);
        mapping[node_id] = new_node_id;
        remapping.push_back({node_id});
        new_node_weight.push_back(node_weight[node_id]);
        new_CP.push_back(CP[node_id]);
        new_node_id++;
    }
    
    for (int node_id = 0; node_id < n; ++node_id) {
    
        if (merged_nodes.find(node_id) != merged_nodes.end())
            continue;
        
        for (CSRMatrix::InnerIterator it(graph, node_id); it; ++it) {
            int node_id2 = it.index();
            if (merged_nodes.find(node_id2) != merged_nodes.end() || node_id2 <= node_id || node_weight[node_id] + node_weight[node_id2] > capacity){
                continue;
            }

            double A = (double)it.value() / (node_weight[node_id] * node_weight[node_id2]);
            double B = (double)it.value() * it.value() / (degree[node_id] * degree[node_id2]);
            
            double priority = A;

            neighbor_priority.push_back({priority, node_id, node_id2});
        }

    }
    
    std::sort(neighbor_priority.begin(), neighbor_priority.end(),
          [](const std::tuple<double, int, int> &a, const std::tuple<double, int, int> &b) {
              return std::get<0>(a) > std::get<0>(b);
          });
    

    for (const auto& [_, node_id, node_id2] : neighbor_priority) {
    
        if (merged_nodes.find(node_id) != merged_nodes.end() || merged_nodes.find(node_id2) != merged_nodes.end()) {
            continue;
        }
        std::unordered_set<int> intersection;
        for (int element : CP[node_id]) {
            if (CP[node_id2].count(element)) {
                intersection.insert(element);
            }
        }
        if (intersection.size()<threshold){
            continue;
        }
        // merge node_id and neighbor_id
        merged_nodes.insert(node_id);
        merged_nodes.insert(node_id2);
        mapping[node_id] = new_node_id;
        mapping[node_id2] = new_node_id;
        remapping.push_back({node_id, node_id2});
        new_node_weight.push_back(node_weight[node_id] + node_weight[node_id2]);
        new_CP.push_back(intersection);
        new_node_id++;
    }
    
    for (int node_id = 0; node_id < n; ++node_id) {
        if (merged_nodes.find(node_id) == merged_nodes.end()){
            merged_nodes.insert(node_id);
            mapping[node_id] = new_node_id;
            remapping.push_back({node_id});
            new_node_weight.push_back(node_weight[node_id]);
            new_CP.push_back(CP[node_id]);
            new_node_id++;
        }
    }
    
    // std::cout<<"pair number:"<<num1<<" pair successful number:"<<num2<<" "<<num3<<" "<<num4<<std::endl;
    // std::cout<<new_fixed_nodes_set.size()<<std::endl;
    
    int k = new_node_id;

    std::vector<Triplet<int>> triplets;
    for (int i = 0; i < n; ++i) {
        triplets.emplace_back(i, mapping[i], 1);
    }
    CSRMatrix X(n, k);
    X.setFromTriplets(triplets.begin(), triplets.end());
    
    CSRMatrix Xt = X.transpose();
    
    CSRMatrix new_graph = Xt * graph * X;
    
    
    std::vector<Triplet<int>> tripletList;

    // Step 2: Iterate through all non-zero elements of new_graph
    for (int k = 0; k < new_graph.outerSize(); ++k) {
        for (CSRMatrix::InnerIterator it(new_graph, k); it; ++it) {
            if (it.row() != it.col()) {  // Only keep non-diagonal elements
                tripletList.emplace_back(it.row(), it.col(), it.value());
            }
        }
    }
    new_graph.setZero();
    new_graph.setFromTriplets(tripletList.begin(), tripletList.end());

    return {new_graph, mapping, remapping, new_node_weight, new_CP, new_fixed_nodes_set};
}

struct CompareFirst {
    bool operator()(const std::tuple<double, int>& a, const std::tuple<double, int>& b) const {
        return std::get<0>(a) < std::get<0>(b);
    }
};

struct CompareFirstInThree {
    bool operator()(const std::tuple<double, int, int>& a, const std::tuple<double, int, int>& b) const {
        return std::get<0>(a) < std::get<0>(b);
    }
};


int initial_partitioning(
    const CSRMatrix& current_graph, 
    std::vector<std::unordered_set<int>> &CP, 
    const std::vector<std::vector<int>> &fpga_distances, 
    const std::vector<int> &node_weight,
    const int capacity,
    const std::unordered_set<int>current_fixed_nodes_set) 
{  
    int num_fpgas = fpga_distances.size();
    int n = current_graph.rows();
    int m = current_graph.nonZeros() / 2;
    
    std::unordered_set<int> fixed_nodes_set;
    
    std::vector<int> nodes_fixed_in_fpga(num_fpgas, 0);
    
    
    for (int node_id = 0; node_id < n; ++node_id) {
        if (CP[node_id].size() == 1){
            int fpga_id_index = *CP[node_id].begin();
            nodes_fixed_in_fpga[fpga_id_index] += node_weight[node_id];
            fixed_nodes_set.insert(node_id);
        }
        else if(CP[node_id].size() == 0){
            return 1;
        }
    }
    
    int cut_size2 = 0;
    int violation2 = 0;
    for(int i = 0; i < current_graph.rows(); i++){
        if (fixed_nodes_set.count(i)==0)
            continue;
        int fpga_id = *CP[i].begin();
        for (CSRMatrix::InnerIterator it(current_graph, i); it; ++it) {
            int neighbor_id = it.index();
            if (fixed_nodes_set.count(neighbor_id)==0 || neighbor_id < i)
                continue;
            int neighbor_fpga_id = *CP[neighbor_id].begin();
            if (fpga_distances[fpga_id][neighbor_fpga_id] >= 1) {
                cut_size2 += it.value();;
            }
            if(fpga_distances[fpga_id][neighbor_fpga_id] > 1) {
                violation2 += it.value();;
            }
        }
    }
    
    std::cout<<"cut_size:"<<cut_size2<<" violation:"<<violation2<<std::endl;
    if(violation2!=0){
        std::cout<<"sth is wrong with violation in init_partitioning (fixed node). "<<std::endl;
        return 1;
    }
    
    
    boost::heap::fibonacci_heap<std::tuple<double, int>, boost::heap::compare<CompareFirst>> node_heaps;
    std::unordered_map<int, boost::heap::fibonacci_heap<std::tuple<double, int>,boost::heap::compare<CompareFirst>>::handle_type> handleMap;
    
    
    std::vector<int> degrees(current_graph.rows(), 0);
    for(int i = 0; i < current_graph.rows(); i++){
        for (CSRMatrix::InnerIterator it(current_graph, i); it; ++it) {
            int j = it.index();
            if (i < j) {
                degrees[i] += it.value();
                degrees[j] += it.value();
            }
        }
    }
    int max_degree = 0;
    for(int i = 0; i < current_graph.rows(); i++){
        if(degrees[i] > max_degree)
            max_degree = degrees[i];
    }
    
    for (int node_id = 0; node_id < n; ++node_id) {
        if (CP[node_id].size() == 1) continue;
        if (CP[node_id].size() == 0){
            std::cout<<"sth is wrong with violation in init_partitioning. "<<std::endl;
            std::exit(0);
        }   
        double priority = -static_cast<double>(CP[node_id].size())+(double)degrees[node_id]/max_degree;
        auto handle1 = node_heaps.push(std::make_tuple(priority, node_id));
        handleMap[node_id] = handle1;
    }
    
    std::vector<int> fpga_degrees(num_fpgas, 0);
    for (int i = 0; i < num_fpgas; ++i) {
        for (int j = 0; j < num_fpgas; ++j) {
            if (i > j && fpga_distances[i][j] == 1) {
                fpga_degrees[i]++;
                fpga_degrees[j]++;
            }
        }
    }
    
    std::vector<int> nodes_weight_fixed_in_this_fpga(num_fpgas, 0);

    for (int node_id = 0; node_id < n; ++node_id) {
        if (CP[node_id].size() == 1){
            int fpga_id_index = *CP[node_id].begin();
            nodes_weight_fixed_in_this_fpga[fpga_id_index] += node_weight[node_id];
        }
    }
    
    std::cout << "nodes_weight_fixed_in_this_fpga: ";
    for (int i = 0; i < num_fpgas; ++i) {
      std::cout << nodes_weight_fixed_in_this_fpga[i] << " ";
    }
    std::cout << std::endl;
    
    while(!node_heaps.empty()){

        auto [priority, node_id] = node_heaps.top();
        
        //std::cout<<"node_id:"<<node_id<<" "<<priority<<std::endl;
        
        node_heaps.pop();
        auto itmap = handleMap.find(node_id);
        handleMap.erase(itmap);
        
        std::vector<int>neighbor_in_the_fpga(num_fpgas, 0);
        
        for (CSRMatrix::InnerIterator it(current_graph, node_id); it; ++it) {
            int neighbor_id = it.index();
            if (CP[neighbor_id].size() == 1) {
                int neighbor_fpga_id = *(CP[neighbor_id].begin());
                neighbor_in_the_fpga[neighbor_fpga_id]++;
            }
        }
        
        std::vector<int> indices(CP[node_id].begin(), CP[node_id].end());
        
        int max_neighbor_in_the_fpga = 0;
        for (int fpga_id : indices){
            if(neighbor_in_the_fpga[fpga_id] > max_neighbor_in_the_fpga)
                max_neighbor_in_the_fpga = neighbor_in_the_fpga[fpga_id];
        }
        
        std::sort(indices.begin(), indices.end(),
                  [&](int a, int b) {
                      return fpga_degrees[a] + (double)neighbor_in_the_fpga[a]/max_neighbor_in_the_fpga > fpga_degrees[b] + (double)neighbor_in_the_fpga[b]/max_neighbor_in_the_fpga;
                  });
        
        int length = CP[node_id].size();
        
        for (int fpga_id : indices) {
            
            if(nodes_fixed_in_fpga[fpga_id] + node_weight[node_id] > capacity){
                length--;
                continue;
            }
            
            CP[node_id] = {fpga_id};
            fixed_nodes_set.insert(node_id);
            nodes_fixed_in_fpga[fpga_id] += node_weight[node_id];

            
            int d = 0;
            for (const auto& distance : fpga_distances[fpga_id]) {
                d = std::max(d, distance);
            }
            std::vector<std::tuple<int, int>> reachable_nodes = find_reachable_nodes(current_graph, node_id, d, fixed_nodes_set);
            for (const auto& [neighbor_id, distance] : reachable_nodes) {
                if (CP[neighbor_id].size() > 1) {
                    std::unordered_set<int> CP_copy = CP[neighbor_id];
                    bool changed = false;
                    for (int candidate_fpga : CP_copy) {
                        if (fpga_distances[fpga_id][candidate_fpga] > distance) {
                            CP[neighbor_id].erase(candidate_fpga);
                            changed = true;
                        }
                    }
                    if(changed==true){
                        auto itmap = handleMap.find(neighbor_id);
                        if (itmap != handleMap.end()) {
                            node_heaps.erase(itmap->second);
                            handleMap.erase(itmap); 
                        }
                    }
                    
                    if (CP[neighbor_id].size() == 0)
                        return 1;
                    auto itmap = handleMap.find(neighbor_id);
                    if (itmap != handleMap.end()) {
                        node_heaps.erase(itmap->second);
                        handleMap.erase(itmap); 
                    }
                    double priority = -static_cast<double>(CP[neighbor_id].size()) + (double)degrees[neighbor_id]/max_degree;
                    auto handle1 = node_heaps.push(std::make_tuple(priority, neighbor_id));
                    handleMap[neighbor_id] = handle1;
                }
            }
            break;
        }
        
        if(length==0){
            std::cout<<"sth is wrong with violation in init_partitioning (for). "<<std::endl;
            return 1;
        }
    }
    
    
    for (int node_id = 0; node_id < n; ++node_id) {
        if (CP[node_id].size() != 1){
            std::cout<<"something is wrong in initial partitioning(CP)"<<std::endl;
            std::exit(0);
        }
    }
    
    int cut_size3 = 0;
    int violation3 = 0;
    for(int i = 0; i < current_graph.rows(); i++){
        if (fixed_nodes_set.count(i)==0)
            continue;
        int fpga_id = *CP[i].begin();
        for (CSRMatrix::InnerIterator it(current_graph, i); it; ++it) {
            int neighbor_id = it.index();
            if (fixed_nodes_set.count(neighbor_id)==0 || neighbor_id < i)
                continue;
            int neighbor_fpga_id = *CP[neighbor_id].begin();
            if (fpga_distances[fpga_id][neighbor_fpga_id] >= 1) {
                cut_size3 += it.value();;
            }
            if(fpga_distances[fpga_id][neighbor_fpga_id] > 1) {
                violation3 += it.value();;
            }
        }
    }

    std::cout<<"success:intial partitioning's cut_size:"<<cut_size3<<"intial partitioning's violation:"<<violation3<<std::endl;
    
    if(violation3!=0)
        return 1;
    
    return 0;
}
    

std::vector<std::unordered_set<int>> FM(
    const CSRMatrix &current_graph,
    std::vector<std::unordered_set<int>> CP,
    const std::vector<std::vector<int>> &fpga_distances, 
    const std::vector<int> &node_weight,
    double beta,
    const int capacity,
    std::unordered_set<int>fixed_nodes_set) 
{

    int n = current_graph.rows();
    int k = fpga_distances.size();
    std::vector<int>nodes_fixed_in_fpga(k,0);
    
    for (int node_id = 0; node_id < n; ++node_id) {
        if(CP[node_id].size()!=1){
            std::cout<<"wrong"<<std::endl;
            std::exit(0);
        }
        int fpga_id = *CP[node_id].begin();
        nodes_fixed_in_fpga[fpga_id] += node_weight[node_id];
    }
    
    boost::heap::fibonacci_heap<std::tuple<double, int, int>, boost::heap::compare<CompareFirstInThree>>FibHeap;
    std::unordered_map<std::pair<int, int>, boost::heap::fibonacci_heap<std::tuple<double, int, int>, boost::heap::compare<CompareFirstInThree>>::handle_type, boost::hash<std::pair<int, int>>> HandleMap;
    std::unordered_map<std::pair<int, int>, double, boost::hash<std::pair<int, int>>> gain_Map;
    
    for (int node_id = 0; node_id < n; ++node_id) {
        std::vector<double> priority(k, 0);
        if (CP[node_id].size() == 1) {
            int current_fpga_id = *CP[node_id].begin();
            for (CSRMatrix::InnerIterator it(current_graph, node_id); it; ++it) {
                int neighbor_id = it.index();
                int neighbor_fpga_id = *CP[neighbor_id].begin();
                priority[neighbor_fpga_id] += it.value(); //python wrong
                for (int fpga_id = 0; fpga_id < k; ++fpga_id) {
                    if (fpga_distances[neighbor_fpga_id][fpga_id] > 1){
                        priority[fpga_id] -= beta * it.value();
                    }
                }
            }
            for (int fpga_id = 0; fpga_id < k; ++fpga_id) {
                if (fpga_id == current_fpga_id) {
                    continue;
                }
                double gain = priority[fpga_id] - priority[current_fpga_id];
                auto handle = FibHeap.push(std::make_tuple(gain, node_id, fpga_id));
                HandleMap[{node_id, fpga_id}] = handle;
                gain_Map[{node_id, fpga_id}] = gain;
            }
        } 
        else {
            std::cout<<"wrong"<<std::endl;
            std::exit(0);
        }
    }

    std::unordered_set<int> moved_nodes = fixed_nodes_set;
    std::vector<std::tuple<int, int, int, double>> move_history;
    double total_gain = 0;

    while (!FibHeap.empty()){

        auto [gain, node_id, fpga_id] = FibHeap.top();

        auto itmap = HandleMap.find({node_id, fpga_id});
        HandleMap.erase(itmap); //need to delete it
        FibHeap.pop();

        if (moved_nodes.count(node_id) || nodes_fixed_in_fpga[fpga_id] + node_weight[node_id] > capacity) {
            continue;
        }

        for (int fpga = 0; fpga < k; ++fpga) {
            if (fpga != fpga_id) {
                auto itmap = HandleMap.find({node_id, fpga});
                if (itmap != HandleMap.end()) {
                    FibHeap.erase(itmap->second);
                    HandleMap.erase(itmap);
                }
            }
        }

        nodes_fixed_in_fpga[fpga_id] += node_weight[node_id];
        int original_fpga = *CP[node_id].begin();
        nodes_fixed_in_fpga[original_fpga] -= node_weight[node_id];
        moved_nodes.insert(node_id);
        total_gain += gain;
        move_history.push_back({node_id, original_fpga, fpga_id, total_gain});
        
        
        for (CSRMatrix::InnerIterator it(current_graph, node_id); it; ++it) {
            int neighbor_id = it.index();
            if (moved_nodes.count(neighbor_id))
                continue;
            int neighbor_fpga_id = *CP[neighbor_id].begin();
            double priority1 = (neighbor_fpga_id == original_fpga) ? it.value() : 
                               ((fpga_distances[neighbor_fpga_id][original_fpga] > 1) ? -beta * it.value() : 0);

            double priority3 = (neighbor_fpga_id == fpga_id) ? it.value() : 
                               ((fpga_distances[neighbor_fpga_id][fpga_id] > 1) ? -beta * it.value() : 0);

            for (int fpga = 0; fpga < k; ++fpga) {

                if (fpga != neighbor_fpga_id) {

                    auto itmap = HandleMap.find({neighbor_id, fpga});
                    double priority2 = (fpga == original_fpga) ? it.value() : 
                                         ((fpga_distances[fpga][original_fpga] > 1) ? -beta * it.value() : 0);
                    double priority_before = priority2 - priority1;

                    double priority4 = (fpga == fpga_id) ? it.value() : 
                                       ((fpga_distances[fpga][fpga_id] > 1) ? -beta * it.value() : 0);
                    double priority_after = priority4 - priority3;
                    double this_gain = priority_after - priority_before;
                    
                    if (itmap != HandleMap.end()) {
                        
                        double old_gain = std::get<0>(*itmap->second);
                        if(old_gain!=gain_Map[{neighbor_id, fpga}]){
                            std::cout<<"wrong"<<std::endl;
                            std::exit(0);
                        }
                        double new_gain = old_gain + this_gain;

                        FibHeap.erase(itmap->second);
                        HandleMap.erase(itmap);

                        auto new_handle = FibHeap.push(std::make_tuple(new_gain, neighbor_id, fpga));
                        HandleMap[{neighbor_id, fpga}] = new_handle;
                        gain_Map[{neighbor_id, fpga}] = new_gain;

                    }
                    else{
                        double old_gain = gain_Map[{neighbor_id, fpga}];
                        double new_gain = old_gain + this_gain;
                    
                        auto new_handle = FibHeap.push(std::make_tuple(new_gain, neighbor_id, fpga));
                        HandleMap[{neighbor_id, fpga}] = new_handle;
                        gain_Map[{neighbor_id, fpga}] = new_gain;
                    }
                }
            }
            
        }
    }

    int max_gain_index = 0;
    double max_gain = std::get<3>(move_history[0]);;
    for (int i=0;i<move_history.size();i++){
        int gain = std::get<3>(move_history[i]);
        if(gain>max_gain){
            max_gain_index = i;
            max_gain = gain;
        }
    }
    if (max_gain<0)
        return CP;
    else {
        for (size_t i = 0; i <= max_gain_index; ++i) {
            auto [node_id, original_fpga_id, new_fpga_id, total_gain] = move_history[i];
            if (CP[node_id].size()==1 && CP[node_id].count(original_fpga_id)) {
                CP[node_id] = {new_fpga_id};
            } else {
                std::cout<<"wrong"<<std::endl;;
            }
        }
    return CP;
    }
}


std::tuple<CSRMatrix,std::vector<std::tuple<int, int>>> remake_netlist(CSRMatrix&netlist_matrix,std::vector<std::tuple<int, int>>&fixed_nodes){

    int nnn = netlist_matrix.rows();
    std::vector<int> component(nnn, -1);
    int label = 0; 
    int maxSize = 0; 
    int maxLabel = -1;
    
    
    for (int i = 0; i < nnn; ++i) {
        if (component[i] == -1) {
            int currentComponentSize = 0;
            DFS(i, netlist_matrix, component, label, currentComponentSize);

            if (currentComponentSize > maxSize) {
                maxSize = currentComponentSize;
                maxLabel = label;
            }
            
            //std::cout<<"part: "<<label<<" size:"<<currentComponentSize<<std::endl;
            
            ++label;
        }
    }
    
    int new_index = 0;
    std::vector<Triplet<int>> triplets;
    std::vector<int>mapping(nnn,-1);
    for (int i = 0; i < nnn; ++i) {
        if(component[i]==maxLabel){
            triplets.emplace_back(i, new_index, 1);
            mapping[i]=new_index;
            new_index++;
        }
    }
    
    std::vector<std::tuple<int, int>>new_fixed_nodes;
    for(auto&[node_id,fpga_id]:fixed_nodes){
        if(component[node_id]==maxLabel){
            new_fixed_nodes.push_back({mapping[node_id], fpga_id}); 
        }
    }
    
    int k = new_index;
    
    CSRMatrix X(nnn, k);
    X.setFromTriplets(triplets.begin(), triplets.end());
    
    CSRMatrix Xt = X.transpose();
    
    CSRMatrix new_graph = Xt * netlist_matrix * X;
    
    
    std::vector<Triplet<int>> tripletList;

    // Step 2: Iterate through all non-zero elements of new_graph
    for (int k = 0; k < new_graph.outerSize(); ++k) {
        for (CSRMatrix::InnerIterator it(new_graph, k); it; ++it) {
            if (it.row() != it.col()) {  // Only keep non-diagonal elements
                tripletList.emplace_back(it.row(), it.col(), it.value());
            }
        }
    }
    new_graph.setZero();
    new_graph.setFromTriplets(tripletList.begin(), tripletList.end());
    
    
    return {new_graph,new_fixed_nodes};

}



int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <data_filename>" << std::endl;
        return 1;
    }

    std::string data = argv[1];
    std::string netlist_file_path = "./ICCAD2021-TopoPart-Benchmarks/generated_large_netlists/MFS1/" + data;  // Generated_Benchmarks/Generated_Benchmarks/"
    //std::string fixed_nodes_path = "./ICCAD2021-TopoPart-Benchmarks/Titan23_Benchmarks/Titan23_Benchmarks/" + data + ".simulated_annealing.8";
    
    // Example usage:
    CSRMatrix fpga_matrix = create_FPGA_matrix("./ICCAD2021-TopoPart-Benchmarks/FPGA_Graph/FPGA_Graph/MFS1");
    auto [netlist_matrix0, init_fixed_nodes0] = create_netlist_matrix(netlist_file_path);
    
    
    //largestComponent

    auto [netlist_matrix, fixed_nodes] = remake_netlist(netlist_matrix0, init_fixed_nodes0);
    
    std::clock_t start_time = std::clock();
    
    int num_fpgas = fpga_matrix.rows();
    
    std::vector<std::tuple<int, int>> ignored_fixed_nodes = find_fixed_nodes(netlist_matrix, num_fpgas);
    
    int capacity = ceil(5*netlist_matrix.rows()/fpga_matrix.rows());
    std::cout << "capacity:" << capacity << std::endl;
    // Print the number of nodes and edges for FPGA matrix
    std::cout << "FPGA Matrix:" << std::endl;
    std::cout << "  Nodes: " << fpga_matrix.rows() << std::endl;
    std::cout << "  Edges: " << fpga_matrix.nonZeros() / 2 << std::endl; // Divided by 2 because of symmetry

    // Print the number of nodes and edges for Netlist matrix
    std::cout << "Netlist Matrix:" << std::endl;
    std::cout << "  Nodes: " << netlist_matrix.rows() << std::endl;
    std::cout << "  Edges: " << netlist_matrix.nonZeros() / 2 << std::endl; // Divided by 2 because of symmetry
    
    int n = netlist_matrix.rows();
    
    
    std::vector<std::vector<int>> fpga_distances(num_fpgas, std::vector<int>(num_fpgas, 10000)); // Initialize with 10000 (unreachable)
    for (int fpga_id = 0; fpga_id < num_fpgas; ++fpga_id) {
        std::vector<std::tuple<int, int>> reachable_fpgas = find_reachable_nodes(fpga_matrix, fpga_id, num_fpgas - 1, std::unordered_set<int>());
        for (const auto& [neighbor_id, distance] : reachable_fpgas) {
            fpga_distances[fpga_id][neighbor_id] = distance;
        }
    }
    
    std::unordered_set<int> fpga_set;
    for (int i = 0; i < num_fpgas; ++i) {
        fpga_set.insert(i);
    }
    
    std::vector<std::unordered_set<int>> CP(n, fpga_set);
    std::unordered_set<int> fixed_nodes_set;
    std::unordered_set<int> violation_set;
    for (const auto& [node_id, fpga_id] : fixed_nodes) {
        CP[node_id] = {fpga_id};
        fixed_nodes_set.insert(node_id);
    }
    
    
    std::queue<int> Q;
    for (const auto& [node_id, fpga_id] : fixed_nodes) {
        Q.push(node_id);
    }

    
    // 3. Iteratively process nodes in the queue
    while (!Q.empty()) {
        int node_id = Q.front();
        Q.pop();
        std::unordered_set<int> fpga_id = CP[node_id];
    
        if (fpga_id.size() != 1) {
            std::cout << "wrong" << std::endl;
            continue;
        }
    
        int fpga_id_index = *fpga_id.begin();
    
        // Calculate maximum hop distance to other FPGAs
        int d = 0;
        for (const auto& distance : fpga_distances[fpga_id_index]) {
            d = std::max(d, distance);
        }
    
        // Find nodes within distance d in the large graph
        std::vector<std::tuple<int, int>> reachable_nodes = find_reachable_nodes(netlist_matrix, node_id, d, fixed_nodes_set);
    
        // Update candidate FPGAs for reachable nodes
        for (const auto& [node_id2, distance] : reachable_nodes) {
            if (fixed_nodes_set.count(node_id2)||violation_set.count(node_id2)) {
                continue;
            }
            std::unordered_set<int> candidates_copy = CP[node_id2];
            for (int candidate_fpga : candidates_copy) {
                if (fpga_distances[fpga_id_index][candidate_fpga] > distance) {
                    CP[node_id2].erase(candidate_fpga);
                }
            }
    
            if (CP[node_id2].size() == 1) {
                fixed_nodes_set.insert(node_id2);
                Q.push(node_id2);
            } 
            else if (CP[node_id2].empty()) {
                violation_set.insert(node_id2);
            }
        }
    }
    
    int count1=0, count2=0, count3=0, count4=0, count5=0;
    for (int i = 0; i < n; ++i) {
        if(CP[i].size()<2)
            count1++;
        else if(CP[i].size()<4)
            count2++;
        else if(CP[i].size()<6)
            count3++;
        else if(CP[i].size()<8)
            count4++;
        else
            count5++;
    }      
    //std::cout << "CP distribution: [0,2):"<< count1 << "[2,4):"<< count2 << "[4,6):"<< count3 << "[6,8):"<< count4 << "[8,9):"<< count5 << std::endl;
    std::cout << "Fixed node number: " << fixed_nodes_set.size() << std::endl;
    
    int fix_number1 = 0;
    int fix_number2 = 0;
    
    for (int i = 0; i < n; ++i) {
        if (CP[i].size() == 1 && fixed_nodes_set.find(i) == fixed_nodes_set.end()) {
            fix_number1++;
        } else if (CP[i].size() != 1 && fixed_nodes_set.find(i) != fixed_nodes_set.end()) {
            fix_number2++;
        }
    }

    std::cout << fix_number1 << " " << fix_number2 << std::endl;
    
    int vio_number1 = 0;
    int vio_number2 = 0;
    
    for (int i = 0; i < n; ++i) {
        if (CP[i].size() == 0 && violation_set.find(i) == violation_set.end()) {
            vio_number1++;
        } else if (CP[i].size() != 0 && violation_set.find(i) != violation_set.end()) {
            vio_number2++;
        }
    }

    std::cout << vio_number1 << " " << vio_number2 << std::endl;
    
    
    std::cout << "Coarsening phase" << std::endl;
    
    std::vector<CSRMatrix> coarsened_graphs;
    std::vector<std::vector<int>> coarsened_mappings;
    std::vector<std::vector<std::vector<int>>> coarsened_remappings;
    std::vector<std::vector<int>> coarsened_node_weight;
    std::vector<std::unordered_set<int>> coarsened_fixed_nodes_set;
    std::vector<std::vector<std::unordered_set<int>>>coarsened_CP;
    
    
    std::vector<int> node_weight(n, 1);
    CSRMatrix current_graph = netlist_matrix;
    std::vector<int> current_node_weight = node_weight;
    std::vector<std::unordered_set<int>> current_CP = CP;
    std::unordered_set<int>current_fixed_nodes_set = fixed_nodes_set;
    
    bool coarsen_continue = true;
    int T = 0;
  
    
    while (coarsen_continue) {
        auto [new_graph, mapping, remapping, new_node_weight, new_CP, new_fixed_nodes_set] = coarsen_graph(current_graph, current_CP, current_node_weight, capacity, current_fixed_nodes_set, num_fpgas);
        if (current_graph.rows() - new_graph.rows() < 100) { 
            coarsen_continue = false;
        }
        
        std::cout << "Coarsening iteration: " << T << " " << new_graph.rows() << " " << new_graph.nonZeros()/2 << " " << new_CP.size() << std::endl;
        T++;
        
        coarsened_graphs.push_back(new_graph);
        coarsened_mappings.push_back(mapping);
        coarsened_remappings.push_back(remapping);
        coarsened_node_weight.push_back(new_node_weight);
        coarsened_fixed_nodes_set.push_back(new_fixed_nodes_set);
        coarsened_CP.push_back(new_CP);
        
        current_graph = new_graph;
        current_node_weight = new_node_weight;
        current_CP = new_CP;
        current_fixed_nodes_set = new_fixed_nodes_set;
    }
    
    std::cout << std::endl;
    std::cout << "Initial_partitioning phase" << std::endl;
    
    int TT = 0;
    for (const auto& set : current_CP) {
        if (set.size() == 1)
            TT += 1;
    }
    
    std::vector<std::unordered_set<int>>uncoarsen_CP;
    
    int result = 1;    
    int coarsen_level = coarsened_graphs.size();
    int try_level= coarsen_level-1;
    int times=0;
    for (try_level = coarsen_level-1; try_level >= 0&&times<25 && result!=0; try_level--){
        current_graph = coarsened_graphs[try_level];
        if(current_graph.rows()>25000)
            break;
        current_node_weight = coarsened_node_weight[try_level];
        current_CP = coarsened_CP[try_level];
        current_fixed_nodes_set = coarsened_fixed_nodes_set[try_level];
        result = initial_partitioning(current_graph, current_CP, fpga_distances, current_node_weight, capacity, current_fixed_nodes_set);
        times++;
        if(result==0)
            uncoarsen_CP = current_CP;
    }
    
    if(result==1){
        std::clock_t end_time2 = std::clock();
        double cpu_time_used2 = static_cast<double>(end_time2 - start_time) / CLOCKS_PER_SEC;
        std::cout << "Initial_partitioning phase failed. times:"<< times << std::endl;
        std::cout << "Running time: " << cpu_time_used2 << " s" << std::endl;
        std::exit(0);
    
    }
        
    
    std::cout << "Initial_partitioning phase is finished. times:"<< times << std::endl;
    
    std::vector<int> nodes_weight_fixed_in_this_fpga1(num_fpgas, 0);

    for (int node_id = 0; node_id < current_graph.rows(); ++node_id) {
        if (uncoarsen_CP[node_id].size() == 1){
            int fpga_id_index = *uncoarsen_CP[node_id].begin();
            nodes_weight_fixed_in_this_fpga1[fpga_id_index] += current_node_weight[node_id];
        }
        else
            std::cout<<"CP's size is not 1 for node:"<<node_id<<"it is:"<<uncoarsen_CP[node_id].size()<<std::endl;
    }
    
    std::cout << "nodes_weight_fixed_in_this_fpga: ";
    for (int i = 0; i < num_fpgas; ++i) {
      std::cout << nodes_weight_fixed_in_this_fpga1[i] << " ";
    }
    std::cout << std::endl;
    
    
    std::cout << std::endl;
    std::cout << "Uncoarsening phase" << std::endl;

    for (int i=try_level+1; i>=0; i--){
        std::vector<std::vector<int>> map_to_previous_level = coarsened_remappings[i];
        CSRMatrix graph_previous_level;
        std::vector<int>node_weight_previous_level;
        std::unordered_set<int>fixed_nodes_set_previous_level;
        if (i>0){
            graph_previous_level = coarsened_graphs[i-1];
            node_weight_previous_level = coarsened_node_weight[i-1];
            fixed_nodes_set_previous_level = coarsened_fixed_nodes_set[i-1];
        }
        else{
            graph_previous_level = netlist_matrix;
            node_weight_previous_level = std::vector<int>(n, 1);
            fixed_nodes_set_previous_level = fixed_nodes_set;
        }
        int nodenum = graph_previous_level.rows();
        int coarsen_nodenum = coarsened_graphs[i].rows();
        std::vector<std::unordered_set<int>> CP_previous_level(nodenum);

        for (int j=0;j<coarsen_nodenum;j++){
            for (int z=0;z<map_to_previous_level[j].size();z++){
                CP_previous_level[map_to_previous_level[j][z]] = uncoarsen_CP[j];
            }
        }
        for (int j=0;j<nodenum;j++)
            if(CP_previous_level[j].size()!=1)
                std::cout<<"wrong:"<<j<<std::endl;

        double beta = 1e9;
        
        uncoarsen_CP = FM(graph_previous_level, CP_previous_level, fpga_distances, node_weight_previous_level, beta, capacity, fixed_nodes_set_previous_level);
        for (int j = 0; j < uncoarsen_CP.size(); ++j) {
            if(uncoarsen_CP[j].size()!=1){
                std::cout<<"uncoarsen wrong"<<std::endl;
                std::exit(0);
            }
        }
    }
    
    int cut_size = 0;
    int violation = 0;
    std::vector<int>num_count(num_fpgas,0);
    for(int i=0;i<n;i++){
        int fpga_id = *uncoarsen_CP[i].begin();
        num_count[fpga_id]++;
        for (CSRMatrix::InnerIterator it(netlist_matrix, i); it; ++it) {
            int neighbor_id = it.index();
            if (neighbor_id < i)
                continue;
            int neighbor_fpga_id = *uncoarsen_CP[neighbor_id].begin();
            if (fpga_distances[fpga_id][neighbor_fpga_id] >= 1) {
                cut_size += it.value();
            }
            if(fpga_distances[fpga_id][neighbor_fpga_id] > 1) {
                violation += it.value();
            }
        }
    }
    std::cout<<"cut_size:"<<cut_size<<" violation:"<<violation<<std::endl;
    
    
    std::vector<int> nodes_weight_fixed_in_this_fpga(num_fpgas, 0);

    for (int node_id = 0; node_id < n; ++node_id) {
        if (uncoarsen_CP[node_id].size() == 1){
            int fpga_id_index = *uncoarsen_CP[node_id].begin();
            nodes_weight_fixed_in_this_fpga[fpga_id_index] += 1;
        }
        else
            std::cout<<"CP's size is not 1 for node:"<<node_id<<std::endl;
    }
    
    std::cout << "nodes_weight_fixed_in_this_fpga: ";
    for (int i = 0; i < num_fpgas; ++i) {
      std::cout << nodes_weight_fixed_in_this_fpga[i] << " ";
    }
    std::cout << std::endl;
    
    
    std::clock_t end_time = std::clock();
    double cpu_time_used = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;
  
    std::cout << "Running time: " << cpu_time_used << " s" << std::endl;
    
    for (auto& [node_id, fpga_id] : fixed_nodes) {
        int fpga_id_index = *uncoarsen_CP[node_id].begin();
        if(fpga_id_index!=fpga_id){
            std::cout<<"wrong: node "<<node_id<<" should be put into "<<fpga_id<<" while it's in "<<fpga_id_index<<std::endl;
            std::exit(0);
        }
    }
    
    
    
    return 0;
}