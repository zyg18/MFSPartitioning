#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <algorithm>
#include <sstream>
#include <queue>
#include <set>
#include <random>
#include <cmath>
#include <numeric>
#include <unordered_set>
#include <unordered_map>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <boost/heap/fibonacci_heap.hpp>
#include <boost/functional/hash.hpp>
#include <stack>
#include <ctime>

using Eigen::SparseMatrix;
using Eigen::Triplet;

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
        if(ignored_nodes_set.count(node_id) == 0)
            reachable_nodes.push_back({node_id, distance});

        if (distance < max_distance) {
            for (CSRMatrix::InnerIterator it(A, node_id); it; ++it) {
                int neighbor = it.index();
                if (visited.count(neighbor) == 0) {
                    queue.push({neighbor, distance + 1});
                }
            }
        }
    }

    return reachable_nodes;
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

std::tuple<CSRMatrix, std::vector<int>, std::vector<std::vector<int>>, std::vector<int>, std::vector<std::unordered_set<int>>, std::unordered_set<int>> coarsen_graph(
    CSRMatrix& graph, 
    std::vector<std::unordered_set<int>>& CP, 
    std::vector<int>& node_weight,
    int capacity,
    std::unordered_set<int>fixed_nodes_set,
    int threshold) 
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


int initial_partitioning(
    const CSRMatrix& current_graph, 
    std::vector<std::unordered_set<int>> &CP, 
    const std::vector<std::vector<int>> &fpga_distances, 
    const std::vector<int> &node_weight,
    const int capacity,
    const std::unordered_set<int>current_fixed_nodes_set,
    int &init_cut_size) 
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
    
    for (int node_id = 0; node_id < n; ++node_id) {
        if (CP[node_id].size() == 1) continue;
        if (CP[node_id].size() == 0){
            std::cout<<"sth is wrong with violation in init_partitioning. "<<std::endl;
            std::exit(0);
        }   
        double priority = -static_cast<double>(CP[node_id].size());
        auto handle1 = node_heaps.push(std::make_tuple(priority, node_id));
        handleMap[node_id] = handle1;
    }
    
    
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
    
        std::sort(indices.begin(), indices.end(),
                  [&](int a, int b) {
                      return neighbor_in_the_fpga[a] > neighbor_in_the_fpga[b];
                  });
        
        int length = CP[node_id].size();
        
        for (int fpga_id : indices) {
            
            bool judge_in_this_fpga = true;
            for (CSRMatrix::InnerIterator it(current_graph, node_id); it; ++it) {
                int neighbor_id = it.index();
                bool judge = false;
                for (int neighbor_fpga_id : CP[neighbor_id]){
                    if(fpga_distances[fpga_id][neighbor_fpga_id] <= 1){
                        judge = true;
                        break;
                    }
                }
                if(judge == false){
                    judge_in_this_fpga = false;
                    break;
                }
            }
            if(judge_in_this_fpga == false || nodes_fixed_in_fpga[fpga_id] + node_weight[node_id] > capacity){
                length--;
                continue;
            }
            
            CP[node_id] = {fpga_id};
            fixed_nodes_set.insert(node_id);
            nodes_fixed_in_fpga[fpga_id] += node_weight[node_id];
            
            
            for (CSRMatrix::InnerIterator it(current_graph, node_id); it; ++it) {
            
                int neighbor_id = it.index();
                
                if(fixed_nodes_set.count(neighbor_id)==1)
                    continue;
                    
                std::unordered_set<int> CP_copy = CP[neighbor_id];
                for (int neighbor_fpga_id : CP_copy){
                    if(fpga_distances[fpga_id][neighbor_fpga_id] > 1){
                        CP[neighbor_id].erase(neighbor_fpga_id);
                    }
                }
                auto itmap = handleMap.find(neighbor_id);
                if (itmap != handleMap.end()) {
                    node_heaps.erase(itmap->second);
                    handleMap.erase(itmap); 
                }
                double priority = -static_cast<double>(CP[neighbor_id].size());
                auto handle1 = node_heaps.push(std::make_tuple(priority, neighbor_id));
                handleMap[neighbor_id] = handle1;
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
    
    std::cout<<"intial partitioning's cut_size:"<<cut_size3<<"intial partitioning's violation:"<<violation3<<std::endl;
    
    init_cut_size = cut_size3;
    
    return 0;
}


std::vector<std::unordered_set<int>> FM(
    const CSRMatrix &current_graph,
    std::vector<std::unordered_set<int>> CP,
    const std::vector<std::vector<int>> &fpga_distances, 
    const std::vector<int> &node_weight,
    const int capacity,
    std::unordered_set<int>fixed_nodes_set,
    int init_cut_size) 
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
    std::vector<std::vector<int>> neighbors_in_fpgas(n, std::vector<int>(k, 0));
    
    //new
    
    for (int node_id = 0; node_id < n; ++node_id) {
        int current_fpga_id = *CP[node_id].begin();
        for (CSRMatrix::InnerIterator it(current_graph, node_id); it; ++it) {
            int neighbor_id = it.index();
            int neighbor_fpga_id = *CP[neighbor_id].begin();
            neighbors_in_fpgas[node_id][neighbor_fpga_id]++;
        }
    }
    
    std::unordered_set<int>boundary_nodes;
    for (int node_id = 0; node_id < n; ++node_id) {
        if(boundary_nodes.count(node_id))
            continue;
        int current_fpga_id = *CP[node_id].begin();
        for (CSRMatrix::InnerIterator it(current_graph, node_id); it; ++it) {
            int neighbor_id = it.index();
            int neighbor_fpga_id = *CP[neighbor_id].begin();
            if(current_fpga_id!=neighbor_fpga_id){
                boundary_nodes.insert(node_id);
                boundary_nodes.insert(neighbor_id);
                break;
            }
        }
    }
    
    
    
    for (int node_id = 0; node_id < n; ++node_id) {

        std::vector<double> priority(k, 0);
        if (CP[node_id].size() == 1) {
            int current_fpga_id = *CP[node_id].begin();

            for (CSRMatrix::InnerIterator it(current_graph, node_id); it; ++it) {
                int neighbor_id = it.index();
                int neighbor_fpga_id = *CP[neighbor_id].begin();
                priority[neighbor_fpga_id] += it.value();
            }
            
            std::vector<int>violated(k,0);
            for(int i=0; i < k; i++){
                if(neighbors_in_fpgas[node_id][i]!=0){
                    for(int viable_fpga_id = 0; viable_fpga_id < k; viable_fpga_id++){
                        int dist = fpga_distances[i][viable_fpga_id];
                        if (dist > 1){
                            violated[viable_fpga_id] = 1;
                        }
                    }
                }
            }
            
            for (int fpga_id = 0; fpga_id < k; ++fpga_id) {
                double gain = priority[fpga_id] - priority[current_fpga_id];
                
                if(boundary_nodes.count(node_id)==1 && fpga_id != current_fpga_id && violated[fpga_id]==0){
                    auto handle = FibHeap.push(std::make_tuple(gain, node_id, fpga_id));
                    HandleMap[{node_id, fpga_id}] = handle;
                }
                gain_Map[{node_id, fpga_id}] = gain;
            }
        } 
        else {
            std::cout<<"wrong"<<std::endl;
            std::exit(0);
        }
    }
    
    //std::cout<<"start move"<<std::endl;
    
    std::unordered_set<int> moved_nodes = fixed_nodes_set;
    std::vector<std::tuple<int, int, int, double>> move_history;
    double total_gain = 0;
    int move_node_num=0;
    
    while (!FibHeap.empty()){

        auto [gain, node_id, fpga_id] = FibHeap.top();

        auto itmap = HandleMap.find({node_id, fpga_id});
        HandleMap.erase(itmap); //need to delete it
        FibHeap.pop();
        
        if (moved_nodes.count(node_id) || nodes_fixed_in_fpga[fpga_id] + node_weight[node_id] > capacity ) {
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
        move_node_num++;

        for (CSRMatrix::InnerIterator it(current_graph, node_id); it; ++it) {
            int neighbor_id = it.index();
            if (moved_nodes.count(neighbor_id))
                continue;
            int neighbor_fpga_id = *CP[neighbor_id].begin();
            neighbors_in_fpgas[neighbor_id][original_fpga]--;
            neighbors_in_fpgas[neighbor_id][fpga_id]++;
        }
        
        for (CSRMatrix::InnerIterator it(current_graph, node_id); it; ++it) {
            int neighbor_id = it.index();
            if (moved_nodes.count(neighbor_id))
                continue;
            int neighbor_fpga_id = *CP[neighbor_id].begin();
            
            //new
            if(boundary_nodes.count(neighbor_id)==0&&neighbor_fpga_id!=original_fpga){
                std::cout<<"something is wrong with boundary_nodes"<<std::endl;
            }
            else if(boundary_nodes.count(neighbor_id)==0)
                boundary_nodes.insert(neighbor_id);
            else if(boundary_nodes.count(neighbor_id)==1&&fpga_id==neighbor_fpga_id){
                bool all_in_the_same_fpga = true;
                for (CSRMatrix::InnerIterator itt(current_graph, neighbor_id); itt; ++itt){
                    int new_neighbor_id = itt.index();
                    int new_neighbor_fpga_id = *CP[new_neighbor_id].begin();
                    if(new_neighbor_fpga_id!=neighbor_fpga_id){
                        all_in_the_same_fpga=false;
                        break;
                    }
                }
                if(all_in_the_same_fpga){
                    auto find_it = boundary_nodes.find(neighbor_id);
                    if (find_it!= boundary_nodes.end()) {
                        boundary_nodes.erase(find_it);
                    }
                }
            }
            
            double priority1 = (neighbor_fpga_id == original_fpga) ? it.value() : 0;
            
            double priority3 = (neighbor_fpga_id == fpga_id) ? it.value() : 0;
            
            
            std::vector<int>violated(k,0);
            if(boundary_nodes.count(neighbor_id)){
                for(int i=0; i < k; i++){
                    if(neighbors_in_fpgas[neighbor_id][i]!=0){
                        for(int viable_fpga_id = 0; viable_fpga_id < k; viable_fpga_id++){
                            int dist = fpga_distances[i][viable_fpga_id];
                            if (dist > 1){
                                violated[viable_fpga_id] = 1;
                            }
                        }
                    }
                }
            }
            
            for (int fpga = 0; fpga < k; ++fpga) {
            
                if (fpga != neighbor_fpga_id) { //here
                    
                    auto itmap = HandleMap.find({neighbor_id, fpga});
                    if (itmap != HandleMap.end()) {
                        FibHeap.erase(itmap->second);
                        HandleMap.erase(itmap);
                    }    
                    
                    double priority2 = (fpga == original_fpga) ? it.value() : 0;
                    double priority_before = priority2 - priority1;
                    
                    double priority4 = (fpga == fpga_id) ? it.value() : 0;
                    double priority_after = priority4 - priority3;
                    
                    double this_gain = priority_after - priority_before;
                    
                    double old_gain = gain_Map[{neighbor_id, fpga}];
                    double cur_gain = old_gain + this_gain;
                    gain_Map[{neighbor_id, fpga}] = cur_gain;
                    
                    if(boundary_nodes.count(neighbor_id)&&violated[fpga]==0){  //violated fpga
                        auto handle = FibHeap.push(std::make_tuple(cur_gain, neighbor_id, fpga));
                        HandleMap[{neighbor_id, fpga}] = handle;
                    }
                    
                }
            }
            
        }
    }

    int max_gain_index = 0;
    double max_gain = std::get<3>(move_history[0]);
    for (int i=0;i<move_history.size();i++){
        int gain = std::get<3>(move_history[i]);
        if(gain>max_gain){
            max_gain_index = i;
            max_gain = gain;
        }
    }

    //std::cout<<"max_gain_index: "<<max_gain_index<<std::endl;
    
    
    if (max_gain<0)
        return CP;
    else {
        for (int i = 0; i <= max_gain_index; ++i) {
            auto [node_id, original_fpga_id, new_fpga_id, total_gain] = move_history[i];
            if (CP[node_id].size()==1 && CP[node_id].count(original_fpga_id)) {
                CP[node_id] = {new_fpga_id};
            } 
            else {
                std::cout<<"wrong"<<std::endl;;
            }
        }
        return CP;
    }
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
                std::cout<<"violation always happened!"<<std::endl;
                return 0;
            }
        }
    }
    
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
    
    std::cout << "Coarsening phase" << std::endl;
    
    std::vector<CSRMatrix> coarsened_graphs;
    std::vector<std::vector<int>> coarsened_mappings;
    std::vector<std::vector<std::vector<int>>> coarsened_remappings;
    std::vector<std::vector<int>> coarsened_node_weight;
    std::vector<std::unordered_set<int>> coarsened_fixed_nodes_set;
    
    std::vector<int> node_weight(n, 1);
    CSRMatrix current_graph = netlist_matrix;
    std::vector<int> current_node_weight = node_weight;
    std::vector<std::unordered_set<int>> current_CP = CP;
    std::unordered_set<int>current_fixed_nodes_set = fixed_nodes_set;
    
    bool coarsen_continue = true;
    int T = 0;
    int threshold = std::round(2*num_fpgas/3);
    
    std::cout << "initial threshold is: "<< threshold << std::endl;
    
    int result = 1;
    int init_cut_size;
    while(result==1 && threshold>=1){
    
        while (coarsen_continue) {
            auto [new_graph, mapping, remapping, new_node_weight, new_CP, new_fixed_nodes_set] = coarsen_graph(current_graph, current_CP, current_node_weight, capacity, current_fixed_nodes_set, threshold);
            if (current_graph.rows() - new_graph.rows() < 300) { 
                coarsen_continue = false;
                std::cout << "Coarsening iteration: " << T << " " << new_graph.rows() << " " << new_graph.nonZeros()/2 << " " << new_CP.size() << std::endl;
            }
            
            //std::cout << "Coarsening iteration: " << T << " " << new_graph.rows() << " " << new_graph.nonZeros()/2 << " " << new_CP.size() << std::endl;
            T++;
            
            coarsened_graphs.push_back(new_graph);
            coarsened_mappings.push_back(mapping);
            coarsened_remappings.push_back(remapping);
            coarsened_node_weight.push_back(new_node_weight);
            coarsened_fixed_nodes_set.push_back(new_fixed_nodes_set);
            
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
        
        std::cout << "fixed nodes size: " << TT << std::endl;
        std::cout << "fixed nodes size: " << current_fixed_nodes_set.size() << std::endl; 
         
        std::cout << "current graph size: " <<current_graph.rows()<<std::endl;
        std::cout << "current CP size: " << current_CP.size()<<std::endl;
        
        init_cut_size = 0;
        result = initial_partitioning(current_graph, current_CP, fpga_distances, current_node_weight, capacity, current_fixed_nodes_set, init_cut_size);
        
        if(result==1){
            threshold--;
            current_graph = netlist_matrix;
            current_node_weight = node_weight;
            current_CP = CP;
            current_fixed_nodes_set = fixed_nodes_set;
            coarsen_continue = true;
            
            coarsened_graphs.clear();
            coarsened_mappings.clear();
            coarsened_remappings.clear();
            coarsened_node_weight.clear();
            coarsened_fixed_nodes_set.clear();
            T = 0;
            std::cout << "threshold is decreased to: "<< threshold << std::endl;
            continue;
        }
        break;
    }
    if(threshold==0){
        std::cout << "Initial_partitioning phase can't be accomplished." << std::endl;
        std::clock_t end_time = std::clock();
        double cpu_time_used = static_cast<double>(end_time - start_time) / CLOCKS_PER_SEC;
        
        std::cout << "Running time: " << cpu_time_used << " s" << std::endl;
        return 0;
    }
    std::cout << "Initial_partitioning phase is finished." << std::endl;
    
    std::cout << "Uncoarsening phase" << std::endl;
    
    
    
    auto uncoarsen_CP = FM(current_graph, current_CP, fpga_distances, current_node_weight, capacity, current_fixed_nodes_set, init_cut_size);
    int coarsen_level = coarsened_graphs.size();
    
    
    for (int i = coarsen_level-1; i >= 0; i--){
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

        uncoarsen_CP = FM(graph_previous_level, CP_previous_level, fpga_distances, node_weight_previous_level, capacity, fixed_nodes_set_previous_level, init_cut_size);
        for (int j = 0; j < uncoarsen_CP.size(); ++j) {
            if(uncoarsen_CP[j].size()!=1){
                std::cout<<"uncoarsen wrong"<<std::endl;
                std::exit(0);
            }
        }

    }
    
    
    int cut_size111 = 0;
    int violation111 = 0;
    for(int i=0;i<n;i++){
        int fpga_id = *uncoarsen_CP[i].begin();
        for (CSRMatrix::InnerIterator it(netlist_matrix, i); it; ++it) {
            int neighbor_id = it.index();
            if (neighbor_id < i)
                continue;
            int neighbor_fpga_id = *uncoarsen_CP[neighbor_id].begin();
            if (fpga_distances[fpga_id][neighbor_fpga_id] >= 1) {
                cut_size111 += it.value();
            }
            if(fpga_distances[fpga_id][neighbor_fpga_id] > 1) {
                violation111 += it.value();
            }
        }
    }
    std::cout<<"cut_size:"<<cut_size111<<" violation:"<<violation111<<std::endl;
    
    
    std::vector<int> nodes_weight_fixed_in_this_fpga(num_fpgas, 0);
    std::vector<int> nodes_fixed_in_this_fpga(num_fpgas, 0);
    std::cout<<n<<std::endl;
    for (int node_id = 0; node_id < n; ++node_id) {
        if (uncoarsen_CP[node_id].size() == 1){
            int fpga_id_index = *uncoarsen_CP[node_id].begin();
            nodes_weight_fixed_in_this_fpga[fpga_id_index] += 1;
            nodes_fixed_in_this_fpga[fpga_id_index] += 1;
        }
        else
            std::cout<<"CP's size is not 1 for node:"<<node_id<<std::endl;
    }
    
    std::cout << "nodes_weight_fixed_in_this_fpga: ";
    for (int i = 0; i < num_fpgas; ++i) {
      std::cout << nodes_weight_fixed_in_this_fpga[i] << " ";
    }
    std::cout << std::endl;
    
    
    std::cout << "nodes_fixed_in_this_fpga: ";
    for (int i = 0; i < num_fpgas; ++i) {
      std::cout << nodes_fixed_in_this_fpga[i] << " ";
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