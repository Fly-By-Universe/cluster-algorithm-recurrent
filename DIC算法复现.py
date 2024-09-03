from numpy import *
import numpy as np
import operator
import pandas as pd

"""读取数据,这里 X 为示例数据"""
X = pd.read_csv("C:\\Users\\DELL\\OneDrive\\桌面\\算法复现\\iris.data",header=None)
X = X.iloc[:,:-1]
k = 10
X = X

def euclidean(point1,point2):
    """计算欧几里得距离"""
    return np.sqrt(np.sum((point1 - point2)**2))

def k_nearest_neighbors(x,X,k):
    """计算距离最近的k个点"""
    distances = []
    for index in range(len(X)):
        data_point = X.iloc[index,:]
        distance = euclidean(x,data_point)
        distances.append(distance)
    distances_index_pairs = list(enumerate(distances))
    sorted_distances_index_pairs = sorted(distances_index_pairs,key = lambda x:x[1])
    nearest_indices = [index for index, _ in sorted_distances_index_pairs[1:k+1]]
    return nearest_indices


def d(x,x_distance):
    """计算最大密度"""
    return max(x_distance) # 返回x_distance中距离的最大值

def Al_1_DEA(X,k):
    """算法1 : 密度估计算法(Density estimation algorithm)"""
    Density = []
    nearest_neightbor = []
    for i in range(len(X)):
        x = X.iloc[i,:]
        x_distance = k_nearest_neighbors(x,X,k)
        Density.append(1/d(x,x_distance)) # 得到的density值为最近的k个点附近的密度最小值
        nearest_neightbor.append(x_distance)
    return Density,nearest_neightbor


result_density,result_neightbor = Al_1_DEA(X,k)
# result_density 是一个一维列表，其中包含每个点的密度
# result_neightbor 是一个二维列表，其中包含每个点最近的k个点的索引
print(result_neightbor)


def Density_chains(result_density,result_neightbor):
    density_chains = []
    """ 算法2 : 找到一条密度链 xi -> xj -> xk (Density following) """
    for i in range(len(X)):
        chain = []
        chain.append(i)
        start = i
        while True:
            density_list = [result_density[j] for j in result_neightbor[start]]
            maximum = max(density_list)
            index = result_neightbor[start][density_list.index(maximum)]
            if result_density[start] <= maximum:
                chain.append(index)
                start = index
                
            else:
                break
        if len(chain) > 1:
            density_chains.append(chain)
            
    return density_chains

result_chains = Density_chains(result_density,result_neightbor)
print(result_chains)


def Centrality(result_chains):
    """ 计算中心距离 (Centrality) """
    number_counts = {}
    for row in result_chains:
        for number in row:
            if number in number_counts:
                number_counts[number] += 1
            else:
                number_counts[number] = 1
    return number_counts

result_Centrality = Centrality(result_chains)

print(result_Centrality)




def Density_group_discovery(result_chains):
    """ 把密度链分为密度组 (Density group discovery) """
    # 初始化一个字典来存储每个点所属的密度组
    density_groups = {}
    # 初始化一个字典来存储每个点及其连接的其他点
    adjacency_list = {}
    
    # 构建邻接表
    for chain in result_chains:
        for point in chain:
            if point not in adjacency_list:
                adjacency_list[point] = set()
            adjacency_list[point].update(chain)
            adjacency_list[point].remove(point)  # 移除自己避免自连接
    
    # DFS 函数来标记所有连通的节点
    def dfs(node, group_id):
        stack = [node]
        while stack:
            current = stack.pop()
            if current not in density_groups:
                density_groups[current] = group_id
                stack.extend(adjacency_list[current] - set(density_groups.keys()))
    
    groups = []
    
    # 遍历所有节点，执行 DFS
    for point in adjacency_list.keys():
        if point not in density_groups:
            group_id = len(groups)  # 新的组ID
            dfs(point, group_id)
            groups.append([key for key, value in density_groups.items() if value == group_id])
    
    return groups
a = Density_group_discovery(result_chains)
