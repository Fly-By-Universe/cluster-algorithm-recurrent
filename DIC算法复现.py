from numpy import *
import numpy as np
import operator
import pandas as pd

"""读取数据,这里 X 为示例数据"""
X = pd.read_csv("C:\\Users\\DELL\\OneDrive\\桌面\\算法复现\\iris.data",header=None)
X = X.iloc[:,:-1]
k = 10

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


# ---------------------------------------------------------------------------------------我是分割线---------------------------------------------------------------------


def Density_chains(X,k):
    """ 算法2 : 找到一条密度链 xi -> xj -> xk (Density following) """
    result_density,result_neightbor = Al_1_DEA(X,k)
    density_chains = []
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
        density_chains.append(chain)        
    return density_chains




def Centrality(X,k):
    """ 计算中心距离 (Centrality) """
    number_counts = {}
    result_chains = Density_chains(X,k)
    for row in result_chains:
        for number in row:
            if number in number_counts:
                number_counts[number] += 1
            else:
                number_counts[number] = 1
    return number_counts

def Density_group_discovery(X,k):
    """ 把密度链分为密度组 (Density group discovery) """
    result_chains = Density_chains(X,k)
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
            adjacency_list[point].remove(point)  
            #移除自己避免自连接
    
    # DFS,标记所有连通的节点
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
            group_id = len(groups)
            dfs(point, group_id)
            groups.append([key for key, value in density_groups.items() if value == group_id])
    
    return groups


def AL_2_DT(X,k):
    """算法2： 密度追踪算法"""
    result_chains = Density_chains(X,k)
    result_Centrality = Centrality(X,k)
    density_groups = Density_group_discovery(X,k)
    
    density_chain = result_chains.copy()
    density_centrality = result_Centrality.copy()
    density_group = density_groups.copy()
    
    return density_chain,density_centrality,density_group

# 算法2返回密度链条、密度中心性、密度组


# ---------------------------------------------------------------------------------------我是分割线---------------------------------------------------------------------

def Impurity_1(X,k):
    """ 计算不纯度Impurity1 """
    Inpurity_1  = []
    result_chains = Density_chains(X,k)
    for i in range(len(X)):
        Prob = len(result_chains[i]) /( k + 1)
        Inpurity_1.append(1 - Prob**2)
    return Inpurity_1

def Impurity_2(X,k): 
    """ 计算不纯度Impurity2 """
    Inpurity_2  = []
    result_density,result_neightbor = Al_1_DEA(X,k)
    result_chains = Density_chains(X,k)
    for i in range(len(X)):
        Inpurity_2.append(1 - (result_density[i]/ result_density[result_chains[i][-1]]))
    return Inpurity_2

def Impurity(X,k):
    """ 计算不纯度Impurity """
    x1 = Impurity_1(X,k)  
    x2 = Impurity_2(X,k)
    Impurity = []
    for i in range(len(x1)):
        Impurity.append(x1[i] * x2[i])
    return Impurity

Impurity_result = Impurity(X,k)

def AL_3_Pp(X,K_UPPER):
    """算法3：Pre-processing algorithm"""
    t = 1
    r = 0.95
    Global_Density = [0]*len(X)
    Global_Impurity = [0]*len(X)

    for k in K_UPPER:
        Density,Density_neightbor = Al_1_DEA(X,k)
        DensityChains,Centrality,DensityGroups = AL_2_DT(X,k)
        Impurity_total = Impurity(X,k)
        for j in range(len(X)):
            Global_Impurity[j] =Global_Impurity[j] + (r**(k + 1 - t)) * Impurity_total[j]
            Global_Density[j] =Global_Density[j] + (r**(k + 1 - t))* Density[j]
        t+=1
    return Global_Impurity,Global_Density
K_UPPER = [k]
x,y = AL_3_Pp(X,K_UPPER)
print(x,y)

# ---------------------------------------------------------------------------------------我是分割线---------------------------------------------------------------------

DensityGroups = Density_group_discovery(X,k)
centrality  = Centrality(X,k)
Density,Density_neightbor = Al_1_DEA(X,k)

def sum_centrality(X,k):
    """ 求一条链上所有点的中心距的和 """
    Sum_Centrality = []
    result_chains = Density_chains(X,k)
    for i in range(len(result_chains)):
        sum = 0
        for j in result_chains[i]:
            sum += centrality[j]
        Sum_Centrality.append(sum)
    return Sum_Centrality #返回每个链上所有点的中心距的和，是一个列表


def AL_4_CSA(X,k,Impurity_result):
    """ 算法4：Cluster Sampling Algorithm """
    lambda_ = 100
    sampling_rate = 3
    density_drop_rate = 0.8
    sets = set()
    
    impurity = Impurity_result  #代表不纯度结果
    combined = list(zip(X.iterrows(), impurity))
    sorted_combined = sorted(combined, key=lambda item: item[1], reverse=True)
    X_sorted, impurity_sorted = zip(*sorted_combined) # 解压后的DataFrame X_sorted 和 library impurity_sorted
    limit = lambda_ // 3
    result_chains = Density_chains(X,k)

    for i in range(limit):
        minimum_x_to = 10**10
        index  =10**3
        x_from = X_sorted[0][0]  # x_from 为一个Series
        #x_from_index = x_from.name  # x_from.name 是该行在原始 DataFrame X 中的索引标签
        #x_from = x_from_index

        end_index = result_chains[x_from][-1]

        if  x_from != end_index:
            x_end = result_chains[end_index][-1]
            x_to = 10**4
            for j in range(2,len(result_chains[x_from])+1):
                x_to = result_chains[x_from][j-1]
                if Density[x_to] >= density_drop_rate * Density[x_end]:
                    break
                # end if
            #end for
            from_ = X.iloc[x_from]
            to_ = X.iloc[x_to]

            sets.update((tuple(from_),tuple(to_)) )
            
            mid_para = euclidean(from_,to_)
            if mid_para < minimum_x_to:
                minimum_x_to = mid_para
                index = x_to

            sets.update((tuple(from_),tuple(to_)))
    #end for

    limit = lambda_ // (3*sampling_rate)
    for i in range(1,limit+1):
        a = sum_centrality(X,k)
        chain =  result_chains[a.index(max(a))]
        step = len(chain) // sampling_rate
        for j in range(0,sampling_rate-1):
            x_from = chain[j*step]
            x_to = chain[(j+1)*step]
            sets.update((tuple(X.iloc[x_from]),tuple(X.iloc[x_to])) )
        for _ in chain:
            centrality[_] = 0

    return sets
        
ans___ = AL_4_CSA(X,k,Impurity_result)