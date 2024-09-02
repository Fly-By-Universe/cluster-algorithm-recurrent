from numpy import *
import numpy as np
import operator
import pandas as pd


"""读取数据,这里 X 为示例数据"""
X = pd.read_csv('data.csv') 

def euclidean(point1,point2):
    """计算欧几里得距离"""
    return np.sqrt(np.sum((point1 - point2)**2))

def k_nearest_neighbors(x,X,k):
    """计算距离最近的k个点"""
    distances = []
    for data_point in X:
        distance = euclidean(x,data_point)
        distances.append(distance)
    distances_index_pairs = list(enumerate(distances))
    sorted_distances_index_pairs = sorted(distances_index_pairs,key = lambda x:x[1])
    nearest_indices = [index for index, _ in sorted_distances_index_pairs[:k]]
    return nearest_indices


def d(x,x_distance):
    """计算最大密度"""
    return max(x_distance) # 返回x_distance中距离的最大值
    
    
def Al_1_DEA(X,k):
    """算法1 : 密度估计算法(Density estimation algorithm)"""
    Density = []
    for x in X:
        x_distance = k_nearest_neighbors(x,k,X)
        Density.append(1/d(x,x_distance)) # 得到的density值为最近的k个点附近的密度最小值
    return Density


def Density_chains(X,x):
    """算法2 : 找到一条密度链 xi -> xj -> xk (Density following)"""
    return 

def Centrality(X,x):
    return 

def Density_group_discovery(X,x):
    return


def AL_2_DTA(*X,D,k):
    """算法3: 找到密度组并绘制密度框架(Density group discovery)"""
    density_chains_list = []
    Centrality_list = []
    Density_group = []
    for x in X:
        density_chains_list.append(Density_chains(X,x))
        Centrality_list.append(Centrality(X,x))
    """
    DensityGroups <- Discover existing density groups
    """
    return density_chains_list,Centrality_list,Density_group



    



