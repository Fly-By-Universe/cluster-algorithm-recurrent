from __future__ import annotations  
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
import random
from time import time
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, rand_score
from sklearn.preprocessing import MinMaxScaler
import datetime
from enum import Enum, unique
import logging
import threading
from functools import wraps
import psutil
import heapq
import matplotlib.pyplot as plt
def data_processing(filePath: str,  # 要读取的文件位置
                    minMaxScaler=True,  # 是否进行归一化
                    drop_duplicates=True,  # 是否去重
                    shuffle=False):  # 是否进行数据打乱
    data = pd.read_csv(filePath, header=None)
    data1 = data
    true_k = data.iloc[:, -1].drop_duplicates().shape[0]
    if drop_duplicates:
        data = data.drop_duplicates().reset_index(drop=True)
    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True)
    if minMaxScaler:
        data.iloc[:, :-1] = MinMaxScaler().fit_transform(data.iloc[:, :-1])
    # print(data.iloc[:, :-1])
    return np.array(data.iloc[:, :-1]), data.iloc[:, -1].values.tolist(),true_k


def print_estimate(label_true, label_pred, dataset: str, testIndex, iterIndex, time, params='无'):
    print('%15s 数据集，参数%10s，第 %2d次测试，第 %2d次聚类，真实簇 %4d个，预测簇 %4d个，用时 %10.2f s,'
          'RI %0.4f,ARI %0.4f,NMI %0.4f，'
          % (dataset.ljust(15)[:15], params.ljust(10)[:10], int(testIndex), int(iterIndex),
             len(list(set(label_true))), len(list(set(label_pred))), time,
             rand_score(label_true, label_pred),
             adjusted_rand_score(label_true, label_pred),
             normalized_mutual_info_score(label_true, label_pred)))


class Node(object):
    # 基础数据
    id: int  # 节点id
    data: tuple  # 节点数据
    label: int  # 节点标签
    labelName: str  # 节点标签名称
    # 迭代过程
    adjacentNode: dict  # 相邻节点
    degree: int  # 度
    iteration: int  # 迭代序数
    isVisited: bool  # 访问标记
    # 结果
    label_pred: int  # 预测标签
    node_uncertainty : float
    must_link_node : list
    cannot_link_node : list    
    parent : Node 
    query : bool
    children : int
    root_judge : list #是否是本层迭代的根节点？  如果是，则列表最后一个值为1，如果不是，则为0
    def __init__(self, node_id, data, label, labelName):
        # 基础数据
        self.id = int(node_id)  # 节点id
        self.data = data  # 节点数据
        self.label = label  # 节点标签
        self.labelName = labelName  # 节点标签名称
        # 迭代过程
        self.adjacentNode = {}  # 相邻节点
        self.degree = 0  # 度
        self.iteration = 0  # 迭代序数
        self.isVisited = False  # 访问标记
        self.node_num = 0
        self.parent = None #双亲节点，如果没找到就是None
        self.children = 0 #存储孩子节点 
        # 结果
        self.label_pred = 0  # 预测标签
        self.node_uncertainty = 0.0 # 节点的不确定度
        self.must_link_node = []  # 必须链接节点
        self.cannot_link_node = []  # 不能链接节点
        self.root_judge = []   # 初始状态：不是根节点
        self.query = False  # 节点是否被查询过
    def add_adjacent_node(self, node: Node):
        self.adjacentNode[node.id] = node
        self.degree += 1

    def set_iteration(self, iteration: int):
        self.iteration = iteration

    def set_node_num(self, node_num: int):
        self.node_num = node_num


class Graph(object):
    nodeList: list[Node]
    node_size: int

    def __init__(self):
        self.nodeList = []
        self.node_size = len(self.nodeList)

    def add_Node(self, node: Node):
        self.nodeList.append(node)
        self.node_size = len(self.nodeList)

class ClusterResult:
    def __init__(self, dataName, iteration, roots, execution_time, ri, ari, nmi):
        self.dataName = dataName  # 数据集名称
        self.iteration = iteration  # 聚类迭代次数
        self.roots = roots  # 聚类树的根节点及其结构
        self.execution_time = execution_time  # 执行时间
        self.ri = ri  # Rand Index
        self.ari = ari  # Adjusted Rand Index
        self.nmi = nmi  # Normalized Mutual Information

    def __str__(self):
        return (f"ClusterResult(dataName={self.dataName}, iteration={self.iteration}, "
                f"execution_time={self.execution_time:.2f}s, ri={self.ri:.4f}, ari={self.ari:.4f}, "
                f"nmi={self.nmi:.4f})")


class ClusterStoreWithHeap:
    def __init__(self):
        self.heap = []  # 堆，用于按优先级存储聚类树

    def add_result(self, cluster_result: ClusterResult):
        heapq.heappush(self.heap, (cluster_result.execution_time, cluster_result))  # 使用执行时间作为优先级
    def get_best_result(self):
        if self.heap:
            return heapq.heappop(self.heap)[1]  # 返回聚类结果
        return None

    def print_results(self):
        """按执行时间打印所有聚类树结果及其结构"""
        for _, result in self.heap:
            print(result)  # 打印基础信息
            # 打印聚类树结构
            #print(f"\n聚类树结构（迭代次数：{result.iteration}）:")
            for root_id in result.roots:
                #print(f"根节点 {root_id} 的子结构:")
                #print_tree(result.roots[root_id])
                #print("----------")
                pass
        


    def __str__(self):
        return #f"ClusterResult(dataName={self.dataName}, iteration={self.iteration}, ri={self.ri}, ari={self.ari}, nmi={self.nmi}, execution_time={self.execution_time})"

def get_distribute(current_roots: list[int], nodeList: dict[int, Node], size: int):
    label_true = [-1 for i in range(size)]
    label_pred = [-1 for i in range(size)]
    i = 0
    for root_id in current_roots:  # 遍历根节点的ID列表
        node = nodeList[root_id]  # 通过节点ID获取实际的节点对象
        next_node = [node]
        visited = {node.id}
        while next_node:
            r = next_node.pop()
            label_pred[r.id] = i
            label_true[r.id] = r.label
            for n in r.adjacentNode:
                if n not in visited:
                    visited.add(n)
                    next_node.append(r.adjacentNode[n])
        i += 1
    return label_true, label_pred



def extract_pairs_distance(roots, iteration):
    scts = []
    for key in roots:
        sct_node = [roots[key]]
        sct_data = [roots[key].data]
        node_num = 1
        next_node = [roots[key]]
        visited = {key}
        while next_node:
            r = next_node.pop()
            for n in r.adjacentNode:
                if r.adjacentNode[n].iteration == iteration and n not in visited:
                    visited.add(n)
                    next_node.append(r.adjacentNode[n])
                    sct_node.append(r.adjacentNode[n])
                    sct_data.append(r.adjacentNode[n].data)
                    node_num += 1
        for node in sct_node:
            node.set_node_num(node_num)
        # print('%d sct共有 %d 个节点' % (key, node_num))
        scts.append(dict(sct_data=sct_data, sct_node=sct_node))
    return scts


def format_distance(distance):
    d = []
    for i in range(len(distance)):
        for j in range(i + 1, len(distance[i])):
            d.append(distance[i][j])
    d.sort()
    return d


# 计算点对之间的局部密度
def compute_local_density(nodeList, distance, cut_off_distance):
    local_density_point = [dict(node=node, local_density_node_num=0) for node in nodeList]
    for i in range(len(distance)):
        for j in range(i + 1, len(distance[i])):
            if distance[i][j] <= cut_off_distance:
                local_density_point[i]['local_density_node_num'] += 1
                local_density_point[j]['local_density_node_num'] += 1
    data = compute_degree_weigh(sorted(local_density_point, key=lambda k: k['local_density_node_num'], reverse=True))
    return data


# 衡量节点的度距离
def compute_degree_weigh(point):
    repeated_point = [p for p in point if p['local_density_node_num'] == point[0]['local_density_node_num']]
    d = repeated_point[0]['node']
    if len(repeated_point) != 1:
        base_degree = repeated_point[0]['node'].degree
        for p in repeated_point:
            if p['node'].degree > base_degree:
                d = p['node']
                base_degree = p['node'].degree
    return d


def findDensityPeak(query_times: int,roots: list[Node], cut_off=0.4, iteration=0):
    scts = extract_pairs_distance(roots, iteration)
    rootList = {}
    for sct in scts:
        if len(sct['sct_data']) > 1:
            distances = pairwise_distances(np.array(sct['sct_data']), metric="euclidean")
            pairs_distance = format_distance(distances)
            cut_off_distance = pairs_distance[round(len(pairs_distance) * cut_off)]
            root = compute_local_density(sct['sct_node'], distances, cut_off_distance)
            rootList[root.id] = root
            
            for node in sct['sct_node']:
                node.parent = root
                if node.id != root.id:
                    root.children += (node.children + 1)
                    node.root_judge.append(0)    # 把本层除了根节点之外的节点的迭代值标记为0
                if query_times > 0:
                    root.must_link_node = list(set(node.must_link_node + root.must_link_node))
                    root.cannot_link_node = list(set(node.cannot_link_node + root.cannot_link_node))
            root.root_judge.append(1)    #把本层根节点的迭代值标记为1
        else:
            root = sct['sct_node'][0]
            rootList[root.id] = root
            root.parent = root

            root.root_judge.append(1)  # 把本层根节点的迭代值标记为1
    return rootList

# 将数据对象的数据提取用于kdTree计算
# input: list[Node]
# output: list: data
def extract_data_from_Node(nodeList: dict[int, Node]):
    dataList = []
    for key in nodeList:
        dataList.append(nodeList[key].data)
    return dataList


# 寻找数据的最近邻
# input: list[Node]
# output: list: Node's NN index
def findNNs(query_times: int,nodeList: list[Node], k=2):
    dataList = extract_data_from_Node(nodeList)
    return kdTree(query_times,dataList, nodeList, k,return_dist  =False)


def kdTree(query_times: int, dataList, nodeList: dict[int, Node], k, return_dist=False):
    #if query_times == 0:
        # 无论 query_times 的值如何，都计算最近邻
    origin = np.array(dataList)
    neighbors = NearestNeighbors(n_neighbors=3).fit(origin)
        # 获取每个样本的最近邻索引，返回数组 shape=(n_samples, 3)
    indices = neighbors.kneighbors(origin, return_distance=False)
    nns = {}
    snns = {}
        # 假设 nodeList 的 key 顺序与 dataList 的顺序一致
    pos = [key for key in nodeList]
    for i, key in enumerate(nodeList):
            # indices[i][0] 通常是样本自身，indices[i][1] 是最近邻，indices[i][2] 是次近邻
        nns[nodeList[key].id] = pos[indices[i][1]]
        snns[nodeList[key].id] = pos[indices[i][2]]
    return nns, snns
    """
    else
    origin = np.array(dataList)
        neighbors = NearestNeighbors(n_neighbors=3).fit(origin)
        # 获取每个样本的最近邻索引，返回数组 shape=(n_samples, 3)
        indices = neighbors.kneighbors(origin, return_distance=False)
        nns = {}
        snns = {}
        # 假设 nodeList 的 key 顺序与 dataList 的顺序一致
        pos = [key for key in nodeList]
        for i, key in enumerate(nodeList):
            flag0 = False
            flag1 = False
            j = 1
            k = 2
            # indices[i][0] 通常是样本自身，indices[i][1] 是最近邻，indices[i][2] 是次近邻
            while flag0 == False and j < len(indices[i]) :
                if indices[i][j] not in nodeList[key].cannot_link_node:
                    nns[nodeList[key].id] = pos[indices[i][j]]
                    j+= 1
                    flag0 = True
            while flag1 == False and k < len(indices[i]):
                if indices[i][k] not in nodeList[key].cannot_link_node:
                    snns[nodeList[key].id] = pos[indices[i][k]]
                    k+= 1
                    flag1 = True
            snns[nodeList[key].id] = pos[indices[i][2]]
        return nns, snns
    """

        



def compute_sct_num(roots: dict[int, Node], iteration: int):
    rebuild_roots = []
    for key in roots:
        sct_node = [roots[key]]
        node_num = 1
        next_node = [roots[key]]
        other_node = 0
        visited = {key}
        while next_node:
            r = next_node.pop() 
            for n in r.adjacentNode:
                if r.adjacentNode[n].iteration == iteration and n not in visited and (n in r.must_link_node) :
                    next_node.append(r.adjacentNode[n])
                    sct_node.append(r.adjacentNode[n])
                    visited.add(n)
                    other_node = n
                    node_num += 1
        if node_num == 2:
            rebuild_roots.append((key, other_node))
        for node in sct_node:
            node.set_node_num(node_num)
        # print('%d sct共有 %d 个节点' % (key, node_num))
    return rebuild_roots


def construction(nodeList: list[Node], nns: dict[int], snns:dict[int],iteration: int, query_times: int):
    if query_times == 0:  # 无监督方式
        nodeDict = {node.id: node for node in nodeList} 
        roots = {}
        candidates = np.array(nodeList)
        while len(candidates) > 0:
            link = np.array([], dtype=int)  
            i = random.randint(0, len(candidates) - 1)
            n = candidates[i]  
            while True:
                link = np.append(link, n.id)
                n.set_iteration(iteration)
                j_id = nns[n.id]
                j = nodeDict[j_id]
                if j_id in link:
                    roots[n.id] = n
                    candidates = np.array([node for node in candidates if node.id not in link])
                    break
                elif j_id not in [node.id for node in candidates]:
                    n.add_adjacent_node(j)
                    j.add_adjacent_node(n)
                    candidates = np.array([node for node in candidates if node.id not in link])
                    break
                else:
                    n.add_adjacent_node(j)
                    j.add_adjacent_node(n)
                    n = j  
        return roots
    
    else:
        nodeDict = {node.id: node for node in nodeList} 
        roots = {}
        candidates = np.array(nodeList)
        step = 0
        prev_candidate_count = len(candidates)
        while len(candidates) > 0:
            if step > 0 and len(candidates) == prev_candidate_count:
                break
            prev_candidate_count = len(candidates)
            if len(candidates) == 0:
                break
            link = np.array([], dtype=int)
            i = random.randint(0, len(candidates) - 1)
            n = candidates[i]
            step += 1
            step2 = 0
            other_list = []
            visited_cannot_update = set()
            while step2 < 1000:
                step2 += 1
                link = np.append(link, n.id)
                n.set_iteration(iteration)

                j_id = nns[n.id]
                if j_id not in n.cannot_link_node and j_id in n.must_link_node: 
                    j = nodeDict[j_id]
                    if j_id in link:
                        root_node = n
                        roots[root_node.id] = root_node
                        candidates = np.array([node for node in candidates if node.id not in link])
                        break
                    elif j_id not in [node.id for node in candidates]:
                        n.add_adjacent_node(j)
                        j.add_adjacent_node(n)
                        candidates = np.array([node for node in candidates if node.id not in link])
                        break
                    else:
                        n.add_adjacent_node(j)
                        j.add_adjacent_node(n)
                    n = j
                else:
                    break
        # 将剩余候选节点作为根节点加入结果中
        for node in candidates:
            roots[node.id] = node

        return roots







def connect_roots(rebuild_roots, roots, snns, nodeList: list[Node], iteration: int, query_times: int):
    if query_times == 0:
        """
        连接根节点对，确保类型一致并尊重 cannot_link 约束。 
        Args:
            rebuild_roots: 需要重建的根节点对列表
            roots (dict): 根节点字典
            snns (dict): 次近邻字典
            nodeList (list[Node]): 节点列表
            iteration (int): 当前迭代次数
            query_times (int): 查询次数
        """
        # 创建 ID 到 Node 的映射字典
        nodeDict = {node.id: node for node in nodeList}
        candidates = np.array(rebuild_roots).reshape(-1)
        for root in rebuild_roots:
            # 使用 nodeDict 访问节点，避免索引越界
            root_node_0 = nodeDict[root[0]]
            root_node_1 = nodeDict[root[1]]      
            # 检查 cannot_link 关系，使用节点 ID
            if (root_node_1.id in root_node_0.cannot_link_node) or (root_node_0.id in root_node_1.cannot_link_node):
                continue
            else:
                roots.pop(root[0])
                left_connect_node = nodeDict[snns[root[0]]].node_num
                right_connect_node = nodeDict[snns[root[1]]].node_num
                # 选择较大的节点作为主节点
                if left_connect_node <= right_connect_node:
                    big_node = nodeDict[snns[root[1]]]
                    small_node = nodeDict[root[1]]
                else:
                    big_node = nodeDict[snns[root[0]]]
                    small_node = nodeDict[root[0]]
                # 进行相邻节点连接
                big_node.add_adjacent_node(small_node)
                small_node.add_adjacent_node(big_node)
                # 如果小节点在 candidates 中，更新 roots
                if small_node.id in candidates:
                    roots[small_node.id] = small_node
                    
    else:
        nodeDict = {node.id: node for node in nodeList}
        candidates = np.array(rebuild_roots).reshape(-1)
        for root in rebuild_roots:
            root_node_0 = nodeDict[root[0]]
            root_node_1 = nodeDict[root[1]]

            if root_node_0.id in root_node_1.cannot_link_node or root_node_1.id in root_node_0.cannot_link_node:
                continue
            else:
                # 获取次近邻的 node_num（簇大小）
                left_connect_node_num = nodeDict[snns[root[0]]].node_num if snns[root[0]] in nodeDict else 0
                right_connect_node_num = nodeDict[snns[root[1]]].node_num if snns[root[1]] in nodeDict else 0

                # 选择较大的簇作为 big_node
                if left_connect_node_num <= right_connect_node_num:
                    big_node = nodeDict[snns[root[1]]] if snns[root[1]] in nodeDict else None
                    small_node = root_node_1
                else:
                    big_node = nodeDict[snns[root[0]]] if snns[root[0]] in nodeDict else None
                    small_node = root_node_0

                if big_node is None:
                    continue

                # 移除旧的根节点记录
                roots.pop(root[0], None)

                # 进行连接
                big_node.add_adjacent_node(small_node)
                small_node.add_adjacent_node(big_node)

                # 更新 must_link_node 和 cannot_link_node
                #big_node.must_link_node = list(set(big_node.must_link_node + small_node.must_link_node))
                #big_node.cannot_link_node = list(set(big_node.cannot_link_node + small_node.cannot_link_node))
                #big_node.must_link_node.append(small_node.id)
                #small_node.must_link_node.append(big_node.id)

                # 更新 roots
                if small_node.id in candidates:
                    roots[small_node.id] = small_node

        # Step 4: 合并有 must_link 关系的簇
        def merge_clusters(node1, node2, roots):
            """合并两个簇，将 node2 的簇合并到 node1 中，但在合并前检查 cannot_link 约束"""
            # 直接检查
            if node2.id in node1.cannot_link_node or node1.id in node2.cannot_link_node:

                return
            # 聚合检查：检查两个簇中是否存在冲突
            if set(node1.cannot_link_node).intersection(set(node2.must_link_node)) or \
               set(node2.cannot_link_node).intersection(set(node1.must_link_node)):
                return
            # 合并操作
            node1.add_adjacent_node(node2)
            node2.add_adjacent_node(node1)
            if node2.id in roots:
                roots.pop(node2.id)
            roots[node1.id] = node1

        # 使用动态迭代当前 roots 中的所有键
        for node_id in list(roots.keys()):
            # 检查当前键是否仍存在
            if node_id not in roots:
                continue
            node = roots[node_id]
            for must_link_id in node.must_link_node:
                if must_link_id in roots and must_link_id != node.id:
                    merge_clusters(node, roots[must_link_id], roots)




def rebuild(snns: dict[int], roots: list[Node], nodeList: list[Node], iteration: int, query_times):
    """
    重建聚类结构，调用 connect_roots 函数。
    
    Args:
        snns (dict[int]): 次近邻字典
        roots (list[Node]): 根节点列表
        nodeList (list[Node]): 节点列表
        iteration (int): 当前迭代次数
        query_times (int): 查询次数
    """
    rebuild_roots = compute_sct_num(roots, iteration)
    connect_roots(rebuild_roots, roots, snns, nodeList, iteration, query_times)





class Task():
    def __init__(self, params, iterIndex: int, dataName: str, path):
        self.params = params
        self.iterIndex = str(iterIndex)
        self.dataName = str(dataName)
        self.filePath = str(path)

    def __str__(self):
        return '{}-{}'.format(self.dataName, self.iterIndex)

# 记录类型类：用于控制输出的结果类型，该名称会被用于创建输出结果的上级目录
@unique
class RecordType(Enum):
    assignment = 'assignment'
    tree = 'tree'

class Assignment():
    def __init__(self, types: str, iter: str, record: dict):
        self.type = types
        self.iter = iter
        self.record = record
# 记录类：保存每次输出的结果

class Record():
    def __init__(self):
        self.record = []
        self.cpuTime = []

    # 保存每轮聚类结果，必须明确输出类型，子迭代序数（多轮迭代结果），标签信息和预测结果
    def save_output(self, types: RecordType, label_true: list, label_pred: list, iter=0):
        assert isinstance(types, RecordType), TypeError(
            '输入类型必须为RecordType枚举类，请检查。当前类型为 ' + str(type(types)))
        assert len(label_pred) > 0, \
            TypeError('label_pred必须为list类型，且长度不为0')
        assert len(label_pred) > 0, \
            TypeError('label_true必须为list类型，且长度不为0')
        self.record.append(
            Assignment(types, str(iter), {'label_true': label_true, 'label_pred': label_pred}))

    # 保存每轮所用的时间，最终计算得到总时间。
    def save_time(self, cpuTime):
        assert isinstance(cpuTime, float), TypeError("输入类型必须为float类型，请检查。当前类型为 " + str(type(cpuTime)))
        self.cpuTime.append(cpuTime)

logging.basicConfig(level=logging.INFO, format=' %(asctime)s - %(levelname)s- %(message)s')


class ExpMonitor():
    def __init__(self, expId: str, algorithmName: str, storgePath="G:\Experiment"):
        self.task = None
        self.expId = expId
        self.algorithmName = algorithmName
        self.storgePath = storgePath
        self.stop_thread = False  # 初始化 stop_thread 属性

# ExpMonitor 类的 __call__ 方法中
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.task = kwargs['task'] if 'task' in kwargs else args[0]
            if not self.repeat_thread_detection("monitor"):
                t = threading.Thread(target=self.out_memory, name="monitor", daemon=True)
                t.start()
            res = func(*args, **kwargs)
            record: Record = res['record']  # 确保 res 中包含有效的 record
            self.out_record(record.record)  # 此处触发断言
            self.out_cpu_time(record.cpuTime)
            logging.info('%15s 数据集第 %2d 轮测试完成，运行时间 %10.2f s，参数信息：%10s' %
                        (self.task.dataName.ljust(15)[:15], int(self.task.iterIndex),
                        sum(record.cpuTime), str(self.task.params).ljust(10)[:10]))
            return res
        return wrapper


    def out_memory(self):
        p = psutil.Process(os.getpid())
        while not self.stop_thread:  # 加入一个退出条件
            cpu_percent = round(((p.cpu_percent(interval=1) / 100) / psutil.cpu_count(logical=False)) * 100, 2)
            mem_percent = round(p.memory_percent(), 2)
            mem_size = p.memory_info().rss
            data = dict(cpu_utilization_ratio=cpu_percent,
                        memory_utilization_ratio=mem_percent,
                        memory_size_mb=round(mem_size / 1024 / 1024, 4))
            path = os.path.join(self.storgePath, self.expId, self.algorithmName, str(self.task.params), self.task.iterIndex,
                                'memory')
            self.lineOutput(path, self.task.dataName, data)


    # 在实验结束后，调用这个方法停止线程
    def stop_monitor_thread(self):
        self.stop_thread = True

    def out_record(self, records: list[Assignment]):
        assert len(records) > 0, \
            ValueError('输出结果record没有记录，请检查')
        for record in records:
            path = os.path.join(self.storgePath, self.expId, self.algorithmName, str(self.task.params), self.task.iterIndex,
                                'output',
                                self.task.dataName, record.type.value)
            self.makeDir(path)
            pd.DataFrame(record.record).to_csv(path + '/' + record.iter + '.csv', index=False)

    def out_cpu_time(self, cpuTime: list):
        assert len(cpuTime) > 0, \
            ValueError('cputime 没有记录，请检查')
        path = os.path.join(self.storgePath, self.expId, self.algorithmName, str(self.task.params), self.task.iterIndex,
                            'cpuTime')
        self.makeDir(path)
        self.lineOutput(path, 'cpuTime', dict(dataName=self.task.dataName, cpuTime=sum(cpuTime)))

    # 输出信息工具
    def lineOutput(self, path, fileName, data: dict):
        self.makeDir(path)
        outputPath = os.path.join(path, fileName + '.csv')
        if not os.path.exists(outputPath):
            pd.DataFrame(data, index=[0]).to_csv(outputPath, index=False, mode='a')
        else:
            pd.DataFrame(data, index=[0]).to_csv(outputPath, index=False, header=False, mode='a')

    # 目录创建工具
    def makeDir(self, path):
        os.makedirs(path, exist_ok=True)

    # 查询线程是否活动
    def repeat_thread_detection(self, tName):
        for item in threading.enumerate():
            if tName == item.name:
                return True
        return False




def query_oracle(node1, node2):
    """
    根据两个节点的真实标签判断它们之间的连接关系。
    """
    if node1.label == node2.label:
        return "must_link"
    else:
        return "cannot_link"


from collections import defaultdict

class UnionFind:
    """并查集（Union-Find）数据结构，用于快速管理 must_link 连通分量"""
    def __init__(self):
        self.parent = dict()  # {node_id: parent_id}
        
    def find(self, node_id):
        if node_id not in self.parent:
            self.parent[node_id] = node_id
        while self.parent[node_id] != node_id:
            self.parent[node_id] = self.parent[self.parent[node_id]]  # 路径压缩
            node_id = self.parent[node_id]
        return node_id
    
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_y] = root_x

def query_top(root_keys, nodes_dict, query_times):
    uf = UnionFind()  # 初始化并查集

    # 为每个节点初始化连通分量的 must_link 和 cannot_link 集合
    # 注意：这里的 key 就用节点的 id（在最初时，每个节点独立）
    group_must = {node_id: {node_id} for node_id in root_keys}
    group_cannot = {node_id: set() for node_id in root_keys}

    # 辅助函数：返回节点当前连通分量的代表及其 must_link 和 cannot_link 集合
    def get_group_info(node_id):
        root = uf.find(node_id)
        return root, group_must[root], group_cannot[root]

    n = len(root_keys)
    for i in range(n):
        for j in range(i + 1, n):
            node_i_id = root_keys[i]
            node_j_id = root_keys[j]
            root_i = uf.find(node_i_id)
            root_j = uf.find(node_j_id)

            # 如果已在同一连通分量，则已通过传递性建立 must_link，无需查询
            if root_i == root_j:
                continue

            # 传递规则：若其中一个分量的 cannot_link 集合中已有对方的某个节点，则关系可推断为 cannot_link
            if (node_j_id in group_cannot[root_i]) or (node_i_id in group_cannot[root_j]):
                # 已知它们必然不能连通，不调用查询函数
                continue

            # 进行真实查询
            query_times += 1
            node_i = nodes_dict[node_i_id]
            node_j = nodes_dict[node_j_id]
            result = query_oracle(node_i, node_j)

            if result == "must_link":
                # 合并两个连通分量：记住合并后新代表可能是root_i或root_j
                uf.union(node_i_id, node_j_id)
                new_root = uf.find(node_i_id)
                old_root = root_j if new_root == root_i else root_i

                # 合并两个分量的 must_link 和 cannot_link 集合
                merged_must = group_must[new_root].union(group_must[old_root])
                merged_cannot = group_cannot[new_root].union(group_cannot[old_root])
                group_must[new_root] = merged_must
                group_cannot[new_root] = merged_cannot

                # 更新所有合并后的节点信息（仅更新列表展示，不影响后续计算，计算时依然以 group 集合为准）
                for node_id in merged_must:
                    node = nodes_dict[node_id]
                    # 必须连通的节点列表（不包含自己）
                    node.must_link_node = list(merged_must - {node_id})
                    node.cannot_link_node = list(merged_cannot)
            else:  # result == "cannot_link"
                # 根据规则：如果A和B为 cannot_link，则A的整个 must_link 集合都与B不能连通
                # 更新各自分量的 cannot_link 集合
                group_cannot[root_i].update(group_must[root_j])
                group_cannot[root_j].update(group_must[root_i])

                # 更新相关节点的 cannot_link 列表
                for node_id in group_must[root_i]:
                    node = nodes_dict[node_id]
                    node.cannot_link_node = list(group_cannot[root_i])
                for node_id in group_must[root_j]:
                    node = nodes_dict[node_id]
                    node.cannot_link_node = list(group_cannot[root_j])
    return query_times







def plot_clustering_iteration(data, labels, nodeList, iteration):
    """
    绘制当前迭代的聚类结果，并显示 Must-Link 和 Cannot-Link 约束
    """
    plt.figure(figsize=(8, 8))

    # 绘制数据点，颜色代表当前聚类类别
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.6, edgecolors='k')

    # 遍历所有节点，绘制 Must-Link 和 Cannot-Link 约束
    for node_id, node in nodeList.items():
        #print(node_id, node.must_link_node, node.cannot_link_node)
        for must_link_id in node.must_link_node:
            # 检查该 must_link_id 是否仍存在于当前的 nodeList 中
            if must_link_id in nodeList:
                must_link_node = nodeList[must_link_id]
                plt.plot([node.data[0], must_link_node.data[0]],
                         [node.data[1], must_link_node.data[1]],
                         'b-', linewidth=0.8)  # Must-Link (红色实线)

        for cannot_link_id in node.cannot_link_node:
            if cannot_link_id in nodeList:
                cannot_link_node = nodeList[cannot_link_id]
                plt.plot([node.data[0], cannot_link_node.data[0]],
                         [node.data[1], cannot_link_node.data[1]],
                         'b--', linewidth=0.8)  # Cannot-Link (红色虚线)

    plt.title(f"Clustering at Iteration {iteration}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


    # 辅助函数：递归改变节点及其所有子节点的标签
def change_label(node: Node, new_label: int, visited: set = None):
    
    if visited is None:
        visited = set()
    if node.id in visited:
        return
    visited.add(node.id)
    node.label_pred = new_label
    if node.iteration == 0:
        return
    for adj_id, adj_node in node.adjacentNode.items():
        change_label(adj_node, new_label, visited)


    
    # 辅助函数：获取节点的所有子节点
def examine(nodeList: dict[int, Node]) -> dict[int, Node]:
    """
    检查并调整nodeList中的密度峰值节点，确保cannot_link的节点不在同一簇中。
    当改变某个节点的标签时，递归更新其所有子节点（包括子节点的子节点）的标签。
        nodeList (dict[int, Node]): 密度峰值节点字典，键为节点ID，值为Node对象。
        dict[int, Node]: 调整后的nodeList。
    """
    # 辅助函数：生成新的唯一标签
    def generate_new_label(current_labels: set[int]) -> int:
        return max(current_labels, default=-1) + 1
    
    # 获取所有密度峰值节点
    peak_nodes = list(nodeList.values())
    # 获取当前所有标签集合
    current_labels = set(node.label_pred for node in peak_nodes)
    
    # 遍历所有密度峰值节点对
    for i in range(len(peak_nodes)):
        for j in range(i + 1, len(peak_nodes)):
            node1 = peak_nodes[i]
            node2 = peak_nodes[j]
            
            # 检查是否为cannot_link且标签相同
            if (node2.id in node1.cannot_link_node or node1.id in node2.cannot_link_node) and \
               node1.label_pred == node2.label_pred:
                # 确定sct_num较小的节点
                if node1.id  in node2.adjacentNode:
                    del node2.adjacentNode[node1.id]
                if node2.id  in node1.adjacentNode:
                    del node1.adjacentNode[node2.id]
                if node1.node_num < node2.node_num:
                    small_node = node1
                    large_node = node2
                else:
                    small_node = node2
                    large_node = node1
                
                # 与其他密度峰值节点查询关系
                all_cannot_link = True
                for other_node in peak_nodes:
                    if other_node != small_node and other_node != large_node:
                        relation = query_oracle(small_node, other_node)  
                        if relation == "must_link" :
                            if other_node.id not in small_node.adjacentNode:
                                small_node.add_adjacent_node(other_node)
                                other_node.add_adjacent_node(small_node)
                            all_cannot_link = False
                            # 比较sct_num并调整标签
                            #if small_node.node_num < other_node.node_num:
                                #small_node.must_link_node = []
                                #change_label(small_node, other_node.label_pred)
                            #else:
                                #other_node.must_link_node = []
                                #change_label(other_node, small_node.label_pred)
                            break
                
                # 如果与所有其他节点均为cannot_link，则分配新标签
                if all_cannot_link:
                    new_label = generate_new_label(current_labels)
                    change_label(small_node, new_label)
                    current_labels.add(new_label)
    
    return nodeList


def set_parents(nodeList_copy: dict[int, Node], node: Node, visited: set = None, depth: int = 0):
    if visited is None:
        visited = set()
    if node.id in visited:
        return
    visited.add(node.id)
    print(f"Depth {depth}, Node {node.id}, iteration {node.iteration}")
    for child_id, child_node in node.adjacentNode.items():
        nodeList_copy[child_id].parent = node.id
        if child_node.iteration > 0:
            set_parents(nodeList_copy, child_node, visited, depth + 1)

def spread(node):
    if node.root_judge[node.iteration] == 1:  # 如果 node 原先是根节点，直接返回
        return  
    elif node.root_judge[node.iteration] == 0 and node.iteration > 0:
        to_delete = []  # 先收集要删除的键
        for i in list(node.adjacentNode.values()):  # 转换为列表，避免修改字典时报错
            if node.iteration == i.iteration:
                to_delete.append(i)  # 记录需要删除的节点

        # 统一删除，避免遍历过程中修改字典
        for i in to_delete:
            del i.adjacentNode[node.id]
            if i.id in node.adjacentNode:  # 如果 i 在 node 的邻接节点中，则删除
                del node.adjacentNode[i.id]
            if node.parent.id not in i.adjacentNode:   # 如果双亲节点不在 i 的邻接节点中，则添加
                i.add_adjacent_node(node.parent)
            if i.id not in node.parent.adjacentNode:
                node.parent.add_adjacent_node(i)

    elif node.root_judge[node.iteration] == 0 and node.iteration == 0:
        to_delete = list(node.adjacentNode.values())  # 先记录需要删除的节点

        for i in to_delete:
            del i.adjacentNode[node.id]
            if i.id in node.adjacentNode:  # 如果 i 在 node 的邻接节点中，则删除
                del node.adjacentNode[i.id]
            if node.parent.id not in i.adjacentNode:   # 如果双亲节点不在 i 的邻接节点中，则添加
                i.add_adjacent_node(node.parent)
            if i.id not in node.parent.adjacentNode:
                node.parent.add_adjacent_node(i)

    
            


C1 = 1
beta1 = 0.75
C2 = 1.2
beta2 = 0.6
# 修改后的 run 函数（部分）
@ExpMonitor(expId='ALDP', algorithmName='ALDP', storgePath="C:/Users/DELL/OneDrive/桌面/dezh123/123")
def run(task: Task, data_override=None):
    global query_times
    query_times = 0
    record = Record()
    # 判断是否使用外部数据
    if data_override is not None:
        data, label, K = data_override
    else:
        data, label, K = data_processing(task.filePath)
    n = len(data)
    nodeList = {i: Node(i, data[i], label[i], label[i]) for i in range(len(data))}
    iteration = 0
    start = time.time()
    current_cluster_count_1 = 0
    current_cluster_count_2 = -1
    # 用于记录每次迭代的根节点序号
    iteration_roots_list = []
    query_iteration = 0
    while len(nodeList) >= K and len(nodeList) >= 3 and current_cluster_count_1 != current_cluster_count_2:
        current_cluster_count_2 = current_cluster_count_1
        nns, snns = findNNs(query_times, nodeList=nodeList, k=3)
        roots = construction(nodeList=list(nodeList.values()), nns=nns, snns = snns,iteration=iteration, query_times=query_times) 
        rebuild(snns, roots, list(nodeList.values()), iteration, query_times)
        nodeList = findDensityPeak(query_times,roots, task.params, iteration=iteration)
        current_roots = list(nodeList.keys())
        if  n <= 1500:
            x = round(C1 * (n ** beta1))
        else:
            x = round(C2 * (n ** beta2))
        # 进行第一次查询
        if len(current_roots) <= x:
            query_times = query_top(current_roots, nodeList, query_times)
            query_iteration += 1
        # 记录当前迭代的根节点
        #if query_times > 0  and query_iteration >=1:
            #nodeList = examine(nodeList)
        current_roots = list(nodeList.keys())
        iteration_roots_list.append(current_roots)
        # 获取当前聚类标
        label_true, label_pred = get_distribute(current_roots, nodeList, len(label))
        
        end = time.time()
        record.save_time(end - start)
        record.save_output(RecordType.assignment, label_true, label_pred, iteration)
        elapsed_time = time.time() - start
        print_estimate(label_true, label_pred, task.dataName, task.iterIndex, 0, elapsed_time)
        current_cluster_count_1 = len(nodeList)
        iteration += 1

        #plot_clustering_iteration(data, label_pred, nodeList, iteration)
    print(f"查询次数{query_times}")
    
    cluster_centers = extract_cluster_centers(current_roots, nodeList)
    # 计算聚类树中所有节点的不确定度（内部会遍历整个树，对每个节点调用 calculate_Uncertainty）
    
    calculate_all_uncertainties(nodeList, cluster_centers)
    #  收集聚类树中所有节点
    def get_all_nodes(roots: dict[int, Node]) -> list[Node]:
        all_nodes = {}
        for root in roots.values():
            stack = [root]
            while stack:
                node = stack.pop()
                if node.id not in all_nodes:
                    all_nodes[node.id] = node
                    for child in node.adjacentNode.values():
                        stack.append(child)
        return list(all_nodes.values())
    all_nodes = get_all_nodes(nodeList)         # 得到所有节点
    top_nodes = get_top_k_uncertain_nodes(all_nodes, top_k = len(data))     
                      # 利用堆排序得到不确定度最高的 top_k 个节点
    for node in top_nodes:
        node_copy = node
        while node_copy.parent.id != node_copy.id:
            node_copy = node_copy.parent
        ans = query_oracle(node_copy, node)
        query_times += 1
        if ans == "must_link":
            continue
        elif ans == "cannot_link":
            flag  = False  # 标记是否找到可以链接的节点
            query_times_2 = 0 
            for i in nodeList.values():
                if i.id == node_copy.id:
                    continue
                else:
                    ans_2 = query_oracle(i, node)
                    if ans_2 == "must_link":
                        flag = True #能找到可连接的双亲节点
                        t = i
                        node.label_pred = i.label_pred
                        break
                    elif ans_2 == "cannot_link":
                        query_times_2 += 1

                    
            query_times += query_times_2
            if  flag == False:
                node.label_pred = len(current_roots)
                current_roots.append(node.id)
                nodeList[node.id] = node   

            spread(node)
            if  node.id in node.parent.adjacentNode:  #将node和其parent互相删除
                del node.parent.adjacentNode[node.id]
            if node.parent.id in node.adjacentNode:
                del node.adjacentNode[node.parent.id]

            if flag:    
                node.add_adjacent_node(t)
                t.add_adjacent_node(node)
                node.parent = t     #将node作为i的子节点（邻接点），互相添加

        label_true, label_pred = get_distribute(current_roots, nodeList, len(label))
        ari = adjusted_rand_score(label_true, label_pred)
        if ari >= 0.9999 :
            break


    print_estimate(label_true, label_pred, task.dataName, task.iterIndex, 0, elapsed_time)   
    ari = adjusted_rand_score(label_true, label_pred)
    print(f"查询次数{query_times},    ari = {ari}")
    print("\n")
    return {'record': record}

#如果先查到iteration小的数据，该数据与根节点的查询结果返回must_link，然后又查到该数据的祖先节点，但是祖先节点与根节点的查询结果返回cannot_link，那么就认为该数据与根节点之间存在 cannot_link 的关系，
#那么该祖先节点就会和根节点分离，iteration小的数据本来连接正确，但是会随着祖先节点的移动而被错误地连接到其他节点上，这又可能导致聚类结果发生错误
#出现这种情况的原因是祖先节点与该节点在一开始就被错误地归为一类
# 但是重新聚类的开销又会很大，我们的要求是最大限度降低查询次数！


# 在主程序中创建ClusterStore实例并保存结果
cluster_store = ClusterStoreWithHeap()

########################################################################################################################################
########################################################################################################################################
# 提取聚类树
########################################################################################################################################
########################################################################################################################################

def print_tree(node, visited=None):
    if visited is None:
        visited = set()
    visited.add(node.id)
    
import numpy as np

def calculate_Uncertainty(node, cluster_centers):
    """
    基于 Softmax 的不确定度计算。
    对每个节点，先计算其到所有聚类中心的距离，然后将距离转化为相似度：
        s_i = exp(-alpha * d_i)
    再归一化得到概率分布 P，并计算熵作为该节点的不确定度。
    参数：
    node (Node): 需要计算不确定性的节点，对象中包含 node.data (坐标信息)
    cluster_centers (list/array): 聚类中心的坐标列表，每个元素是一个坐标向量
    alpha (float): 控制距离对相似度影响的尺度因子，alpha 越大分布越尖锐
    返回：
    float: 该节点的熵（不确定度）
    """
    epslion = 1e-20
    alpha=1.0
    node_data = np.array(node.data)
    centers = np.array(cluster_centers)

    distances = np.linalg.norm(centers - node_data, axis=1)

    s = np.exp(-alpha * distances)
    sum_s = np.sum(s) + epslion
    P = s / sum_s

    # 计算熵: -∑ P_i * log(P_i)
    entropy = -np.sum(P * np.log(P + epslion))

    # 将结果存入节点对象的属性
    node.node_uncertainty = entropy

    return entropy

def calculate_all_uncertainties(roots, cluster_centers):
    """
    计算聚类树中所有节点的不确定性。

    参数：
    roots (dict): 聚类树中根节点的字典
    cluster_centers (list): 聚类中心列表

    返回：
    dict: 节点ID及其不确定性值的字典
    """
    uncertainty_values = {}
    visited = set()
    for root_id, root_node in roots.items():
        nodes_to_process = [root_node]
        
        while nodes_to_process:
            current_node = nodes_to_process.pop()

            if current_node.id in visited:
                continue
            visited.add(current_node.id)
            
            uncertainty = calculate_Uncertainty(current_node, cluster_centers)
            uncertainty_values[current_node.id] = uncertainty

            for adj_id, adj_node in current_node.adjacentNode.items():
                if adj_id not in visited:
                    nodes_to_process.append(adj_node)
    
    return uncertainty_values


def extract_cluster_centers(current_roots, nodeList):
    """
    寻找聚类中心
    current_roots: 一个节点 ID 列表
    nodeList: 包含所有节点的字典，节点 ID -> 节点对象
    """
    centers = []
    for root_id in current_roots:  # 遍历 current_roots 中的每个节点 ID
        node = nodeList[root_id]  # 通过 ID 获取节点
        centers.append(node.data)  # 获取该节点的 data 作为聚类中心
    return centers



def get_all_nodes(roots: dict[int, Node]) -> list[Node]:
    """从聚类树中收集所有节点，避免重复访问"""
    all_nodes = {}
    for root in roots.values():
        stack = [root]
        while stack:
            node = stack.pop()
            if node.id not in all_nodes:
                all_nodes[node.id] = node
                # 将相邻节点加入遍历队列
                for child in node.adjacentNode.values():
                    stack.append(child)
    return list(all_nodes.values())



def get_top_k_uncertain_nodes(nodes, top_k=30):
    # 使用 heapq.nlargest 快速提取不确定度最高的 top_k 个节点
    return heapq.nlargest(top_k, nodes, key=lambda node: (node.iteration, node.node_uncertainty))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

def generate_synthetic_data(n_samples=3000, n_features=2, n_clusters=3, cluster_std=2.2, random_state=100):
    """
    生成二维平面数据点，并增加随机性。
    cluster_std 控制数据分布的散度，值越大数据越乱。
    """
    data, labels = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, 
                              cluster_std=cluster_std, random_state=random_state)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)  # 归一化
    return data, labels, n_clusters


if __name__ == '__main__':
    """
    # 生成数据
    data, label, K = generate_synthetic_data(n_samples=2000, n_clusters=5)

    # 适配原框架（但不使用文件路径）
    task = Task(params=0.05, iterIndex=1, dataName="synthetic", path=None)
    
    # 运行 ALDP 聚类算法，直接传递数据
    result = run(task=task, data_override=(data, label, K))

    
    # 可视化聚类结果
    label_pred = result['record'].record[-1].record['label_pred']
    
    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], c=label_pred, cmap='viridis', alpha=0.6, edgecolors='k')
    plt.title("ALDP Clustering on Synthetic Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
    """
    path = "D:/ALDP-master/data/small"
    cut = 0.05
    cluster_store = ClusterStoreWithHeap()  # 确保在外部初始化

    for dataName in os.listdir(path):
        data, label, K = data_processing(path + '/' + dataName)
        task = Task(round(cut * 1, 2), 1, dataName.split('.')[0], path + '/' + dataName)
        run(task=task)

    
    # 所有实验完成后打印堆结果
    cluster_store.print_results()
    best_result = cluster_store.get_best_result()
    print("\n")
    
    