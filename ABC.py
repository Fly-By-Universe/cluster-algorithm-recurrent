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
def data_processing(filePath: str,  # 要读取的文件位置
                    minMaxScaler=True,  # 是否进行归一化
                    drop_duplicates=True,  # 是否去重
                    shuffle=True):  # 是否进行数据打乱
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
    def __init__(self, id, data, label, labelName):
        # 基础数据
        self.id = id  # 节点id
        self.data = data  # 节点数据
        self.label = label  # 节点标签
        self.labelName = labelName  # 节点标签名称
        # 迭代过程
        self.adjacentNode = {}  # 相邻节点
        self.degree = 0  # 度
        self.iteration = 0  # 迭代序数
        self.isVisited = False  # 访问标记
        self.node_num = 0
        # 结果
        self.label_pred = 0  # 预测标签
        self.node_uncertainty = 0.0 # 节点的不确定度

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
                print(f"根节点 {root_id} 的子结构:")
                print_tree(result.roots[root_id])
                print("----------")
        


    def __str__(self):
        return f"ClusterResult(dataName={self.dataName}, iteration={self.iteration}, ri={self.ri}, ari={self.ari}, nmi={self.nmi}, execution_time={self.execution_time})"

def get_distribute(roots: dict[int, Node], size: int):
    label_true = [-1 for i in range(size)]
    label_pred = [-1 for i in range(size)]
    i = 0
    for key in roots:
        next_node = [roots[key]]
        visited = {roots[key].id}
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


def findDensityPeak(roots: list[Node], cut_off=0.4, iteration=0):
    scts = extract_pairs_distance(roots, iteration)
    rootList = {}
    for sct in scts:
        if len(sct['sct_data']) > 1:
            distances = pairwise_distances(np.array(sct['sct_data']), metric="euclidean")
            pairs_distance = format_distance(distances)
            cut_off_distance = pairs_distance[round(len(pairs_distance) * cut_off)]
            root = compute_local_density(sct['sct_node'], distances, cut_off_distance)
            rootList[root.id] = root
        else:
            root = sct['sct_node'][0]
            rootList[root.id] = root
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
def findNNs(nodeList: list[Node], k=2):
    dataList = extract_data_from_Node(nodeList)
    return kdTree(dataList, nodeList, k)


def kdTree(dataList, nodeList: dict[int, Node], k, return_dist=False):
    origin = np.array(dataList)
    neighbors = NearestNeighbors(n_neighbors=3).fit(origin)
    dist = neighbors.kneighbors(origin, return_distance=False).tolist()
    nns = {}
    snns = {}
    i = 0
    pos = [key for key in nodeList]
    for key in nodeList:
        nns[nodeList[key].id] = pos[dist[i][1:][0]]
        snns[nodeList[key].id] = pos[dist[i][2:][0]]
        i += 1
    return nns, snns

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
                if r.adjacentNode[n].iteration == iteration and n not in visited:
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


def connect_roots(rebuild_roots, roots, snns, nodeList: list[Node]):
    candidates = np.array(rebuild_roots).reshape(-1)
    for root in rebuild_roots:
        roots.pop(root[0])
        left_connect_node = nodeList[snns[root[0]]].node_num
        right_connect_node = nodeList[snns[root[1]]].node_num
        if left_connect_node <= right_connect_node:
            nodeList[snns[root[0]]].add_adjacent_node(nodeList[root[0]])
            nodeList[root[0]].add_adjacent_node(nodeList[snns[root[0]]])
            if snns[root[0]] in candidates:
                roots[snns[root[0]]] = nodeList[snns[root[0]]]
            # print('%d 0 link 0 %d' % (root[0], snns[root[0]]))
        else:
            nodeList[snns[root[1]]].add_adjacent_node(nodeList[root[1]])
            nodeList[root[1]].add_adjacent_node(nodeList[snns[root[1]]])
            if snns[root[1]] in candidates:
                roots[snns[root[1]]] = nodeList[snns[root[1]]]
            # print('%d 1 link 1 %d' % (root[1], snns[root[1]]))

def rebuild(snns: dict[int], roots: list[Node], nodeList: list[Node], iteration: int):
    rebuild_roots = compute_sct_num(roots, iteration)
    connect_roots(rebuild_roots, roots, snns, nodeList)

def construction(nodeList: list[Node], nns: dict[int], iteration: int):
    roots = {}
    candidates = np.array([key for key in nodeList])
    while len(candidates) > 0:
        link = np.array([])
        # i 代表随机选取的数据点的数组索引
        i = random.randint(0, len(candidates) - 1)
        n = candidates[i]
        while True:
            link = np.append(link, n)
            nodeList[n].set_iteration(iteration)
            # j 代表随机选取的i的数据点的对应最近邻的数组索引
            j = nns[n]
            if j in link:
                roots[nodeList[n].id] = nodeList[n]
                candidates = np.setdiff1d(candidates, link)
                # print('link: %s ,root: %d' % (output + '->' + str(j), n))
                break
            elif j not in candidates:
                nodeList[n].add_adjacent_node(nodeList[j])
                nodeList[j].add_adjacent_node(nodeList[n])
                candidates = np.setdiff1d(candidates, link)
                # print(output + '->' + str(j))
                break
            nodeList[n].add_adjacent_node(nodeList[j])
            nodeList[j].add_adjacent_node(nodeList[n])
            n = j
    return roots


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
            time.sleep(0.1)

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



# 修改后的 run 函数（部分）
@ExpMonitor(expId='ALDP', algorithmName='ALDP', storgePath= "C:/Users/DELL/OneDrive/桌面/dezh123/123")
def run(task: Task):
    record = Record()
    data, label, K = data_processing(task.filePath)
    nodeList = {i: Node(i, data[i], label[i], label[i]) for i in range(len(data))}
    iteration = 1
    start = time.time()
    while len(nodeList) > K:
        nns, snns = findNNs(nodeList=nodeList, k=3)
        roots = construction(nodeList=nodeList, nns=nns, iteration=iteration)
        rebuild(snns, roots, nodeList, iteration)
        nodeList = findDensityPeak(roots, task.params, iteration=iteration)
        label_true, label_pred = get_distribute(roots, len(label))
        end = time.time()
        record.save_time(end - start)
        record.save_output(RecordType.assignment, label_true, label_pred, iteration)  # 关键：确保调用
        elapsed_time = time.time() - start
        print_estimate(label_true, label_pred, task.dataName, task.iterIndex, 0, elapsed_time)
        # 生成 ClusterResult 并存入堆
        iteration += 1
    cluster_result = ClusterResult(
            dataName=task.dataName,
            iteration=iteration,       
            roots=roots,
            execution_time=end - start,
            ri=rand_score(label_true, label_pred),
            ari=adjusted_rand_score(label_true, label_pred),
            nmi=normalized_mutual_info_score(label_true, label_pred)
        )
    cluster_store.add_result(cluster_result)
    # print(ClusterResult.roots)
    return {'record': record}  



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
    #print(f"Node {node.id} -> {', '.join(str(neighbor.id) for neighbor in node.adjacentNode.values())}")

    #for neighbor in node.adjacentNode.values():
        #if neighbor.id not in visited:
            #print_tree(neighbor, visited)



    
def calculate_Uncertainty(node, cluster_centers):
    """
通过计算到所有聚类中心的欧几里得距离之和来确定一个节点的不确定性。

参数：
node (Node): 需要计算不确定性的节点
cluster_centers (list): 聚类中心列表（每个中心是一个坐标的元组/数组）

返回：
float: 欧几里得距离之和（不确定性值）
    """
    epslion = 1e-8
    node_data = np.array(node.data)
    centers = np.array(cluster_centers)

    # 计算节点到所有聚类中心的欧几里得距离，利用向量化操作
    distances = np.linalg.norm(centers - node_data, axis=1)
    total_distance = np.sum(distances)
    # 防止除零，加一个极小值
    P = distances / (total_distance + epslion)
    #熵（Entropy）方法
    entropy = -np.sum(P * np.log(P + epslion))
    node.uncertainty = entropy
    """
    边际差方法
    sorted_P = np.sort(P)[::-1] 
    margin = sorted_P[0] - sorted_P[1]
    node.uncertainty = 1 - margin
    """
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


def extract_cluster_centers(roots):
    """
    寻找聚类中心
    """
    centers = []
    for root_id, root_node in roots.items():
        centers.append(root_node.data)
    return centers



"""
# 在找到密度峰值并确定聚类之后：
cluster_centers = extract_cluster_centers(roots)  # 从根节点中提取聚类中心
uncertainty_values = calculate_all_uncertainties(roots, cluster_centers)  # 计算所有节点的不确定性值

# 为了找到最不确定的节点（可能是离群点或边界点）：
sorted_uncertainties = sorted(uncertainty_values.items(), key=lambda x: x[1], reverse=True)  # 按不确定性值从高到低排序
most_uncertain_nodes = sorted_uncertainties[:10]  # 获取前10个最不确定的节点

# 为了在下一轮迭代中使用不确定性进行决策：
for node_id, node in nodeList.items():
    if node.node_uncertainty > threshold:  # 如果节点的不确定性超过阈值
        # 对高不确定性节点进行特殊处理
        pass
"""









# 主程序
if __name__ == '__main__':
    path = "D:/ALDP-master/data/small"
    cut = 0.05
    cluster_store = ClusterStoreWithHeap()  # 确保在外部初始化

    for dataName in os.listdir(path):
        data, label, K = data_processing(path + '/' + dataName)
        task = Task(round(cut * 1, 2), 1, dataName.split('.')[0], path + '/' + dataName)
        run(task=task)

    
    # 所有实验完成后打印堆结果
    cluster_store.print_results()
    time.sleep(1)
    print(cluster_store.heap)
    best_result = cluster_store.get_best_result()

    # 检查是否存在结果
    if best_result:
        print("Roots:", best_result.roots)  # 通过实例访问 roots 属性
    else:
        print("未找到聚类结果。")

        











