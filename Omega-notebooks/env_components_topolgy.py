import numpy as np
import networkx as nx
from env_components_gpu import Gpu
from env_components_parameters import Parameters

__all__ = ['NetworkTopology']


class NetworkTopology:
    """
        The class to hold the topology  of the environment.

        This class must be created once in beginning, it constructs the cluster topology,
        the adjacency matrix and then initialise all gpus.

        Attributes
        ----------
        G : networkx
            it is the created graph
        node_index : int
            used to index each node in the graph, it is increased by 1 when a node is added
        dic_nam_idx : dictionary
            this dictionary contains each node in the graph as a key and its index as value
        edges_array: ndarray
            contain all edges in the graph
        nvLink: int 40
            the weight of the edge between gpus sharing same host
            NVLink between two nodes within the same host
        infiniBand: int 4
            the weight of the edge between gpus sharing same n
            InfiniBand between GPUs across hosts
        peer_to_peer: int 2
            Peer-to-peer direct link within racks
        resources: ndarray
            ndarray containing the shape of all resource in the cluster as host rack gpu
        gpu_number: int
            total number of gpu in the cluster
        cluster_gpus: ndarray
            ndarray containing all gpus object
        adjacency_matrix: 2d-array
            matrix representing the adjacency between gpus, each cell contain the weigh of the path.
    """
    def __init__(self):
        
        
        num_gpus_per_machine = 4
        num_machines_per_rack = 4
        num_racks_per_cluster = 2
        max_gpu_request = 8
        max_job_len = 30
        jobqueue_maxlen = 10
        max_backlog_len = 10
        new_job_rate = 0.6
        target_num_job_done = 50
        delay_penalty = -1
        hold_penalty = -1
        dismiss_penalty = -1
    
    
        pa = Parameters(num_gpus_per_machine=num_gpus_per_machine,
                        num_racks_per_cluster=num_racks_per_cluster,
                        num_machines_per_rack=num_machines_per_rack,
                        max_gpu_request=max_gpu_request,
                        max_job_len=max_job_len,
                        jobqueue_maxlen=jobqueue_maxlen,
                        max_backlog_len=max_backlog_len,
                        new_job_rate=new_job_rate,
                        hold_penalty=hold_penalty,
                        delay_penalty=delay_penalty,
                        dismiss_penalty=dismiss_penalty,
                        target_num_job_done=target_num_job_done,
                        max_num_timesteps=60000)
        
        
        
        self.G = None
        self.node_index = 0
        self.dic_nam_idx = {}
        self.edges_array = []
        self.nvLink = 40
        self.infiniBand = 4
        self.peer_to_peer = 2
        # rack Host class, contains gpu
        self.resources = -np.ones((pa.num_racks_per_cluster,
                                   pa.num_machines_per_rack,
                                   pa.num_gpus_per_machine), dtype=np.int32)
        self.gpu_number = pa.num_racks_per_cluster * \
                          pa.num_machines_per_rack * \
                          pa.num_gpus_per_machine
        self.cluster_gpus = [] * self.gpu_number
        self.adjacency_matrix = np.ones((self.gpu_number, self.gpu_number))
        self.node_index_build()
        self.create_graph()
        self.create_adjacency_matrix()
        
    def node_index_build(self):
        """

        :return:
        """
        for i, racks in enumerate(self.resources):
            # r==> racks
            namer = 'r' + str(i)
            self.dic_nam_idx[namer] = self.node_index
            self.node_index += 1
            for l, _ in enumerate(self.resources):
                if i > l:
                    namer1 = 'r' + str(l)
                    self.edges_array.append((namer, namer1))
            for j, node in enumerate(racks):
                # h==>host
                nameh = namer + 'h' + str(j)
                self.dic_nam_idx[nameh] = self.node_index
                self.node_index += 1
                self.edges_array.append((namer, nameh))
                for k, gpu in enumerate(node):
                    #  g ==> gpu
                    nameg = nameh + 'g' + str(k)
                    self.dic_nam_idx[nameg] = self.node_index
                    self.node_index += 1
                    self.edges_array.append((nameh, nameg))
                    self.cluster_gpus.append(Gpu(gpu_id=self.dic_nam_idx[nameg],
                                            host_id=self.dic_nam_idx[nameh],
                                            rack_id=self.dic_nam_idx[namer],
                                            remaining_time=0,
                                            status=0
                                            ))

        # dic_nam_idx: {'r0': 0, 'r0h0': 1, 'r0h0g0': 2, 'r0h0g1': 3, 'r0h1g0': 4, 'r0h1g1': 5, ...}
        # edges_array= [('r0', 'r0h0'), ('r0h0', 'r0h0g0'), ('r0h0', 'r0h0g1'), ('r0h1', 'r0h1g0'),
        # ('r0', 'r1'), ('r1', 'r1h0'), ...]
        #  cluster= [Gpu0, Gpu1, Gpu2, Gpu3, ...]
        #  Gpu0={'gpu_id' = 2, 'node_id' = 1, 'rack_id' = 0, 'remaining_time'=0}

    def create_graph(self):
        self.G = nx.Graph()
        for nid in self.dic_nam_idx.values():
            self.G.add_node(nid)

        for edg in self.edges_array:
            self.G.add_edge(self.dic_nam_idx[edg[0]], self.dic_nam_idx[edg[1]])

    def create_adjacency_matrix(self):
        for gpua in range(len(self.cluster_gpus)):
            for gpub in range(len(self.cluster_gpus)):
                if self.cluster_gpus[gpua].rack_id == self.cluster_gpus[gpub].rack_id:
                    if self.cluster_gpus[gpua].host_id == self.cluster_gpus[gpub].host_id:
                        if self.cluster_gpus[gpua].gpu_id == self.cluster_gpus[gpub].gpu_id:
                            self.adjacency_matrix[gpua, gpub] = 0
                        else:
                            self.adjacency_matrix[gpua, gpub] = 1 / self.nvLink
                    else:
                        self.adjacency_matrix[gpua, gpub] = 1 / self.infiniBand
                else:
                    # print(f'{gpua.gpu_id} - {gpub.gpu_id}')
                    self.adjacency_matrix[gpua, gpub] = 1 / self.peer_to_peer
                    
                    
                    
    def get_gpu_distance_from_all(self,Gpu_ID):
      # Gpu_ID int   
        return self.adjacency_matrix[Gpu_ID]
        
    def get_gpu_distance_gpu(self,Gpu_ID1,Gpu_ID2):
      # Gpu_ID int
        list = self.adjacency_matrix[Gpu_ID1]
    
        return list[Gpu_ID2]
        
    def get_gpu_sort_nearest(self,Gpu_ID):
      # Gpu_ID int   
      # returns 
        return self.adjacency_matrix[Gpu_ID].argsort()# first one is the node itself
    
    def return_adj_matrx(self):
        return self.adjacency_matrix