import warnings
import heapq as heap

from collections import OrderedDict, defaultdict

import numpy as np
import torch


class Graph:
    def __init__(self):
        self.edges = OrderedDict()
        self.nodes = self._get_nodes()
        self.adj = None

    def _get_nodes(self):
        nodes = OrderedDict()
        for edgeID, edge in self.edges.items():
            node = edge.fromNode
            nodes[node.id] = node
            node = edge.toNode
            nodes[node.id] = node
        return nodes

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges

    def get_edge(self, edgeID):
        return self.edges[edgeID]

    def get_node_ids(self):
        nodes = self.get_nodes()
        return list(nodes.keys())

    def get_edge_ids(self):
        edges = self.get_edges()
        return list(edges.keys())

    def get_edges_fromNode(self, fromNode_id):
        """
        Get all edges with the requested fromNode id
        :param fromNode_id:
        :return: list of all edges that include the specified from node id
        """
        wanted = []
        edges = self.get_edges()
        for _, edge in edges.items():
            if edge.fromNode.id == fromNode_id:
                wanted.append(edge)
        return wanted

    def get_edges_toNode(self, toNode_id):
        """
        Get all edges with the requested toNode id
        :param toNode_id:
        :return: list of all edges that include the specified to node id
        """
        wanted = []
        edges = self.get_edges()
        for _, edge in edges.items():
            if edge.toNode.id == toNode_id:
                wanted.append(edge)
        return wanted

    def get_downstream_nodes(self, fromNode_id):
        """
        Get all downstream nodes with the requested fromNode id
        :param fromNode_id:
        :return: list of all edges that include the specified from node id
        """
        wanted = []
        travel_time = []
        edges = self.get_edges()
        for _, edge in edges.items():
            if edge.fromNode.id == fromNode_id:
                wanted.append(edge.toNode.id)
                travel_time.append(edge.freeflowTravelTime)
        return zip(wanted, travel_time)

    def get_upstream_nodes(self, toNode_id):
        """
        Get all upstream nodes with the requested toNode id
        :param toNode_id:
        :return: list of all edges that include the specified from node id
        """
        wanted = []
        travel_time = []
        edges = self.get_edges()
        for _, edge in edges.items():
            if edge.toNode.id == toNode_id:
                wanted.append(edge.fromNode.id)
                travel_time.append(edge.freeflowTravelTime)
        return zip(wanted, travel_time)

    def add_edge(self, edgeID, fromNode, toNode, **kwargs):
        add_fromNode = self.add_node(fromNode)
        add_toNode = self.add_node(toNode)
        edge = Edge(edgeID, add_fromNode, add_toNode, **kwargs)
        self.edges[edge.id] = edge

    def add_node(self, node):
        if node.id in self.nodes.keys():
            return self.nodes[node.id]
        else:
            self.nodes[node.id] = node
            return node

    def set_adj(self, adj):
        self.adj = adj

    def dijkstra(self, start_node, end_node, gpu):
        visited = set()
        parents_map = {}
        pq = []
        node_cost = defaultdict(lambda: np.inf)
        node_cost[start_node] = torch.tensor([0.0]).to(gpu)
        heap.heappush(pq, (0, start_node))

        while pq:
            # go greedily by always extending the shorter cost nodes first
            _, node = heap.heappop(pq)
            visited.add(node)

            for adjacent, travel_time in self.get_downstream_nodes(node):
                if adjacent in visited:
                    continue

                cost_new = node_cost[node] + travel_time
                if node_cost[adjacent] > cost_new:
                    parents_map[adjacent] = node
                    node_cost[adjacent] = cost_new
                    heap.heappush(pq, (cost_new, adjacent))
        return node_cost[end_node]

    def refresh(self, gpu):
        """
        Update nhead, nbeforehead, and pi0 of each node
        """
        nodes = self.get_nodes()
        for node_id, node in nodes.items():
            edges_fromNode = self.get_edges_fromNode(node_id)
            edges_toNode = self.get_edges_toNode(node_id)

            # update nhead and nbeforehead
            nhead_all = []
            nbeforehead_all = []
            for edge in edges_fromNode:
                nhead_all.append(edge.freeflowTravelTime)
            for edge in edges_toNode:
                nbeforehead_all.append(edge.freeflowTravelTime)

            nhead = min(nhead_all).clone() if len(nhead_all) > 0 else torch.tensor([0.0])
            nbeforehead = min(nbeforehead_all).clone() if len(nbeforehead_all) > 0 else torch.tensor([0.0])
            node.set_nhead(nhead)
            node.set_nbeforehead(nbeforehead)

            # update pi0 using dijkstra
            pi0 = self.dijkstra(node_id, self.get_node_ids()[-1], gpu)
            node.set_pi0(pi0)


class Edge:
    def __init__(self, edgeID, fromNode, toNode, **kwargs):
        self.id = edgeID
        self.fromNode = fromNode
        self.toNode = toNode
        given = ["freeflowTravelTime",
                 "queueUpstream",
                 "queueDownstream",
                 "flowCap",
                 "queueCap",
                 "shockwaveTravelTime",
                 "inflow",
                 "outflow",
                 "withheld"]

        if kwargs == {}:
            self.freeflowTravelTime = 0.0
            self.queueUpstream = 0.0
            self.queueDownstream = 0.0
            self.flowCap = 0.0
            self.shockwaveTravelTime = 0.0
            self.queueCap = 0.0
            self.inflow = 0.0
            self.outflow = 0.0
            self.withheld = 0.0
            self.priority = None
            warnings.warn("Input should contain {}".format(given))
        else:
            self.freeflowTravelTime = kwargs["freeflowTravelTime"]
            self.queueUpstream = kwargs["queueUpstream"]
            self.queueDownstream = kwargs["queueDownstream"]
            self.flowCap = kwargs["flowCap"]
            self.shockwaveTravelTime = kwargs["shockwaveTravelTime"]
            self.queueCap = kwargs["queueCap"]
            self.inflow = kwargs["inflow"]
            self.outflow = kwargs["outflow"]
            self.withheld = kwargs["withheld"]

    def get_fromNode(self):
        return self.fromNode

    def get_toNode(self):
        return self.toNode

    def set_fromNode(self, fromNode):
        self.fromNode = fromNode

    def set_toNode(self, toNode):
        self.toNode = toNode

    def set_freeflowTravelTime(self, freeflowTravelTime):
        self.freeflowTravelTime = freeflowTravelTime

    def set_shockwaveTravelTime(self, shockwaveTravelTime):
        self.shockwaveTravelTime = shockwaveTravelTime

    def set_priority(self, priority):
        self.priority = priority

    def update_queueUpstream(self, queueUpstream):
        self.queueUpstream = queueUpstream

    def update_queueDownstream(self, queueDownstream):
        self.queueDownstream = queueDownstream

    def update_flowCap(self, flowCap):
        self.flowCap = flowCap

    def update_queueCap(self, queueCap):
        self.queueCap = queueCap

    def update_inflow(self, inflow):
        self.inflow = inflow

    def update_outflow(self, outflow):
        self.outflow = outflow

    def update_withheld(self, withheld):
        self.withheld = withheld


class Node:
    def __init__(self, nodeID, x=0.0, y=0.0,
                 demand=None, nhead=None, nbeforehead=None, pi0=None):
        self.id = nodeID
        self.x = x
        self.y = y
        self.demand = demand  # demand to the destination
        self.nhead = nhead
        self.nbeforehead = nbeforehead
        self.pi0 = pi0

    def set_coordinates(self, x, y):
        self.x = x
        self.y = y

    def set_demand(self, demand):
        self.demand = demand

    def set_nhead(self, nhead):
        """
        min free flow travel time to the downstream
        """
        self.nhead = nhead

    def set_nbeforehead(self, nbeforehead):
        """
        min free flow travel time from the upstream
        """
        self.nbeforehead = nbeforehead

    def set_pi0(self, pi0):
        """
        min free flow travel time to the destination
        """
        self.pi0 = pi0


def make(game):
    if game == "six-link":
        """
          1---3---5
        7-1-2-3-4-5-6
        """
        graph = Graph()
        node_1 = Node(1)
        node_2 = Node(2)
        node_3 = Node(3)
        node_4 = Node(4)
        node_5 = Node(5)
        node_6 = Node(6)
        node_7 = Node(7)
        graph.add_edge(0, node_1, node_2)
        graph.add_edge(1, node_1, node_3)
        graph.add_edge(2, node_2, node_3)
        graph.add_edge(3, node_3, node_4)
        graph.add_edge(4, node_3, node_5)
        graph.add_edge(5, node_4, node_5)
        graph.add_edge(6, node_7, node_1)
        graph.add_edge(7, node_5, node_6)
    if game == "Chain":
        """
        9--1-2-3-4-5-6-7--8
        """
        graph = Graph()
        node_1 = Node(1)
        node_2 = Node(2)
        node_3 = Node(3)
        node_4 = Node(4)
        node_5 = Node(5)
        node_6 = Node(6)
        node_7 = Node(7)
        node_8 = Node(8)
        node_9 = Node(9)
        graph.add_edge(0, node_1, node_2)
        graph.add_edge(1, node_2, node_3)
        graph.add_edge(2, node_3, node_4)
        graph.add_edge(3, node_4, node_5)
        graph.add_edge(4, node_5, node_6)
        graph.add_edge(5, node_6, node_7)
        graph.add_edge(6, node_9, node_1)
        graph.add_edge(7, node_7, node_8)
    return graph

