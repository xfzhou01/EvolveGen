# given a graph, check it with mutiple rules
# 1. whether there is a useless loop node
# 
from graph_manager import GraphManager
import networkx as nx
from node import LoopNode, BranchNode
class GraphRuler:

    def __init__(self):
        self.graph_instance:nx.MultiDiGraph = None
        self.graph_instance_copy_1:nx.MultiDiGraph = None
        self.graph_instance_copy_2:nx.MultiDiGraph = None

    def check(self, gm_instance:GraphManager):
        self.graph_instance = gm_instance.program_graph

    def _check_loop_repeat(self):
        for n in self.graph_instance.nodes():
            if isinstance(n, LoopNode):
                pass
    