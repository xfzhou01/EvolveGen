from matplotlib import pyplot as plt
import networkx as nx
from node import OpNode, OperationType, ResultDataType, BranchNode, LoopNode, Node, ArrayNode
from node import DepNode

from queue import Queue

from node import QuantizationMode, OverflowMode
from node import BRAM_TYPE
import shutil
import subprocess
from typing import Union, List

class GraphManager:

    def __init__(self):
        self.program_graph = nx.DiGraph()
        self.program_graph_function_pipeline = False


        self.op_counter = 0
        self.loop_node_counter = 0
        self.branch_node_counter = 0
        self.array_node_counter = 0
        self.visit_node_counter = 0
        self.write_node_counter = 0
        self.dep_node_counter = 0

        self.function_name = "top"

        # the program graphs for dumping various verilog
        self.program_graph_copy_1 = nx.DiGraph()
        self.program_graph_copy_2 = nx.DiGraph()
        self.cp_1 = 10
        self.cp_2 = 10
        
        self.function_pipeline_1 = False
        self.function_pipeline_2 = False

    def get_clock_period_1(self):
        return self.cp_1
    
    def get_clock_period_2(self):
        return self.cp_2
    
    def _get_code_block_node_list_under_code_block_node(self, code_block_node):
        code_block_node_list = []
        q = Queue()
        if not (isinstance(code_block_node, BranchNode) or \
                isinstance(code_block_node, LoopNode)):
            raise TypeError()
        q.put(code_block_node)
        # do bfs
        while not q.empty():
            bfs_n = q.get_nowait()
            cb_node_succs = self.program_graph.successors(bfs_n)
            for cbns in cb_node_succs:
                if isinstance(cbns, BranchNode) or isinstance(cbns, LoopNode):
                    q.put(cbns)
                    code_block_node_list.append(cbns)
        # 
        return code_block_node_list
    
    def _get_lv_set_under_code_block_node(self, code_block_node):
        lv_set = set()
        q = Queue()
        if not (isinstance(code_block_node, BranchNode) or \
                isinstance(code_block_node, LoopNode)):
            raise TypeError()
        q.put(code_block_node)
        # do bfs
        while not q.empty():
            bfs_n = q.get_nowait()
            cb_node_succs = self.program_graph.successors(bfs_n)
            for cbns in cb_node_succs:
                if isinstance(cbns, BranchNode) or isinstance(cbns, LoopNode):
                    q.put(cbns)
                elif isinstance(cbns, OpNode):
                    lv_set.add(cbns)
        return lv_set
    
    def _get_rv_set_under_code_block_node(self, code_block_node):
        rv_set = set()
        q = Queue()
        if not (isinstance(code_block_node, BranchNode) or \
                isinstance(code_block_node, LoopNode)):
            raise TypeError()
        q.put(code_block_node)
        # do bfs
        while not q.empty():
            bfs_n = q.get_nowait()
            cb_node_succs = self.program_graph.successors(bfs_n)
            for cbns in cb_node_succs:
                if isinstance(cbns, BranchNode) or isinstance(cbns, LoopNode):
                    q.put(cbns)
                elif isinstance(cbns, OpNode):
                    pl = list(self.program_graph.predecessors(cbns))
                    for p in pl:
                        if isinstance(p, OpNode):
                            rv_set.add(p)
        return rv_set

    def _is_loop_empty_loop(self, loop_node):
        # if not there is a op node under loop node, we judge it as an empty loop
        q = Queue()
        if not isinstance(loop_node, LoopNode):
            raise TypeError()
        q.put(loop_node)
        # do bfs
        while not q.empty():
            bfs_n = q.get_nowait()
            cb_node_succs = self.program_graph.successors(bfs_n)
            for cbns in cb_node_succs:
                if isinstance(cbns, BranchNode) or isinstance(cbns, LoopNode):
                    q.put(cbns)
                elif isinstance(cbns, OpNode):
                    return False
        return True

    def _get_op_node_list(self):
        """Traverse the graph and return all OpNode instances as a list, excluding WRITE operation types."""
        op_nodes = []
        for node in self.program_graph.nodes():
            if isinstance(node, OpNode) and node.op_type != OperationType.WRITE:
                op_nodes.append(node)
        return op_nodes

    def _get_array_node_list(self):
        """Traverse the graph and return all ArrayNode instances as a list."""
        array_nodes = []
        for node in self.program_graph.nodes():
            if isinstance(node, ArrayNode):
                array_nodes.append(node)
        return array_nodes

    def _get_visit_node_list(self):
        """Traverse the graph and return all OpNode instances with VISIT operation type as a list."""
        visit_nodes = []
        for node in self.program_graph.nodes():
            if isinstance(node, OpNode) and node.op_type == OperationType.VISIT:
                visit_nodes.append(node)
        return visit_nodes

    def _get_write_node_list(self):
        """Traverse the graph and return all OpNode instances with WRITE operation type as a list."""
        write_nodes = []
        for node in self.program_graph.nodes():
            if isinstance(node, OpNode) and node.op_type == OperationType.WRITE:
                write_nodes.append(node)
        return write_nodes

    def _get_loop_node_list(self):
        """Traverse the graph and return all LoopNode instances as a list."""
        loop_nodes = []
        for node in self.program_graph.nodes():
            if isinstance(node, LoopNode):
                loop_nodes.append(node)
        return loop_nodes

    def _get_branch_node_list(self):
        """Traverse the graph and return all BranchNode instances as a list."""
        branch_nodes = []
        for node in self.program_graph.nodes():
            if isinstance(node, BranchNode):
                branch_nodes.append(node)
        return branch_nodes
    
    def _get_dep_node_list(self):
        dep_nodes = []
        for node in self.program_graph.nodes():
            if isinstance(node, dep_nodes):
                dep_nodes.append(node)
        return dep_nodes

    def get_function_name(self):
        return self.function_name
    
    def set_function_name(self, new_fucntion_name:str):
        if new_fucntion_name is None:
            raise ValueError("the new function name is None")
        if not isinstance(new_fucntion_name,str):
            raise TypeError(f"expected type is str but got {type(new_fucntion_name)}") 
        if new_fucntion_name == "":
            raise ValueError("new function name should not be empty string")
        self.function_name = new_fucntion_name

    def add_array_node(self, array_node_created = None):
        if array_node_created is None:
            raise NotImplementedError()
        else:
            array_node_instance = array_node_created
            array_node_instance.name = f"array_{self.array_node_counter}"
        array_node_instance:ArrayNode
        self.program_graph.add_node(array_node_instance)
        self.array_node_counter += 1

    def add_dep_node(self, dep_node_created = None,
                     predecessor_node:OpNode = None,
                     successor_node:OpNode = None,
                     loop_node:LoopNode = None,
                     branch_node:BranchNode = None):
        # dep node must belongs to a loop node / or can no directly belongs to 
        # case: outside loop
        #   v1 = v0 + 1
        #   v2 = v1 + v0
        #   v1 = v2 (dep)
        # * in this case the dep is useless can will reduce the ability of expression
        if dep_node_created is not Node:
            print("use existing created dep node")
            dep_node_instance = dep_node_created
            dep_node_instance:DepNode
            dep_node_instance.name = f"dep_{self.dep_node_counter}"
        else:
            if not isinstance(predecessor_node, OpNode):
                raise ValueError("dep node's predecessor node should be op node")
            if not isinstance(successor_node, OpNode):
                raise ValueError("dep node's successor node should be op node")
            dep_node_instance = DepNode(
                name=f"dep_{self.dep_node_counter}",
                predecessor=predecessor_node,
                successor=successor_node
            )
        self.program_graph.add_node(dep_node_instance)

        if loop_node is not None and branch_node is not None:
            raise ValueError("loop node and branch node should not be `NOT none` at the same time")
        if loop_node is not None:
            self.program_graph.add_edge(loop_node, dep_node_instance)
        if branch_node is not None:
            self.program_graph.add_edge(branch_node, dep_node_instance, direction = br_node_branch)

        self.dep_node_counter += 1
        return dep_node_instance



    def add_op_node(self, op_node_created = None,
                    op_type: Union[str, OperationType] = "add", 
                    predecessor_list: List[OpNode] = [],
                    result_type = ResultDataType.AP_INT,
                    result_width = 32,
                    result_int_width_ap_fixed = 16,
                    result_wrap_mode = "AP_RND",
                    result_rounding_mode = "AP_WRAP",
                    loop_node:LoopNode = None,
                    br_node:BranchNode = None,
                    br_node_branch:bool = True):

        # 
        if op_node_created is not None:
            print("use existing created op node")
            op_node_instance = op_node_created
            op_node_instance:OpNode
            op_node_instance.name = f"op_{self.op_counter}"
        else:
            if isinstance(op_type, str):
                op_type_enum = OperationType[op_type.upper()]
            else:
                op_type_enum = op_type
            op_node_instance = OpNode(
                name=f"op_{self.op_counter}",
                op_type=op_type_enum,
                result_type=result_type,
                result_width=result_width,
                result_int_width_ap_fixed=result_int_width_ap_fixed,
                result_wrap_mode=result_wrap_mode,
                result_rounding_mode=result_rounding_mode
            )
        
        self.program_graph.add_node(op_node_instance)
        for pred in predecessor_list:
            self.program_graph.add_edge(pred, op_node_instance)
        if loop_node is not None and br_node is not None:
            raise ValueError("loop node and branch node should not be `NOT none` at the same time")
        if loop_node is not None:
            self.program_graph.add_edge(loop_node, op_node_instance)
        if br_node is not None:
            self.program_graph.add_edge(br_node, op_node_instance, direction = br_node_branch)

        self.op_counter += 1
        return op_node_instance
    
    def add_loop_node(self, 
                      start_index = 0, 
                      end_index = 1023, 
                      step = 1,
                      loop_node_predecessor:LoopNode = None,
                      br_node_predecessor:BranchNode = None,
                      br_node_branch:bool = True,
                      loop_node_created:LoopNode = None):
        """
        add LoopNode to program_graph。
        start_index, end_index: int or op node
        step: int
        """
        if loop_node_created is not None:
            if not isinstance(loop_node_created, LoopNode):
                raise TypeError()
            loop_node_instance = loop_node_created
            loop_node_instance:LoopNode
            loop_node_instance.name = f"loop_{self.loop_node_counter}"
        else:
            loop_node_instance = LoopNode(
                name=f"loop_{self.loop_node_counter}",
                start_index=start_index,
                end_index=end_index,
                step=step
            )
        self.program_graph.add_node(loop_node_instance)

        # the code block it belongs to
        if loop_node_predecessor is not None and br_node_predecessor is not None:
            raise ValueError("loop node and branch node should not be `NOT none` at the same time")
        if loop_node_predecessor is not None:
            self.program_graph.add_edge(loop_node_predecessor, loop_node_instance)
        if br_node_predecessor is not None:
            self.program_graph.add_edge(br_node_predecessor, loop_node_instance, direction = br_node_branch)
        self.loop_node_counter += 1
        return loop_node_instance
    
    def _loop_node_head_to_str(self, node: LoopNode):
        if isinstance(node.start_index, int) and isinstance(node.end_index, int):
            return f"for (int {node.get_loop_var_name()} = {node.start_index};" +\
                f"{node.get_loop_var_name()} <= {node.end_index};" +\
                f"{node.get_loop_var_name()} += {node.step}) {{"
        elif isinstance(node.start_index, OpNode) and isinstance(node.end_index, int):
            return f"for (int {node.get_loop_var_name()} = {node.start_index.name};" +\
                f"{node.get_loop_var_name()} <= {node.end_index};" +\
                f"{node.get_loop_var_name()} += {node.step}) {{"
        elif isinstance(node.start_index, int) and isinstance(node.end_index, OpNode):
            return f"for (int {node.get_loop_var_name()} = {node.start_index};" +\
                f"{node.get_loop_var_name()} <= {node.end_index.name};" +\
                f"{node.get_loop_var_name()} += {node.step}){{"
        elif isinstance(node.start_index, OpNode) and isinstance(node.end_index, OpNode):
            return f"for (int {node.get_loop_var_name()} = {node.start_index.name};" +\
                f"{node.get_loop_var_name()} <= {node.end_index.name};" +\
                f"{node.get_loop_var_name()} += {node.step}){{"
        else:
            raise NotImplementedError(f"Unsupported combination of start_index and end_index types in LoopNode." +\
                                      f"start index = {node.start_index}, end index = {node.end_index}")
    def _loop_node_pragma_to_str(self, node:LoopNode):
        node.check_pragma_status() 
        loop_node_pragma_str = ""
        if node.is_pipelined:
            loop_node_pragma_str += "\n#pragma HLS pipeline\n"
        else:
            loop_node_pragma_str += "\n#pragma HLS pipeline off\n"

        if node.is_flattened:
            loop_node_pragma_str += "\n#pragma HLS loop_flatten\n"
        
        if node.is_unrolled:
            if node.is_fully_unrolled:
                loop_node_pragma_str += "\n#pragma HLS unroll\n"
            else:
                loop_node_pragma_str += f"\n#pragma HLS unroll factor={node.unroll_factor}\n"
        else:
            pass
        return loop_node_pragma_str
    
    def _array_node_pragma_to_str(self, node:ArrayNode):
        # This setting only applies to verison vitis 2020.2
        # pragma HLS interface ap_memory storage_type=RAM_1P port=array_1
        array_node_pragma_str = ""
        if not isinstance(node, ArrayNode):
            raise TypeError()
        if not isinstance(node.memory_type, BRAM_TYPE):
            raise TypeError()
        array_node_pragma_str += "\n#pragma HLS interface "+\
            f"ap_memory storage_type={node.memory_type.value} "+\
            f"port={node.name}\n"
        return array_node_pragma_str

    def add_branch_node(self, conditional_op:OpNode, 
                branch_node_created = None,
                loop_node_predecessor = None,
                br_node_predecessor = None,
                br_node_branch = True):
        if (not isinstance(conditional_op, OpNode)) and branch_node_created is None:
            raise TypeError()
        if branch_node_created is not None:
            br_node_instance = branch_node_created
            br_node_instance:BranchNode
            br_node_instance.name = f"br_{self.branch_node_counter}"
        else:
            br_node_instance = \
                BranchNode(name=f"br_{self.branch_node_counter}")

        self.branch_node_counter += 1

        self.program_graph.add_node(br_node_instance)
        self.program_graph.add_edge(conditional_op, br_node_instance)

        # the code block it belongs to
        if loop_node_predecessor is not None and br_node_predecessor is not None:
            raise ValueError("loop node and branch node should not be `NOT none` at the same time")
        if loop_node_predecessor is not None:
            self.program_graph.add_edge(loop_node_predecessor, br_node_instance)
        if br_node_predecessor is not None:
            self.program_graph.add_edge(br_node_predecessor, br_node_instance, direction = br_node_branch)
        
    def _loop_node_tail_to_str(self, node:LoopNode):
        return f"}}"
    
    def _loop_node_to_str(self, node:LoopNode):
        head_str = self._loop_node_head_to_str(node=node)
        pragma_str = self._loop_node_pragma_to_str(node=node)
        instr_str = ""

        GraphManager._graph_to_function_body_visitor_for_sanity_check.add(node)
        # Sort successors: DepNode instances at the end
        successors = list(self.program_graph.successors(node))
        successors_sorted = sorted(successors, key=lambda n: isinstance(n, DepNode))
        for n in successors_sorted:
            if isinstance(n, LoopNode):
                instr_str += self._loop_node_to_str(n)
            elif isinstance(n, BranchNode):
                instr_str += self._br_node_to_str(n)
            elif isinstance(n, OpNode):
                instr_str += self._op_node_to_assignment_str(n)
            elif isinstance(n, DepNode):
                instr_str += self._dep_node_to_str(n)
        
        tail_str = self._loop_node_tail_to_str(node=node)
        return head_str + pragma_str + instr_str + tail_str
    
    def _general_node_to_str(self, node:Node):

        GraphManager._graph_to_function_body_visitor_for_sanity_check.add(node)
        instr_str = ""
        if node is None:
            raise ValueError("node is None")
        if isinstance(node, LoopNode):
            instr_str += self._loop_node_to_str(node)
        elif isinstance(node, BranchNode):
            instr_str += self._br_node_to_str(node)
        elif isinstance(node, OpNode):
            instr_str += self._op_node_to_assignment_str(node)
        else:
            raise TypeError(f"unsupported node type {type(node)}")
        return instr_str
        
    def add_array_visit(self, array_node:ArrayNode,
                        address_node:Union[OpNode, int, LoopNode]):
        # Handle the case where address_node is a list (select first element)
        # if isinstance(address_node, list):
        #     if len(address_node) == 0:
        #         raise ValueError("address_node list is empty")
        #     address_node = address_node[0]
        #     print(f"[WARNING] address_node was a list, selecting first element: {address_node}")
        
        op_node_instance = OpNode(
            name=f"visit_{self.visit_node_counter}",
            op_type=OperationType.VISIT,
            result_type=array_node.result_type,
            result_width=array_node.result_width,
            result_int_width_ap_fixed=array_node.result_int_width_ap_fixed,
            result_wrap_mode=array_node.result_wrap_mode,
            result_rounding_mode=array_node.result_rounding_mode
        )
        self.visit_node_counter += 1
        self.program_graph.add_node(op_node_instance)
        self.program_graph.add_edge(array_node, op_node_instance, description="array")
        if isinstance(address_node, OpNode):
            self.program_graph.add_edge(address_node, op_node_instance, description="address")
        elif isinstance(address_node, int):
            self.program_graph.add_node(address_node)
            self.program_graph.add_edge(address_node, op_node_instance, description="address")
        elif isinstance(address_node, LoopNode):
            self.program_graph.add_edge(address_node, op_node_instance, description="address")
        else:
            raise TypeError(f"unsupported address_node type {type(address_node)}, address node {address_node}")
        
    def add_array_write(self, array_node:ArrayNode,
                        write_value_node:OpNode,
                        address_node:Union[OpNode, int, LoopNode]):
        # Handle the case where address_node is a list (select first element)
        # if isinstance(address_node, list):
        #     if len(address_node) == 0:
        #         raise ValueError("address_node list is empty")
        #     address_node = address_node[0]
        #     print(f"[WARNING] address_node was a list, selecting first element: {address_node}")
        
        # type check
        if not isinstance(array_node, ArrayNode):
            raise TypeError(f"array_node should be ArrayNode type but got {type(array_node)}")
        if not isinstance(write_value_node, OpNode):
            raise TypeError(f"write_value_node should be OpNode type but got {type(write_value_node)}")
        if not isinstance(address_node, (OpNode, int, LoopNode)):
            raise TypeError(f"address_node should be OpNode or int or LoopNode type but got {type(address_node)}")
        # create write node
        op_node_instance = OpNode(
            name=f"write_{self.write_node_counter}",
            op_type=OperationType.WRITE,
            result_type=array_node.result_type,
            result_width=array_node.result_width,
            result_int_width_ap_fixed=array_node.result_int_width_ap_fixed,
            result_wrap_mode=array_node.result_wrap_mode,
            result_rounding_mode=array_node.result_rounding_mode
        )
        self.write_node_counter += 1
        self.program_graph.add_node(op_node_instance)
        self.program_graph.add_edge(write_value_node, op_node_instance, description="write_value", key="write_value_edge")
        # add edge to write node
        if isinstance(address_node, OpNode):
            if address_node == write_value_node:
                # If address_node and write_value_node are the same, we need to add two edges with different keys
                # The write_value edge was already added above, now add the address edge with a different key
                print(f"[WARNING] address_node and write_value_node are the same for write node {op_node_instance.name}")
                # Force adding a second edge by providing a different key
                self.program_graph.add_edge(address_node, op_node_instance, description="address", key="address_edge")
            else:
                self.program_graph.add_edge(address_node, op_node_instance, description="address")
        elif isinstance(address_node, int):
            self.program_graph.add_node(address_node)
            self.program_graph.add_edge(address_node, op_node_instance, description="address")
        elif isinstance(address_node, LoopNode):
            self.program_graph.add_edge(address_node, op_node_instance, description="address")
        else:
            raise TypeError(f"unsupported address_node type {type(address_node)}")
        # add edge to array node
        self.program_graph.add_edge(op_node_instance, array_node, description="array")
    

    def _extract_opnode_from_a_list(self, list_of_nodes:list[Node]):
        return [_ for _ in list_of_nodes if isinstance(_, OpNode)]
    
    def _dep_node_to_str(self, node:DepNode):
        right_value = node.predecessor.name
        left_value = node.successor.name
        return f"{right_value} = {left_value};"

    def _br_node_to_str(self, node:BranchNode):
        GraphManager._graph_to_function_body_visitor_for_sanity_check.add(node)
        condition_node = list(self.program_graph.predecessors(node))
        condition_node = self._extract_opnode_from_a_list(condition_node)
        if not len(condition_node) == 1:
            raise ValueError(f"the branch node should have 1 condition but got {condition_node}")
        condition_node = condition_node[0]
        if not isinstance(condition_node, OpNode):
            raise TypeError(f"the condition node should be opNode type but got {type(condition_node)}")
        
        n_true_list = []
        n_false_list = []
        
        for succ in self.program_graph.successors(node):
            edge_data_dict = self.program_graph.get_edge_data(node, succ)
            if edge_data_dict:
                # Handle both cases: single edge (returns dict) or multiple edges (returns dict of dicts)
                if isinstance(edge_data_dict, dict):
                    # Check if this is a single edge (has 'direction' key) or multiple edges (dict of dicts)
                    if 'direction' in edge_data_dict:
                        # Single edge case - edge_data_dict is the actual edge data
                        direction = edge_data_dict.get('direction', False)
                        if direction is True:
                            n_true_list.append(succ)
                        elif direction is False:
                            n_false_list.append(succ)
                    else:
                        # Multiple edges case - edge_data_dict is a dict of edge keys to edge data
                        for edge_key, edge_data in edge_data_dict.items():
                            if isinstance(edge_data, dict):
                                direction = edge_data.get('direction', False)
                                if direction is True:
                                    n_true_list.append(succ)
                                elif direction is False:
                                    n_false_list.append(succ)
        if not len(set(n_true_list)) + len(set(n_false_list)) == len(list(self.program_graph.successors(node))):
            raise ValueError()
        
        # Sort n_true_list in topological order
        n_true_list_sorted = list(nx.topological_sort(self.program_graph.subgraph(n_true_list)))

        # Sort n_false_list in topological order
        n_false_list_sorted = list(nx.topological_sort(self.program_graph.subgraph(n_false_list)))


        br_node_str = f"if ({condition_node.name}) {{"
        br_node_true_instr_str = ""
        for true_node in n_true_list_sorted:
            br_node_true_instr_str += self._general_node_to_str(node=true_node)
        br_node_false_instr_str = ""
        for false_node in n_false_list_sorted:
            br_node_false_instr_str += self._general_node_to_str(node=false_node)
        br_node_str += br_node_true_instr_str
        br_node_str += f"}} else {{"
        br_node_str += br_node_false_instr_str
        br_node_str += f"}}"
        return br_node_str
        


    def _op_node_to_decl_str(self, node: OpNode):
        # Sanity checks
        if not isinstance(node.result_type, ResultDataType):
            raise ValueError(f"Invalid result_type: {node.result_type}")
        if not isinstance(node.result_width, int) or node.result_width <= 0:
            raise ValueError(f"Invalid result_width: {node.result_width}")
        if node.result_type == ResultDataType.AP_FIXED:
            if not isinstance(node.result_int_width_ap_fixed, int) or node.result_int_width_ap_fixed <= 0:
                raise ValueError(f"Invalid result_int_width_ap_fixed: {node.result_int_width_ap_fixed}")
            if not isinstance(node.result_rounding_mode, QuantizationMode):
                raise TypeError(f"Invalid node.result_rounding_mode = {node.result_rounding_mode}, "+\
                                f"expected to have type QuantizationMode, "+\
                                f"but got type {type(node.result_rounding_mode)}")
            if not isinstance(node.result_wrap_mode, OverflowMode):
                raise TypeError(f"Invalid node.result_wrap_mode = {node.result_wrap_mode}, "+\
                                f"expected to have type OverflowMode")
            c_type = f"ap_fixed<{node.result_width},{node.result_int_width_ap_fixed},"+\
                f"{node.result_rounding_mode.value},{node.result_wrap_mode.value}>"
        elif node.result_type == ResultDataType.AP_INT:
            c_type = f"ap_int<{node.result_width}>"
        elif node.result_type == ResultDataType.AP_UINT:
            c_type = f"ap_uint<{node.result_width}>"
        else:
            raise ValueError(f"Unsupported result_type: {node.result_type}")
        # 
        return f"{c_type} {node.name};"
    
    def _op_node_to_assignment_str(self, node:OpNode):
        # 
        GraphManager._graph_to_function_body_visitor_for_sanity_check.add(node)

        if len(list(self.program_graph.predecessors(node))) == 0:
            print(f"[WARNING]the OpNode node with name {node.name}, is a argument, will not generate assignment str")
            return ""

        op_map = {
            OperationType.ADD: '+',
            OperationType.SUB: '-',
            OperationType.MUL: '*',
            OperationType.DIV: '/',
            OperationType.MOD: '%',
            OperationType.AND: '&',
            OperationType.OR: '|',
            OperationType.XOR: '^',
            OperationType.SHL: '<<',
            OperationType.SHR: '>>',
            OperationType.EQ: '==',
            OperationType.NEQ: '!=',
            OperationType.LT: '<',
            OperationType.GT: '>',
            OperationType.LE: '<=',
            OperationType.GE: '>=',
        }

        if node.op_type == OperationType.VISIT:
            visit_predecessor = list(self.program_graph.predecessors(node))
            array_node_instance = None
            address_node_instance = None
            for v in visit_predecessor:
                # For MultiDiGraph, get_edge_data returns different structures
                edge_data_dict = self.program_graph.get_edge_data(v, node)
                if edge_data_dict:
                    # Handle both cases: single edge or multiple edges
                    if isinstance(edge_data_dict, dict):
                        # Check if this is a single edge (has 'description' key) or multiple edges
                        if 'description' in edge_data_dict:
                            # Single edge case
                            desc = edge_data_dict.get("description", "")
                            if desc == "array":
                                array_node_instance = v
                            elif desc == "address":
                                address_node_instance = v
                        else:
                            # Multiple edges case
                            for edge_key, edge_data in edge_data_dict.items():
                                if isinstance(edge_data, dict):
                                    desc = edge_data.get("description", "")
                                    if desc == "array":
                                        array_node_instance = v
                                    elif desc == "address":
                                        address_node_instance = v
            if array_node_instance is None:
                raise ValueError("Visit operation must have an array node as predecessor.")
            if address_node_instance is None:
                raise ValueError("Visit operation must have an address node as predecessor.")
            # Generate the visit operation string
            if isinstance(address_node_instance, OpNode):
                return f"{node.name} = {array_node_instance.name}[{address_node_instance.name}];"
            elif isinstance(address_node_instance, int):
                return f"{node.name} = {array_node_instance.name}[{address_node_instance}];"
            elif isinstance(address_node_instance, LoopNode):
                return f"{node.name} = {array_node_instance.name}[{address_node_instance.get_loop_var_name()}];"
            else:
                raise TypeError(f"Unsupported address_node type {type(address_node_instance)} for visit operation.")
        if node.op_type == OperationType.WRITE:
            write_predecessor = list(self.program_graph.predecessors(node))
            write_successor = list(self.program_graph.successors(node))
            if len(write_successor) != 1:
                raise ValueError(f"Write operation {node.name} expects 1 successor, got {len(write_successor)}")
            array_node_instance = None
            address_node_instance = None
            write_value_node_instance = None
            
            for w in write_predecessor:
                # For MultiDiGraph, get_edge_data returns different structures
                edge_data_dict = self.program_graph.get_edge_data(w, node)
                
                if edge_data_dict:
                    # Handle both cases: single edge or multiple edges
                    if isinstance(edge_data_dict, dict):
                        # Check if this is a single edge (has 'description' key) or multiple edges
                        if 'description' in edge_data_dict:
                            # Single edge case
                            desc = edge_data_dict.get("description", "")
                            if desc == "write_value":
                                write_value_node_instance = w
                            elif desc == "address":
                                address_node_instance = w
                        else:
                            # Multiple edges case - check if values are dicts
                            for edge_key, edge_data in edge_data_dict.items():
                                if isinstance(edge_data, dict):
                                    desc = edge_data.get("description", "")
                                    if desc == "write_value":
                                        write_value_node_instance = w
                                    elif desc == "address":
                                        address_node_instance = w
            
            # Special case: if we only have one predecessor but no write_value found,
            # it might be the case where the same node serves both roles
            if write_value_node_instance is None and address_node_instance is not None and len(write_predecessor) == 1:
                w = write_predecessor[0]
                if w == address_node_instance:
                    # The same node serves both roles
                    write_value_node_instance = w
                    
            for w in write_successor:
                # For MultiDiGraph, get_edge_data returns different structures
                edge_data_dict = self.program_graph.get_edge_data(node, w)
                if edge_data_dict:
                    # Handle both cases: single edge or multiple edges
                    if isinstance(edge_data_dict, dict):
                        # Check if this is a single edge (has 'description' key) or multiple edges
                        if 'description' in edge_data_dict:
                            # Single edge case
                            desc = edge_data_dict.get("description", "")
                            if desc == "array":
                                array_node_instance = w
                        else:
                            # Multiple edges case
                            for edge_key, edge_data in edge_data_dict.items():
                                if isinstance(edge_data, dict):
                                    desc = edge_data.get("description", "")
                                    if desc == "array":
                                        array_node_instance = w
            if array_node_instance is None:
                raise ValueError("Write operation must have an array node as successor.")
            if address_node_instance is None:
                raise ValueError("Write operation must have an address node as predecessor.")
            if write_value_node_instance is None:
                raise ValueError("Write operation must have a write value node as predecessor. "+\
                                 f"current write node = {node}, write predecessor = {write_predecessor}, "+\
                                 f"write predecessor length = {len(write_predecessor)} "+\
                                 f"write successor length = {len(write_successor)} "
                                 f"write successor = {write_successor}")
            # Generate the write operation string
            if isinstance(address_node_instance, OpNode):
                return f"{array_node_instance.name}[{address_node_instance.name}] = {write_value_node_instance.name};"
            elif isinstance(address_node_instance, int):
                return f"{array_node_instance.name}[{address_node_instance}] = {write_value_node_instance.name};"
            elif isinstance(address_node_instance, LoopNode):
                return f"{array_node_instance.name}[{address_node_instance.get_loop_var_name()}] = {write_value_node_instance.name};"

        preds = list(self.program_graph.predecessors(node))
        preds = [p for p in preds if not (isinstance(p, LoopNode) or isinstance(p, BranchNode))]
        preds:list[OpNode]
        if node.op_type in op_map:
            if len(preds) != 2:
                print(f"[WARNING] op node {node.name} with type {node.op_type} has the same predecessor")
                if (node.op_type == OperationType.SHL or node.op_type == OperationType.SHR) and \
                    (preds[0].result_type == ResultDataType.AP_FIXED):
                    return f"{node.name} = {preds[0].name} {op_map[node.op_type]} (int){preds[0].name};"
                return f"{node.name} = {preds[0].name} {op_map[node.op_type]} {preds[0].name};"
            
            if (node.op_type == OperationType.SHL or node.op_type == OperationType.SHR) and \
                    (preds[1].result_type == ResultDataType.AP_FIXED):
                return f"{node.name} = {preds[0].name} {op_map[node.op_type]} (int){preds[1].name};"
            return f"{node.name} = {preds[0].name} {op_map[node.op_type]} {preds[1].name};"
        elif node.op_type == OperationType.NOT:
            if len(preds) != 1:
                raise ValueError(f"Operation NOT expects 1 operand, got {len(preds)}")
            return f"{node.name} = ~{preds[0].name};"
        elif node.op_type == OperationType.VISIT:
            raise NotImplementedError(f"Operation {node.op_type} not supported.")
        else:
            raise NotImplementedError(f"Operation {node.op_type} not supported.")
        
    def _op_node_to_ref_str(self, node:OpNode):
        return node.name
    
    def _remove_all_standalone_nodes_in_graph(self):
        for n in self.program_graph.nodes():
            if len(list(self.program_graph.predecessors(n))) == 0 and \
            len(list(self.program_graph.successors(n))) == 0:
                print(f"[WARNING]: standalone node in graph, {n}, removing ...")

    def _check_multiple_code_blocks_in_graph(self):
        # the node should belongs to at most one code block
        for n in self.program_graph.nodes():
            n_pred =  list(self.program_graph.predecessors(n))
            n_pred_code_block = [_ for _ in n_pred if isinstance(_, LoopNode) or isinstance(_, BranchNode)]
            if len(n_pred_code_block) > 1:
                raise ValueError(f"node {n} got multiple code blocks")
            
    def _check_access_array_node_in_graph(self):
        # read and write array must through the visit node
        for n in self.program_graph.nodes():
            if isinstance(n, ArrayNode):
                n_pred = list(self.program_graph.predecessors(n))
                for np in n_pred:
                    if not isinstance(np, OpNode):
                        raise TypeError(f"the array should write with OpNode `write`, but got {np}")
                    else:
                        np:OpNode
                        if not np.op_type == OperationType.WRITE:
                            raise ValueError(f"the array should write with OpNode `write`, but got {np}")
                n_succ = list(self.program_graph.successors(n))
                for ns in n_succ:
                    if not isinstance(ns, OpNode):
                        raise TypeError(f"the array should read with OpNode `visit`, but got {ns}")
                    else:
                        ns:OpNode
                        if not ns.op_type == OperationType.VISIT:
                            raise ValueError(f"the array should read with OpNode `visit`, but got {ns}")

    def sanity_check_graph(self):
        self._remove_all_standalone_nodes_in_graph()
        self._check_multiple_code_blocks_in_graph()
        self._check_access_array_node_in_graph()


    def _select_function_arg_list(self):
        function_arg_nodes_input = []
        function_arg_nodes_output = []
        function_arg_nodes_array = []
        for n in self.program_graph.nodes():
            if isinstance(n, ArrayNode):
                function_arg_nodes_array.append(n)
                continue
            if len(list(self.program_graph.predecessors(n))) == 0 \
                and (isinstance(n, OpNode)):
                function_arg_nodes_input.append(n)
                continue
            if len(list(self.program_graph.successors(n))) == 0 \
                and (isinstance(n, OpNode)):
                function_arg_nodes_output.append(n)
                continue
        return function_arg_nodes_input, function_arg_nodes_output, function_arg_nodes_array
    
    
            
    def _graph_to_function_decl(self):
        function_arg_nodes_input, \
        function_arg_nodes_output,\
        function_arg_nodes_array = self._select_function_arg_list()
        print("[INFO] function input args: ")
        for arg_node_input in function_arg_nodes_input:
            print("[INFO]", arg_node_input)
        print("[INFO] function interface arrays:")
        for array_n in function_arg_nodes_array:
            print("[INFO]", array_n)
        print("[INFO] function output args")
        for arg_node_output in function_arg_nodes_output:
            print("[INFO]", arg_node_output)
        if len(function_arg_nodes_input) == 0 and len(function_arg_nodes_output) == 0 and len(function_arg_nodes_array) == 0:
            raise ValueError("the function should have at least one input or output argument")
        # generate function declaration for input
        function_decl = f"void {self.function_name}("
        for i, arg_node_input in enumerate(function_arg_nodes_input):
            arg_node_input: OpNode
            if i > 0:
                function_decl += ", "

            # arg declaration
            if arg_node_input.result_type == ResultDataType.AP_FIXED:

                if not isinstance(arg_node_input.result_rounding_mode, QuantizationMode):
                    raise TypeError("expected have type QuantizationMode, but got "+\
                                    f"{arg_node_input.result_rounding_mode}, with type {type(arg_node_input.result_rounding_mode)}")
                if not isinstance(arg_node_input.result_wrap_mode, OverflowMode):
                    raise TypeError("expected have type OverflowMode but got "+\
                                    f"{arg_node_input.result_wrap_mode}, with type {type(arg_node_input.result_wrap_mode)}")

                function_decl += f"ap_fixed<{arg_node_input.result_width},{arg_node_input.result_int_width_ap_fixed},"+\
                    f"{arg_node_input.result_rounding_mode.value},{arg_node_input.result_wrap_mode.value}> "+\
                    f"{arg_node_input.name}"
            elif arg_node_input.result_type == ResultDataType.AP_INT:
                function_decl += f"ap_int<{arg_node_input.result_width}> {arg_node_input.name}"
            elif arg_node_input.result_type == ResultDataType.AP_UINT:
                function_decl += f"ap_uint<{arg_node_input.result_width}> {arg_node_input.name}"
            else:
                raise ValueError(f"Unsupported result type: {arg_node_input.result_type}")
        for i, arg_node_output in enumerate(function_arg_nodes_output):
            if i > 0 or len(function_arg_nodes_input) > 0:
                function_decl += ", "
            arg_node_output: OpNode
            # arg declaration
            if arg_node_output.result_type == ResultDataType.AP_FIXED:

                if not isinstance(arg_node_output.result_rounding_mode, QuantizationMode):
                    raise TypeError("expected have type QuantizationMode, but got "+\
                                    f"{arg_node_output.result_rounding_mode}, with type {type(arg_node_output.result_rounding_mode)}")
                if not isinstance(arg_node_output.result_wrap_mode, OverflowMode):
                    raise TypeError("expected have type OverflowMode but got "+\
                                    f"{arg_node_output.result_wrap_mode}, with type {type(arg_node_output.result_wrap_mode)}")
                function_decl += \
                    f"ap_fixed<{arg_node_output.result_width},{arg_node_output.result_int_width_ap_fixed},"+\
                    f"{arg_node_output.result_rounding_mode.value},{arg_node_output.result_wrap_mode.value}> " +\
                    f"&{arg_node_output.name}"
            elif arg_node_output.result_type == ResultDataType.AP_INT:
                function_decl += f"ap_int<{arg_node_output.result_width}> &{arg_node_output.name}"
            elif arg_node_output.result_type == ResultDataType.AP_UINT:
                function_decl += f"ap_uint<{arg_node_output.result_width}> &{arg_node_output.name}"
            else:
                raise ValueError(f"Unsupported result type: {arg_node_output.result_type}")
        for i, arg_node_array in enumerate(function_arg_nodes_array):
            if i > 0 or len(function_arg_nodes_input) > 0 or len(function_arg_nodes_output) > 0:
                function_decl += ", "
            arg_node_array: ArrayNode
            # arg declaration
            if arg_node_array.result_type == ResultDataType.AP_FIXED:


                if not isinstance(arg_node_array.result_rounding_mode, QuantizationMode):
                    raise TypeError("expected have type QuantizationMode, but got "+\
                                    f"{arg_node_array.result_rounding_mode}, with type {type(arg_node_array.result_rounding_mode)}")
                if not isinstance(arg_node_array.result_wrap_mode, OverflowMode):
                    raise TypeError("expected have type OverflowMode but got "+\
                                    f"{arg_node_array.result_wrap_mode}, with type {type(arg_node_array.result_wrap_mode)}")

                function_decl += f"ap_fixed<{arg_node_array.result_width},{arg_node_array.result_int_width_ap_fixed},"+\
                    f"{arg_node_array.result_rounding_mode.value},{arg_node_array.result_wrap_mode.value}> "+\
                    f"{arg_node_array.name}[{arg_node_array.length}]"
            elif arg_node_array.result_type == ResultDataType.AP_INT:
                function_decl += f"ap_int<{arg_node_array.result_width}> {arg_node_array.name}[{arg_node_array.length}]"
            elif arg_node_array.result_type == ResultDataType.AP_UINT:
                function_decl += f"ap_uint<{arg_node_array.result_width}> {arg_node_array.name}[{arg_node_array.length}]"
        function_decl += ") {\n"
        return function_decl
    
    def _graph_to_function_variable_decl(self):
        variable_decl = ""
        function_arg_nodes_input, \
        function_arg_nodes_output,\
        function_arg_nodes_array = self._select_function_arg_list()
        for n in self.program_graph.nodes():
            if isinstance(n, OpNode):
                if n in function_arg_nodes_input:
                    continue
                if n in function_arg_nodes_output:
                    continue
                if n in function_arg_nodes_array:
                    continue
                variable_decl += self._op_node_to_decl_str(n) + "\n"

        return variable_decl

    _graph_to_function_body_visitor_for_sanity_check = set()
    def _graph_to_function_body(self):
        function_body = ""
        GraphManager._graph_to_function_body_visitor_for_sanity_check.clear()
        program_dag = self.program_graph.copy()
        # remove all array nodes from the program_dag
        for n in list(program_dag.nodes()):
            if isinstance(n, ArrayNode):
                program_dag.remove_node(n)
        # check dag is acyclic
        if not nx.is_directed_acyclic_graph(program_dag):
            raise ValueError("The program graph is not a directed acyclic graph (DAG).")
        for n in nx.topological_sort(program_dag):

            n_predecessor_loop_node_list = self._get_n_predecessor_loop_node(n)
            n_predecessor_branch_node_list = self._get_n_predecessor_branch_node(n)
            n_predecessor_loop_node = None
            n_predecessor_branch_node = None
            if len(n_predecessor_loop_node_list) == 1:
                n_predecessor_loop_node = n_predecessor_loop_node_list[0]
            if len(n_predecessor_branch_node_list) == 1:
                n_predecessor_branch_node = n_predecessor_branch_node_list[0]

            if n_predecessor_loop_node is not None:
                if not n_predecessor_loop_node in GraphManager._graph_to_function_body_visitor_for_sanity_check:
                    raise ValueError("it is expected that the predecessor loop node has been visited already")
            if n_predecessor_branch_node is not None:
                if not n_predecessor_branch_node in GraphManager._graph_to_function_body_visitor_for_sanity_check:
                    raise ValueError("it is expected that the predecessor branch node has been visited already")
            
            # may be visited already
            if n_predecessor_loop_node is not None or n_predecessor_branch_node is not None:
                continue
            
            if isinstance(n, LoopNode):
                function_body += self._loop_node_to_str(n) + "\n"
            elif isinstance(n, BranchNode):
                function_body += self._br_node_to_str(n) + "\n"
            elif isinstance(n, OpNode):
                function_body += self._op_node_to_assignment_str(n) + "\n"
            elif isinstance(n, DepNode):
                function_body += self._dep_node_to_str(n) + "\n"
            elif isinstance(n, ArrayNode):
                raise NotImplementedError()

        GraphManager._graph_to_function_body_visitor_for_sanity_check.clear()
        return function_body


    def _get_n_predecessor_loop_node(self, n:Node):
        """
        Get the predecessor LoopNode of the given node n.
        """
        predecessors = list(self.program_graph.predecessors(n))
        loop_predecessors = [pred for pred in predecessors if isinstance(pred, LoopNode)]
        if len(loop_predecessors) > 1:
            raise ValueError(f"Node {n} has multiple predecessor LoopNodes: {loop_predecessors}")
        return loop_predecessors
    
    def _get_n_predecessor_branch_node(self, n:Node):
        """
        Get the predecessor BranchNode of the given node n.
        """
        predecessors = list(self.program_graph.predecessors(n))
        branch_predecessors = [pred for pred in predecessors if isinstance(pred, BranchNode)]
        if len(branch_predecessors) > 1:
            raise ValueError(f"Node {n} has multiple predecessor BranchNodes: {branch_predecessors}")
        return branch_predecessors
    

    def _cpp_head_generation(self):
        
        from datetime import datetime
        
        now = datetime.now()

        # Format as ddmmyyHHMMSS
        formatted_datetime = now.strftime("%d%m%y%H%M%S")

        return f'//{formatted_datetime}\n#include"ap_int.h"\n#include"ap_fixed.h"\n'

    def _function_pipeline_pragma(self):
        if self.program_graph_function_pipeline:
            return "#pragma HLS pipeline\n"
        else:
            return ""

    def _dump_cpp(self):
        # dump the program to cpp
        cpp_head = self._cpp_head_generation()
        function_decl = self._graph_to_function_decl()
        function_pipeline_str = self._function_pipeline_pragma()
        interface_pragmas = self._graph_to_interface_pragmas()
        variable_decl = self._graph_to_function_variable_decl()
        function_body = self._graph_to_function_body()
        return f"{cpp_head}\n{function_decl}\n{interface_pragmas}"+\
            f"\n{function_pipeline_str}\n"+\
            f"\n{variable_decl}\n{function_body}\n}}"
    
    def _graph_to_interface_pragmas(self):
        interface_pragmas_str = ""
        for node in self.program_graph.nodes():
            if isinstance(node, ArrayNode):
                interface_pragmas_str += self._array_node_pragma_to_str(node)
        return interface_pragmas_str


    def _dump_cp_1_cpp(self):
        tmp_origin = self.program_graph
        tmp_function_pipeline = self.program_graph_function_pipeline
        self.program_graph = self.program_graph_copy_1 
        self.program_graph_function_pipeline = self.function_pipeline_1

        cp_1_cpp_str = self._dump_cpp()

        self.program_graph = tmp_origin
        self.program_graph_function_pipeline = tmp_function_pipeline
        return cp_1_cpp_str
    
    def _dump_cp_2_cpp(self):
        tmp_origin = self.program_graph
        tmp_function_pipeline = self.program_graph_function_pipeline
        self.program_graph = self.program_graph_copy_2
        self.program_graph_function_pipeline = self.function_pipeline_2

        cp_2_cpp_str = self._dump_cpp()

        self.program_graph = tmp_origin
        self.program_graph_function_pipeline = tmp_function_pipeline
        return cp_2_cpp_str

    def dump_cpp_std(self, file_path: str = "output.cpp"):
        """
        Dump the program to a C++ file.
        :param file_path: The path to the output C++ file.
        """
        contain_clang_format = False
        if shutil.which("clang-format") is None:
            print("[WARNING] clang-format not found in system environment.")
        else:
            print("[INFO] clang-format found in system environment.")
            contain_clang_format = True
        self.dump_png()

        cpp_code = self._dump_cpp()
        with open(file_path, 'w') as f:
            f.write(cpp_code)
        print(f"[INFO] C++ code dumped to {file_path}")
        if contain_clang_format:
            print("[INFO] Running clang-format on the dumped C++ code...")
            result = subprocess.run(["clang-format", "-i", "--style=Google", file_path], capture_output=True, text=True)
            if result.returncode == 0:
                print("[INFO] clang-format completed successfully.")
            else:
                print("[WARNING] clang-format encountered issues:")
                print(result.stdout)
                print(result.stderr)

    def generate_cmp_graphs(self):
        print("[INFO] generate gragmas for 2 graphs for comparsion")
        print("[INFO] reset the graph pointer to empty")
        self.program_graph_copy_1 = nx.DiGraph()
        self.program_graph_copy_2 = nx.DiGraph()
        self._copy_graph_and_insert_pragmas()

        # simple check, have same number of nodes
        if not (self.program_graph_copy_1.number_of_nodes() == \
                self.program_graph_copy_2.number_of_nodes() == \
                self.program_graph.number_of_nodes()):
            raise ValueError("expected the copied graph and the original "+\
            "one to have the same number of nodes, but got "+\
            f"self.program_graph_copy_1.number_of_nodes() = "+\
            f"{self.program_graph_copy_1.number_of_nodes()}; "+\
            f"self.program_graph_copy_2.number_of_nodes() = "+\
            f"{self.program_graph_copy_2.number_of_nodes()}; "+\
            f"self.program_graph.number_of_nodes() = "+\
            f"{self.program_graph.number_of_nodes()}")
        if not (self.program_graph_copy_1.number_of_edges() == \
                self.program_graph_copy_2.number_of_edges() == \
                self.program_graph.number_of_edges()):
            raise ValueError("expected the copied graph and the original "+\
            "one to have the same number of edges, but got "+\
            f"self.program_graph_copy_1.number_of_edges() = "+\
            f"{self.program_graph_copy_1.number_of_edges()}; "+\
            f"self.program_graph_copy_2.number_of_edges() = "+\
            f"{self.program_graph_copy_2.number_of_edges()}; "+\
            f"self.program_graph.number_of_edges() = "+\
            f"{self.program_graph.number_of_edges()}")
        print("[INFO] finished pragma generation")
        return True


    def dump_cpp_comparsion(self, file_path_1:str = "output_1.cpp",
                            file_path_2:str = "output_2.cpp"):
        contain_clang_format = False
        if shutil.which("clang-format") is None:
            print("[WARNING] clang-format not found in system environment.")
        else:
            print("[INFO] clang-format found in system environment.")
            contain_clang_format = True
        self.dump_png()

        # Generate comparison graphs with different pragmas
        self.generate_cmp_graphs()

        cpp_code_1 = self._dump_cp_1_cpp()
        cpp_code_2 = self._dump_cp_2_cpp()

        with open(file_path_1, 'w') as f:
            f.write(cpp_code_1)
        print(f"[INFO] C++ code dumped to {file_path_1}")

        with open(file_path_2, 'w') as f:
            f.write(cpp_code_2)
        print(f"[INFO] C++ code dumped to {file_path_2}")

        if contain_clang_format:
            print("[INFO] Running clang-format on the dumped C++ code...")
            result = subprocess.run(["clang-format", 
                "-i", "--style=Google", file_path_1], capture_output=True, text=True)
            if result.returncode == 0:
                print("[INFO] clang-format completed successfully.")
            else:
                print("[WARNING] clang-format encountered issues:")
                print(result.stdout)
                print(result.stderr)
        if contain_clang_format:
            print("[INFO] Running clang-format on the dumped C++ code...")
            result = subprocess.run(["clang-format", 
                "-i", "--style=Google", file_path_2], capture_output=True, text=True)
            if result.returncode == 0:
                print("[INFO] clang-format completed successfully.")
            else:
                print("[WARNING] clang-format encountered issues:")
                print(result.stdout)
                print(result.stderr)

    

    def _set_loop_node_pragmas(self, node):
        raise NotImplementedError("not overloaded")
    
    def _set_design_cp_in_ns(self):
        raise NotImplementedError("not overloaded")
    

    

    def _insert_pragmas_to_graph(self, program_graph_to_be_inserted:nx.DiGraph):
        for node in program_graph_to_be_inserted.nodes():
            if isinstance(node, LoopNode):
                self._set_loop_node_pragmas(node)
                node.check_pragma_status()

    def _copy_graph_and_insert_pragmas(self):
        print("[INFO] call GraphManager::_copy_graph_and_insert_pragmas")
        self.program_graph_copy_1 = self.program_graph.copy()
        self.program_graph_copy_2 = self.program_graph.copy()       

        self._insert_pragmas_to_graph(self.program_graph_copy_1)
        self._insert_pragmas_to_graph(self.program_graph_copy_2)

        self.cp_1 = self._set_design_cp_in_ns()
        self.cp_2 = self._set_design_cp_in_ns() 
        print("[INFO] end call GraphManager::_copy_graph_and_insert_pragmas")

    
    
    def dump_png(self, file_path: str = "output_graph.png"):
        """
        Dump the program graph to a PNG file with only node names displayed.
        :param file_path: The path to the output PNG file.
        """
        # Create a mapping of node objects to their names for labels
        labels = {}
        for node in self.program_graph.nodes():
            if hasattr(node, 'name'):
                labels[node] = node.name
            else:
                labels[node] = str(node)
        
        # Clear any previous plots
        plt.clf()
        
        # Draw the graph with custom labels showing only node names
        pos = nx.spring_layout(self.program_graph)  # Use spring layout for better visualization
        nx.draw(self.program_graph, pos, labels=labels, with_labels=True, 
                node_size=2000, node_color='lightblue', font_size=10, 
                font_color='black', font_weight='bold', arrows=True, 
                arrowsize=20, edge_color='gray')
        
        plt.title("Program Graph")
        plt.axis('off')  # Turn off axis for cleaner look
        # plt.tight_layout()
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.clf()  # Clear the plot after saving
        print(f"[INFO] Program graph dumped to {file_path}")

    def _has_branch_node(self):
        return self.branch_node_counter > 0
    
    def _has_array_node(self):
        return self.array_node_counter > 0
    
    def _has_loop_node(self):
        # use in debug only, delete after
        loop_node_list = self._get_loop_node_list()
        if not len(loop_node_list) == self.loop_node_counter:
            raise ValueError(f"loop node list: {loop_node_list}, loop node counter: {self.loop_node_counter}")
        return self.loop_node_counter > 0
    
    
    
    def _make_single_output(self):

        single_node_list = []
        for node in self.program_graph.nodes():
            if isinstance(node, ArrayNode):
                raise ValueError("Currently we do not support ArrayNode")
            if isinstance(node, OpNode):
                if node.op_type == OperationType.VISIT:
                    raise ValueError("Currently we do not support visit node")
                if node.op_type == OperationType.WRITE:
                    raise ValueError("Currently we do not support write node")
            if len(list(self.program_graph.successors(node))) == 0 and \
                isinstance(node, OpNode):
                single_node_list.append(node)
        
        if len(single_node_list) <= 1:
            return
        print("[INFO] detect multiple output nodes, use add to make it simple output")
        froniter_node = single_node_list[0]
        for i in range(len(single_node_list) - 1):
            froniter_node = self.add_op_node(
                op_type=OperationType.ADD,
                predecessor_list=[froniter_node, single_node_list[i+1]],
                result_type=ResultDataType.AP_INT,
                result_width=32
            )
        single_node_list_check = []
        for node in self.program_graph.nodes():
            if isinstance(node, ArrayNode):
                raise ValueError("Currently we do not support ArrayNode")
            if isinstance(node, OpNode):
                if node.op_type == OperationType.VISIT:
                    raise ValueError("Currently we do not support visit node")
                if node.op_type == OperationType.WRITE:
                    raise ValueError("Currently we do not support write node")
            if len(list(self.program_graph.successors(node))) == 0 and \
                isinstance(node, OpNode):
                single_node_list_check.append(node)
        if len(single_node_list_check) != 1:
            raise ValueError()
            
        

    
    def print_node_list(self, ident = ""):
        print(f"{ident} node list info: ")
        for node in self.program_graph.nodes():
            print(f"{ident} {node}")
        print(f"{ident} End node list info: ")

    def find_and_print_edges_between_nodes(self, node_a, node_b):
        """
        Find and print all edges from node_a to node_b in the Digraph.
        
        Args:
            node_a: Source node
            node_b: Target node
        """
        print(f"Finding all edges from {node_a} to {node_b}:")
        
        # Method 1: Using edges() method
        edges_a_to_b = list(self.program_graph.edges(node_a, node_b, data=True, keys=True))
        
        if not edges_a_to_b:
            print(f"No edges found from {node_a} to {node_b}")
            return
        
        print(f"Found {len(edges_a_to_b)} edge(s):")
        for i, (u, v, key, data) in enumerate(edges_a_to_b, 1):
            print(f"  Edge {i}: {u} -> {v} (key: {key})")
            if data:
                for attr_name, attr_value in data.items():
                    print(f"    {attr_name}: {attr_value}")
            else:
                print("    No attributes")
        
        # Method 2: Using get_edge_data() method (alternative)
        print("\nAlternative method using get_edge_data():")
        edge_data_dict = self.program_graph.get_edge_data(node_a, node_b)
        
        if edge_data_dict:
            # Handle both single edge and multiple edges cases
            if isinstance(edge_data_dict, dict):
                # Check if this is a single edge (has direct attributes) or multiple edges
                if any(key for key in edge_data_dict.keys() if isinstance(edge_data_dict[key], dict)):
                    # Multiple edges case
                    for key, data in edge_data_dict.items():
                        print(f"  Edge key {key}: {data}")
                else:
                    # Single edge case
                    print(f"  Single edge: {edge_data_dict}")
        else:
            print(f"  No edges found from {node_a} to {node_b}")

    def find_edges_by_description(self, node_a, node_b, description):
        """
        Find edges between two nodes with a specific description attribute.
        
        Args:
            node_a: Source node
            node_b: Target node  
            description: Description value to search for
            
        Returns:
            List of (key, data) tuples for matching edges
        """
        matching_edges = []
        edge_data_dict = self.program_graph.get_edge_data(node_a, node_b)
        
        if edge_data_dict and isinstance(edge_data_dict, dict):
            # Handle both single edge and multiple edges cases
            if 'description' in edge_data_dict:
                # Single edge case
                if edge_data_dict.get("description") == description:
                    matching_edges.append((0, edge_data_dict))
            else:
                # Multiple edges case
                for key, data in edge_data_dict.items():
                    if isinstance(data, dict) and data.get("description") == description:
                        matching_edges.append((key, data))
        
        return matching_edges


    def _is_loop_node_inner_loop(self, loop_node:LoopNode):
        if not isinstance(loop_node, LoopNode):
            raise TypeError()
        loop_node_succ_list = list(self.program_graph.successors(loop_node))
        loop_node_succ_list = [_ for _ in loop_node_succ_list if isinstance(_, LoopNode)]
        if len(loop_node_succ_list) == 0:
            return True
        return False
        
