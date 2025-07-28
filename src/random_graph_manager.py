from graph_manager import GraphManager
from node import Node, LoopNode, BranchNode, OpNode, ArrayNode, ResultDataType
from node import OperationType
from enum import Enum
from dataclasses import dataclass
from random_type_generator import RandomTypeGenerator
from random_op_type_generator import RandomOpTypeGenerator
import random
import networkx as nx
import numpy as np
from node import QuantizationMode, OverflowMode
from random_pragma_generator import RandomPragmaGenerator
# from typing import overload


class RandomGraphManager(GraphManager):
    """
    RandomGraphManager is a subclass of GraphManager that manages the generation
    and manipulation of random graphs, specifically designed for HLS model checking.
    It extends the functionality of GraphManager to include random graph generation.
    """

    def _generate_random_op_node(self) -> Node:
        """
        Generate a random operation node with a given name.
        """
        result_type = self.rand_type_gen.generate()
        if len(result_type) == 2:
            result_type_str = result_type[0]
            result_width = result_type[1]
        elif len(result_type) == 5:
            result_type_str = result_type[0]
            result_width = result_type[1]
            result_int_width = result_type[2]
            quant_mode = result_type[3]
            overflow_mode = result_type[4]
        else:
            raise ValueError("err info", result_type)
        if len(result_type) == 5:
            if not isinstance(quant_mode, QuantizationMode):
                raise TypeError("expected quant_mode have type QuantizationMode "+\
                                f"but got quant_mode = {quant_mode} "+\
                                f"with type = {type(quant_mode)}")
            if not isinstance(overflow_mode, OverflowMode):
                raise TypeError("expected overflow_mode have type OverflowMode "+\
                                f"but got overflow_mode = {overflow_mode} "+\
                                f"with type = {type(overflow_mode)}")
        
        result_type_enum = None

        if result_type_str == "ap_int":
            result_type_enum = ResultDataType.AP_INT
        elif result_type_str == "ap_fixed":
            result_type_enum = ResultDataType.AP_FIXED
        elif result_type_str == "ap_uint":
            result_type_enum = ResultDataType.AP_UINT
        result_op_type_enum = self.rand_op_type_gen.generate()

        if result_type_enum == ResultDataType.AP_FIXED:
            op_node_instance = OpNode(
                name="",
                op_type=result_op_type_enum,
                result_type=result_type_enum,
                result_width=result_width,
                result_int_width_ap_fixed=result_int_width,
                result_wrap_mode=overflow_mode,
                result_rounding_mode=quant_mode
            )
        else:
            op_node_instance = OpNode(
                name="",
                op_type=result_op_type_enum,
                result_type=result_type_enum,
                result_width=result_width
            )
        return op_node_instance
    
    def _generate_random_loop_node(self, op_node_list):
        if op_node_list is None:
            raise ValueError()
        if len(op_node_list) == 0:
            raise ValueError()
        
        # Decide whether to use integer values or OpNodes for start/end indices
        use_op_node_for_start = random.choice([True, False])
        use_op_node_for_end = random.choice([True, False])
        
        # Generate start index
        if use_op_node_for_start and len(op_node_list) > 0:
            start_index = random.choice(op_node_list)
        else:
            start_index = random.randint(0, 10)
        
        # Generate end index  
        if use_op_node_for_end and len(op_node_list) > 0:
            end_index = random.choice(op_node_list)
        else:
            # Ensure end_index is greater than start_index when both are integers
            if isinstance(start_index, int):
                end_index = random.randint(start_index + 1, start_index + 100)
            else:
                end_index = random.randint(10, 100)
        
        # Generate step (always an integer)
        step = random.choice([1, 2, 4, 8])

        return LoopNode(
            name="",
            start_index=start_index,
            end_index=end_index,
            step=step
        )
    
    def _generate_random_branch_node(self):
        return BranchNode(name="")
    
    def _generate_random_array_node(self):
        """
        Generate a random array node with random type and length.
        """
        # Generate random type using rand_type_gen
        result_type = self.rand_type_gen.generate()
        
        # Parse the result type similar to _generate_random_op_node
        if len(result_type) == 2:
            result_type_str = result_type[0]
            result_width = result_type[1]
            result_int_width = 0  # Default for non-AP_FIXED types
            quant_mode_str = "AP_RND"  # Default
            overflow_mode_str = "AP_WRAP"  # Default
        elif len(result_type) == 5:
            result_type_str = result_type[0]
            result_width = result_type[1]
            result_int_width = result_type[2]
            quant_mode_str = result_type[3]
            overflow_mode_str = result_type[4]
        else:
            raise ValueError("Invalid result type format", result_type)
        
        # Convert string to enum
        result_type_enum = None
        if result_type_str == "ap_int":
            result_type_enum = ResultDataType.AP_INT
        elif result_type_str == "ap_fixed":
            result_type_enum = ResultDataType.AP_FIXED
        elif result_type_str == "ap_uint":
            result_type_enum = ResultDataType.AP_UINT
        else:
            raise ValueError(f"Unknown result type: {result_type_str}")
        
        # Generate random array length
        array_length = random.choice([64, 128, 256, 512, 1024, 2048, 4096])
        
        # Create ArrayNode instance
        array_node_instance = ArrayNode(
            name="",
            length=array_length,
            result_type=result_type_enum,
            result_width=result_width,
            result_int_width_ap_fixed=result_int_width,
            result_wrap_mode=overflow_mode_str,
            result_rounding_mode=quant_mode_str
        )
        
        return array_node_instance
    
    def _random_pick_from_list_with_normal_distribution(self, l):
        """
        Pick an element from a list using normal distribution weights.
        Elements near the center of the list have higher probability of being selected.
        
        Args:
            l: List to pick from
            
        Returns:
            A randomly selected element from the list
        """
        if not l:
            raise ValueError("Cannot pick from empty list")
        
        if len(l) == 1:
            return l[0]
        
        # Create indices for the list
        indices = np.arange(len(l))
        
        # Calculate the center of the list
        center = (len(l) - 1) / 2
        
        # Generate weights using normal distribution centered at the middle
        # Standard deviation is set to be about 1/3 of the list length for good spread
        std_dev = len(l) / 6
        weights = np.exp(-0.5 * ((indices - center) / std_dev) ** 2)
        
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        
        # Use random.choices with weights to select an index
        selected_index = random.choices(indices, weights=weights, k=1)[0]
        
        return l[selected_index]
    
    def _random_binary_choice(self):
        # do a equal random binary choice that return boolean
        return random.choice([True, False])

    def _action_random_add_array(self):
        # randomly generate an array node and add to graph
    
        print("[INFO] Do action: randomly add array node")
        array_node = self._generate_random_array_node()
        self.add_array_node(array_node_created=array_node)
        return True
    
    def _action_random_add_input(self):
        print("[INFO] Do action: randomly add input")
        op_node_r =  self._generate_random_op_node()
        self.add_op_node(
            op_node_created=op_node_r,
            predecessor_list=[]
        )
        return True
    
    def _random_get_op_node_predecessor_list(self,op_node_r:OpNode):
        op_node_list = self._get_op_node_list()
        op_node_type = op_node_r.op_type
        predecessor_list = []
        if op_node_type == OperationType.NOT:
            op_node_pick = self._random_pick_from_list_with_normal_distribution(
                op_node_list
            )
            predecessor_list.append(op_node_pick)
        else:
            op_node_pick_0 = self._random_pick_from_list_with_normal_distribution(
                op_node_list
            )
            op_node_pick_1 = self._random_pick_from_list_with_normal_distribution(
                op_node_list
            )
            predecessor_list.append(op_node_pick_0)
            predecessor_list.append(op_node_pick_1)
        return predecessor_list
    
    def _random_get_branch_predecessor(self):
        br_node_list = self._get_branch_node_list()
        br_node_pick = self._random_pick_from_list_with_normal_distribution(br_node_list)
        return br_node_pick
    
    def _random_get_loop_predecessor(self):
        loop_node_list = self._get_loop_node_list()
        loop_node_pick = self._random_pick_from_list_with_normal_distribution(loop_node_list)
        return loop_node_pick

    def _action_random_add_op(self):
        print("[INFO] Do action: randomly add op node")
        op_node_r = self._generate_random_op_node()
        op_node_r:OpNode
        is_belong_to_code_block = self._random_binary_choice()
        is_belong_to_loop_block = self._random_binary_choice()
        is_belong_to_branch_block = not is_belong_to_loop_block

        is_belong_to_code_block, \
        is_belong_to_loop_block, \
        is_belong_to_branch_block = self._fix_random_choice(
            is_belong_to_code_block=is_belong_to_code_block,
            is_belong_to_loop_block=is_belong_to_loop_block,
            is_belong_to_branch_block=is_belong_to_branch_block
        )

        if not is_belong_to_code_block:
            # Check if we have enough nodes for predecessors
            op_node_list = self._get_op_node_list()
            if len(op_node_list) < 1:
                # If no predecessors available, add as input node
                self.add_op_node(
                    op_node_created=op_node_r,
                    predecessor_list=[]
                )
            else:
                predecessor_list = self._random_get_op_node_predecessor_list(
                    op_node_r=op_node_r
                )
                self.add_op_node(
                    op_node_created=op_node_r,
                    predecessor_list=predecessor_list
                )
        else:
            if is_belong_to_loop_block:
                if not self._has_loop_node():
                    raise ValueError("there is expected to be loop nodes")
                loop_node_p = self._random_get_loop_predecessor()
                predecessor_list = self._random_get_op_node_predecessor_list(
                op_node_r=op_node_r)
                self.add_op_node(
                    op_node_created=op_node_r,
                    predecessor_list=predecessor_list,
                    loop_node=loop_node_p
                )
            elif is_belong_to_branch_block:
                if not self._has_branch_node():
                    raise ValueError("there is expected to be branch nodes")
                branch_direction = self._random_binary_choice()
                br_node_p = self._random_get_branch_predecessor()
                predecessor_list = self._random_get_op_node_predecessor_list(
                op_node_r=op_node_r)
                self.add_op_node(
                    op_node_created=op_node_r,
                    predecessor_list=predecessor_list,
                    br_node=br_node_p,
                    br_node_branch=branch_direction
                )
            else:
                raise NotImplementedError("how do you get here")
        return True
    
    def _action_random_add_loop(self):
        print("[INFO] Do action: randomly add loop node")
        op_node_list = self._get_op_node_list()
        if len(op_node_list) < 1:
            return False
        loop_node_r = self._generate_random_loop_node(op_node_list)
        is_belong_to_code_block = self._random_binary_choice()
        is_belong_to_loop_block = self._random_binary_choice()
        is_belong_to_branch_block = not is_belong_to_loop_block

        is_belong_to_code_block, \
        is_belong_to_loop_block, \
        is_belong_to_branch_block = self._fix_random_choice(
            is_belong_to_code_block=is_belong_to_code_block,
            is_belong_to_loop_block=is_belong_to_loop_block,
            is_belong_to_branch_block=is_belong_to_branch_block
        )

        if not is_belong_to_code_block:
            self.add_loop_node(
                loop_node_created=loop_node_r
            )
        else:
            if is_belong_to_loop_block:
                if not self._has_loop_node():
                    raise ValueError("there is no loop node")
                loop_node_p = self._random_get_loop_predecessor()
                self.add_loop_node(
                    loop_node_created=loop_node_r,
                    loop_node_predecessor=loop_node_p
                )
            elif is_belong_to_branch_block:
                if not self._has_branch_node():
                    raise ValueError("there is no branch node")
                branch_direction = self._random_binary_choice()
                br_node_p = self._random_get_branch_predecessor()
                self.add_loop_node(
                    loop_node_created=loop_node_r,
                    br_node_predecessor=br_node_p,
                    br_node_branch=branch_direction
                )
            else:
                raise NotImplementedError("how do you get here")
        return True
    
    def _fix_random_choice(self, is_belong_to_code_block:bool,
                           is_belong_to_loop_block:bool,
                           is_belong_to_branch_block:bool):
        if not is_belong_to_code_block:
            return is_belong_to_code_block, is_belong_to_loop_block, is_belong_to_branch_block
        else:
            if is_belong_to_branch_block and not self._has_branch_node():
                return False, is_belong_to_loop_block, is_belong_to_branch_block
            elif is_belong_to_loop_block and not self._has_loop_node():
                return False, is_belong_to_loop_block, is_belong_to_branch_block
            else:
                return is_belong_to_code_block, is_belong_to_loop_block, is_belong_to_branch_block

    def _action_random_add_branch(self):
        print("[INFO] Do action: randomly add branch node")
        br_node_r = self._generate_random_branch_node()
        
        op_node_list = self._get_op_node_list()
        if len(op_node_list) < 1:
            return False
        conditional_op_node_r = \
            self._random_pick_from_list_with_normal_distribution(op_node_list)

        is_belong_to_code_block = self._random_binary_choice()
        is_belong_to_loop_block = self._random_binary_choice() 
        is_belong_to_branch_block = not is_belong_to_loop_block

        is_belong_to_code_block, \
        is_belong_to_loop_block, \
        is_belong_to_branch_block = self._fix_random_choice(
            is_belong_to_code_block=is_belong_to_code_block,
            is_belong_to_loop_block=is_belong_to_loop_block,
            is_belong_to_branch_block=is_belong_to_branch_block
        )

        if not is_belong_to_code_block:
            self.add_branch_node(
                conditional_op=conditional_op_node_r,
                branch_node_created=br_node_r,
            )
        else:
            if is_belong_to_loop_block:
                if not self._has_loop_node():
                    raise ValueError("there is no loop node here")
                loop_node_p = self._random_get_loop_predecessor()
                self.add_branch_node(
                    conditional_op=conditional_op_node_r,
                    branch_node_created=br_node_r,
                    loop_node_predecessor=loop_node_p
                )
            elif is_belong_to_branch_block:
                if not self._has_branch_node():
                    raise ValueError("there is no branch node here")
                branch_direction = self._random_binary_choice()
                br_node_p = self._random_get_branch_predecessor()
                self.add_branch_node(
                    conditional_op=conditional_op_node_r,
                    branch_node_created=br_node_r,
                    br_node_predecessor=br_node_p,
                    br_node_branch=branch_direction
                )
            else:
                raise NotImplementedError("how do you get here")
        return True
    
    def _action_random_add_array_visit(self):
        print("[INFO] Do action: randomly add array visiting node")
        array_node_list = self._get_array_node_list()
        op_node_list = self._get_op_node_list()

        if len(op_node_list) < 2 or len(array_node_list) < 1:
            return False

        array_node_r = self._random_pick_from_list_with_normal_distribution(array_node_list)
        array_node_r:ArrayNode
        if not isinstance(array_node_r, ArrayNode):
            raise TypeError()
        array_node_r_len = array_node_r.length
        index = random.randint(0, array_node_r_len - 1)
        
        loop_node_list = self._get_loop_node_list()
        r_sel_list = [index] + op_node_list + loop_node_list

        address_node_r = self._random_pick_from_list_with_normal_distribution(r_sel_list)

        if not isinstance(address_node_r, (OpNode, int, LoopNode)):
            raise TypeError(f"expected type: OpNode, int, LoopNode, got type {type(address_node_r)}")

        self.add_array_visit(array_node=array_node_r,
                             address_node=address_node_r)
        return True

    def _action_random_add_array_write(self):
        print("[INFO] Do action: randomly add array write node")
        op_node_list = self._get_op_node_list()
        array_node_list = self._get_array_node_list()

        if len(op_node_list) < 2 or len(array_node_list) < 1:
            return False

        array_node_r = self._random_pick_from_list_with_normal_distribution(array_node_list)
        array_node_r:ArrayNode
        array_node_r_len = array_node_r.length
        index = random.randint(0, array_node_r_len - 1)
        
        loop_node_list = self._get_loop_node_list()
        r_sel_list = [index] + op_node_list + loop_node_list
        
        address_node_r = self._random_pick_from_list_with_normal_distribution(r_sel_list)
        write_value_node_r = self._random_pick_from_list_with_normal_distribution(op_node_list)
        
        
        self.add_array_write(array_node=array_node_r,
                             write_value_node=write_value_node_r,
                             address_node=address_node_r)
        return True


    def _generate_random_graph(self):
        """
        Generate a random graph by performing 100 random actions.
        Each action adds a different type of node or operation to the graph.
        """
        self._reset_all()
        action_list = [
            # self._action_random_add_array,
            self._action_random_add_input, 
            self._action_random_add_op,
            # self._action_random_add_loop,
            # self._action_random_add_branch,
            # self._action_random_add_array_visit,
            # self._action_random_add_array_write
        ]
        
        print("[INFO] Starting random graph generation with 100 actions...")
        successful_actions = 0
        action_number_total = 20
        for i in range(action_number_total):
            # Randomly select an action from the list
            action = random.choice(action_list)
            
            try:
                # Execute the action and check if it was successful
                success = action()
                if success:
                    successful_actions += 1
                    print(f"[INFO] Action {i+1}/{action_number_total} completed successfully")
                else:
                    print(f"[INFO] Action {i+1}/{action_number_total} skipped (insufficient nodes)")
            except Exception as e:
                print(f"[ERROR] Action {i+1}/{action_number_total} failed with error: {e}")
                raise e
        self._make_single_output()
        print(f"[INFO] Random graph generation completed. {successful_actions}/100 actions were successful.")
        return True
    

    

    def _reset_all(self):
        print("[WARNING] resetting the generated graph and counters .....")
        self.program_graph = nx.DiGraph()
        self.loop_node_counter = 0
        self.op_counter = 0
        self.array_node_counter = 0
        self.visit_node_counter = 0
        self.write_node_counter = 0
        self.branch_node_counter = 0

    def generate_random_graph(self):
        try:
            ret_code = self._generate_random_graph()
        except Exception as e:
            print("[ERROR] encounter some errors during graph generation")
            self.print_node_list(ident="    ")
            raise e
        if not ret_code:
            print("[INFO] fails to generate random graph")
            return False
        return True

    def generate_random_c(self):
        # try:
        ret_code = self.generate_random_graph()
        # except Exception as e:
        #     print("[ERROR] encounter some errors during graph generation")
        #     self.print_node_list(ident="    ")
        #     raise e
        if ret_code:
            self.dump_cpp_std("output.cpp")

    
    def _set_loop_node_pragmas(self, loop_node):
        self.rand_pg_gen.generate_pragma_for_loop_node(loop_node=loop_node)

    def _set_design_cp_in_ns(self):
        return self.rand_pg_gen.generate_cp_ns()
        

    def __init__(self, seed = 42):
        super().__init__()
        self.seed = seed
        random.seed(seed)
        self.rand_type_gen = RandomTypeGenerator()
        self.rand_op_type_gen = RandomOpTypeGenerator()
        self.rand_pg_gen = RandomPragmaGenerator()

    def _copy_graph_and_insert_pragmas(self):
        """
        Override parent method to ensure different pragma generation for comparison files.
        This method creates two copies of the graph and inserts different random pragmas
        into each copy by using different random seeds.
        """
        import copy
        print("[INFO] call RandomGraphManager::_copy_graph_and_insert_pragmas")
        
        # Create deep copies of the graph to ensure node objects are independent
        self.program_graph_copy_1 = nx.MultiDiGraph()
        self.program_graph_copy_2 = nx.MultiDiGraph()
        
        # Deep copy nodes to ensure independence
        node_mapping_1 = {}
        node_mapping_2 = {}
        
        for node in self.program_graph.nodes():
            node_copy_1 = copy.deepcopy(node)
            node_copy_2 = copy.deepcopy(node)
            self.program_graph_copy_1.add_node(node_copy_1)
            self.program_graph_copy_2.add_node(node_copy_2)
            node_mapping_1[node] = node_copy_1
            node_mapping_2[node] = node_copy_2
        
        # Copy edges using the new node mappings
        for edge in self.program_graph.edges(data=True):
            source, target, data = edge
            self.program_graph_copy_1.add_edge(node_mapping_1[source], node_mapping_1[target], **data)
            self.program_graph_copy_2.add_edge(node_mapping_2[source], node_mapping_2[target], **data)

        # Save current random state
        current_state = random.getstate()
        
        # Generate pragmas for copy 1 with original seed
        random.seed(self.seed * 2 + 1)  # Use a derived seed for copy 1
        self._insert_pragmas_to_graph(self.program_graph_copy_1)
        
        # Generate pragmas for copy 2 with different seed
        random.seed(self.seed * 2 + 2)  # Use a different derived seed for copy 2
        self._insert_pragmas_to_graph(self.program_graph_copy_2)
        
        # Restore random state
        random.setstate(current_state)

        # Generate different clock period values
        random.seed(self.seed * 3 + 1)
        self.cp_1 = self._set_design_cp_in_ns()
        random.seed(self.seed * 3 + 2) 
        self.cp_2 = self._set_design_cp_in_ns()
        
        # Restore random state again
        random.setstate(current_state)

        print("[INFO] end call RandomGraphManager::_copy_graph_and_insert_pragmas")


