

from node import LoopNode
import random

class RandomPragmaGenerator:

    def _random_binary_choice(self):
        # do a equal random binary choice that return boolean
        return random.choice([True, False])

    def generate_pragma_for_loop_node(self, loop_node:LoopNode):
        if not isinstance(loop_node, LoopNode):
            raise TypeError(f"unexpected type for loop node type is {type(loop_node)}")
        loop_node.is_pipelined = self._random_binary_choice()
        loop_node.is_flattened = self._random_binary_choice()
        loop_node.is_unrolled = self._random_binary_choice()

        if not loop_node.is_unrolled:
            loop_node.unroll_factor = 1
            loop_node.is_fully_unrolled = False
        else:
            loop_node.is_fully_unrolled = self._random_binary_choice()
            if not loop_node.is_fully_unrolled:
                loop_node.unroll_factor = random.choice([2, 4, 8, 16, 32])
            else:
                loop_node.unroll_factor = 999
        loop_node.check_pragma_status()

    def generate_pragma_for_loop_node_no_opt(self, loop_node:LoopNode):
        if not isinstance(loop_node, LoopNode):
            raise TypeError(f"unexpected type for loop node type is {type(loop_node)}")
        loop_node.is_pipelined = False
        loop_node.is_flattened = False
        loop_node.is_fully_unrolled = False
        loop_node.unroll_factor = 1
        loop_node.is_unrolled = False
        loop_node.check_pragma_status()

    def generate_pragma_for_loop_node_flatten_only(self, loop_node:LoopNode):
        loop_node.is_pipelined = False
        loop_node.is_flattened = True
        loop_node.is_unrolled = False
        loop_node.is_fully_unrolled = False
        loop_node.unroll_factor = 1
        loop_node.check_pragma_status()

    def generate_pragma_for_loop_node_full_unroll_only(self, loop_node:LoopNode):
        loop_node.is_pipelined = False
        loop_node.is_flattened = False
        loop_node.is_unrolled = True
        loop_node.is_fully_unrolled = True
        loop_node.unroll_factor = 999
        loop_node.check_pragma_status()

    def generate_pragma_for_loop_node_pipeline_only(self, loop_node:LoopNode):
        loop_node.is_pipelined = True
        loop_node.is_flattened = False
        loop_node.is_unrolled = False
        loop_node.is_fully_unrolled = False
        loop_node.unroll_factor = 1
        loop_node.check_pragma_status()


    def generate_cp_ns(self):
        return random.choice([1,2,3,4,5,6,7,8,9,10])