from enum import Enum, auto
from dataclasses import dataclass
from typing import Union


class OperationType(Enum):
    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"
    DIV = "DIV"
    MOD = "MOD"
    AND = "AND"
    OR = "OR"
    XOR = "XOR"
    NOT = "NOT"
    SHL = "SHL"
    SHR = "SHR"
    EQ = "EQ"
    NEQ = "NEQ"
    LT = "LT"
    GT = "GT"
    LE = "LE"
    GE = "GE"
    VISIT = "VISIT"
    WRITE = "WRITE"

class ResultDataType(Enum):
    AP_INT = "ap_int"
    AP_FIXED = "ap_fixed"
    AP_UINT = "ap_uint"

class QuantizationMode(Enum):
    """
    Enum for quantization modes.
    """
    AP_RND = "AP_RND"
    AP_RND_ZERO = "AP_RND_ZERO"
    AP_RND_MIN_INF = "AP_RND_MIN_INF"
    AP_RND_INF = "AP_RND_INF"
    AP_RND_CONV = "AP_RND_CONV"
    AP_TRN = "AP_TRN"
    AP_TRN_ZERO = "AP_TRN_ZERO"

class OverflowMode(Enum):
    AP_SAT = "AP_SAT"
    AP_SAT_ZERO = "AP_SAT_ZERO"
    AP_SAT_SYM = "AP_SAT_SYM"
    AP_WRAP = "AP_WRAP"
    AP_WRAP_SM = "AP_WRAP_SM"

# ram_1p, ram_1wnr, ram_2p, ram_s2p, ram_t2p, rom_1p, rom_2p, and rom_np
class BRAM_TYPE(Enum):
    RAM_1P = "RAM_1P"
    RAM_1WNR = "RAM_1WNR"
    RAM_2P = "RAM_2P"
    RAM_S2P = "RAM_S2P"
    RAM_T2P = "RAM_T2P"
    ROM_1P = "ROM_1P"
    ROM_2P = "ROM_2P"
    ROM_NP = "ROM_NP"



@dataclass
class Node:
    name: str

    def __hash__(self):
        return hash(self.name)

@dataclass
class OpNode(Node):
    op_type : OperationType
    result_type : ResultDataType
    result_width : int
    result_int_width_ap_fixed : int = 0
    result_wrap_mode : OverflowMode = OverflowMode.AP_SAT
    result_rounding_mode : QuantizationMode = QuantizationMode.AP_RND

    def to_dict(self):
        return {
            'name': self.name,
            'op_type': self.op_type.value if hasattr(self.op_type, 'value') else self.op_type,
            'result_type': self.result_type.value if hasattr(self.result_type, 'value') else self.result_type,
            'result_width': self.result_width,
            'result_int_width_ap_fixed': self.result_int_width_ap_fixed,
            'result_wrap_mode': self.result_wrap_mode,
            'result_rounding_mode': self.result_rounding_mode
        }
    
    def __hash__(self):
        return hash((self.name, self.op_type, self.result_type, self.result_width, 
                    self.result_int_width_ap_fixed, self.result_wrap_mode, self.result_rounding_mode))
    
@dataclass
class DepNode(Node):
    predecessor:OpNode
    successor:OpNode

    def __hash__(self):
        return hash(self.name)


@dataclass
class BranchNode(Node):
    def __hash__(self):
        return hash(self.name)

@dataclass
class LoopNode(Node):
    start_index : Union[int, 'OpNode']
    end_index : Union[int, 'OpNode']
    step : int

    is_pipelined : bool = False
    is_flattened : bool = False
    is_unrolled : bool = False
    is_fully_unrolled : bool = False
    unroll_factor : int = 1


    def get_loop_var_name(self):
        return f"{self.name}_loop_var"
    
    def __hash__(self):
        # Handle the case where start_index or end_index might be OpNode objects
        start_hash = hash(self.start_index) if isinstance(self.start_index, int) else hash(self.start_index.name)
        end_hash = hash(self.end_index) if isinstance(self.end_index, int) else hash(self.end_index.name)
        return hash((self.name, start_hash, end_hash, self.step))
    
    def check_pragma_status(self):
        if self.is_unrolled:
            if self.unroll_factor <= 1:
                raise ValueError("illegal loop node, when unrolled but with factor <= 1 "+\
                                 f"loop node: {self.__repr__()}")
        else:
            if self.unroll_factor != 1:
                raise ValueError("illegal loop node, when not unrolled but with factor != 1 "+\
                                 f"loop node: {self.__repr__()}")
            if self.is_fully_unrolled:
                raise ValueError("illegal loop node, when not unrolled but with fully unrolled set true "+\
                                 f"loop node: {self.__repr__()}")

@dataclass
class ArrayNode(Node):
    result_type : ResultDataType
    result_width : int
    result_int_width_ap_fixed : int
    length : int = 1024
    result_wrap_mode : OverflowMode = OverflowMode.AP_WRAP
    result_rounding_mode : QuantizationMode = QuantizationMode.AP_RND
    # #pragma HLS interface ap_memory storage_type=RAM_1P port=array_4
    memory_type : BRAM_TYPE = BRAM_TYPE.RAM_1P
    
    def __hash__(self):
        return hash((self.name, self.result_type, self.result_width, 
                    self.result_int_width_ap_fixed, self.length, 
                    self.result_wrap_mode, self.result_rounding_mode))
