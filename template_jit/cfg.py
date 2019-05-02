import collections
import dis
import opcode

from types import FunctionType
from typing import Iterable, Iterator, List, Tuple


Label = str


class Node:
    pass


class EntryNode(Node):
    pass


class ExitNode(Node):
    pass


class BasicBlock(Node):
    def __init__(self, label: Label, instructions: List[dis.Instruction]) -> None:
        if len(instructions) == 0:
            raise ValueError('Basic blocks cannot be empty')
        self.label = label
        self.instructions = instructions
        self.head = instructions[0]
        self.terminator = instructions[-1]

    def __str__(self) -> str:
        lines = [
            f'{self.label}:',
        ]
        for instr in self.instructions:
            if instr.arg is None:
                line = "  %-20s" % (instr.opname)
            elif instr.arg == instr.argval:
                line = "  %-20s %-4s" % (instr.opname, instr.arg)
            else:
                line = "  %-20s %-4s (%s)" % (instr.opname, instr.arg, instr.argval)
            line = line.rstrip()
            lines.append(line)
        return "\n".join(lines)


class CFGIterator:
    """Iterates through the basic blocks in a control flow graph in reverse
    post order (tsort).
    """

    def __init__(self, cfg: 'ControlFlowGraph') -> None:
        self.cfg = cfg
        self.queue: Deque[Node] = collections.deque([cfg.entry_node])
        self.visited: Set[Node] = set()

    def __iter__(self) -> 'CFGIterator':
        return self

    def __next__(self) -> BasicBlock:
        while self.queue:
            node = self.queue.popleft()
            if node in self.visited:
                continue
            succs = self.cfg.get_successors(node)
            self.visited.add(node)
            if isinstance(node, BasicBlock):
                terminator = node.terminator
                if terminator.opcode in CONDITIONAL_BRANCH_OPCODES:
                    true_block = self.cfg.blocks[terminator.true_branch]
                    false_block = self.cfg.blocks[terminator.false_branch]
                    # extendleft inserts into the deque in order, so items that
                    # are to appear at the front of the queue should appear at
                    # the end of this list
                    succs = false_block, true_block
                self.queue.extendleft(succs)
            else:
                self.queue.extendleft(succs)
                continue
            return node
        raise StopIteration


def labels(nodes):
    labels = []
    for node in nodes:
        if isinstance(node, BasicBlock):
            labels.append(node.label)
        else:
            labels.append(node.__class__.__name__)
    return labels


class WorkStack:
    def __init__(self) -> None:
        self.stack: List[Node] = []
        self.index: Set[Node] = set()

    def append(self, node: Node) -> None:
        if node in self.index:
            return
        self.stack.append(node)
        self.index.add(node)

    def extend(self, nodes: Iterable[Node]) -> None:
        for node in nodes:
            self.append(node)

    def has_node(self, node: Node) -> bool:
        return node in self.index

    def pop(self) -> Node:
        node = self.stack.pop()
        self.index.remove(node)
        return node

    def is_empty(self) -> bool:
        return len(self.stack) == 0


class PostOrderCFGIterator:
    """Iterates through the basic blocks in a control flow graph in post
    order.
    """

    def __init__(self, cfg: 'ControlFlowGraph') -> None:
        self.cfg = cfg
        self.stack = WorkStack()
        self.stack.append(cfg.entry_node)
        self.visited: Set[Node] = set()
        self.processed: Set[Node] = set()

    def __iter__(self) -> 'CFGIterator':
        return self

    def __next__(self) -> BasicBlock:
        while not self.stack.is_empty():
            node = self.stack.pop()
            if node in self.visited:
                continue
            succs = self.cfg.get_successors(node)
            if not isinstance(node, BasicBlock):
                self.stack.extend(succs)
                continue
            if not succs or node in self.processed:
                self.visited.add(node)
                return node
            self.stack.append(node)
            self.stack.extend(succs)
            self.processed.add(node)
        raise StopIteration


class ControlFlowGraph:
    def __init__(self) -> None:
        self.entry_node = EntryNode()
        self.exit_node = ExitNode()
        # src -> dst
        self.edges: Dict[Node, Set[Node]] = {
            self.entry_node: set(),
            self.exit_node: set(),
        }
        self.blocks: Dict[Label, BasicBlock] = {}

    def add_block(self, block: BasicBlock) -> None:
        self.edges[block] = set()
        self.blocks[block.label] = block

    def add_edge(self, src: Node, dst: Node) -> None:
        self.edges[src].add(dst)

    def get_successors(self, node: Node) -> Iterable[Node]:
        return self.edges.get(node, set())

    def __iter__(self) -> Iterator[BasicBlock]:
        return CFGIterator(self)

    def __str__(self) -> str:
        output = ['entry:']
        blocks = [node for node in self if isinstance(node, BasicBlock)]
        blocks = sorted(blocks, key=lambda b: b.label)
        for block in blocks:
            output.append(str(block))
        return "\n".join(output)


def build_initial_cfg(blocks: List[BasicBlock]) -> ControlFlowGraph:
    """Build a CFG from a list of basic blocks.

    Assumes that the blocks are in order, with the first block as the entry
    block.
    """
    cfg = ControlFlowGraph()
    if not blocks:
        return cfg
    cfg.add_edge(cfg.entry_node, blocks[0])
    block_index = {block.label: block for block in blocks}
    for i, block in enumerate(blocks):
        cfg.add_block(block)
        # Outgoing edges are as follows if the terminator is a:
        #   - Direct branch      => block of branch target
        #   - Conditional branch => block of branch target and next block
        #   - Return             => exit node
        #   - Otherwise          => next block
        terminator = block.terminator
        if terminator.opname == "RETURN_VALUE":
            cfg.add_edge(block, cfg.exit_node)
        elif terminator.opcode in CONDITIONAL_BRANCH_OPCODES:
            cfg.add_edge(block, block_index[terminator.true_branch])
            cfg.add_edge(block, block_index[terminator.false_branch])
        else:
            cfg.add_edge(block, blocks[i + 1])
    return cfg


DIRECT_BRANCH_OPCODES = {
    opcode.opmap["RETURN_VALUE"],
}


CONDITIONAL_BRANCH_OPCODES = {
    opcode.opmap["POP_JUMP_IF_FALSE"],
}


BRANCH_OPCODES = DIRECT_BRANCH_OPCODES | CONDITIONAL_BRANCH_OPCODES


# Branching instructions for which the argument is an absolute offset
ABSOLUTE_BRANCH_OPCODES = {
    opcode.opmap["POP_JUMP_IF_FALSE"],
}


# Branching instructions for which the argument is relative to the next
# instruction offset
RELATIVE_BRANCH_OPCODES = {}


INSTRUCTION_SIZE_B = 2


def compute_block_boundaries(func: FunctionType) -> List[Tuple[int, int]]:
    """Compute the offsets of basic blocks.

    An offset starts a new basic block if:
      - It is the first instruction
      - It is the target of a branch
      - It follows a  branch

    Returns:
        A list of half open intervals, where each interval contains a
        basic block.
    """
    code_len= len(func.__code__.co_code)
    if code_len == 0:
        return []
    block_starts = {0}
    last_offset = code_len
    for instr in dis.get_instructions(func):
        opcode = instr.opcode
        next_instr_offset = instr.offset + INSTRUCTION_SIZE_B
        if opcode in BRANCH_OPCODES and next_instr_offset < last_offset:
            block_starts.add(next_instr_offset)
        if opcode in RELATIVE_BRANCH_OPCODES:
            block_starts.add(next_instr_offset + instr.argument)
        elif opcode in ABSOLUTE_BRANCH_OPCODES:
            block_starts.add(instr.argval)
    sorted_block_starts = sorted(block_starts)
    sorted_block_starts.append(code_len)
    boundaries: List[Tuple[int, int]] = []
    for i in range(0, len(sorted_block_starts) - 1):
        boundaries.append((sorted_block_starts[i], sorted_block_starts[i + 1]))
    return boundaries


def make_branch_symbolic(instr, labels):
    if instr.opcode == opcode.opmap["POP_JUMP_IF_FALSE"]:
        instr.true_branch = labels[instr.offset + INSTRUCTION_SIZE_B]
        instr.false_branch = labels[instr.arg]


def build_cfg(func: FunctionType) -> ControlFlowGraph:
    """Build a CFG from the supplied function."""
    # Label blocks
    labels = {}
    block_boundaries = compute_block_boundaries(func)
    for i, interval in enumerate(block_boundaries):
        labels[interval[0]] = f'bb{i}'
    # Construct blocks
    blocks = []
    boundary_iter = iter(block_boundaries)
    instrs = []
    boundary = next(boundary_iter)
    for instr in dis.get_instructions(func):
        if instr.opcode in CONDITIONAL_BRANCH_OPCODES:
            make_branch_symbolic(instr, labels)
        if instr.offset < boundary[1]:
            # Instruction belongs to the current block
            instrs.append(instr)
        else:
            # Instruction starts a new block
            block = BasicBlock(labels[boundary[0]], instrs)
            blocks.append(block)
            instrs = [instr]
            boundary = next(boundary_iter)
    if instrs:
        block = BasicBlock(labels[boundary[0]], instrs)
        blocks.append(block)
    return build_initial_cfg(blocks)
