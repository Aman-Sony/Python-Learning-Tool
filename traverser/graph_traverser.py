# --- traverser/graph_traverser.py ---

from Myparser.graphml_parser import Node, Edge
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Set


def find_start_node(nodes: List[Node]) -> str | None:
    """Attempts to find the most likely start node."""
    for node in nodes:
        label = (node.label or "").lower()
        shape = (node.shape or "").lower()
        if "start" in label or shape in ("start1", "terminator", "ellipse"):
            return node.id
    return nodes[0].id if nodes else None


def build_adjacency(edges: List[Edge]) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], str]]:
    adj = defaultdict(list)
    edge_labels = {}
    for edge in edges:
        adj[edge.source].append(edge.target)
        edge_labels[(edge.source, edge.target)] = (edge.label or "").strip().lower()
    return adj, edge_labels


def traverse_graph(nodes: List[Node], edges: List[Edge]) -> Tuple[List[str], Dict[str, str]]:
    """
    Returns a traversal order of node IDs (BFS-based for now),
    and inferred rough roles (deprecated, now handled by classifier).
    """
    node_map = {n.id: n for n in nodes}
    adj, edge_labels = build_adjacency(edges)
    start_node = find_start_node(nodes)

    if not start_node:
        return [], {}

    visited: Set[str] = set()
    order: List[str] = []
    queue = deque([start_node])
    visit_count = defaultdict(int)

    while queue:
        current = queue.popleft()

        if current not in visited:
            visited.add(current)
            order.append(current)

        for neighbor in adj.get(current, []):
            if visit_count[neighbor] < 3:  # Prevent infinite loop in cycle-heavy diagrams
                queue.append(neighbor)
                visit_count[neighbor] += 1

    # Detect unreachable nodes and append at end
    all_node_ids = {node.id for node in nodes}
    unreachable = list(all_node_ids - set(order))
    if unreachable:
        order.extend(unreachable)

    # Legacy role fallback (now discouraged, kept for backward compatibility)
    roles = {node.id: "Unknown" for node in nodes}

    return order, roles
