# --- interpreter/interpreter.py ---

from Myparser.graphml_parser import Node, Edge
from typing import Dict, Any, List
import os
import json

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "logic_templates.json")

def load_logic_templates():
    if os.path.exists(TEMPLATE_PATH):
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def match_template(label: str, role: str, templates: Dict[str, Any]) -> str | None:
    """
    Finds a logic template by first looking for an exact keyword match,
    then falling back to a partial keyword match within the node's role.
    """
    label_lower = label.lower()
    role_templates = templates.get(role, [])

    # 1. Exact match pass
    for entry in role_templates:
        for kw in entry.get("keywords", []):
            if label_lower == kw.lower():
                return entry["logic"]

    # 2. Partial match pass
    for entry in role_templates:
        for kw in entry.get("keywords", []):
            if kw.lower() in label_lower:
                return entry["logic"]

    return None

def interpret_flowchart(
    nodes: List[Node],
    edges: List[Edge],
    roles: Dict[str, str],
    traversal_order: List[str],
    diagram_type: str
) -> Dict[str, Any]:
    templates = load_logic_templates()
    id_to_node = {node.id: node for node in nodes}

    # Edge map: from -> list of {"target_id", "label"}
    next_map = {}
    for edge in edges:
        next_map.setdefault(edge.source, []).append({
            "target_id": edge.target,
            "label": (edge.label or "").strip().lower()
        })

    # Final output
    flowchart_data = {
        "diagram_type": diagram_type,
        "nodes": [
            {
                "id": n.id,
                "label": n.label.strip(),
                "shape": n.shape.strip(),
                "role": roles.get(n.id, "Unknown")
            } for n in nodes
        ],
        "edges": [
            {
                "source_id": e.source,
                "target_id": e.target,
                "label": (e.label or "").strip()
            } for e in edges
        ],
        "execution_flow": []
    }

    visited = set()

    def visit_node(node_id: str):
        if node_id in visited:
            return
        visited.add(node_id)

        node = id_to_node.get(node_id)
        if not node:
            return

        label = node.label.strip()
        role = roles.get(node.id, "Unknown")
        shape = node.shape.strip()
        next_nodes = next_map.get(node.id, [])
        logic = match_template(label, role, templates)

        step: Dict[str, Any] = {
            "node_id": node.id,
            "label": label,
            "role": role,
            "type": shape,
            "logic_template": logic,
            "flow_type": "action",  # Default fallback
        }

        # Determine flow type from role
        role_lower = role.lower()
        if role_lower == "start":
            step["flow_type"] = "start"
        elif role_lower == "end":
            step["flow_type"] = "end"
        elif role_lower == "decision":
            step["flow_type"] = "decision"
            step["branches"] = []
            for out in next_nodes:
                target = id_to_node.get(out["target_id"])
                if target:
                    condition_raw = out["label"]
                    condition = ""
                    if condition_raw in ["yes", "true", "y"]:
                        condition = "yes"
                    elif condition_raw in ["no", "false", "n"]:
                        condition = "no"

                    step["branches"].append({
                        "condition_label": condition_raw,
                        "condition": condition,
                        "target_node_id": out["target_id"],
                        "target_label": target.label.strip()
                    })
        elif role_lower == "loop":
            step["flow_type"] = "loop"
        elif role_lower in ("input", "output", "process", "task"):
            step["flow_type"] = role_lower
        elif role_lower == "jump":
            step["flow_type"] = "jump"
        elif role_lower in ("class", "interface", "method", "attribute"):
            step["flow_type"] = "uml_element"
        else:
            step["flow_type"] = "generic_action"

        # Add next_id if exactly one successor (for linear flows)
        if len(next_nodes) == 1 and role_lower not in ("decision", "end"):
            step["next_id"] = next_nodes[0]["target_id"]

        flowchart_data["execution_flow"].append(step)

        # Continue traversal
        for out in next_nodes:
            visit_node(out["target_id"])

    # Use the traversal order to guide processing
    for nid in traversal_order:
        visit_node(nid)

    return flowchart_data
