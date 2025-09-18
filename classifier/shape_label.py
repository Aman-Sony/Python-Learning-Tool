# --- classifier/shape_label.py ---

from Myparser.graphml_parser import Node
import re

def classify_node_roles(nodes: list[Node], diagram_type: str, edges: list = None) -> dict:
    node_roles = {}
    edge_label_map = {}

    if edges:
        for edge in edges:
            key = (edge.source, edge.target)
            edge_label_map[key] = (edge.label or "").strip().lower()

    for node in nodes:
        label = (node.label or "").strip().lower()
        shape = (node.shape or "").strip().lower()
        role = "Unknown"

        # ===== FLOWCHART DETECTION =====
        if diagram_type == "Flowchart":
            if shape in ["ellipse", "terminator", "start1"] or "start" in label:
                role = "Start"
            elif "end" in label or "stop" in label:
                role = "End"
            elif shape == "diamond" or "?" in label or any(kw in label for kw in ["if", "check", "compare", "do", "can", "should", "else"]):
                role = "Decision"
            elif any(kw in label for kw in ["repeat", "loop", "again", "while", "for"]):
                role = "Loop"
            elif shape in ["parallelogram", "input", "document"] or any(kw in label for kw in ["input", "scan", "read", "enter"]):
                role = "Input"
            elif shape in ["output", "display"] or any(kw in label for kw in ["print", "display", "output", "show"]):
                role = "Output"
            elif shape in ["rectangle", "roundrectangle", "box"]:
                role = "Process"
            elif shape in ["offpageconnector", "jump", "connector"]:
                role = "Jump"
            else:
                # fallback guess based on label
                if "print" in label:
                    role = "Output"
                elif "input" in label or "scan" in label:
                    role = "Input"
                else:
                    role = "Process"

        # ===== UML CLASS DIAGRAM DETECTION =====
        elif diagram_type == "UML Class Diagram":
            if any(tag in label for tag in ["<<interface>>", "<<abstract>>", "interface", "abstract"]):
                role = "Interface"
            elif label.startswith("class") or "class" in label:
                role = "Class"
            elif "{" in label and "}" in label:
                role = "Class"  # typical yEd auto format
            elif re.match(r"^\w+\(.*\)$", label):
                role = "Method"
            elif ":" in label or "=" in label or any(t in label for t in ["int", "str", "float", "bool", "string", "boolean", "void"]):
                role = "Attribute"
            elif shape == "rectangle" and label.isidentifier():
                role = "Attribute"
            else:
                role = "Unknown UML Part"

        # ===== FUTURE DIAGRAMS (Sequence, ER, Activity) =====
        elif diagram_type == "UML Sequence Diagram":
            if "lifeline" in label:
                role = "Lifeline"
            elif "message" in label or "-->" in label:
                role = "Message"
            else:
                role = "Actor"

        elif diagram_type == "ER Diagram":
            if "entity" in label:
                role = "Entity"
            elif "attribute" in label:
                role = "Attribute"
            elif "relation" in label or "relationship" in label:
                role = "Relationship"

        elif diagram_type == "UML Activity Diagram":
            if "fork" in label:
                role = "Fork"
            elif "join" in label:
                role = "Join"
            elif "action" in label or "activity" in label:
                role = "Action"
            elif "swimlane" in label:
                role = "Swimlane"

        node_roles[node.id] = role

    return node_roles
