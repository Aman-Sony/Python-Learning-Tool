# classifier/diagram_type.py

from Myparser.graphml_parser import Node

def classify_diagram_type(nodes: list[Node]) -> str:
    uml_keywords = ['class', 'attribute', 'method', '<<interface>>', '<<abstract>>']
    flowchart_keywords = ['start', 'stop', 'input', 'output', 'decision', 'process', 'end']
    sequence_keywords = ['activate', 'deactivate', 'message', 'return', 'lifeline']
    er_keywords = ['entity', 'relation', 'attribute', 'primary key', 'foreign key']
    activity_keywords = ['activity', 'fork', 'join', 'action', 'swimlane']

    scores = {
        "UML Class Diagram": 0,
        "Flowchart": 0,
        "UML Sequence Diagram": 0,
        "ER Diagram": 0,
        "UML Activity Diagram": 0
    }

    for node in nodes:
        label = (node.label or "").lower()
        shape = (node.shape or "").lower()

        # === Step 1: Strong shape-based identification ===
        if shape in ['diamond', 'parallelogram', 'ellipse', 'document', 'offpageconnector']:
            scores["Flowchart"] += 2
        elif shape in ['rectangle', 'roundrectangle']:
            # Ambiguous: could be UML class, flowchart process, or ER entity
            scores["Flowchart"] += 1
            scores["UML Class Diagram"] += 1
        elif shape == 'hexagon':
            scores["UML Activity Diagram"] += 2

        # === Step 2: Keyword matching ===
        if any(kw in label for kw in flowchart_keywords):
            scores["Flowchart"] += 2
        if any(kw in label for kw in uml_keywords):
            scores["UML Class Diagram"] += 2
        if any(kw in label for kw in sequence_keywords):
            scores["UML Sequence Diagram"] += 2
        if any(kw in label for kw in er_keywords):
            scores["ER Diagram"] += 2
        if any(kw in label for kw in activity_keywords):
            scores["UML Activity Diagram"] += 2

        # === Step 3: Label structure hints (UML) ===
        if "::" in label or ("class" in label and "{" in label and "}" in label):
            scores["UML Class Diagram"] += 3
        if "(" in label and ")" in label and ":" in label:
            # Might be method signature
            scores["UML Class Diagram"] += 2

    # === Step 4: Final decision ===
    top_type, top_score = max(scores.items(), key=lambda x: x[1])
    second_score = sorted(scores.values(), reverse=True)[1]

    # Threshold to avoid misclassification
    if top_score == 0:
        return "Unknown"

    if top_type == "UML Class Diagram" and scores["Flowchart"] > 2 and (top_score - scores["Flowchart"]) <= 1:
        # Probably misclassified due to generic words like 'int', 'method'
        return "Flowchart"

    if top_score - second_score >= 2:
        return top_type
    elif top_score >= 4:
        return top_type  # Allow if strong enough signal
    else:
        return "Flowchart"  # Default fallback for ambiguous diagrams
