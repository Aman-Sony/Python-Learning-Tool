import xml.etree.ElementTree as ET
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Node:
    def __init__(self, node_id, label="", shape="", metadata=None):
        self.id = node_id
        self.label = label
        self.shape = shape
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Node(id='{self.id}', label='{self.label}', shape='{self.shape}')"

class Edge:
    def __init__(self, source, target, label=""):
        self.source = source
        self.target = target
        self.label = label

    def __repr__(self):
        return f"Edge(source='{self.source}', target='{self.target}', label='{self.label}')"

def parse_graphml(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        ns = {
            'graphml': 'http://graphml.graphdrawing.org/xmlns',
            'y': 'http://www.yworks.com/xml/graphml'
        }

        nodes = []
        edges = []

        graph = root.find("graphml:graph", ns)
        if graph is None:
            logger.warning("No <graph> element found in GraphML.")
            return [], []

        for node in graph.findall("graphml:node", ns):
            node_id = node.attrib['id']
            label = ""
            shape = ""
            metadata = {}

            # Label
            label_elem = node.find(".//y:NodeLabel", ns)
            if label_elem is not None:
                label = (label_elem.text or "").strip()

            # Shape from <y:Shape>
            shape_elem = node.find(".//y:Shape", ns)
            if shape_elem is not None:
                shape = shape_elem.attrib.get("type", "").lower()

            # Shape from <y:FlowchartShape>
            if not shape:
                fc_shape = node.find(".//y:FlowchartShape", ns)
                if fc_shape is not None:
                    shape = fc_shape.attrib.get("type", "").lower()

            # Shape from <y:GenericNode>
            if not shape:
                generic = node.find(".//y:GenericNode", ns)
                if generic is not None:
                    config = generic.attrib.get("configuration", "").lower()
                    if "terminator" in config or "start" in config:
                        shape = "ellipse"
                    elif "decision" in config:
                        shape = "diamond"
                    elif "data" in config or "input" in config:
                        shape = "parallelogram"
                    elif "process" in config:
                        shape = "rectangle"
                    else:
                        shape = config.split(".")[-1]  # fallback

            # Optional: Save shape label or raw config
            metadata['raw_shape'] = shape
            metadata['raw_config'] = generic.attrib.get("configuration", "") if generic is not None else ""

            nodes.append(Node(node_id, label, shape, metadata))

        for edge in graph.findall("graphml:edge", ns):
            source = edge.attrib['source']
            target = edge.attrib['target']
            label = ""

            label_elem = edge.find(".//y:EdgeLabel", ns)
            if label_elem is not None:
                label = (label_elem.text or "").strip().lower()

            edges.append(Edge(source, target, label))

        logger.info(f"Parsed {len(nodes)} nodes and {len(edges)} edges.")
        return nodes, edges

    except Exception as e:
        logger.error(f"Error parsing GraphML file: {e}")
        return [], []
