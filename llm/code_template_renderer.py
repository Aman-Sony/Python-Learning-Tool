from jinja2 import Environment, FileSystemLoader, TemplateNotFound, TemplateError
import os
from datetime import datetime
from typing import Dict, Any

class CodeTemplateRenderer:
    def __init__(self, template_dir: str, default_template: str = "flowchart_base.j2"):
        """
        Renders code from structured diagram data using Jinja2 templates.
        Args:
            template_dir: path to templates folder (should contain .j2 files)
            default_template: fallback template name (default: flowchart_base.j2)
        """
        self.template_dir = template_dir
        self.default_template = default_template
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        self.env.filters['slugify_label'] = self._slugify_label

    def _slugify_label(self, label: str) -> str:
        """Converts a label into a Python-friendly identifier."""
        import re
        label = re.sub(r'\W+', '_', label).strip('_')
        if not label:
            return "unnamed_function"
        if not label[0].isalpha() and label[0] != '_':
            label = '_' + label
        return label.lower()

    def get_template_for_type(self, diagram_type: str) -> str:
        """
        Selects a template file based on diagram type.
        Extend this map as new diagram types are supported.
        """
        mapping = {
            "Flowchart": "flowchart_base.j2",
            "UML Class Diagram": "uml_class_base.j2",   # You can create this template next
            "ER Diagram": "er_diagram_base.j2",
            "UML Sequence Diagram": "sequence_diagram_base.j2"
        }
        return mapping.get(diagram_type.strip(), self.default_template)

    def render_code(
        self,
        flowchart_data: Dict[str, Any],
        override_template: str = None
    ) -> str:
        """
        Renders code from the diagram using appropriate template.
        Args:
            flowchart_data: structured dict of parsed and interpreted diagram
            override_template: if provided, overrides default template logic
        Returns:
            Rendered code string
        """
        diagram_type = flowchart_data.get("diagram_type", "").strip()
        template_name = override_template or self.get_template_for_type(diagram_type)

        try:
            template = self.env.get_template(template_name)
        except TemplateNotFound:
            raise FileNotFoundError(f"Template '{template_name}' not found in {self.template_dir}")
        except Exception as e:
            raise RuntimeError(f"Error loading template: {e}")

        context = {
            "flowchart_data": flowchart_data,
            "current_datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        try:
            return template.render(context)
        except TemplateError as te:
            raise RuntimeError(f"Template rendering failed: {te}")

# ✅ Example usage (for test run)
if __name__ == "__main__":
    dummy_flowchart_data = {
        "diagram_type": "Flowchart",
        "nodes": [
            {"id": "n1", "label": "Start", "shape": "ellipse", "role": "Start"},
            {"id": "n2", "label": "Input Value", "shape": "parallelogram", "role": "Input"},
            {"id": "n3", "label": "Print Result", "shape": "parallelogram", "role": "Output"},
            {"id": "n4", "label": "End", "shape": "ellipse", "role": "End"},
        ],
        "edges": [],
        "execution_flow": [
            {"node_id": "n1", "label": "Start", "role": "Start", "type": "ellipse", "flow_type": "start"},
            {"node_id": "n2", "label": "Input Value", "role": "Input", "type": "parallelogram", "flow_type": "input", "next_id": "n3"},
            {"node_id": "n3", "label": "Print Result", "role": "Output", "type": "parallelogram", "flow_type": "output", "next_id": "n4"},
            {"node_id": "n4", "label": "End", "role": "End", "type": "ellipse", "flow_type": "end"},
        ]
    }

    template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
    renderer = CodeTemplateRenderer(template_dir)
    try:
        output = renderer.render_code(dummy_flowchart_data)
        print(output)
    except Exception as e:
        print("❌ Error:", e)
