import sys
import os
import streamlit as st
import re
import json
import logging
import pandas as pd
from pathlib import Path
import hashlib
import threading
import queue
import time


# --- Redis Setup ---
try:
    import redis
    r = redis.Redis(host='localhost', port=6379, db=0)
except ImportError:
    r = None
    st.warning("Redis is not installed. Run 'pip install redis' and ensure Redis server is running.")

def get_diagram_hash(flowchart_data):
    flowchart_json = json.dumps(flowchart_data, sort_keys=True)
    return hashlib.sha256(flowchart_json.encode('utf-8')).hexdigest()

def cache_generated_code(diagram_hash, code, metadata=None, source="unknown"):
    if r:
        key = f"{diagram_hash}_{source}"
        value = {
            "code": code,
            "metadata": metadata or {},
            "source": source,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        r.set(key, json.dumps(value))

def get_cached_code(diagram_hash, source=None):
    if r:
        if source:
            key = f"{diagram_hash}_{source}"
            val = r.get(key)
            if val:
                return json.loads(val)
        else:
            gemini_val = r.get(f"{diagram_hash}_gemini")
            ollama_val = r.get(f"{diagram_hash}_ollama")
            result = {}
            if gemini_val:
                result['gemini'] = json.loads(gemini_val)
            if ollama_val:
                result['ollama'] = json.loads(ollama_val)
            return result if result else None
    return None

def make_serializable(obj):
    if isinstance(obj, list):
        
        return [make_serializable(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif hasattr(obj, "__dict__"):
        return vars(obj)
    return obj

def extract_pure_code(code_string: str) -> str:
    if not code_string:
        return ""
    code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", code_string, re.DOTALL)
    code = "\n".join(code_blocks).strip() if code_blocks else code_string.strip()
    return strip_comments(code)

def strip_comments(code: str) -> str:
    """
    Remove comments from Python code while preserving docstrings and string literals.
    """
    if not code:
        return code
    
    # Pattern to match comments (lines starting with #, but not inside strings)
    # This is a simplified approach - for more robust comment removal, we'd need a proper parser
    lines = code.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Skip empty lines
        if not line.strip():
            cleaned_lines.append(line)
            continue
            
        # Check if line is a comment (starts with # after optional whitespace)
        stripped_line = line.lstrip()
        if stripped_line.startswith('#'):
            # Skip comment lines
            continue
            
        # Check for inline comments (but preserve string literals)
        if '#' in line:
            # Simple check to avoid removing # inside strings
            # This is a basic approach - for production use, consider using ast module
            in_string = False
            string_char = None
            new_line = []
            
            for char in line:
                if char in ('"', "'") and not in_string:
                    in_string = True
                    string_char = char
                    new_line.append(char)
                elif char == string_char and in_string:
                    in_string = False
                    string_char = None
                    new_line.append(char)
                elif char == '#' and not in_string:
                    break  # Stop at the first comment outside strings
                else:
                    new_line.append(char)
            
            cleaned_line = ''.join(new_line).rstrip()
            if cleaned_line.strip():  # Only add if not empty after stripping
                cleaned_lines.append(cleaned_line)
        else:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def is_placeholder_code(code: str) -> bool:
    if not code:
        return True
    return any(keyword in code for keyword in ["simulate_output(", "pass", "# TODO", "# Placeholder", "return True"])

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from Myparser.graphml_parser import parse_graphml
    from classifier.diagram_type import classify_diagram_type
    from classifier.shape_label import classify_node_roles
    from traverser.graph_traverser import traverse_graph
    from interpreter.interpreter import interpret_flowchart
    from llm.ollama_client import generate_code_with_ollama
    from llm.code_template_renderer import CodeTemplateRenderer
except ImportError as e:
    st.error(f"Error importing custom modules: {e}. Make sure your project structure is correct and all modules are in the Python path.")
    st.stop()

ollama_queue = queue.Queue()
ollama_results = {}

def ollama_worker():
    while True:
        try:
            task = ollama_queue.get(timeout=1)
            if task is None:
                break
            diagram_hash, prompt = task
            try:
                raw_output = generate_code_with_ollama(prompt, model="qwen3:0.6b")
                code = extract_pure_code(raw_output)
                cache_generated_code(
                    diagram_hash,
                    code,
                    metadata={
                        "source": "ollama",
                        "timestamp": pd.Timestamp.now().isoformat(),
                        "raw_output": raw_output
                    },
                    source="ollama"
                )
                ollama_results[diagram_hash] = {
                    "code": code,
                    "raw_output": raw_output,
                    "status": "completed"
                }
            except Exception as e:
                logger.error(f"Ollama generation error for {diagram_hash}: {e}")
                ollama_results[diagram_hash] = {
                    "error": str(e),
                    "status": "error"
                }
            ollama_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Ollama worker error: {e}")

if "ollama_worker_started" not in st.session_state:
    threading.Thread(target=ollama_worker, daemon=True).start()
    st.session_state.ollama_worker_started = True

# =============================================================================
# CUSTOM CSS STYLING WITH DARK MODE THEME (NO TOGGLE)
# =============================================================================

def load_custom_css():
    """
    Loads custom CSS styles with fixed dark mode theme and no blank space.
    """
    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');
        
        /* FIXED: Remove all padding and margins that cause blank space */
        .main .block-container {
            font-family: 'Inter', sans-serif;
            max-width: 100% !important;
            padding: 0 !important;
            margin: 0 !important;
            height: 100vh !important;
            overflow: hidden !important;
            background-color: #0E1117 !important;
            color: #FAFAFA !important;
        }

        html, body, .main, [data-testid="stAppViewContainer"] {
            height: 100vh !important;
            overflow: hidden !important;
            background-color: #0E1117 !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        
        /* FIXED: Remove all spacing from Streamlit containers */
        .st-emotion-cache-1cpx91a,
        .st-emotion-cache-6q9sum, 
        .st-emotion-cache-16p0e1w,
        .st-emotion-cache-z5fcl4,
        .st-emotion-cache-18e3th9,
        .st-emotion-cache-1d391kg,
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] > div:first-child {
            background-color: #0E1117 !important;
            padding: 0 !important;
            margin: 0 !important;
            height: 100vh !important;
            overflow: hidden !important;
        }

        /* FIXED: Header with exact positioning - no gaps */
        .main-header {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 1000;
            text-align: center;
            padding: 1rem 2rem;
            background: linear-gradient(135deg, #1A1A2E 0%, #16213E 50%, #0F3460 100%);
            color: #ECF0F1;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            height: 80px;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .main-header h1 {
            font-weight: 600;
            margin: 0;
            font-size: 1.6rem;
            line-height: 1.2;
            color: #FAFAFA;
        }
        
        .main-header p {
            font-weight: 300;
            opacity: 0.9;
            font-size: 0.85rem;
            margin: -0.5rem 0 0 0; /* Updated to move text up */
            color: #FAFAFA;
        }

        /* FIXED: Main content positioned with a small gap below header */
        .main-content {
            position: fixed;
            top: 90px; /* Updated to move content down */
            left: 0;
            right: 0;
            bottom: 40px;
            padding: 0.25rem;
            display: flex;
            gap: 0.25rem;
            overflow: hidden;
            background-color: #0E1117;
            margin: 0 !important;
        }

        .input-panel, .results-panel {
            flex: 1;
            display: flex;
            flex-direction: column;
            height: 100%;
            background: #1E1E1E;
            border-radius: 12px;
            padding: 1rem;
            border: 1px solid #333333;
            overflow: hidden;
        }

        .main-content h3 {
            color: #FAFAFA;
            margin: 0 0 0.5rem 0;
            font-size: 1.1rem;
            font-weight: 600;
        }

        .main-content h4 {
            color: #FAFAFA;
            margin: 0.8rem 0 0.4rem 0;
            font-size: 1rem;
            font-weight: 500;
        }

        .stTextArea textarea {
            border-radius: 8px !important;
            border: 2px solid #333333 !important;
            font-family: 'JetBrains Mono', monospace !important;
            background-color: #262626 !important;
            color: #FAFAFA !important;
            height: 180px !important;
            resize: none !important;
            font-size: 14px !important;
        }
        
        .stTextArea textarea:focus {
            border-color: #3B82F6 !important;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
        }

        .stTextArea label {
            color: #FAFAFA !important;
            font-weight: 500 !important;
            margin-bottom: 0.3rem !important;
        }

        .stFileUploader {
            margin-top: 0.5rem;
        }

        .stFileUploader > div {
            border: 2px dashed #333333;
            border-radius: 8px;
            background: #262626;
            padding: 0.8rem;
            text-align: center;
        }

        .stFileUploader label {
            color: #FAFAFA !important;
            font-weight: 500 !important;
        }

        .stButton > button {
            background: linear-gradient(135deg, #3B82F6, #1D4ED8) !important; /* Updated to blue gradient */
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 0.4rem 1.2rem !important;
            font-weight: 500 !important;
            font-family: 'Inter', sans-serif !important;
            transition: all 0.3s ease !important;
            width: 100% !important;
            margin-top: 0.3rem !important;
            font-size: 14px !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important; /* Updated hover shadow to blue */
        }
        
        .stDownloadButton > button {
            background: linear-gradient(135deg, #10B981, #047857) !important;
            border-radius: 8px !important;
            padding: 0.4rem 1.2rem !important;
            font-weight: 500 !important;
            width: 100% !important;
            color: white !important;
            border: none !important;
            margin-top: 0.3rem !important;
            font-size: 14px !important;
        }

        .explanation-container {
            flex: 1;
            background: #262626;
            border: 1px solid #333333;
            border-radius: 8px;
            padding: 0.8rem;
            overflow-y: auto;
            margin-top: 0.3rem;
            max-height: 180px;
        }

        .explanation-container p {
            color: #FAFAFA;
            line-height: 1.6;
            margin: 0;
            font-size: 14px;
        }

        .info-box {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 1rem;
            border-radius: 12px;
            text-align: center;
            margin: 0.5rem 0;
        }

        .info-box h4 {
            margin: 0 0 0.3rem 0;
            color: white;
            font-size: 1rem;
        }

        .info-box p {
            margin: 0;
            opacity: 0.9;
            font-size: 14px;
        }

        .navigation-instructions {
            background: linear-gradient(135deg, #4F46E5, #7C3AED);
            color: white;
            padding: 0.8rem;
            border-radius: 8px;
            margin-top: 0.3rem;
            font-size: 13px;
            font-weight: 500;
            text-align: center;
        }

        .navigation-instructions strong {
            color: #FEF3C7;
        }

        .main-footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: linear-gradient(135deg, #1E1E1E, #2D2D2D);
            border-top: 1px solid #333333;
            padding: 0.4rem;
            text-align: center;
            color: #FAFAFA;
            font-size: 0.75rem;
            font-weight: 400;
            z-index: 1000;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        /* FIXED: Hide ALL Streamlit default elements that cause spacing */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .css-1d391kg {display: none !important;}
        .css-18e3th9 {display: none !important;}
        .css-1lcbmhc {display: none !important;}
        .css-1y4p8pa {display: none !important;}
        div.css-1r6slb0 {display: none !important;}
        div.css-12ttj6m {display: none !important;}
        div.css-1wivap2 {display: none !important;}
        .st-emotion-cache-z5fcl4 {padding-top: 0 !important; margin-top: 0 !important;}
        .st-emotion-cache-18e3th9 {padding-top: 0 !important; margin-top: 0 !important;}
        .st-emotion-cache-1d391kg {padding-top: 0 !important; margin-top: 0 !important;}
        .block-container {padding-top: 0 !important; margin-top: 0 !important;}

        [data-testid="column"] {
            padding: 0 !important;
            margin: 0 !important;
        }

        /* Additional spacing killers */
        .element-container {margin: 0 !important; padding: 0 !important;}
        .stMarkdown {margin: 0 !important;}
        .main > div {margin: 0 !important; padding: 0 !important;}

        /* Hide Streamlit elements */
        .stApp > header {display: none;}
        .stDeployButton {display: none;}
        div[data-testid="stDecoration"] {display: none;}
        div[data-testid="stToolbar"] {display: none;}
        
        @media (max-width: 768px) {
            .main-header {
                padding: 0.5rem 1rem;
                height: 70px;
            }
            
            .main-header h1 {
                font-size: 1.3rem;
            }
            
            .main-header p {
                font-size: 0.75rem;
            }
            
            .main-content {
                top: 70px;
                flex-direction: column;
                gap: 0.3rem;
                padding: 0.3rem;
            }
            
            .input-panel, .results-panel {
                flex: none;
                height: calc(50% - 0.15rem);
            }
        }
    </style>
    """
    
    st.markdown(css, unsafe_allow_html=True)

def render_header():
    """
    Renders the main application header with centered content.
    """
    st.markdown("""
    <div class="main-header">
        <div>
            <h1>UML to Python Converter</h1>
            <p>Professional UML diagram conversion with AI-powered code generation</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_footer():
    """
    Renders the fixed footer.
    """
    st.markdown("""
    <div class="main-footer">
        <p>This project is done for the internship in Freethought Labs (OPC) Private Limited.</p>
    </div>
    """, unsafe_allow_html=True)

def render_input_section():
    """
    Renders the UML input section with file uploader and selection.
    """
    st.markdown("### Upload UML Diagram")
    
    diagrams_dir = "diagrams"
    os.makedirs(diagrams_dir, exist_ok=True)
    files = [f for f in os.listdir(diagrams_dir) if f.endswith(".graphml")]
    selected_file = None
    
    if files:
        st.markdown("**Choose a .graphml file from folder:**")
        selected_file = st.selectbox("Select file:", files, key="file_select")
    
    st.markdown("**Or upload your own .graphml file:**")
    uploaded_file = st.file_uploader("Upload a .graphml file", type="graphml", key="file_upload")
    
    if uploaded_file:
        temp_path = os.path.join(diagrams_dir, f"temp_{uploaded_file.name}")
        try:
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            selected_file = f"temp_{uploaded_file.name}"
        except Exception as e:
            st.error(f"Error saving uploaded file: {e}")
    
    # Render the Parse Diagram button below the file uploader
    if st.button("Parse Diagram", type="primary", key="parse_btn"):
        full_path = os.path.join("diagrams", selected_file)
        with st.spinner("Parsing diagram..."):
            try:
                nodes, edges = parse_graphml(full_path)
                diagram_type = classify_diagram_type(nodes)
                order, _ = traverse_graph(nodes, edges)
                roles = classify_node_roles(nodes, diagram_type, edges)
                flowchart_data = interpret_flowchart(nodes, edges, roles, order, diagram_type)
                st.session_state.nodes = nodes
                st.session_state.edges = edges
                st.session_state.roles = roles
                st.session_state.diagram_type = diagram_type
                st.session_state.order = order
                st.session_state.flowchart_data = flowchart_data
                st.session_state.parsed = True
                st.session_state.code_gemini = ""
                st.session_state.initial_templated_code = ""
                st.session_state.qwen_llm_prompt = ""
                st.session_state.qwen_raw_llm_output = ""
                st.session_state.gemini_llm_prompt = ""
                st.session_state.gemini_raw_llm_output = ""
                st.success("Diagram parsed successfully.")
                if selected_file.startswith("temp_"):
                    try:
                        os.remove(full_path)
                    except:
                        pass
            except Exception as e:
                st.error(f"Error parsing diagram: {e}. Please check the .graphml file format.")
                logger.error(f"Diagram parsing error: {e}")
                st.session_state.parsed = False

    return selected_file

def render_results_section(selected_file):
    """
    Renders the results section with code generation and display.
    """
    st.markdown("### Conversion Results")
    
    if not selected_file:
        st.markdown("""
        <div class="info-box">
            <h4>Ready to Convert</h4>
            <p>Upload or select a UML diagram to generate Python code.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    if "parsed" in st.session_state and st.session_state.parsed:
        st.markdown("#### Generate Python Code")
        diagram_hash = get_diagram_hash(st.session_state.flowchart_data)
        cached = get_cached_code(diagram_hash, "gemini")
        
        if cached:
            st.session_state.code_gemini = cached["code"]
            st.success("Loaded cached code from Redis.")
        elif st.button("Generate Code", type="primary", key="generate_btn"):
            with st.spinner("Generating code..."):
                try:
                    if not st.session_state.template_renderer:
                        st.error("Template renderer not initialized.")
                        st.stop()
                    st.session_state.initial_templated_code = st.session_state.template_renderer.render_code(
                        st.session_state.flowchart_data
                    )
                except Exception as e:
                    st.error(f"Template rendering failed: {e}")
                    logger.error(f"Template rendering error: {e}")
                    st.stop()
                try:
                    flowchart_json = json.dumps(make_serializable(st.session_state.flowchart_data), indent=2)
                except Exception as e:
                    st.error(f"Error serializing flowchart data: {e}")
                    st.stop()

                llm_prompt = f"""
You are a senior Python developer specializing in converting flowchart logic into clean, executable Python code.
The user has provided a flowchart diagram, which has been parsed and interpreted into the following structured JSON data:

```json
{flowchart_json}
```

Based on this structured data, and considering the following initial templated code (which might contain placeholders or basic structure):

```python
{st.session_state.initial_templated_code}
```

Your task is to generate a complete and functional Python program that follows clean, idiomatic Python practices.
DO NOT simulate node traversal using a variable like `current_node` or `while True` state machines.
Instead:
1. Use standard Python constructs: `if`, `else`, `while`, `for`, and function calls.
2. Map flowchart **decision branches** to actual `if/elif/else` structures using condition labels.
3. For input/output nodes:
   * Use `input()` for user input.
   * Use `print()` for output.
   * Wrap input in `try/except` for safety if numeric conversion is implied.
4. For process nodes:
   * Perform the actual operation in-line using Python expressions.
5. For loop nodes:
   * Use `while <condition>` and place the inner logic inside it.
6. Omit placeholder logic (e.g., `simulate_output`, `return True`, `# TODO`). Instead, infer intent from the label and implement sample logic where possible.
7. Add clear, detailed comments for each block of code that explain what the code does in simple terms However, do not oversimplify the logic - implement the full complexity shown in the flowchart using proper programming constructs.
8. Encapsulate the entire flow inside `main()`, and use:

```python
if __name__ == "__main__":
    main()
```

Here are the key requirements for the generated code:
1. **Implement the exact logic**: Translate the flowchart's steps into Python code. Pay close attention to:
   * **Start/End**: These indicate the program's beginning and end.
   * **Input Nodes**: For nodes with `type: "input"`, use `input()` to get values from the user. Ensure proper type conversion (e.g., `int()` or `float()`) if the context implies numbers. Add error handling for invalid input (e.g., `try-except` for `ValueError`).
   * **Process Nodes**: For nodes with `type: "process"`, implement the specified operation (e.e., `C = A * B`).
   * **Output Nodes**: For nodes with `type: "output"`, use `print()` to display the results.
   * **Decisions/Loops**: If there are decision (diamond) or loop nodes, implement them using `if/elif/else` or `while/for` loops.
2. **Beginner-friendly but complete**: Add clear, detailed inline comments to explain each significant part of the code, especially for input/output, calculations, and control flow. Make the comments educational so someone learning programming can understand what each part does and why. However, implement the full complexity and logic shown in the flowchart - do not oversimplify the code structure.
3. **No placeholders**: Replace any placeholder comments or generic `pass` statements from the template with actual, working implementation.
4. **Function encapsulation**: Encapsulate the main flowchart logic within a `def main():` function, and call `main()` if `__name__ == "__main__"`.
5. **No external libraries unless explicitly needed**: Stick to standard Python library functions unless the flowchart implies a need for a specific library (e.g., `math` for complex calculations).
6. **Return only the Python code**: Do not include any conversational text, explanations, or markdown outside the code block.
7. **Format**: Enclose the entire generated Python code within a single markdown code block (`python ... `).
"""
                
                st.session_state.qwen_llm_prompt = llm_prompt
                st.session_state.gemini_llm_prompt = llm_prompt
                
                # Queue Ollama task in background
                ollama_queue.put((diagram_hash, llm_prompt))
                
                # Generate Gemini code and show immediately
                # Generate code using Ollama qwen3:0.6b
            try:
                from llm.ollama_client import OllamaCodeGenerator
                ollama_generator = OllamaCodeGenerator(model_name="qwen3:0.6b")
                result = ollama_generator.generate_python_code(llm_prompt)
                st.session_state.gemini_raw_llm_output = result
                st.session_state.code_gemini = extract_pure_code(result)

                if is_placeholder_code(st.session_state.code_gemini):
                    st.warning("⚠️ Ollama generated incomplete or placeholder code.")
                else:
                    st.success("✅ Code generated successfully")

                cache_generated_code(
                    diagram_hash,
                    st.session_state.code_gemini,
                    metadata={
                        "source": "gemini",
                        "timestamp": pd.Timestamp.now().isoformat(),
                        "diagram_type": st.session_state.diagram_type,
                        "raw_output": result
                    },
                    source="gemini"
                )

            except Exception as e:
                st.error(f"Error generating code with Ollama (qwen3:0.6b): {e}")
                logger.error(f"Ollama code generation error: {e}")
                st.session_state.code_gemini = ""

    # Code Output and Save Functionality
    if st.session_state.code_gemini:
        # Explanation section with manual generation button - MOVED ABOVE CODE DISPLAY
        st.markdown("#### Code Explanation")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("Generate Explanation", key="explain_btn"):
                with st.spinner("Generating explanation..."):
                    try:
                        # Create explanation prompt
                        explanation_prompt = f"""
Please explain the following Python code in simple, beginner-friendly terms that someone without a computer science background can understand:


```python
{st.session_state.code_gemini}
```

Your explanation should:
1. Describe what the program does in plain English
2. Explain each major section of the code using simple analogies
3. Teach basic programming concepts (variables, loops, conditions) in an accessible way
4. Use real-world examples to illustrate technical concepts
5. Be educational and encouraging for someone learning to code
6. Avoid technical jargon unless you clearly explain it
7. Break down complex concepts step by step
8. Focus on what the code accomplishes rather than how it works technically

Please provide a clear, friendly explanation that would help a complete beginner understand programming concepts.
"""
                        
                        from llm.ollama_client import OllamaCodeGenerator
                        ollama_generator = OllamaCodeGenerator(model_name="qwen3:0.6b")
                        explanation_result = ollama_generator.generate_explanation(
                            st.session_state.code_gemini, model="gemma3:270m"
                        )
                        st.session_state.code_explanation = explanation_result.strip()
                    except Exception as e:
                        st.error(f"Error generating explanation: {e}")
                        logger.error(f"Explanation generation error: {e}")
                        st.session_state.code_explanation = ""
        
        if st.session_state.code_explanation:
            st.markdown(f"""
            <div class="explanation-container">
                <p>{st.session_state.code_explanation}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Display code in a scrollable text box - MOVED BELOW EXPLANATION BUTTON
        st.markdown("#### Generated Python Code")
        
        # Create a scrollable code display
        st.markdown(f"""
        <div style="background: #262626; border: 1px solid #333333; border-radius: 8px; padding: 1rem; max-height: 300px; overflow-y: auto; font-family: 'JetBrains Mono', monospace; font-size: 14px; color: #FAFAFA;">
            <pre style="margin: 0; white-space: pre-wrap; word-wrap: break-word;">{st.session_state.code_gemini}</pre>
        </div>
        """, unsafe_allow_html=True)
        
        # Download button for the code
        st.download_button(
            label="📥 Download Code",
            data=st.session_state.code_gemini,
            file_name="generated_code.py",
            mime="text/x-python",
            key="download_code"
        )

# =============================================================================
# MAIN APPLICATION
# =============================================================================

# Override theme and custom CSS for modern UI
st.set_page_config(
    page_title="UML to Python Converter",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🔍"
)

# --- Initialize session state defaults ---
if "parsed" not in st.session_state:
    st.session_state.parsed = False
    st.session_state.flowchart_data = None
    st.session_state.nodes = []
    st.session_state.edges = []
    st.session_state.roles = {}
    st.session_state.order = []
    st.session_state.diagram_type = ""
    st.session_state.filename_gemini = "code_gemini"
    st.session_state.initial_templated_code = ""
    st.session_state.qwen_llm_prompt = ""
    st.session_state.qwen_raw_llm_output = ""
    st.session_state.gemini_llm_prompt = ""
    st.session_state.gemini_raw_llm_output = ""
    st.session_state.code_gemini = ""
    st.session_state.code_explanation = ""
    st.session_state.explanation_loading = False
    
    # Initialize template renderer and Gemini generator
    try:
        template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
        st.session_state.template_renderer = CodeTemplateRenderer(template_dir)
    except Exception as e:
        st.error(f"Failed to initialize template renderer: {e}")
        logger.error(f"Template renderer initialization error: {e}")
        st.session_state.template_renderer = None
    
    st.session_state.gemini_generator = None  # Gemini API not used in offline mode

def main():
    """
    Main application function that orchestrates the entire user interface.
    """
    # Load custom CSS first
    load_custom_css()
    
    # Render header
    render_header()
    
    # Main content container - FIXED HEIGHT, NO SCROLLING, NO BLANK SPACE
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.markdown('<div class="input-panel">', unsafe_allow_html=True)
        selected_file = render_input_section()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="results-panel">', unsafe_allow_html=True)
        render_results_section(selected_file)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    render_footer()

if __name__ == "__main__":
    main()
