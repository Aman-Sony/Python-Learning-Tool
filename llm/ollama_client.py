# llm/ollama_client.py

import requests
import json
import re
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Code Generator Interface ---
class CodeGenerator(ABC):
    @abstractmethod
    def generate_python_code(self, diagram_prompt: str, generation_context: Dict[str, Any] = None) -> str:
        pass

# --- Clean up Ollama Response ---
def _strip_thinking_and_markdown(response_text: str) -> str:
    """
    Cleans Ollama response: removes <think> tags and extracts Python code blocks or raw fallback.
    """
    # Remove custom LLM reasoning sections
    response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL)

    # Try extracting markdown code blocks
    code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", response_text, re.DOTALL)
    if code_blocks:
        return "\n".join(code_blocks).strip()

    # Fallback: remove any remaining markdown characters
    return re.sub(r"^`+|`+$", "", response_text.strip(), flags=re.MULTILINE)

# --- Main Call for Code Generation ---
def generate_code_with_ollama(prompt: str, model: str = "qwen3:0.6b", timeout: int = 1800) -> str:
    """
    Sends prompt to Ollama server and returns clean Python code.

    Args:
        prompt (str): Full diagram description to be turned into Python code.
        model (str): Ollama model to use, default is 'qwen3:0.6b'.
        timeout (int): Timeout in seconds for the HTTP request.

    Returns:
        str: Clean generated Python code or error message.
    """
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}

    system_prompt = (
        "You are a Python code generator. Output only valid executable Python code. "
        "No comments, no explanations, no markdown, no headers. Your goal is to convert the following description into actual working code."
    )

    payload = {
        "model": model,
        "prompt": f"{system_prompt}\n{prompt.strip()}",
        "stream": False,
        "options": {
            "temperature": 0.6
        }
    }

    try:
        logger.info(f"Sending request to Ollama using model '{model}'...")
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)

        if response.status_code == 200:
            result = response.json()
            raw_output = result.get("response", "")
            logger.info("Code received from Ollama.")
            return _strip_thinking_and_markdown(raw_output)

        else:
            error_msg = f"Ollama returned status {response.status_code}: {response.text}"
            logger.error(error_msg)
            return error_msg

    except requests.exceptions.Timeout:
        return f"Ollama request timed out. Please ensure 'ollama run {model}' is running."
    except requests.exceptions.RequestException as e:
        return f"Ollama request failed: {str(e)}. Is the Ollama server running on localhost:11434?"

# --- Explanation Generation with Gemma Model ---
def generate_explanation_with_ollama(code: str, model: str = "gemma3:270m", timeout: int = 1800) -> str:
    """
    Generates explanation for Python code using Ollama's Gemma model.

    Args:
        code (str): Python code to explain
        model (str): Ollama model to use, default is 'gemma3:270m'
        timeout (int): Timeout in seconds for the HTTP request.

    Returns:
        str: Explanation text or error message.
    """
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}

    explanation_prompt = f"""
Please explain the following Python code in simple, beginner-friendly terms that someone without a computer science background can understand:

```python
{code}
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

    payload = {
        "model": model,
        "prompt": explanation_prompt.strip(),
        "stream": False,
        "options": {
            "temperature": 0.7
        }
    }

    try:
        logger.info(f"Generating explanation with Ollama using model '{model}'...")
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)

        if response.status_code == 200:
            result = response.json()
            explanation = result.get("response", "")
            logger.info("Explanation received from Ollama.")
            return explanation.strip()

        else:
            error_msg = f"Ollama returned status {response.status_code}: {response.text}"
            logger.error(error_msg)
            return error_msg

    except requests.exceptions.Timeout:
        return f"Ollama request timed out. Please ensure 'ollama run {model}' is running."
    except requests.exceptions.RequestException as e:
        return f"Ollama request failed: {str(e)}. Is the Ollama server running on localhost:11434?"

# --- Ollama Code Generator Implementation ---
class OllamaCodeGenerator(CodeGenerator):
    def __init__(self, model_name: str = "qwen3:0.6b"):
        """
        Initialize Ollama code generator.
        
        Args:
            model_name (str): Ollama model to use for code generation
        """
        self.model_name = model_name
        logger.info(f"OllamaCodeGenerator initialized with model: {self.model_name}")

    def generate_python_code(self, diagram_prompt: str, generation_context: Dict[str, Any] = None) -> str:
        """
        Generate Python code using Ollama model.
        
        Args:
            diagram_prompt (str): Prompt for code generation
            generation_context (Dict[str, Any]): Optional generation context
            
        Returns:
            str: Generated Python code or error message
        """
        logger.info("Generating Python code using Ollama...")
        logger.debug(f"Prompt preview: {diagram_prompt[:300]}...")
        if generation_context:
            logger.debug(f"Context passed: {generation_context}")

        try:
            result = generate_code_with_ollama(diagram_prompt, model=self.model_name)
            logger.info("Ollama code generation completed.")
            return result
        except Exception as e:
            error_msg = f"Ollama code generation error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    def generate_explanation(self, code: str, model: str = "gemma3:270m") -> str:
        """
        Generate explanation for Python code using Ollama model.
        
        Args:
            code (str): Python code to explain
            model (str): Ollama model to use for explanation
            
        Returns:
            str: Explanation text or error message
        """
        logger.info("Generating explanation using Ollama...")
        try:
            result = generate_explanation_with_ollama(code, model=model)
            logger.info("Ollama explanation generation completed.")
            return result
        except Exception as e:
            error_msg = f"Ollama explanation generation error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg
