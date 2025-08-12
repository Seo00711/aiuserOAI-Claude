import logging
from typing import Dict, Any, List

logger = logging.getLogger("red.gpt5assistant.tools.code_interpreter")


class CodeInterpreterTool:
    def __init__(self):
        pass
    
    def is_enabled_in_tools(self, tools_config: Dict[str, Any]) -> bool:
        return tools_config.get("code_interpreter", False)
    
    def get_tool_config(self) -> Dict[str, Any]:
        return {
            "type": "code_interpreter",
            "container": {"type": "auto"}
        }
    
    def get_capabilities(self) -> List[str]:
        return [
            "Execute Python code in a secure sandbox",
            "Process and analyze data files",
            "Generate visualizations and charts",
            "Perform mathematical calculations",
            "Create and manipulate files",
            "Debug and iterate on code solutions"
        ]
    
    def get_usage_guidelines(self) -> str:
        return """
        Code Interpreter capabilities:
        - Python execution in a secure, sandboxed environment
        - Data analysis with pandas, numpy, matplotlib
        - File processing and generation
        - Mathematical computations and visualizations
        - Iterative problem solving with debugging
        
        Usage tips:
        - Ask for data analysis, calculations, or file processing
        - Request visualizations and charts
        - Use for complex mathematical problems
        - Good for debugging and code development
        """
    
    def get_supported_libraries(self) -> List[str]:
        return [
            "pandas", "numpy", "matplotlib", "seaborn", "scipy",
            "scikit-learn", "requests", "json", "csv", "xml",
            "PIL", "io", "base64", "datetime", "re", "os", "sys"
        ]
    
    def get_limitations(self) -> List[str]:
        return [
            "No internet access from code execution",
            "Limited to Python language",
            "Temporary file storage only",
            "No persistent state between sessions",
            "Sandboxed environment restrictions"
        ]
    
    async def suggest_use_cases(self, user_query: str) -> List[str]:
        query_lower = user_query.lower()
        suggestions = []
        
        # Data analysis suggestions
        if any(keyword in query_lower for keyword in ["data", "analyze", "csv", "excel", "statistics"]):
            suggestions.append("Data analysis and statistical computations")
            suggestions.append("CSV/Excel file processing and visualization")
        
        # Math suggestions
        if any(keyword in query_lower for keyword in ["calculate", "math", "equation", "solve"]):
            suggestions.append("Mathematical calculations and equation solving")
            suggestions.append("Numerical analysis and computations")
        
        # Visualization suggestions
        if any(keyword in query_lower for keyword in ["plot", "chart", "graph", "visualize"]):
            suggestions.append("Data visualization with matplotlib/seaborn")
            suggestions.append("Chart and graph generation")
        
        # Programming suggestions
        if any(keyword in query_lower for keyword in ["code", "script", "program", "function"]):
            suggestions.append("Python code development and debugging")
            suggestions.append("Algorithm implementation and testing")
        
        # File processing suggestions
        if any(keyword in query_lower for keyword in ["file", "process", "convert", "transform"]):
            suggestions.append("File processing and format conversion")
            suggestions.append("Text manipulation and parsing")
        
        return suggestions or ["General Python programming and data analysis"]