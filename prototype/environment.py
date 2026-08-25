class Environment:
    """Environemt responsible for executing agent tools"""

    def __init__(self):
        self.tools = {}
        
    def register_tool(self, tool_name, tool):
        """Register a tool to the environment"""
        self.tools[tool_name] = tool

    def execute_tool(self,name, argument):
        if name not in self.tools:
            return f'ERROR: Unknown tool: {name}'

        return self.tools[name](argument)
    