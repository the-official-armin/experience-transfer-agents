def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""

    try:
        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )
        return str(result)

    except Exception as e:
        return f"ERROR: {e}"