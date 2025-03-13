import os
import re

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get current package directory
INIT_FILE = os.path.join(PACKAGE_DIR, "__init__.py")

def extract_functions(file_path):
    """Extract function names from a Python file."""
    functions = []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    matches = re.findall(r"^def (\w+)\(", content, re.MULTILINE)
    return matches

def update_init():
    """Auto-update __init__.py to include all functions from modules."""
    modules = []
    all_functions = []

    for file in os.listdir(PACKAGE_DIR):
        if file.endswith(".py") and file != "__init__.py" and not file.startswith("_"):
            module_name = file[:-3]  # Remove .py extension
            file_path = os.path.join(PACKAGE_DIR, file)
            
            functions = extract_functions(file_path)
            if functions:
                modules.append((module_name, functions))
                all_functions.extend(functions)

    # Generate import statements
    import_lines = [f"from .{mod} import {', '.join(funcs)}" for mod, funcs in modules]
    all_line = f"__all__ = {all_functions}"

    # Write to __init__.py
    with open(INIT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(import_lines) + "\n\n" + all_line + "\n")

    print("✅ __init__.py updated successfully!")

if __name__ == "__main__":
    update_init()
