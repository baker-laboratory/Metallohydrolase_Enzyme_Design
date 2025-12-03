#!/usr/bin/env python3
"""
List all Python files imported (directly or indirectly) by a given main script,
and also report external Python scripts referenced in os.system/subprocess.run
commands (e.g., containerized calls to other .py scripts). This is done
recursively: any discovered external .py scripts are also analyzed.

Usage:
    python list_imported_files.py /path/to/main_script.py
"""

import os
import sys
import re
import ast
import sysconfig
from modulefinder import ModuleFinder

# Global accumulators so we can combine info across recursive analysis
ALL_IMPORTED_MODULES = set()    # (name, path)
ALL_EXTERNAL_SCRIPTS = set()    # absolute paths to .py files (from shell commands / command strings)
OTHER_FOUND_PY_FILES = set()    # any other *.py strings that resolve to real files
VISITED_SCRIPTS = set()         # scripts we've already analyzed (by absolute path)


# ----------------------------- IMPORT ANALYSIS ----------------------------- #

def analyze_imports(script_path: str):
    """
    Use ModuleFinder to list imported modules and their files, filtering out stdlib.
    Results go into ALL_IMPORTED_MODULES.
    """
    finder = ModuleFinder()
    finder.run_script(script_path)

    stdlib_dir = sysconfig.get_paths().get("stdlib", None)

    for name, module in finder.modules.items():
        filename = module.__file__
        if not filename:
            continue

        filename = os.path.abspath(filename)

        # Skip stdlib
        if stdlib_dir and filename.startswith(stdlib_dir):
            continue

        ALL_IMPORTED_MODULES.add((name, filename))


# ------------------------ STRING EVALUATION HELPERS ------------------------ #

def eval_string_expr(node, env):
    """
    Best-effort extraction of (mostly) static text from an AST node,
    using a simple environment 'env' for string variables.

    Handles:
      - ast.Constant(str)
      - ast.JoinedStr (f-strings), including embedded Names from env
      - ast.BinOp(Add) for string concatenation
      - ast.Name, if that name is in env as a string
    """
    # Literal string
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value

    # Name reference
    if isinstance(node, ast.Name):
        return env.get(node.id)

    # f-string
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                inner = eval_string_expr(value.value, env)
                if inner:
                    parts.append(inner)
        return "".join(parts) if parts else None

    # String concatenation: "foo" + var + "bar"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = eval_string_expr(node.left, env)
        right = eval_string_expr(node.right, env)
        # If both sides resolve, concatenate
        if left and right:
            return left + right
        # If only one side resolves, return that (partial info still useful)
        return left or right

    return None


class StringEnvBuilder(ast.NodeVisitor):
    """
    First pass over the AST: collect simple string assignments like

        cmd = "/path/to/script.py"
        build_residue_script = "/software/.../build_full_residue_from_tips.py"

    into an environment dict {variable_name: string_value}.

    ALSO: as a robustness boost, any assignment whose string value contains
    a *.py path is immediately added to ALL_EXTERNAL_SCRIPTS.
    """

    def __init__(self, base_dir):
        self.env = {}
        self.base_dir = base_dir
        self.py_path_pattern = re.compile(r'([^\s\'"]+\.py)\b')

    def visit_Assign(self, node: ast.Assign):
        # Only handle simple "x = <expr>" assignments
        if len(node.targets) != 1:
            return

        target = node.targets[0]
        if isinstance(target, ast.Name):
            value_str = eval_string_expr(node.value, self.env)
            if isinstance(value_str, str):
                self.env[target.id] = value_str

                # If this assigned string contains a .py path, treat it as an external script reference
                for match in self.py_path_pattern.findall(value_str):
                    raw = match
                    if os.path.isabs(raw):
                        full = os.path.abspath(raw)
                    else:
                        full = os.path.abspath(os.path.join(self.base_dir, raw))
                    ALL_EXTERNAL_SCRIPTS.add(full)

        self.generic_visit(node)


class AssignmentIndex(ast.NodeVisitor):
    """
    Build an index of assignments so we can, for example, find the RHS of
    'command = f"...{build_script}..."' when we later see os.system(command).

    Stores: {var_name: [value_node1, value_node2, ...]} in textual order.
    """

    def __init__(self):
        self.assignments = {}

    def visit_Assign(self, node: ast.Assign):
        if len(node.targets) != 1:
            return

        target = node.targets[0]
        if isinstance(target, ast.Name):
            var_name = target.id
            self.assignments.setdefault(var_name, []).append(node.value)

        self.generic_visit(node)


# ------------------------- EXTERNAL SCRIPT ANALYSIS ------------------------ #

def scan_for_py_strings(script_path: str):
    """
    Safety net: scan the raw source for ANY 'something.py' tokens.
    If they resolve to real files, stash them in OTHER_FOUND_PY_FILES.
    """
    base_dir = os.path.dirname(os.path.abspath(script_path))
    py_path_pattern = re.compile(r'([^\s\'"]+\.py)\b')

    with open(script_path, "r") as f:
        src = f.read()

    for match in py_path_pattern.findall(src):
        raw = match
        if os.path.isabs(raw):
            full = os.path.abspath(raw)
        else:
            full = os.path.abspath(os.path.join(base_dir, raw))

        if os.path.isfile(full):
            OTHER_FOUND_PY_FILES.add(full)


def extract_external_scripts_from_ast(script_path: str):
    """
    Parse the script's AST and look for:
      - os.system("...") / os.system(f"...") / os.system("foo" + var) / os.system(command)
      - subprocess.run([...]) / subprocess.run("...", shell=True) / subprocess.run(command, ...)

    Then extract any *.py paths from those commands (including ones that appear
    via variables or f-strings), resolve them relative to 'script_path' if
    they are not absolute, and add them to ALL_EXTERNAL_SCRIPTS.

    Also runs a raw-text safety-net scan for any *.py strings.
    """
    with open(script_path, "r") as f:
        source = f.read()

    tree = ast.parse(source, filename=script_path)
    base_dir = os.path.dirname(os.path.abspath(script_path))

    # First pass: build env of simple string variables (also seeds ALL_EXTERNAL_SCRIPTS)
    env_builder = StringEnvBuilder(base_dir)
    env_builder.visit(tree)
    env = env_builder.env

    # Second pass: index of assignments for each variable name
    assign_index_builder = AssignmentIndex()
    assign_index_builder.visit(tree)
    assign_index = assign_index_builder.assignments

    # Regex to grab *.py tokens from a command string
    py_path_pattern = re.compile(r'([^\s\'"]+\.py)\b')

    class CommandVisitor(ast.NodeVisitor):
        def _resolve_command_arg(self, arg_node):
            """
            Resolve the argument to os.system / subprocess.run into a string,
            using env and, if necessary, looking up the last assignment to
            a Name like 'command'.
            """
            # First try direct evaluation (handles constants, f-strings, etc.)
            cmd_str = eval_string_expr(arg_node, env)
            if isinstance(cmd_str, str):
                return cmd_str

            # If it's a Name and not in env, try to resolve its last assignment
            if isinstance(arg_node, ast.Name):
                var_name = arg_node.id
                if var_name in assign_index and assign_index[var_name]:
                    last_value_node = assign_index[var_name][-1]
                    cmd_str = eval_string_expr(last_value_node, env)
                    if isinstance(cmd_str, str):
                        return cmd_str

            return None

        def visit_Call(self, node: ast.Call):
            # os.system(...)
            if isinstance(node.func, ast.Attribute):
                # os.system(...)
                if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "os"
                    and node.func.attr == "system"
                ):
                    if node.args:
                        cmd_str = self._resolve_command_arg(node.args[0])
                        if cmd_str:
                            self._extract_paths(cmd_str)

                # subprocess.run(...)
                if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "subprocess"
                    and node.func.attr == "run"
                ):
                    if node.args:
                        first_arg = node.args[0]

                        # subprocess.run(["python", "/path/to/script.py", ...])
                        if isinstance(first_arg, ast.List):
                            for elt in first_arg.elts:
                                piece_str = self._resolve_command_arg(elt)
                                if piece_str:
                                    self._extract_paths(piece_str)
                        else:
                            # subprocess.run("python /path/to/script.py ...", shell=True)
                            cmd_str = self._resolve_command_arg(first_arg)
                            if cmd_str:
                                self._extract_paths(cmd_str)

            self.generic_visit(node)

        def _extract_paths(self, cmd_str: str):
            for match in py_path_pattern.findall(cmd_str):
                raw = match
                if os.path.isabs(raw):
                    full = os.path.abspath(raw)
                else:
                    full = os.path.abspath(os.path.join(base_dir, raw))
                ALL_EXTERNAL_SCRIPTS.add(full)

    CommandVisitor().visit(tree)

    # Safety net: scan the raw source for any *.py strings that exist on disk
    scan_for_py_strings(script_path)


# ------------------------------- RECURSION --------------------------------- #

def analyze_script_recursive(script_path: str):
    """
    Analyze one script:
      - add its imported modules to ALL_IMPORTED_MODULES
      - add its external shell-called scripts to ALL_EXTERNAL_SCRIPTS
      - then recursively analyze those external scripts (if they exist)
    """
    script_path = os.path.abspath(script_path)
    if script_path in VISITED_SCRIPTS:
        return
    VISITED_SCRIPTS.add(script_path)

    print(f"### Analyzing: {script_path} ###")

    # 1) Imports (non-stdlib)
    analyze_imports(script_path)

    # 2) External .py scripts called via shell + safety-net scan
    extract_external_scripts_from_ast(script_path)

    # 3) Recurse into any newly discovered external scripts
    for ext_script in list(ALL_EXTERNAL_SCRIPTS):
        if ext_script not in VISITED_SCRIPTS and os.path.isfile(ext_script):
            analyze_script_recursive(ext_script)


# --------------------------------- MAIN ------------------------------------ #

def main(entry_script_path: str):
    entry_script_path = os.path.abspath(entry_script_path)

    if not os.path.isfile(entry_script_path):
        print(f"ERROR: {entry_script_path} does not exist or is not a file.")
        sys.exit(1)

    analyze_script_recursive(entry_script_path)

    print("\n### IMPORTED PYTHON MODULE FILES (non-stdlib) ###")
    if not ALL_IMPORTED_MODULES:
        print("(none found)")
    else:
        for name, path in sorted(ALL_IMPORTED_MODULES, key=lambda x: x[1]):
            print(f"{name:30s} -> {path}")

    print("\n### EXTERNAL PYTHON SCRIPTS CALLED VIA SHELL (RECURSIVE) ###")
    if not ALL_EXTERNAL_SCRIPTS:
        print("(none found)")
    else:
        for path in sorted(ALL_EXTERNAL_SCRIPTS):
            print(path)

    # Safety-net section: any real *.py paths seen in source that aren't already
    # in ALL_EXTERNAL_SCRIPTS.
    extra_py_files = OTHER_FOUND_PY_FILES - ALL_EXTERNAL_SCRIPTS
    print("\n### OTHER FOUND .py FILES (EXISTING ON DISK) ###")
    if not extra_py_files:
        print("(none found)")
    else:
        for path in sorted(extra_py_files):
            print(path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python list_imported_files.py /path/to/main_script.py")
        sys.exit(1)

    main(sys.argv[1])
