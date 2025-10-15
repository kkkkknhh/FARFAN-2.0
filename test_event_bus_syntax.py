#!/usr/bin/env python3
"""
Syntax and structure validation for refactored event_bus.py
"""

import ast
import sys


def analyze_complexity(source_code):
    """Analyze cognitive complexity of methods."""
    tree = ast.parse(source_code)

    complexities = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            complexity = calculate_cognitive_complexity(node)
            complexities[node.name] = complexity

    return complexities


def calculate_cognitive_complexity(func_node):
    """
    Calculate cognitive complexity based on control flow.
    Simplified version counting: if/else, for, while, try, and nested structures.
    """
    complexity = 0
    nesting_level = 0

    def visit_node(node, level):
        nonlocal complexity

        if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
            complexity += 1 + level
        elif isinstance(node, (ast.And, ast.Or)):
            complexity += 1

        # Increment nesting for control structures
        new_level = level
        if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
            new_level = level + 1

        for child in ast.iter_child_nodes(node):
            visit_node(child, new_level)

    visit_node(func_node, 0)
    return complexity


def main():
    """Validate refactored code structure."""
    print("Analyzing event_bus.py structure...\n")

    with open("choreography/event_bus.py", "r") as f:
        source = f.read()

    # Check syntax
    try:
        ast.parse(source)
        print("✓ Syntax is valid")
    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        return 1

    # Analyze complexity
    complexities = analyze_complexity(source)

    print("\nMethod complexity analysis:")
    print("-" * 50)

    publish_complexity = complexities.get("publish", None)

    for name, complexity in sorted(complexities.items()):
        if name in [
            "publish",
            "_prepare_handler_task",
            "_collect_handler_tasks",
            "_log_handler_exceptions",
        ]:
            marker = "→" if name == "publish" else " "
            print(f"{marker} {name:30} complexity: {complexity}")

    print("-" * 50)

    if publish_complexity is None:
        print("\n✗ publish method not found")
        return 1

    # Check for helper methods
    helper_methods = [
        "_prepare_handler_task",
        "_collect_handler_tasks",
        "_log_handler_exceptions",
    ]
    found_helpers = [m for m in helper_methods if m in complexities]

    print(f"\n✓ Found {len(found_helpers)}/{len(helper_methods)} helper methods:")
    for helper in found_helpers:
        print(f"  • {helper}")

    # Verify event types are preserved
    required_strings = [
        "graph.edge_added",
        "contradiction.detected",
        "posterior.updated",
        "asyncio.gather",
    ]

    print("\n✓ Required patterns preserved:")
    for pattern in required_strings:
        if pattern in source:
            print(f"  • {pattern}")
        else:
            print(f"  ✗ Missing: {pattern}")
            return 1

    print(f"\n✓ publish method complexity: {publish_complexity}")

    if publish_complexity <= 15:
        print(f"✓ Complexity target met (≤15)")
    else:
        print(
            f"✗ Complexity still too high (target: ≤15, actual: {publish_complexity})"
        )
        return 1

    print("\n✅ Refactoring validation passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
