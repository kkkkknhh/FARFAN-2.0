#!/usr/bin/env python3
"""
Static EventBus Flow Analysis
==============================
Traces all publish/subscribe calls to map event flows.
"""

import ast
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set

class EventFlowAnalyzer:
    """Analyzes EventBus publish/subscribe patterns."""

    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.publish_calls: List[Dict[str, Any]] = []
        self.subscribe_calls: List[Dict[str, Any]] = []
        self.event_types: Set[str] = set()

    def analyze_file(self, filepath: Path) -> None:
        """Parse a Python file and extract EventBus calls."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                tree = ast.parse(content, filename=str(filepath))

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if self._is_publish_call(node):
                        self._extract_publish(node, filepath, content)
                    elif self._is_subscribe_call(node):
                        self._extract_subscribe(node, filepath, content)

        except Exception as e:
            print(f"  ⚠️  Could not parse {filepath.name}: {e}")

    def _is_publish_call(self, node: ast.Call) -> bool:
        """Check if node is a .publish() call."""
        if isinstance(node.func, ast.Attribute):
            return node.func.attr == "publish"
        return False

    def _is_subscribe_call(self, node: ast.Call) -> bool:
        """Check if node is a .subscribe() call."""
        if isinstance(node.func, ast.Attribute):
            return node.func.attr == "subscribe"
        return False

    def _extract_publish(self, node: ast.Call, filepath: Path, content: str) -> None:
        """Extract details from a publish() call."""
        event_type = None
        payload_keys = []

        # Try to extract event_type from PDMEvent constructor
        if node.args and isinstance(node.args[0], ast.Call):
            event_node = node.args[0]
            for keyword in event_node.keywords:
                if keyword.arg == "event_type":
                    if isinstance(keyword.value, ast.Constant):
                        event_type = keyword.value.value
                        self.event_types.add(event_type)
                    elif isinstance(keyword.value, ast.Str):
                        event_type = keyword.value.s
                        self.event_types.add(event_type)
                elif keyword.arg == "payload":
                    if isinstance(keyword.value, ast.Dict):
                        payload_keys = []
                        for k in keyword.value.keys:
                            if isinstance(k, ast.Constant):
                                payload_keys.append(k.value)
                            elif isinstance(k, ast.Str):
                                payload_keys.append(k.s)

        # Get context (class and function)
        context = self._get_context(content, node.lineno)

        self.publish_calls.append(
            {
                "file": str(filepath.relative_to(self.base_path)),
                "line": node.lineno,
                "event_type": event_type,
                "payload_keys": payload_keys,
                "context": context,
            }
        )

    def _extract_subscribe(self, node: ast.Call, filepath: Path, content: str) -> None:
        """Extract details from a subscribe() call."""
        event_type = None
        handler_name = None

        # Extract event_type (first argument)
        if node.args:
            arg0 = node.args[0]
            if isinstance(arg0, ast.Constant):
                event_type = arg0.value
                self.event_types.add(event_type)
            elif isinstance(arg0, ast.Str):
                event_type = arg0.s
                self.event_types.add(event_type)

        # Extract handler name (second argument)
        if len(node.args) >= 2:
            handler_arg = node.args[1]
            if isinstance(handler_arg, ast.Attribute):
                handler_name = handler_arg.attr
            elif isinstance(handler_arg, ast.Name):
                handler_name = handler_arg.id

        context = self._get_context(content, node.lineno)

        self.subscribe_calls.append(
            {
                "file": str(filepath.relative_to(self.base_path)),
                "line": node.lineno,
                "event_type": event_type,
                "handler": handler_name,
                "context": context,
            }
        )

    def _get_context(self, content: str, lineno: int) -> str:
        """Get class/function context for a line."""
        lines = content.split('\n')
        context_parts = []
        
        # Look backwards for class or function definitions
        for i in range(lineno - 1, max(0, lineno - 50), -1):
            line = lines[i].strip()
            if line.startswith('class '):
                context_parts.insert(0, line.split('(')[0].replace('class ', ''))
                break
            elif line.startswith('def ') or line.startswith('async def '):
                func_name = line.split('(')[0].replace('def ', '').replace('async ', '').strip()
                context_parts.insert(0, func_name)
                
        return '.'.join(context_parts) if context_parts else 'module_level'

    def analyze_directory(self, pattern: str = "**/*.py") -> None:
        """Analyze all Python files matching pattern."""
        files = [f for f in self.base_path.glob(pattern) 
                 if '.venv' not in str(f) and '.git' not in str(f)]
        
        print(f"Analyzing {len(files)} Python files...")
        
        for filepath in files:
            self.analyze_file(filepath)

    def generate_event_flow_map(self) -> Dict[str, Dict[str, List]]:
        """Generate mapping of event flows."""
        event_map = defaultdict(lambda: {"publishers": [], "subscribers": []})

        for pub in self.publish_calls:
            if pub["event_type"]:
                event_map[pub["event_type"]]["publishers"].append(
                    {
                        "file": pub["file"],
                        "line": pub["line"],
                        "context": pub["context"],
                        "payload_keys": pub["payload_keys"],
                    }
                )

        for sub in self.subscribe_calls:
            if sub["event_type"]:
                event_map[sub["event_type"]]["subscribers"].append(
                    {
                        "file": sub["file"],
                        "line": sub["line"],
                        "handler": sub["handler"],
                        "context": sub["context"],
                    }
                )

        return dict(event_map)


def print_section(title: str, width: int = 80):
    """Print formatted section header."""
    print("\n" + "=" * width)
    print(f"{title:^{width}}")
    print("=" * width + "\n")


def analyze_decoupling(flow_map: Dict) -> Dict[str, Any]:
    """Analyze event-based communication patterns."""
    event_coverage = {}
    
    for event_type, flows in flow_map.items():
        pub_count = len(flows['publishers'])
        sub_count = len(flows['subscribers'])
        
        # Check for orphaned events (published but no subscribers)
        # or unused subscriptions (subscribed but never published)
        event_coverage[event_type] = {
            'publishers': pub_count,
            'subscribers': sub_count,
            'orphaned': pub_count > 0 and sub_count == 0,
            'unused_subscription': sub_count > 0 and pub_count == 0,
            'active': pub_count > 0 and sub_count > 0,
        }
    
    return event_coverage


def main():
    """Run static analysis."""
    print_section("CHOREOGRAPHY MODULE - EVENT FLOW ANALYSIS")

    analyzer = EventFlowAnalyzer(".")
    analyzer.analyze_directory()

    print(f"📊 Analysis Results:")
    print(f"   Total Event Types: {len(analyzer.event_types)}")
    print(f"   Publish Calls: {len(analyzer.publish_calls)}")
    print(f"   Subscribe Calls: {len(analyzer.subscribe_calls)}")
    print()

    # Generate event flow map
    event_flow_map = analyzer.generate_event_flow_map()
    
    print_section("EVENT FLOW MAP")
    
    for event_type in sorted(analyzer.event_types):
        flows = event_flow_map.get(event_type, {"publishers": [], "subscribers": []})
        pub_count = len(flows["publishers"])
        sub_count = len(flows["subscribers"])
        
        status = "✓" if pub_count > 0 and sub_count > 0 else "⚠️"
        print(f"{status} {event_type}")
        print(f"   Publishers: {pub_count}")
        
        for pub in flows["publishers"][:5]:
            print(f"      📤 {pub['file']}:{pub['line']} ({pub['context']})")
            if pub['payload_keys']:
                print(f"         Payload: {', '.join(pub['payload_keys'])}")
        
        print(f"   Subscribers: {sub_count}")
        for sub in flows["subscribers"][:5]:
            print(f"      📥 {sub['file']}:{sub['line']} ({sub['context']}) -> {sub['handler']}")
        print()

    # Analyze decoupling
    print_section("DECOUPLING ANALYSIS")
    coverage = analyze_decoupling(event_flow_map)
    
    active_events = sum(1 for c in coverage.values() if c['active'])
    orphaned_events = sum(1 for c in coverage.values() if c['orphaned'])
    unused_subscriptions = sum(1 for c in coverage.values() if c['unused_subscription'])
    
    print(f"Active Event Types: {active_events}/{len(coverage)}")
    print(f"Orphaned Events (published but no subscribers): {orphaned_events}")
    print(f"Unused Subscriptions (subscribed but never published): {unused_subscriptions}")
    print()
    
    if orphaned_events > 0:
        print("⚠️  Orphaned Events:")
        for event_type, stats in coverage.items():
            if stats['orphaned']:
                print(f"   - {event_type}")
    
    if unused_subscriptions > 0:
        print("\n⚠️  Unused Subscriptions:")
        for event_type, stats in coverage.items():
            if stats['unused_subscription']:
                print(f"   - {event_type}")
    
    # Check ContradictionDetectorV2
    print_section("CONTRADICTION DETECTOR V2 VERIFICATION")
    
    graph_edge_added_subs = event_flow_map.get("graph.edge_added", {}).get("subscribers", [])
    contradiction_detector_subs = [
        sub for sub in graph_edge_added_subs 
        if 'ContradictionDetectorV2' in sub.get('context', '')
    ]
    
    if contradiction_detector_subs:
        print("✓ ContradictionDetectorV2 subscribes to 'graph.edge_added'")
        for sub in contradiction_detector_subs:
            print(f"  Location: {sub['file']}:{sub['line']}")
            print(f"  Handler: {sub['handler']}")
    else:
        print("⚠️  ContradictionDetectorV2 does NOT subscribe to 'graph.edge_added'")
    
    # Check StreamingBayesianUpdater
    print_section("STREAMING BAYESIAN UPDATER VERIFICATION")
    
    posterior_updated_pubs = event_flow_map.get("posterior.updated", {}).get("publishers", [])
    streaming_pubs = [
        pub for pub in posterior_updated_pubs 
        if 'StreamingBayesianUpdater' in pub.get('context', '') or 'update_from_stream' in pub.get('context', '')
    ]
    
    if streaming_pubs:
        print("✓ StreamingBayesianUpdater publishes 'posterior.updated' events")
        for pub in streaming_pubs:
            print(f"  Location: {pub['file']}:{pub['line']}")
            print(f"  Context: {pub['context']}")
            if pub['payload_keys']:
                print(f"  Payload keys: {', '.join(pub['payload_keys'])}")
    else:
        print("⚠️  StreamingBayesianUpdater does NOT publish 'posterior.updated' events")
    
    # Validation triggers
    print_section("REAL-TIME VALIDATION TRIGGERS")
    
    validation_events = [
        "graph.edge_added",
        "graph.node_added", 
        "posterior.updated",
        "contradiction.detected",
        "evidence.extracted",
        "validation.completed",
    ]
    
    print("Expected real-time triggers:")
    for event in validation_events:
        if event in event_flow_map:
            flows = event_flow_map[event]
            pub_count = len(flows['publishers'])
            sub_count = len(flows['subscribers'])
            
            if pub_count > 0 and sub_count > 0:
                print(f"  ✓ {event} ({pub_count} publishers, {sub_count} subscribers)")
            elif pub_count > 0:
                print(f"  ⚠️  {event} ({pub_count} publishers, NO subscribers)")
            elif sub_count > 0:
                print(f"  ⚠️  {event} (NO publishers, {sub_count} subscribers)")
            else:
                print(f"  ❌ {event} (not used)")
        else:
            print(f"  ❌ {event} (not found)")
    
    # Save detailed report
    report = {
        "event_types": list(analyzer.event_types),
        "event_flow_map": event_flow_map,
        "statistics": {
            "total_event_types": len(analyzer.event_types),
            "total_publishes": len(analyzer.publish_calls),
            "total_subscribes": len(analyzer.subscribe_calls),
            "active_events": active_events,
            "orphaned_events": orphaned_events,
            "unused_subscriptions": unused_subscriptions,
        },
        "decoupling_coverage": coverage,
    }
    
    output_file = "choreography_event_flow_report.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print_section("REPORT SAVED")
    print(f"Detailed report saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()
