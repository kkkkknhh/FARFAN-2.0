#!/usr/bin/env python3
"""
EventBus Choreography Analysis - Comprehensive Publisher-Subscriber Mapping
=============================================================================

Analyzes all EventBus publisher-subscriber relationships across the codebase
to ensure proper decoupling, identify missing subscriptions, and detect
potential event storms.

SIN_CARRETA Compliance:
- Deterministic analysis with fixed seed
- Contract-based validation of event flows
- Audit trail generation for all findings
"""

import ast
import hashlib
import json
import logging
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EventPublication:
    """Contract: Single event publication instance"""

    event_type: str
    file_path: str
    line_number: int
    function_name: str
    payload_keys: List[str]
    publisher_component: str

    def __hash__(self):
        return hash(f"{self.file_path}:{self.line_number}:{self.event_type}")


@dataclass
class EventSubscription:
    """Contract: Single event subscription instance"""

    event_type: str
    file_path: str
    line_number: int
    handler_name: str
    subscriber_component: str

    def __hash__(self):
        return hash(
            f"{self.file_path}:{self.line_number}:{self.event_type}:{self.handler_name}"
        )


@dataclass
class EventFlow:
    """Contract: Complete event flow from publisher to subscriber(s)"""

    event_type: str
    publishers: List[EventPublication]
    subscribers: List[EventSubscription]
    is_orphaned: bool
    is_unused: bool

    @property
    def health_status(self) -> str:
        if self.is_orphaned:
            return "ORPHANED (published but no subscribers)"
        elif self.is_unused:
            return "UNUSED (subscribed but no publishers)"
        else:
            return "HEALTHY"


@dataclass
class ComponentAnalysis:
    """Contract: Component-level event bus usage"""

    component_name: str
    publishes: Set[str]
    subscribes: Set[str]
    is_extractor: bool = False
    is_validator: bool = False
    is_auditor: bool = False
    has_direct_coupling: bool = False  # Should always be False


@dataclass
class EventStormRisk:
    """Contract: Potential event storm scenario"""

    event_chain: List[str]
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    description: str
    affected_components: List[str]


@dataclass
class AnalysisReport:
    """Contract: Complete analysis report"""

    event_flows: Dict[str, EventFlow]
    components: Dict[str, ComponentAnalysis]
    orphaned_events: List[str]
    unused_subscriptions: List[str]
    event_storm_risks: List[EventStormRisk]
    missing_subscriptions: List[Tuple[str, str]]  # (component, expected_event)
    decoupling_violations: List[Tuple[str, str]]  # (source, target)
    analysis_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "event_flows": {k: asdict(v) for k, v in self.event_flows.items()},
            "components": {
                k: {
                    "component_name": v.component_name,
                    "publishes": list(v.publishes),
                    "subscribes": list(v.subscribes),
                    "is_extractor": v.is_extractor,
                    "is_validator": v.is_validator,
                    "is_auditor": v.is_auditor,
                    "has_direct_coupling": v.has_direct_coupling,
                }
                for k, v in self.components.items()
            },
            "orphaned_events": self.orphaned_events,
            "unused_subscriptions": self.unused_subscriptions,
            "event_storm_risks": [asdict(r) for r in self.event_storm_risks],
            "missing_subscriptions": self.missing_subscriptions,
            "decoupling_violations": self.decoupling_violations,
            "analysis_hash": self.analysis_hash,
        }


class EventBusAnalyzer:
    """
    Analyzes EventBus usage patterns across codebase.

    SIN_CARRETA Compliance:
    - Deterministic AST traversal
    - No external network calls
    - Reproducible analysis with fixed ordering
    """

    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.publications: List[EventPublication] = []
        self.subscriptions: List[EventSubscription] = []
        self.components: Dict[str, ComponentAnalysis] = {}

    def analyze(self) -> AnalysisReport:
        """
        Execute complete analysis.

        Contract: Deterministic analysis with fixed ordering.
        """
        logger.info("Starting EventBus choreography analysis...")

        # Phase 1: Scan codebase for publications and subscriptions
        self._scan_codebase()

        # Phase 2: Build event flows
        event_flows = self._build_event_flows()

        # Phase 3: Analyze components
        self._analyze_components()

        # Phase 4: Detect orphaned and unused events
        orphaned = self._detect_orphaned_events(event_flows)
        unused = self._detect_unused_subscriptions(event_flows)

        # Phase 5: Analyze event storm risks
        storm_risks = self._analyze_event_storm_risks(event_flows)

        # Phase 6: Detect missing subscriptions
        missing_subs = self._detect_missing_subscriptions()

        # Phase 7: Detect direct coupling violations
        violations = self._detect_coupling_violations()

        # Generate deterministic hash
        analysis_hash = self._compute_analysis_hash(event_flows)

        report = AnalysisReport(
            event_flows=event_flows,
            components=self.components,
            orphaned_events=orphaned,
            unused_subscriptions=unused,
            event_storm_risks=storm_risks,
            missing_subscriptions=missing_subs,
            decoupling_violations=violations,
            analysis_hash=analysis_hash,
        )

        logger.info(f"Analysis complete. Hash: {analysis_hash[:16]}...")
        return report

    def _scan_codebase(self):
        """Scan Python files for EventBus usage"""
        python_files = []
        for root, dirs, files in os.walk(self.root_dir):
            # Skip venv and .git
            dirs[:] = [d for d in dirs if d not in ["venv", ".git", "__pycache__"]]

            for file in sorted(files):  # Sort for determinism
                if file.endswith(".py"):
                    python_files.append(Path(root) / file)

        for file_path in python_files:
            self._scan_file(file_path)

    def _scan_file(self, file_path: Path):
        """Scan single Python file for EventBus patterns"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()

            tree = ast.parse(source, filename=str(file_path))
            visitor = EventBusVisitor(file_path, source)
            visitor.visit(tree)

            self.publications.extend(visitor.publications)
            self.subscriptions.extend(visitor.subscriptions)

        except Exception as e:
            logger.warning(f"Error scanning {file_path}: {e}")

    def _build_event_flows(self) -> Dict[str, EventFlow]:
        """Build event flows from publications and subscriptions"""
        flows = {}

        # Group by event type
        pubs_by_type = defaultdict(list)
        subs_by_type = defaultdict(list)

        for pub in self.publications:
            pubs_by_type[pub.event_type].append(pub)

        for sub in self.subscriptions:
            subs_by_type[sub.event_type].append(sub)

        # Combine all event types
        all_events = set(pubs_by_type.keys()) | set(subs_by_type.keys())

        for event_type in sorted(all_events):  # Sort for determinism
            pubs = pubs_by_type.get(event_type, [])
            subs = subs_by_type.get(event_type, [])

            flows[event_type] = EventFlow(
                event_type=event_type,
                publishers=pubs,
                subscribers=subs,
                is_orphaned=len(pubs) > 0 and len(subs) == 0,
                is_unused=len(pubs) == 0 and len(subs) > 0,
            )

        return flows

    def _analyze_components(self):
        """Analyze component-level event bus usage"""
        # Build component mapping
        for pub in self.publications:
            comp = pub.publisher_component
            if comp not in self.components:
                self.components[comp] = ComponentAnalysis(
                    component_name=comp,
                    publishes=set(),
                    subscribes=set(),
                    is_extractor="extractor" in comp.lower()
                    or "extraction" in comp.lower(),
                    is_validator="validator" in comp.lower()
                    or "contradiction" in comp.lower(),
                    is_auditor="auditor" in comp.lower() or "audit" in comp.lower(),
                )
            self.components[comp].publishes.add(pub.event_type)

        for sub in self.subscriptions:
            comp = sub.subscriber_component
            if comp not in self.components:
                self.components[comp] = ComponentAnalysis(
                    component_name=comp,
                    publishes=set(),
                    subscribes=set(),
                    is_extractor="extractor" in comp.lower()
                    or "extraction" in comp.lower(),
                    is_validator="validator" in comp.lower()
                    or "contradiction" in comp.lower(),
                    is_auditor="auditor" in comp.lower() or "audit" in comp.lower(),
                )
            self.components[comp].subscribes.add(sub.event_type)

    def _detect_orphaned_events(self, flows: Dict[str, EventFlow]) -> List[str]:
        """Detect events that are published but have no subscribers"""
        return [event_type for event_type, flow in flows.items() if flow.is_orphaned]

    def _detect_unused_subscriptions(self, flows: Dict[str, EventFlow]) -> List[str]:
        """Detect subscriptions that have no publishers"""
        return [event_type for event_type, flow in flows.items() if flow.is_unused]

    def _analyze_event_storm_risks(
        self, flows: Dict[str, EventFlow]
    ) -> List[EventStormRisk]:
        """Detect potential event storm scenarios"""
        risks = []

        # Build event dependency graph
        event_deps = defaultdict(set)
        for pub in self.publications:
            for sub in self.subscriptions:
                if pub.event_type == sub.event_type:
                    # Check if subscriber also publishes events
                    subscriber_pubs = [
                        p
                        for p in self.publications
                        if p.publisher_component == sub.subscriber_component
                    ]
                    for sp in subscriber_pubs:
                        event_deps[pub.event_type].add(sp.event_type)

        # Detect cycles (feedback loops)
        cycles = self._detect_cycles(event_deps)
        for cycle in cycles:
            risks.append(
                EventStormRisk(
                    event_chain=cycle,
                    severity="HIGH",
                    description=f"Feedback loop detected: {' -> '.join(cycle)}",
                    affected_components=[],
                )
            )

        # Detect fan-out scenarios
        for event_type, flow in flows.items():
            if len(flow.subscribers) > 10:
                risks.append(
                    EventStormRisk(
                        event_chain=[event_type],
                        severity="MEDIUM",
                        description=f"High fan-out: {len(flow.subscribers)} subscribers for {event_type}",
                        affected_components=[
                            s.subscriber_component for s in flow.subscribers
                        ],
                    )
                )

        return risks

    def _detect_cycles(self, graph: Dict[str, Set[str]]) -> List[List[str]]:
        """Detect cycles in event dependency graph"""
        cycles = []
        visited = set()
        rec_stack = []

        def dfs(node, path):
            if node in rec_stack:
                # Cycle detected
                cycle_start = rec_stack.index(node)
                cycles.append(rec_stack[cycle_start:] + [node])
                return

            if node in visited:
                return

            visited.add(node)
            rec_stack.append(node)

            for neighbor in graph.get(node, []):
                dfs(neighbor, path + [neighbor])

            rec_stack.remove(node)

        for node in sorted(graph.keys()):  # Sort for determinism
            dfs(node, [node])

        return cycles

    def _detect_missing_subscriptions(self) -> List[Tuple[str, str]]:
        """Detect components that should subscribe to events based on architecture"""
        missing = []

        # Expected subscriptions based on architecture
        expected_subscriptions = {
            "ContradictionDetectorV2": {"graph.edge_added", "graph.node_added"},
            "StreamingBayesianUpdater": set(),  # Only publishes
            "AxiomaticValidator": {"contradiction.detected", "posterior.updated"},
            "ExtractionPipeline": set(),  # Only publishes
        }

        for component_pattern, expected_events in expected_subscriptions.items():
            # Find matching components
            matching_comps = [
                c for c in self.components.keys() if component_pattern in c
            ]

            for comp in matching_comps:
                actual_subs = self.components[comp].subscribes
                missing_events = expected_events - actual_subs

                for event in missing_events:
                    missing.append((comp, event))

        return missing

    def _detect_coupling_violations(self) -> List[Tuple[str, str]]:
        """Detect direct coupling between components (violates event-driven architecture)"""
        # This would require more sophisticated analysis (imports, direct method calls)
        # For now, return empty list - can be extended
        return []

    def _compute_analysis_hash(self, flows: Dict[str, EventFlow]) -> str:
        """Compute deterministic hash of analysis results"""
        # Create canonical representation
        canonical = {
            "event_types": sorted(flows.keys()),
            "pub_count": len(self.publications),
            "sub_count": len(self.subscriptions),
            "component_count": len(self.components),
        }

        canonical_str = json.dumps(canonical, sort_keys=True)
        return hashlib.sha256(canonical_str.encode()).hexdigest()


class EventBusVisitor(ast.NodeVisitor):
    """AST visitor to extract EventBus publish/subscribe calls"""

    def __init__(self, file_path: Path, source: str):
        self.file_path = file_path
        self.source_lines = source.split("\n")
        self.publications = []
        self.subscriptions = []
        self.current_function = None
        self.current_class = None

    def visit_ClassDef(self, node):
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node):
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function

    def visit_Call(self, node):
        # Check for publish calls
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "publish":
                self._extract_publication(node)
            elif node.func.attr == "subscribe":
                self._extract_subscription(node)

        self.generic_visit(node)

    def _extract_publication(self, node):
        """Extract publication information"""
        # Get event type from PDMEvent constructor
        if node.args and isinstance(node.args[0], ast.Call):
            event_node = node.args[0]
            event_type = self._extract_event_type(event_node)
            payload_keys = self._extract_payload_keys(event_node)

            if event_type:
                pub = EventPublication(
                    event_type=event_type,
                    file_path=str(self.file_path),
                    line_number=node.lineno,
                    function_name=self.current_function or "module_level",
                    payload_keys=payload_keys,
                    publisher_component=self.current_class or self.file_path.stem,
                )
                self.publications.append(pub)

    def _extract_subscription(self, node):
        """Extract subscription information"""
        if len(node.args) >= 2:
            # First arg is event type
            if isinstance(node.args[0], ast.Constant):
                event_type = node.args[0].value
            elif isinstance(node.args[0], ast.Str):  # Python 3.7 compat
                event_type = node.args[0].s
            else:
                return

            # Second arg is handler
            handler_name = self._extract_handler_name(node.args[1])

            sub = EventSubscription(
                event_type=event_type,
                file_path=str(self.file_path),
                line_number=node.lineno,
                handler_name=handler_name,
                subscriber_component=self.current_class or self.file_path.stem,
            )
            self.subscriptions.append(sub)

    def _extract_event_type(self, event_node):
        """Extract event_type from PDMEvent constructor"""
        for keyword in event_node.keywords:
            if keyword.arg == "event_type":
                if isinstance(keyword.value, ast.Constant):
                    return keyword.value.value
                elif isinstance(keyword.value, ast.Str):
                    return keyword.value.s
        return None

    def _extract_payload_keys(self, event_node):
        """Extract payload keys from PDMEvent constructor"""
        for keyword in event_node.keywords:
            if keyword.arg == "payload":
                if isinstance(keyword.value, ast.Dict):
                    keys = []
                    for key in keyword.value.keys:
                        if isinstance(key, ast.Constant):
                            keys.append(key.value)
                        elif isinstance(key, ast.Str):
                            keys.append(key.s)
                    return keys
        return []

    def _extract_handler_name(self, handler_node):
        """Extract handler function name"""
        if isinstance(handler_node, ast.Attribute):
            return handler_node.attr
        elif isinstance(handler_node, ast.Name):
            return handler_node.id
        else:
            return "unknown_handler"


def main():
    """Execute analysis and generate report"""
    analyzer = EventBusAnalyzer(".")
    report = analyzer.analyze()

    # Print summary
    print("\n" + "=" * 80)
    print("EVENTBUS CHOREOGRAPHY ANALYSIS REPORT")
    print("=" * 80 + "\n")

    print(f"Total Event Types: {len(report.event_flows)}")
    print(
        f"Total Publications: {sum(len(f.publishers) for f in report.event_flows.values())}"
    )
    print(
        f"Total Subscriptions: {sum(len(f.subscribers) for f in report.event_flows.values())}"
    )
    print(f"Total Components: {len(report.components)}\n")

    print("Event Flow Health:")
    for event_type, flow in sorted(report.event_flows.items()):
        print(
            f"  {event_type}: {flow.health_status} ({len(flow.publishers)} pub, {len(flow.subscribers)} sub)"
        )

    if report.orphaned_events:
        print(f"\n⚠️  Orphaned Events: {len(report.orphaned_events)}")
        for event in report.orphaned_events:
            print(f"    - {event}")

    if report.unused_subscriptions:
        print(f"\n⚠️  Unused Subscriptions: {len(report.unused_subscriptions)}")
        for event in report.unused_subscriptions:
            print(f"    - {event}")

    if report.event_storm_risks:
        print(f"\n⚠️  Event Storm Risks: {len(report.event_storm_risks)}")
        for risk in report.event_storm_risks:
            print(f"    - {risk.severity}: {risk.description}")

    if report.missing_subscriptions:
        print(f"\n⚠️  Missing Subscriptions: {len(report.missing_subscriptions)}")
        for comp, event in report.missing_subscriptions:
            print(f"    - {comp} should subscribe to {event}")

    print(f"\nAnalysis Hash: {report.analysis_hash}\n")

    # Save full report to JSON
    output_path = "eventbus_analysis_report.json"
    with open(output_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    print(f"Full report saved to: {output_path}")


if __name__ == "__main__":
    main()
