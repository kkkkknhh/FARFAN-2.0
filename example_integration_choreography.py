"""
Integration Example: FRENTE 3 with Existing FARFAN-2.0 Modules
===============================================================
Demonstrates how to integrate the choreography module with existing
components like contradiction detection, embedding, and causal analysis.
"""

import asyncio
from typing import Any, Dict, List

from choreography.event_bus import EventBus, PDMEvent
from choreography.evidence_stream import (
    EvidenceStream,
    MechanismPrior,
    StreamingBayesianUpdater,
)

# ============================================================================
# MOCK INTEGRATIONS (Replace with actual imports in production)
# ============================================================================


class MockPolicyAnalyzer:
    """
    Mock analyzer simulating policy_processor.py integration.
    In production, replace with actual PolicyAnalysisPipeline.
    """

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.analysis_results = {}

    async def analyze_document(self, text: str, run_id: str) -> Dict[str, Any]:
        """Analyze document and publish events."""
        print(f"📄 Analyzing document (run_id={run_id})...")

        # Simulate analysis phases
        phases = [
            ("extraction.started", {"text_length": len(text)}),
            ("semantic.chunking", {"chunk_count": 10}),
            ("embedding.generated", {"embedding_dim": 768}),
            ("evidence.extracted", {"evidence_count": 25}),
            ("analysis.completed", {"status": "success"}),
        ]

        for event_type, payload in phases:
            await self.event_bus.publish(
                PDMEvent(event_type=event_type, run_id=run_id, payload=payload)
            )
            await asyncio.sleep(0.1)

        return {"status": "success", "run_id": run_id}


class MockContradictionDetector:
    """
    Mock detector simulating contradiction_deteccion.py integration.
    In production, replace with PolicyContradictionDetectorV2.
    """

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.contradictions = []

        # Subscribe to relevant events
        event_bus.subscribe("graph.edge_added", self.on_graph_update)
        event_bus.subscribe("evidence.extracted", self.on_evidence_extracted)

    async def on_graph_update(self, event: PDMEvent):
        """Check for contradictions when graph is updated."""
        payload = event.payload
        print(f"  🔍 Checking contradiction for edge: {payload.get('edge_data')}")

        # Simulate contradiction detection
        edge_data = payload.get("edge_data", {})
        if self._has_temporal_conflict(edge_data):
            contradiction = {
                "type": "temporal_conflict",
                "severity": "high",
                "edge": edge_data,
            }
            self.contradictions.append(contradiction)

            await self.event_bus.publish(
                PDMEvent(
                    event_type="contradiction.detected",
                    run_id=event.run_id,
                    payload=contradiction,
                )
            )

    async def on_evidence_extracted(self, event: PDMEvent):
        """Process extracted evidence."""
        evidence_count = event.payload.get("evidence_count", 0)
        print(f"  📊 Processing {evidence_count} evidence items for contradictions")

    def _has_temporal_conflict(self, edge_data: Dict) -> bool:
        """Simulate temporal conflict detection."""
        # For demo: detect if edge has 'conflict' keyword
        return "conflict" in str(edge_data).lower()


class MockCausalGraph:
    """
    Mock causal graph simulating dereck_beach CDAF framework.
    In production, replace with actual CDAFFramework.
    """

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.nodes = {}
        self.edges = []

    async def add_node(self, node_id: str, node_type: str, run_id: str):
        """Add node and publish event."""
        self.nodes[node_id] = {"type": node_type}

        await self.event_bus.publish(
            PDMEvent(
                event_type="graph.node_added",
                run_id=run_id,
                payload={
                    "node_id": node_id,
                    "node_type": node_type,
                    "total_nodes": len(self.nodes),
                },
            )
        )

    async def add_edge(self, source: str, target: str, relation: str, run_id: str):
        """Add edge and publish event."""
        edge_data = {"source": source, "target": target, "relation": relation}
        self.edges.append(edge_data)

        await self.event_bus.publish(
            PDMEvent(
                event_type="graph.edge_added",
                run_id=run_id,
                payload={"edge_data": edge_data, "total_edges": len(self.edges)},
            )
        )


# ============================================================================
# INTEGRATION ORCHESTRATOR
# ============================================================================


class IntegratedPDMAnalyzer:
    """
    Integrated analyzer combining all components via event bus.
    This demonstrates the power of event-driven architecture.
    """

    def __init__(self):
        # Create shared event bus
        self.event_bus = EventBus()

        # Initialize components (all share the same event bus)
        self.analyzer = MockPolicyAnalyzer(self.event_bus)
        self.detector = MockContradictionDetector(self.event_bus)
        self.graph = MockCausalGraph(self.event_bus)
        self.bayesian_updater = StreamingBayesianUpdater(self.event_bus)

        # Subscribe to key events for monitoring
        self._setup_monitoring()

    def _setup_monitoring(self):
        """Subscribe to events for monitoring and logging."""

        async def log_contradiction(event: PDMEvent):
            severity = event.payload.get("severity", "unknown")
            print(f"  ⚠️  ALERT: Contradiction detected (severity={severity})")

        async def log_posterior_update(event: PDMEvent):
            posterior = event.payload.get("posterior", {})
            mean = posterior.get("posterior_mean", 0)
            print(f"  📈 Bayesian update: posterior_mean={mean:.3f}")

        async def log_analysis_milestone(event: PDMEvent):
            print(f"  ✓ {event.event_type}: {event.payload}")

        self.event_bus.subscribe("contradiction.detected", log_contradiction)
        self.event_bus.subscribe("posterior.updated", log_posterior_update)
        self.event_bus.subscribe("analysis.completed", log_analysis_milestone)

    async def analyze_pdm(self, text: str, run_id: str = "integrated_run"):
        """
        Perform integrated PDM analysis.
        All components communicate via events automatically.
        """
        print("=" * 70)
        print("Integrated PDM Analysis with Event-Driven Architecture")
        print("=" * 70)
        print()

        # Phase 1: Document analysis
        print("Phase 1: Document Analysis")
        print("-" * 70)
        await self.analyzer.analyze_document(text, run_id)
        await asyncio.sleep(0.2)

        # Phase 2: Causal graph construction
        print("\nPhase 2: Causal Graph Construction")
        print("-" * 70)

        # Add nodes
        await self.graph.add_node("Programa_Educacion", "programa", run_id)
        await self.graph.add_node("Resultado_Cobertura", "resultado", run_id)
        await self.graph.add_node("Impacto_Calidad", "impacto", run_id)

        # Add edges (contradiction detector auto-validates)
        await self.graph.add_edge(
            "Programa_Educacion", "Resultado_Cobertura", "contributes_to", run_id
        )
        await self.graph.add_edge(
            "Resultado_Cobertura", "Impacto_Calidad", "leads_to", run_id
        )

        # Add edge with potential conflict
        await self.graph.add_edge(
            "Programa_Conflict", "Resultado_Conflict", "temporal_conflict", run_id
        )

        await asyncio.sleep(0.2)

        # Phase 3: Streaming evidence analysis
        print("\nPhase 3: Streaming Evidence Analysis")
        print("-" * 70)

        # Create sample chunks
        chunks = [
            {
                "chunk_id": f"chunk_{i}",
                "content": f"Evidencia sobre educación y calidad educativa. Chunk {i}.",
                "embedding": None,
                "metadata": {},
                "pdq_context": None,
                "token_count": 15,
                "position": (i * 100, (i + 1) * 100),
            }
            for i in range(5)
        ]

        stream = EvidenceStream(chunks)
        prior = MechanismPrior("educación", 0.5, 0.2, 0.5)

        print(f"Processing {len(chunks)} evidence chunks...")
        posterior = await self.bayesian_updater.update_from_stream(
            stream, prior, run_id
        )

        await asyncio.sleep(0.2)

        # Results summary
        print("\n" + "=" * 70)
        print("Analysis Results")
        print("-" * 70)
        print(f"Run ID: {run_id}")
        print(f"Total Events: {len(self.event_bus.event_log)}")
        print(f"Graph Nodes: {len(self.graph.nodes)}")
        print(f"Graph Edges: {len(self.graph.edges)}")
        print(f"Contradictions: {len(self.detector.contradictions)}")
        print(f"\nBayesian Analysis:")
        print(f"  Mechanism: {posterior.mechanism_name}")
        print(f"  Posterior Mean: {posterior.posterior_mean:.3f}")
        print(f"  Evidence Count: {posterior.evidence_count}")
        print(f"  Confidence: {posterior._compute_confidence()}")

        # Event breakdown
        print(f"\nEvent Breakdown:")
        event_types = {}
        for event in self.event_bus.event_log:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1

        for event_type, count in sorted(event_types.items()):
            print(f"  {event_type}: {count}")

        print("\n" + "=" * 70)
        print("✓ Integrated analysis complete!")
        print("=" * 70)

        return {
            "run_id": run_id,
            "events": len(self.event_bus.event_log),
            "contradictions": len(self.detector.contradictions),
            "posterior": posterior.to_dict(),
        }


# ============================================================================
# DEMONSTRATION
# ============================================================================


async def main():
    """Run integration demonstration."""

    # Create integrated analyzer
    analyzer = IntegratedPDMAnalyzer()

    # Sample PDM text
    pdm_text = """
    PLAN DE DESARROLLO MUNICIPAL 2024-2027
    
    Eje Estratégico 1: Educación de Calidad
    
    Objetivo: Mejorar la calidad educativa mediante programas innovadores
    y fortalecimiento de infraestructura.
    
    Metas:
    - Aumentar cobertura educativa del 85% al 95%
    - Reducir deserción escolar del 12% al 5%
    - Incrementar resultados en pruebas SABER en 15 puntos
    
    Programas:
    1. Infraestructura Educativa
    2. Capacitación Docente
    3. Alimentación Escolar
    
    Presupuesto: $5,000 millones de pesos
    """

    # Run analysis
    results = await analyzer.analyze_pdm(pdm_text, run_id="demo_integration_001")

    print(f"\n🎉 Success! Generated {results['events']} events")


if __name__ == "__main__":
    asyncio.run(main())
