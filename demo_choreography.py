"""
Demonstration of FRENTE 3: COREOGRAFÍA Implementation
======================================================
Shows practical usage of event bus and streaming evidence pipeline.
"""

import asyncio
from datetime import datetime

from choreography.event_bus import ContradictionDetectorV2, EventBus, PDMEvent
from choreography.evidence_stream import (
    EvidenceStream,
    MechanismPrior,
    StreamingBayesianUpdater,
)


async def main():
    """Demonstrate the choreography components."""

    print("=" * 70)
    print("FRENTE 3: COREOGRAFÍA - Demonstration")
    print("=" * 70)
    print()

    # ========================================================================
    # Part 1: Event Bus Demonstration
    # ========================================================================

    print("Part 1: Event Bus for Phase Transitions")
    print("-" * 70)

    # Create event bus
    bus = EventBus()
    print("✓ Event bus created")

    # Subscribe handlers
    async def on_graph_update(event: PDMEvent):
        print(f"  📊 Graph updated: {event.payload.get('action')}")

    async def on_validation_result(event: PDMEvent):
        status = event.payload.get("status")
        emoji = "✓" if status == "passed" else "✗"
        print(f"  {emoji} Validation: {status}")

    bus.subscribe("graph.updated", on_graph_update)
    bus.subscribe("validation.completed", on_validation_result)
    print("✓ Handlers subscribed")

    # Publish events
    print("\nPublishing events...")

    await bus.publish(
        PDMEvent(
            event_type="graph.updated",
            run_id="demo_run",
            payload={"action": "Added causal edge: Programa A -> Resultado B"},
        )
    )

    await bus.publish(
        PDMEvent(
            event_type="validation.completed",
            run_id="demo_run",
            payload={"status": "passed", "validator": "TemporalConsistency"},
        )
    )

    await asyncio.sleep(0.1)  # Let handlers execute

    print(f"\n✓ Event log contains {len(bus.event_log)} events")

    # ========================================================================
    # Part 2: Contradiction Detector
    # ========================================================================

    print("\n" + "=" * 70)
    print("Part 2: Real-time Contradiction Detection")
    print("-" * 70)

    # Create detector (automatically subscribes to graph events)
    detector = ContradictionDetectorV2(bus)
    print("✓ Contradiction detector initialized")

    # Monitor for contradictions
    detected = []

    async def on_contradiction(event: PDMEvent):
        severity = event.payload.get("severity")
        edge = event.payload.get("edge")
        print(
            f"  ⚠️  CONTRADICTION ({severity}): {edge.get('source')} -> {edge.get('target')}"
        )
        detected.append(event)

    bus.subscribe("contradiction.detected", on_contradiction)

    # Test valid edge
    print("\nAdding valid edge...")
    await bus.publish(
        PDMEvent(
            event_type="graph.edge_added",
            run_id="demo_run",
            payload={
                "source": "Objetivo_A",
                "target": "Resultado_B",
                "relation": "leads_to",
            },
        )
    )
    await asyncio.sleep(0.1)

    # Test self-loop (contradiction)
    print("\nAdding self-loop edge (should trigger contradiction)...")
    await bus.publish(
        PDMEvent(
            event_type="graph.edge_added",
            run_id="demo_run",
            payload={
                "source": "Programa_X",
                "target": "Programa_X",
                "relation": "depends_on",
            },
        )
    )
    await asyncio.sleep(0.1)

    print(f"\n✓ Detected {len(detected)} contradiction(s)")

    # ========================================================================
    # Part 3: Streaming Evidence Pipeline
    # ========================================================================

    print("\n" + "=" * 70)
    print("Part 3: Streaming Evidence Analysis")
    print("-" * 70)

    # Create sample evidence chunks (simulating a PDM document)
    evidence_chunks = [
        {
            "chunk_id": "chunk_1",
            "content": "El municipio implementará programas de educación de calidad para mejorar los indicadores académicos.",
            "embedding": None,
            "metadata": {"section": "Eje Educación", "page": 15},
            "pdq_context": None,
            "token_count": 25,
            "position": (0, 100),
        },
        {
            "chunk_id": "chunk_2",
            "content": "Se asignarán $500 millones para infraestructura educativa y capacitación docente.",
            "embedding": None,
            "metadata": {"section": "Eje Educación", "page": 16},
            "pdq_context": None,
            "token_count": 20,
            "position": (100, 200),
        },
        {
            "chunk_id": "chunk_3",
            "content": "Meta: Aumentar cobertura educativa del 85% al 95% en educación media.",
            "embedding": None,
            "metadata": {"section": "Metas", "page": 20},
            "pdq_context": None,
            "token_count": 18,
            "position": (200, 300),
        },
        {
            "chunk_id": "chunk_4",
            "content": "Indicador: Tasa de deserción escolar. Línea base: 12%. Meta 2027: 5%.",
            "embedding": None,
            "metadata": {"section": "Indicadores", "page": 25},
            "pdq_context": None,
            "token_count": 22,
            "position": (300, 400),
        },
        {
            "chunk_id": "chunk_5",
            "content": "Se fortalecerá la articulación educación-sector productivo mediante convenios con empresas locales.",
            "embedding": None,
            "metadata": {"section": "Estrategias", "page": 30},
            "pdq_context": None,
            "token_count": 20,
            "position": (400, 500),
        },
    ]

    # Create evidence stream
    stream = EvidenceStream(evidence_chunks)
    print(f"✓ Created evidence stream with {len(evidence_chunks)} chunks")

    # Create prior belief about education mechanism
    prior = MechanismPrior(
        mechanism_name="educación",
        prior_mean=0.5,  # Neutral prior
        prior_std=0.2,
        confidence=0.5,
    )
    print(f"✓ Prior belief: μ={prior.prior_mean:.2f}, σ={prior.prior_std:.2f}")

    # Create streaming updater with event bus
    updater = StreamingBayesianUpdater(bus)
    print("✓ Streaming Bayesian updater initialized")

    # Monitor posterior updates
    update_count = [0]

    async def on_posterior_update(event: PDMEvent):
        update_count[0] += 1
        posterior_data = event.payload.get("posterior", {})
        progress = event.payload.get("progress", 0)
        print(
            f"  📈 Update {update_count[0]}: "
            f"μ={posterior_data.get('posterior_mean', 0):.3f}, "
            f"evidence={posterior_data.get('evidence_count', 0)}, "
            f"progress={progress:.0%}"
        )

    bus.subscribe("posterior.updated", on_posterior_update)

    # Perform streaming update
    print("\nProcessing evidence stream...")
    posterior = await updater.update_from_stream(stream, prior, run_id="demo_run")
    await asyncio.sleep(0.2)

    # Display final results
    print("\n" + "-" * 70)
    print("Final Results:")
    print(f"  Mechanism: {posterior.mechanism_name}")
    print(f"  Posterior Mean: {posterior.posterior_mean:.3f}")
    print(f"  Posterior Std: {posterior.posterior_std:.3f}")
    print(f"  Evidence Count: {posterior.evidence_count}")
    print(
        f"  Credible Interval (95%): [{posterior.credible_interval_95[0]:.3f}, {posterior.credible_interval_95[1]:.3f}]"
    )
    print(f"  Confidence Level: {posterior._compute_confidence()}")

    # ========================================================================
    # Summary
    # ========================================================================

    print("\n" + "=" * 70)
    print("Summary")
    print("-" * 70)
    print(f"Total events published: {len(bus.event_log)}")
    print(f"Contradictions detected: {len(detector.detected_contradictions)}")
    print(f"Posterior updates: {update_count[0]}")
    print(f"Evidence chunks processed: {len(evidence_chunks)}")
    print()
    print("✓ FRENTE 3: COREOGRAFÍA demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
