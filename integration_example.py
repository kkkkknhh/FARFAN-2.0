#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration Example: Orchestrator with Contradiction Detection
===============================================================

This example demonstrates how to integrate the orchestrator with the existing
contradiction_deteccion.py module while maintaining calibration constants
and audit trail.

NOTE: This is a demonstration of integration patterns. In production, you would
      replace the placeholder implementations in orchestrator.py with actual
      calls to the analytical modules.
"""

from datetime import datetime
from pathlib import Path

from orchestrator import PhaseResult, create_orchestrator


class IntegratedOrchestrator:
    """
    Extended orchestrator with real module integration.

    This class demonstrates how to integrate actual analytical modules
    while preserving the orchestrator's guarantees:
    - Deterministic execution
    - Calibration constant management
    - Audit trail generation
    - Error handling
    """

    def __init__(self, log_dir: Path = None, **calibration_overrides):
        """Initialize with base orchestrator."""
        self.base_orchestrator = create_orchestrator(
            log_dir=log_dir, **calibration_overrides
        )

        # Import actual modules when available
        self.detector = None
        try:
            # Uncomment when ready to integrate:
            # from contradiction_deteccion import ContradictionDetector
            # self.detector = ContradictionDetector()
            pass
        except ImportError:
            self.base_orchestrator.logger.warning(
                "Contradiction detector not available - using placeholder"
            )

    def _extract_statements_integrated(
        self, text: str, plan_name: str, dimension: str
    ) -> PhaseResult:
        """
        Integrated statement extraction with actual module.

        Pattern:
        1. Call actual module with calibration constants
        2. Wrap result in PhaseResult contract
        3. Handle errors gracefully
        4. Return structured result
        """
        timestamp = datetime.now().isoformat()

        try:
            # Example integration (actual module would be called here)
            # statements = self.detector._extract_policy_statements(text, dimension)
            statements = []  # Placeholder

            return PhaseResult(
                phase_name="extract_statements",
                inputs={
                    "text_length": len(text),
                    "plan_name": plan_name,
                    "dimension": dimension,
                },
                outputs={"statements": statements},
                metrics={
                    "statements_count": len(statements),
                    "avg_statement_length": (
                        sum(len(s.text) for s in statements) / max(1, len(statements))
                        if statements
                        else 0
                    ),
                },
                timestamp=timestamp,
                status="success",
            )

        except Exception as e:
            self.base_orchestrator.logger.error(
                f"Statement extraction failed: {e}", exc_info=True
            )

            # Return fallback result
            return PhaseResult(
                phase_name="extract_statements",
                inputs={"text_length": len(text)},
                outputs={"statements": []},
                metrics={"statements_count": 0},
                timestamp=timestamp,
                status="error",
                error=str(e),
            )

    def orchestrate_analysis(
        self, text: str, plan_name: str = "PDM", dimension: str = "estratégico"
    ):
        """
        Execute analysis with integrated modules.

        This method shows how to override specific phases while maintaining
        orchestrator guarantees.
        """
        # Could override specific phase implementations here
        # For now, delegate to base orchestrator
        return self.base_orchestrator.orchestrate_analysis(text, plan_name, dimension)


def demonstrate_integration():
    """Demonstrate integration patterns."""
    print("\n" + "=" * 70)
    print("ORCHESTRATOR INTEGRATION DEMONSTRATION")
    print("=" * 70 + "\n")

    # Create integrated orchestrator
    orchestrator = IntegratedOrchestrator(
        coherence_threshold=0.75, causal_incoherence_limit=4
    )

    print("1. Orchestrator created with custom calibration:")
    print(f"   - Coherence threshold: 0.75 (override from default 0.7)")
    print(f"   - Causal incoherence limit: 4 (override from default 5)")
    print()

    # Execute analysis
    test_text = """
    Plan de Desarrollo Municipal 2024-2028
    
    Objetivo estratégico: Mejorar la calidad de vida de los ciudadanos.
    
    Meta: Reducir la pobreza multidimensional en 10%.
    Inversión: $1,000,000,000 en programas sociales.
    Plazo: 4 años.
    """

    print("2. Executing analysis pipeline...")
    result = orchestrator.orchestrate_analysis(
        text=test_text, plan_name="PDM_Demo_2024", dimension="estratégico"
    )
    print("   ✓ Pipeline completed")
    print()

    print("3. Result structure:")
    print(f"   - Plan: {result['plan_name']}")
    print(f"   - Dimension: {result['dimension']}")
    print(f"   - Total statements: {result['total_statements']}")
    print(f"   - Total contradictions: {result['total_contradictions']}")
    print()

    print("4. Phase execution summary:")
    for phase in [
        "extract_statements",
        "detect_contradictions",
        "analyze_regulatory_constraints",
        "calculate_coherence_metrics",
        "generate_audit_summary",
    ]:
        if phase in result:
            status = result[phase]["status"]
            print(f"   ✓ {phase:35s} - {status}")
    print()

    print("5. Calibration preserved in output:")
    calibration = result["orchestration_metadata"]["calibration"]
    print(f"   - coherence_threshold: {calibration['coherence_threshold']}")
    print(f"   - causal_incoherence_limit: {calibration['causal_incoherence_limit']}")
    print()

    print("6. Audit log generated:")
    log_dir = Path("logs/orchestrator")
    log_files = list(log_dir.glob("audit_log_PDM_Demo_2024_*.json"))
    if log_files:
        print(f"   ✓ {log_files[0].name}")
    else:
        print("   (No log file found in demonstration mode)")
    print()

    print("=" * 70)
    print("INTEGRATION PATTERNS DEMONSTRATED:")
    print("=" * 70)
    print()
    print("✓ Calibration constant management")
    print("✓ Structured PhaseResult data contracts")
    print("✓ Error handling with fallback values")
    print("✓ Audit trail generation")
    print("✓ Deterministic execution flow")
    print()
    print("=" * 70 + "\n")


if __name__ == "__main__":
    demonstrate_integration()
