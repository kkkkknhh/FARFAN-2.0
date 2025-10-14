#!/usr/bin/env python3
"""
Example: Integration of Risk Mitigation Layer with Orchestrator
Demonstrates how to integrate risk assessment into the FARFAN pipeline
"""

from risk_mitigation_layer import (
    RiskSeverity, RiskCategory, Risk, RiskRegistry, RiskMitigationLayer,
    create_default_risk_registry,
    CriticalRiskUnmitigatedException, HighRiskUnmitigatedException
)
from dataclasses import dataclass, field
from typing import Dict, List, Any
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("example_risk_integration")


@dataclass
class ExamplePipelineContext:
    """Simplified version of PipelineContext for demonstration"""
    pdf_path: str = ""
    raw_text: str = ""
    sections: Dict[str, str] = field(default_factory=dict)
    causal_chains: List[Dict] = field(default_factory=list)
    nodes: Dict[str, Any] = field(default_factory=dict)
    financial_allocations: Dict[str, float] = field(default_factory=dict)
    degradations: List[Dict] = field(default_factory=list)


class EnhancedOrchestrator:
    """
    Example Orchestrator with Risk Mitigation Layer integrated
    
    Demonstrates:
    1. How to create and configure RiskRegistry
    2. How to wrap stage execution with risk assessment
    3. How to handle different severity levels
    4. How to generate mitigation reports
    """
    
    def __init__(self):
        # Create default registry with common risks
        self.risk_registry = create_default_risk_registry()
        
        # Add custom risks specific to this orchestrator
        self._register_custom_risks()
        
        # Create mitigation layer
        self.mitigation_layer = RiskMitigationLayer(self.risk_registry)
        
        logger.info("EnhancedOrchestrator initialized with risk mitigation")
    
    def _register_custom_risks(self):
        """Register custom application-specific risks"""
        
        # Custom risk: PDF path doesn't exist
        self.risk_registry.register_risk(
            "STAGE_1_2",
            Risk(
                category=RiskCategory.PDF_UNREADABLE,
                severity=RiskSeverity.CRITICAL,
                probability=0.05,
                impact=1.0,
                detector_predicate=lambda ctx: not ctx.pdf_path or len(ctx.pdf_path) == 0,
                mitigation_strategy=lambda ctx: "Cannot proceed without valid PDF path",
                description="PDF path is missing or empty"
            )
        )
        
        # Custom risk: Very long document (performance concern)
        self.risk_registry.register_risk(
            "STAGE_3",
            Risk(
                category=RiskCategory.TEXT_TOO_SHORT,  # Reusing enum
                severity=RiskSeverity.LOW,
                probability=0.2,
                impact=0.3,
                detector_predicate=lambda ctx: len(ctx.raw_text) > 1000000,
                mitigation_strategy=lambda ctx: "Using chunked processing for large document",
                description="Document is very large, may impact performance"
            )
        )
        
        logger.info("Custom risks registered")
    
    def _stage_extract_document(self, ctx: ExamplePipelineContext) -> ExamplePipelineContext:
        """Stage 1-2: Document Extraction (mock implementation)"""
        logger.info("[STAGE 1-2] Extracting document...")
        
        # Simulate extraction - always populate with sufficient content
        ctx.raw_text = "Sample document text with sufficient length for processing. " * 10
        ctx.sections = {
            "introduction": "Introduction text",
            "objectives": "Objectives text",
            "budget": "Budget text",
            "implementation": "Implementation details"
        }
        
        logger.info(f"  Extracted {len(ctx.raw_text)} characters")
        return ctx
    
    def _stage_causal_extraction(self, ctx: ExamplePipelineContext) -> ExamplePipelineContext:
        """Stage 4: Causal Extraction (mock implementation)"""
        logger.info("[STAGE 4] Causal extraction...")
        
        # Simulate causal extraction
        if len(ctx.raw_text) > 100:
            ctx.causal_chains = [
                {"from": "input", "to": "output", "mechanism": "process"}
            ]
            ctx.nodes = {
                "node1": {"type": "input", "text": "Input node"},
                "node2": {"type": "output", "text": "Output node"}
            }
        
        logger.info(f"  Extracted {len(ctx.causal_chains)} causal chains")
        return ctx
    
    def _stage_financial_audit(self, ctx: ExamplePipelineContext) -> ExamplePipelineContext:
        """Stage 6: Financial Audit (mock implementation)"""
        logger.info("[STAGE 6] Financial audit...")
        
        # Simulate financial audit
        ctx.financial_allocations = {
            "education": 1000000,
            "health": 800000
        }
        
        logger.info(f"  Found {len(ctx.financial_allocations)} budget items")
        return ctx
    
    def process_with_risk_mitigation(self, pdf_path: str) -> ExamplePipelineContext:
        """
        Process pipeline with full risk mitigation
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Processed context with results and degradation documentation
        """
        logger.info("="*80)
        logger.info("Starting pipeline with risk mitigation")
        logger.info("="*80)
        
        ctx = ExamplePipelineContext(pdf_path=pdf_path)
        
        try:
            # STAGE 1-2: Document Extraction with risk assessment
            ctx = self.mitigation_layer.wrap_stage_execution(
                stage="STAGE_1_2",
                stage_function=self._stage_extract_document,
                context=ctx
            )
            
            # STAGE 4: Causal Extraction with risk assessment
            ctx = self.mitigation_layer.wrap_stage_execution(
                stage="STAGE_4",
                stage_function=self._stage_causal_extraction,
                context=ctx
            )
            
            # STAGE 6: Financial Audit with risk assessment
            ctx = self.mitigation_layer.wrap_stage_execution(
                stage="STAGE_6",
                stage_function=self._stage_financial_audit,
                context=ctx
            )
            
            logger.info("="*80)
            logger.info("✅ Pipeline completed successfully")
            logger.info("="*80)
            
        except CriticalRiskUnmitigatedException as e:
            logger.critical(f"Pipeline aborted due to CRITICAL risk: {e}")
            raise
        
        except HighRiskUnmitigatedException as e:
            logger.error(f"Pipeline aborted due to HIGH risk: {e}")
            raise
        
        return ctx
    
    def generate_risk_report(self):
        """Generate comprehensive risk mitigation report"""
        report = self.mitigation_layer.get_mitigation_report()
        
        print("\n" + "="*80)
        print("RISK MITIGATION REPORT")
        print("="*80)
        
        if report['total_mitigations'] == 0:
            print("No mitigations were executed (no risks detected)")
            return
        
        print(f"Total Mitigations: {report['total_mitigations']}")
        print(f"Successful: {report['successful']}")
        print(f"Failed: {report['failed']}")
        print(f"Success Rate: {report['success_rate']:.1%}")
        print(f"Total Time: {report['total_time']:.2f}s")
        print(f"Average Time: {report['average_time']:.2f}s")
        
        print("\nBy Severity:")
        for severity, stats in report['by_severity'].items():
            print(f"  {severity}: {stats['successful']}/{stats['total']} successful")
        
        print("\nDetailed Events:")
        for detail in report['details']:
            status = "✓" if detail['success'] else "✗"
            print(f"  {status} [{detail['severity']}] {detail['category']}")
            print(f"     Attempts: {detail['attempts']}, Time: {detail['time']:.2f}s")
            print(f"     Outcome: {detail['outcome']}")


def example_successful_pipeline():
    """Example: Pipeline with no risks detected"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Successful Pipeline (No Risks)")
    print("="*80)
    
    orchestrator = EnhancedOrchestrator()
    
    # Process with valid input
    ctx = orchestrator.process_with_risk_mitigation(pdf_path="/valid/path/to/document.pdf")
    
    print(f"\nResults:")
    print(f"  Text extracted: {len(ctx.raw_text)} chars")
    print(f"  Sections: {len(ctx.sections)}")
    print(f"  Causal chains: {len(ctx.causal_chains)}")
    print(f"  Degradations: {len(ctx.degradations)}")
    
    orchestrator.generate_risk_report()


def example_with_medium_risk():
    """Example: Pipeline with MEDIUM risk that gets mitigated"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Pipeline with MEDIUM Risk (Mitigated)")
    print("="*80)
    
    orchestrator = EnhancedOrchestrator()
    
    # Create context that will trigger MISSING_SECTIONS risk
    ctx = ExamplePipelineContext(pdf_path="/valid/path.pdf")
    ctx.raw_text = "x" * 200  # Sufficient text
    ctx.sections = {"only_one": "section"}  # Only 1 section (triggers risk)
    
    # Manually assess and mitigate for demonstration
    detected = orchestrator.mitigation_layer.assess_stage_risks("STAGE_1_2", ctx)
    
    print(f"\nDetected {len(detected)} risks")
    for risk in detected:
        print(f"  - {risk.category.value} [{risk.severity.name}]")
        result = orchestrator.mitigation_layer.execute_mitigation(risk, ctx)
        print(f"    Mitigation: {'Success' if result.success else 'Failed'}")
    
    orchestrator.generate_risk_report()


def example_with_critical_risk():
    """Example: Pipeline with CRITICAL risk that aborts execution"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Pipeline with CRITICAL Risk (Aborts)")
    print("="*80)
    
    orchestrator = EnhancedOrchestrator()
    
    try:
        # Process with empty PDF path (triggers CRITICAL risk)
        ctx = orchestrator.process_with_risk_mitigation(pdf_path="")
        
    except CriticalRiskUnmitigatedException as e:
        print(f"\n❌ Pipeline aborted as expected:")
        print(f"   {e}")
        
        orchestrator.generate_risk_report()


def main():
    """Run all examples"""
    print("="*80)
    print("RISK MITIGATION LAYER - INTEGRATION EXAMPLES")
    print("="*80)
    
    try:
        example_successful_pipeline()
        example_with_medium_risk()
        example_with_critical_risk()
        
        print("\n" + "="*80)
        print("✅ ALL EXAMPLES COMPLETED")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ EXAMPLE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
