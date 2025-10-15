#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Circuit Breaker Usage Example for FARFAN 2.0
============================================

Demonstrates how to use the Circuit Breaker pattern with external services
including DNP validation and other APIs.

Author: AI Systems Architect
Version: 1.0.0
"""

import asyncio
import logging
from typing import Dict, Any

from infrastructure import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    ResilientDNPValidator,
    ValidationResult,
    PDMData,
    DNPAPIClient
)


# ============================================================================
# Example 1: Basic Circuit Breaker Usage
# ============================================================================

async def example_basic_circuit_breaker():
    """Demonstrate basic circuit breaker usage"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Circuit Breaker Usage")
    print("="*70 + "\n")
    
    # Simulate external API function
    call_count = [0]  # Use list for closure
    
    async def external_api_call(data: str) -> dict:
        """Simulated external API that fails first 3 times"""
        call_count[0] += 1
        print(f"  API Call #{call_count[0]}: Processing '{data}'...")
        
        if call_count[0] <= 3:
            raise ConnectionError(f"API temporarily unavailable (attempt {call_count[0]})")
        
        return {"status": "success", "data": data.upper()}
    
    # Create circuit breaker
    breaker = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=2
    )
    
    # Try multiple calls
    for i in range(5):
        try:
            print(f"\nAttempt {i+1}:")
            result = await breaker.call(external_api_call, f"request_{i+1}")
            print(f"  ✓ Success: {result}")
            
        except ConnectionError as e:
            print(f"  ✗ Failed: {e}")
            print(f"  Circuit state: {breaker.get_state().value}")
            
        except CircuitOpenError as e:
            print(f"  ⚠ Circuit OPEN - request rejected")
            print(f"  Waiting for recovery... ({e.failure_count} failures)")
            
            # Wait for recovery timeout
            if i == 3:
                print(f"  Sleeping {breaker.recovery_timeout}s for recovery...")
                await asyncio.sleep(breaker.recovery_timeout)
    
    # Show final metrics
    metrics = breaker.get_metrics()
    print(f"\n📊 Final Metrics:")
    print(f"  Total calls: {metrics.total_calls}")
    print(f"  Successful: {metrics.successful_calls}")
    print(f"  Failed: {metrics.failed_calls}")
    print(f"  Rejected: {metrics.rejected_calls}")
    print(f"  State transitions: {metrics.state_transitions}")


# ============================================================================
# Example 2: Resilient DNP Validator
# ============================================================================

class ExampleDNPClient(DNPAPIClient):
    """Example DNP API client implementation"""
    
    def __init__(self, failure_mode: str = "none"):
        """
        Args:
            failure_mode: 'none', 'intermittent', 'permanent'
        """
        self.failure_mode = failure_mode
        self.call_count = 0
    
    async def validate_compliance(self, data: PDMData) -> Dict[str, Any]:
        """Simulated DNP validation API"""
        self.call_count += 1
        
        # Simulate different failure scenarios
        if self.failure_mode == 'permanent':
            raise ConnectionError("DNP service permanently unavailable")
        
        if self.failure_mode == 'intermittent' and self.call_count <= 2:
            raise TimeoutError(f"DNP service timeout (attempt {self.call_count})")
        
        # Success case
        return {
            'cumple': True,
            'score_total': 85.5,
            'nivel_cumplimiento': 'BUENO',
            'competencias_validadas': ['salud', 'educacion'],
            'indicadores_mga_usados': ['IND001', 'IND002'],
            'recomendaciones': ['Incluir más indicadores de género']
        }


async def example_resilient_dnp_validator():
    """Demonstrate resilient DNP validator with fail-open policy"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Resilient DNP Validator")
    print("="*70 + "\n")
    
    # Create sample PDM data
    pdm_data = PDMData(
        sector="salud",
        descripcion="Programa integral de atención primaria en salud",
        indicadores_propuestos=["IND001", "IND002", "IND003"],
        presupuesto=500_000_000,
        es_rural=True,
        poblacion_victimas=False
    )
    
    # Scenario 1: Normal operation
    print("Scenario 1: Normal Operation")
    print("-" * 70)
    client1 = ExampleDNPClient(failure_mode='none')
    validator1 = ResilientDNPValidator(client1, failure_threshold=2)
    
    result = await validator1.validate(pdm_data)
    print(f"Status: {result.status}")
    print(f"Score: {result.score:.2f}")
    print(f"Penalty: {result.score_penalty}")
    print(f"Details: {result.details.get('nivel_cumplimiento', 'N/A')}")
    
    # Scenario 2: Intermittent failures with recovery
    print("\n\nScenario 2: Intermittent Failures (Recovers)")
    print("-" * 70)
    client2 = ExampleDNPClient(failure_mode='intermittent')
    validator2 = ResilientDNPValidator(
        client2,
        failure_threshold=3,
        recovery_timeout=1
    )
    
    for i in range(4):
        result = await validator2.validate(pdm_data)
        print(f"Attempt {i+1}: status={result.status}, score={result.score:.2f}, "
              f"penalty={result.score_penalty}")
    
    # Scenario 3: Permanent failure with fail-open
    print("\n\nScenario 3: Permanent Failure (Fail-Open Policy)")
    print("-" * 70)
    client3 = ExampleDNPClient(failure_mode='permanent')
    validator3 = ResilientDNPValidator(
        client3,
        failure_threshold=2,
        recovery_timeout=1,
        fail_open_penalty=0.05
    )
    
    for i in range(4):
        result = await validator3.validate(pdm_data)
        print(f"Attempt {i+1}:")
        print(f"  Status: {result.status}")
        print(f"  Score: {result.score:.2f} (penalty: {result.score_penalty})")
        print(f"  Reason: {result.reason}")
        print(f"  Circuit: {result.circuit_state}")
    
    # Show metrics
    print("\n📊 Circuit Breaker Metrics:")
    metrics = validator3.get_circuit_metrics()
    for key, value in metrics.items():
        if key != 'last_state_change' and key != 'last_failure_time':
            print(f"  {key}: {value}")


# ============================================================================
# Example 3: Integration with CDAF Pipeline
# ============================================================================

async def example_cdaf_integration():
    """Demonstrate integration with CDAF pipeline"""
    print("\n" + "="*70)
    print("EXAMPLE 3: CDAF Pipeline Integration")
    print("="*70 + "\n")
    
    # Simulated CDAF pipeline
    class SimulatedCDAFPipeline:
        def __init__(self):
            dnp_client = ExampleDNPClient(failure_mode='intermittent')
            self.dnp_validator = ResilientDNPValidator(
                dnp_client,
                failure_threshold=3,
                recovery_timeout=2,
                fail_open_penalty=0.05
            )
            self.logger = logging.getLogger("CDAF")
        
        async def process_pdm(self, pdm_data: PDMData) -> Dict[str, Any]:
            """Process PDM with resilient validation"""
            print("🔄 Processing PDM through pipeline...")
            
            # Phase 1: Extract and analyze
            print("  Phase 1: Extraction and analysis... ✓")
            
            # Phase 2: DNP Validation (with circuit breaker)
            print("  Phase 2: DNP validation...")
            validation_result = await self.dnp_validator.validate(pdm_data)
            
            if validation_result.status == 'passed':
                print(f"    ✓ Validation passed (score: {validation_result.score:.2f})")
            elif validation_result.status == 'skipped':
                print(f"    ⚠ Validation skipped - fail-open policy active")
                print(f"    📉 Applied penalty: {validation_result.score_penalty}")
                print(f"    ▶ Pipeline continues with degraded validation")
            else:
                print(f"    ✗ Validation failed: {validation_result.reason}")
            
            # Phase 3: Coherence analysis
            print("  Phase 3: Coherence analysis... ✓")
            
            # Phase 4: Reporting
            final_score = 0.85 * (1.0 - validation_result.score_penalty)
            print(f"  Phase 4: Report generation... ✓")
            
            return {
                'status': 'completed',
                'final_score': final_score,
                'validation_status': validation_result.status,
                'validation_score': validation_result.score,
                'penalty_applied': validation_result.score_penalty
            }
    
    # Run pipeline
    pipeline = SimulatedCDAFPipeline()
    
    pdm_data = PDMData(
        sector="educacion",
        descripcion="Infraestructura educativa rural",
        indicadores_propuestos=["IND004", "IND005"],
        presupuesto=750_000_000,
        es_rural=True
    )
    
    # Process multiple times to show circuit breaker in action
    for i in range(4):
        print(f"\n--- Run {i+1} ---")
        result = await pipeline.process_pdm(pdm_data)
        print(f"📊 Pipeline Result: {result['status']} "
              f"(score: {result['final_score']:.3f})")
        
        if i < 3:
            await asyncio.sleep(0.5)  # Small delay between runs


# ============================================================================
# Example 4: Manual Circuit Control
# ============================================================================

async def example_manual_circuit_control():
    """Demonstrate manual circuit breaker control"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Manual Circuit Control")
    print("="*70 + "\n")
    
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=5)
    
    async def flaky_service():
        raise Exception("Service error")
    
    # Trigger failures to open circuit
    print("Opening circuit with failures...")
    for i in range(2):
        try:
            await breaker.call(flaky_service)
        except Exception:
            print(f"  Failure {i+1}/2 tracked")
    
    print(f"Circuit state: {breaker.get_state().value}")
    
    # Manual reset
    print("\n🔧 Performing manual circuit reset...")
    breaker.reset()
    print(f"Circuit state after reset: {breaker.get_state().value}")
    print(f"Failure count: {breaker.failure_count}")
    
    # Circuit should work now
    async def working_service():
        return "success"
    
    result = await breaker.call(working_service)
    print(f"✓ Service call after reset: {result}")


# ============================================================================
# Main Runner
# ============================================================================

async def main():
    """Run all examples"""
    # Configure logging
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*70)
    print("CIRCUIT BREAKER PATTERN - USAGE EXAMPLES")
    print("="*70)
    
    await example_basic_circuit_breaker()
    await example_resilient_dnp_validator()
    await example_cdaf_integration()
    await example_manual_circuit_control()
    
    print("\n" + "="*70)
    print("✅ ALL EXAMPLES COMPLETED")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
