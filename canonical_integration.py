#!/usr/bin/env python3
"""
Integration utility between canonical notation and guia_cuestionario
====================================================================

This module provides integration between the canonical notation system
and the existing guia_cuestionario JSON configuration.

Author: AI Systems Architect
Version: 2.0.0
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from canonical_notation import (
    CanonicalID,
    RubricKey,
    EvidenceEntry,
    PolicyArea,
    AnalyticalDimension
)


class GuiaCuestionarioIntegration:
    """
    Integration layer between canonical notation and guia_cuestionario
    
    This class provides utilities to:
    - Load and validate guia_cuestionario configuration
    - Map between canonical IDs and questionnaire structure
    - Generate canonical evidence from questionnaire data
    """
    
    def __init__(self, guia_path: Optional[Path] = None):
        """
        Initialize integration
        
        Args:
            guia_path: Path to guia_cuestionario file (defaults to ./guia_cuestionario)
        """
        if guia_path is None:
            guia_path = Path(__file__).parent / "guia_cuestionario"
        
        self.guia_path = guia_path
        self.config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """Load guia_cuestionario configuration"""
        try:
            with open(self.guia_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Try to parse as JSON, handling potential format issues
                self.config = json.loads(content)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"guia_cuestionario not found at {self.guia_path}. "
                "Ensure the file exists in the repository."
            )
        except json.JSONDecodeError as e:
            # If JSON parsing fails, try to extract just the valid JSON object
            # This handles cases where the file might have trailing data
            try:
                with open(self.guia_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Find the first complete JSON object
                    brace_count = 0
                    in_string = False
                    escape_next = False
                    end_pos = 0
                    
                    for i, char in enumerate(content):
                        if escape_next:
                            escape_next = False
                            continue
                        if char == '\\':
                            escape_next = True
                            continue
                        if char == '"' and not escape_next:
                            in_string = not in_string
                        if not in_string:
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    end_pos = i + 1
                                    break
                    
                    if end_pos > 0:
                        valid_json = content[:end_pos]
                        self.config = json.loads(valid_json)
                    else:
                        raise ValueError(f"Could not extract valid JSON from guia_cuestionario: {e}")
            except Exception as inner_e:
                raise ValueError(f"Invalid JSON in guia_cuestionario: {e}. Recovery attempt failed: {inner_e}")
    
    def get_dimension_mapping(self, policy: str) -> Dict[str, Any]:
        """
        Get dimension mapping for a policy
        
        Args:
            policy: Policy ID (P1-P10)
            
        Returns:
            Dictionary with dimension weights and critical dimensions
        """
        if policy not in self.config.get("decalogo_dimension_mapping", {}):
            raise ValueError(f"Policy {policy} not found in guia_cuestionario")
        
        return self.config["decalogo_dimension_mapping"][policy]
    
    def get_dimension_template(self, dimension: str) -> Dict[str, Any]:
        """
        Get causal verification template for a dimension
        
        Args:
            dimension: Dimension ID (D1-D6)
            
        Returns:
            Dictionary with dimension template
        """
        if dimension not in self.config.get("causal_verification_templates", {}):
            raise ValueError(f"Dimension {dimension} not found in guia_cuestionario")
        
        return self.config["causal_verification_templates"][dimension]
    
    def get_critical_dimensions(self, policy: str) -> List[str]:
        """
        Get critical dimensions for a policy
        
        Args:
            policy: Policy ID (P1-P10)
            
        Returns:
            List of critical dimension IDs
        """
        mapping = self.get_dimension_mapping(policy)
        return mapping.get("critical_dimensions", [])
    
    def get_dimension_weight(self, policy: str, dimension: str) -> float:
        """
        Get weight of a dimension for a policy
        
        Args:
            policy: Policy ID (P1-P10)
            dimension: Dimension ID (D1-D6)
            
        Returns:
            Weight as float
        """
        mapping = self.get_dimension_mapping(policy)
        weight_key = f"{dimension}_weight"
        
        if weight_key not in mapping:
            raise ValueError(f"Weight for {dimension} not found in {policy} mapping")
        
        return mapping[weight_key]
    
    def is_critical_dimension(self, policy: str, dimension: str) -> bool:
        """
        Check if dimension is critical for a policy
        
        Args:
            policy: Policy ID (P1-P10)
            dimension: Dimension ID (D1-D6)
            
        Returns:
            True if dimension is critical
        """
        return dimension in self.get_critical_dimensions(policy)
    
    def get_required_elements(self, dimension: str) -> List[str]:
        """
        Get required elements for a dimension
        
        Args:
            dimension: Dimension ID (D1-D6)
            
        Returns:
            List of required element names
        """
        template = self.get_dimension_template(dimension)
        return template.get("required_elements", [])
    
    def get_validation_patterns(self, dimension: str) -> List[str]:
        """
        Get validation patterns for a dimension
        
        Args:
            dimension: Dimension ID (D1-D6)
            
        Returns:
            List of validation regex patterns
        """
        template = self.get_dimension_template(dimension)
        
        # Different dimensions have different pattern keys
        pattern_keys = [
            "validation_patterns",
            "table_structure_indicators",
            "causal_mechanism_patterns",
            "indicator_verification_patterns",
            "outcome_patterns",
            "impact_patterns",
            "theory_patterns"
        ]
        
        patterns = []
        for key in pattern_keys:
            if key in template:
                patterns.extend(template[key])
        
        return patterns
    
    def create_canonical_evidence(
        self,
        policy: str,
        dimension: str,
        question: int,
        score: float,
        confidence: float,
        stage: str = "questionnaire_validation",
        metadata: Optional[Dict[str, Any]] = None
    ) -> EvidenceEntry:
        """
        Create canonical evidence entry with questionnaire metadata
        
        Args:
            policy: Policy ID (P1-P10)
            dimension: Dimension ID (D1-D6)
            question: Question number
            score: Evaluation score (0.0-1.0)
            confidence: Confidence level (0.0-1.0)
            stage: Processing stage
            metadata: Additional metadata
            
        Returns:
            EvidenceEntry with enriched metadata
        """
        # Get dimension weight and critical status
        dimension_weight = self.get_dimension_weight(policy, dimension)
        is_critical = self.is_critical_dimension(policy, dimension)
        required_elements = self.get_required_elements(dimension)
        
        # Enrich metadata
        enriched_metadata = metadata or {}
        enriched_metadata.update({
            "dimension_weight": dimension_weight,
            "is_critical_dimension": is_critical,
            "required_elements": required_elements,
            "policy_title": PolicyArea.get_title(policy),
            "dimension_name": AnalyticalDimension.get_name(dimension),
            "dimension_focus": AnalyticalDimension.get_focus(dimension)
        })
        
        # Create evidence entry
        evidence = EvidenceEntry.create(
            policy=policy,
            dimension=dimension,
            question=question,
            score=score,
            confidence=confidence,
            stage=stage,
            evidence_id_prefix="guia_",
            metadata=enriched_metadata
        )
        
        return evidence
    
    def validate_questionnaire_structure(self) -> Dict[str, Any]:
        """
        Validate that questionnaire structure matches canonical notation
        
        Returns:
            Dictionary with validation results
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "summary": {}
        }
        
        # Check metadata
        metadata = self.config.get("metadata", {})
        expected_policies = 10
        expected_dimensions = 6
        expected_total = metadata.get("compatibility", {}).get("total_questions", 0)
        
        # Validate policy mappings
        policy_mappings = self.config.get("decalogo_dimension_mapping", {})
        if len(policy_mappings) != expected_policies:
            results["errors"].append(
                f"Expected {expected_policies} policies, found {len(policy_mappings)}"
            )
            results["valid"] = False
        
        # Validate all policies P1-P10 exist
        for i in range(1, 11):
            policy_id = f"P{i}"
            if policy_id not in policy_mappings:
                results["errors"].append(f"Policy {policy_id} missing from mappings")
                results["valid"] = False
        
        # Validate dimension templates
        dimension_templates = self.config.get("causal_verification_templates", {})
        if len(dimension_templates) != expected_dimensions:
            results["errors"].append(
                f"Expected {expected_dimensions} dimensions, found {len(dimension_templates)}"
            )
            results["valid"] = False
        
        # Validate all dimensions D1-D6 exist
        for i in range(1, 7):
            dimension_id = f"D{i}"
            if dimension_id not in dimension_templates:
                results["errors"].append(f"Dimension {dimension_id} missing from templates")
                results["valid"] = False
        
        # Summary
        results["summary"] = {
            "policies_found": len(policy_mappings),
            "dimensions_found": len(dimension_templates),
            "expected_total_questions": expected_total,
            "canonical_total_questions": expected_policies * expected_dimensions * 5
        }
        
        return results
    
    def get_all_canonical_ids(
        self,
        max_questions_per_dimension: int = 5
    ) -> List[CanonicalID]:
        """
        Generate all canonical IDs based on questionnaire structure
        
        Args:
            max_questions_per_dimension: Maximum questions per dimension
            
        Returns:
            List of CanonicalID instances
        """
        canonical_ids = []
        
        policy_mappings = self.config.get("decalogo_dimension_mapping", {})
        
        for policy in sorted(policy_mappings.keys()):
            for i in range(1, 7):  # D1-D6
                dimension = f"D{i}"
                for question in range(1, max_questions_per_dimension + 1):
                    canonical_id = CanonicalID(
                        policy=policy,
                        dimension=dimension,
                        question=question
                    )
                    canonical_ids.append(canonical_id)
        
        return canonical_ids


def demo_integration():
    """Demonstrate guia_cuestionario integration"""
    print("=" * 70)
    print("GUIA CUESTIONARIO INTEGRATION DEMO")
    print("=" * 70)
    
    try:
        integration = GuiaCuestionarioIntegration()
        
        # 1. Validate structure
        print("\n1. Validating questionnaire structure:")
        validation = integration.validate_questionnaire_structure()
        print(f"   Valid: {validation['valid']}")
        print(f"   Policies found: {validation['summary']['policies_found']}")
        print(f"   Dimensions found: {validation['summary']['dimensions_found']}")
        print(f"   Expected total questions: {validation['summary']['expected_total_questions']}")
        
        if validation['errors']:
            print("\n   Errors:")
            for error in validation['errors']:
                print(f"     - {error}")
        
        # 2. Get dimension mapping
        print("\n2. Dimension mapping for P4 (Economic rights):")
        mapping = integration.get_dimension_mapping("P4")
        print(f"   Critical dimensions: {mapping['critical_dimensions']}")
        for i in range(1, 7):
            dim = f"D{i}"
            weight = mapping.get(f"{dim}_weight", 0)
            critical = "✓" if dim in mapping['critical_dimensions'] else " "
            print(f"   [{critical}] {dim}: {weight:.2f}")
        
        # 3. Get dimension template
        print("\n3. Dimension template for D2 (Design):")
        template = integration.get_dimension_template("D2")
        print(f"   Name: {template['dimension_name']}")
        print(f"   Required elements: {len(template['required_elements'])}")
        for element in template['required_elements']:
            print(f"     - {element}")
        
        # 4. Create canonical evidence
        print("\n4. Creating canonical evidence:")
        evidence = integration.create_canonical_evidence(
            policy="P4",
            dimension="D2",
            question=3,
            score=0.85,
            confidence=0.90,
            stage="questionnaire_validation"
        )
        print(f"   Evidence ID: {evidence.evidence_id}")
        print(f"   Question ID: {evidence.question_unique_id}")
        print(f"   Is critical: {evidence.metadata['is_critical_dimension']}")
        print(f"   Weight: {evidence.metadata['dimension_weight']}")
        print(f"   Policy: {evidence.metadata['policy_title'][:50]}...")
        
        # 5. Generate all canonical IDs
        print("\n5. Generating all canonical IDs:")
        all_ids = integration.get_all_canonical_ids(max_questions_per_dimension=5)
        print(f"   Total IDs generated: {len(all_ids)}")
        print(f"   First 5: {[str(id) for id in all_ids[:5]]}")
        print(f"   Last 5: {[str(id) for id in all_ids[-5:]]}")
        
        print("\n" + "=" * 70)
        print("INTEGRATION SUCCESSFUL")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_integration()
