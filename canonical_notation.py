#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canonical Notation System for PDM Evaluation
==============================================

Sistema Canónico de Evaluación de PDM uses a standardized canonical notation 
for all question identifiers and rubric keys. This ensures consistency, 
traceability, and deterministic evaluation across the entire system.

Canonical Format Components:
- P# = Policy Point (Punto del Decálogo OR POLICY AREA)
  Range: P1 through P10
  Represents one of 10 thematic policy areas in Colombian Municipal Development Plans

- D# = Analytical Dimension (Dimensión analítica)
  Range: D1 through D6
  Represents evaluation dimensions (Diagnóstico, Diseño, Productos, Resultados, Impactos, Teoría de Cambio)

- Q# = Question Number
  Range: Q1 and up (positive integers)
  Unique question identifier within a dimension

Identifiers:
- question_unique_id: Format P#-D#-Q# (e.g., P4-D2-Q3)
- rubric_key: Format D#-Q# (e.g., D2-Q3)

Author: AI Systems Architect
Version: 2.0.0
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import json


class PolicyArea(Enum):
    """10 Policy Areas (Puntos del Decálogo)"""
    P1 = "Derechos de las mujeres e igualdad de género"
    P2 = "Prevención de la violencia y protección frente al conflicto"
    P3 = "Ambiente sano, cambio climático, prevención y atención a desastres"
    P4 = "Derechos económicos, sociales y culturales"
    P5 = "Derechos de las víctimas y construcción de paz"
    P6 = "Derecho al buen futuro de la niñez, adolescencia, juventud"
    P7 = "Tierras y territorios"
    P8 = "Líderes y defensores de derechos humanos"
    P9 = "Crisis de derechos de personas privadas de la libertad"
    P10 = "Migración transfronteriza"

    @classmethod
    def get_title(cls, policy_id: str) -> str:
        """Get title for a policy ID (e.g., 'P1' -> title)"""
        try:
            return cls[policy_id].value
        except KeyError:
            raise ValueError(f"Invalid policy ID: {policy_id}")


class AnalyticalDimension(Enum):
    """6 Analytical Dimensions"""
    D1 = "Diagnóstico y Recursos"
    D2 = "Diseño de Intervención"
    D3 = "Productos y Outputs"
    D4 = "Resultados y Outcomes"
    D5 = "Impactos y Efectos de Largo Plazo"
    D6 = "Teoría de Cambio y Coherencia Causal"

    @classmethod
    def get_name(cls, dimension_id: str) -> str:
        """Get name for a dimension ID (e.g., 'D1' -> name)"""
        try:
            return cls[dimension_id].value
        except KeyError:
            raise ValueError(f"Invalid dimension ID: {dimension_id}")

    @classmethod
    def get_focus(cls, dimension_id: str) -> str:
        """Get focus description for a dimension"""
        focus_map = {
            "D1": "Baseline, problem magnitude, resources, institutional capacity",
            "D2": "Activities, target population, intervention design",
            "D3": "Technical standards, proportionality, quantification, accountability",
            "D4": "Result indicators, differentiation, magnitude of change, attribution",
            "D5": "Impact indicators, temporal horizons, systemic effects, sustainability",
            "D6": "Theory of change, assumptions, logical framework, monitoring"
        }
        return focus_map.get(dimension_id, "Unknown dimension")


# Regex patterns for validation
QUESTION_UNIQUE_ID_PATTERN = re.compile(r'^P(10|[1-9])-D[1-6]-Q[1-9][0-9]*$')
RUBRIC_KEY_PATTERN = re.compile(r'^D[1-6]-Q[1-9][0-9]*$')
POLICY_PATTERN = re.compile(r'^P(10|[1-9])$')
DIMENSION_PATTERN = re.compile(r'^D[1-6]$')


@dataclass
class RubricKey:
    """
    Rubric Key identifier (D#-Q#)
    Format: D#-Q# where D is dimension (1-6) and Q is question number (positive integer)
    Examples: D2-Q3, D1-Q1, D6-Q30
    """
    dimension: str
    question: int

    def __post_init__(self):
        """Validate rubric key components"""
        if not DIMENSION_PATTERN.match(self.dimension):
            raise ValueError(f"Invalid dimension format: {self.dimension}. Must match D[1-6]")
        if self.question < 1:
            raise ValueError(f"Invalid question number: {self.question}. Must be positive integer")

    def __str__(self) -> str:
        """String representation in canonical format"""
        return f"{self.dimension}-Q{self.question}"

    @classmethod
    def from_string(cls, rubric_key_str: str) -> 'RubricKey':
        """
        Parse rubric key from string format
        
        Args:
            rubric_key_str: String in format D#-Q#
            
        Returns:
            RubricKey instance
            
        Raises:
            ValueError: If format is invalid
        """
        if not RUBRIC_KEY_PATTERN.match(rubric_key_str):
            raise ValueError(f"Invalid rubric key format: {rubric_key_str}. Must match D[1-6]-Q[1-9][0-9]*")
        
        parts = rubric_key_str.split('-')
        dimension = parts[0]
        question = int(parts[1][1:])  # Remove 'Q' prefix
        
        return cls(dimension=dimension, question=question)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "dimension": self.dimension,
            "question": self.question,
            "rubric_key": str(self)
        }


@dataclass
class CanonicalID:
    """
    Canonical question identifier (P#-D#-Q#)
    Format: P#-D#-Q# where:
    - P is policy area (1-10)
    - D is dimension (1-6)
    - Q is question number (positive integer)
    
    Examples: P4-D2-Q3, P1-D1-Q1, P10-D6-Q30
    """
    policy: str
    dimension: str
    question: int

    def __post_init__(self):
        """Validate canonical ID components"""
        if not POLICY_PATTERN.match(self.policy):
            raise ValueError(f"Invalid policy format: {self.policy}. Must match P(10|[1-9])")
        if not DIMENSION_PATTERN.match(self.dimension):
            raise ValueError(f"Invalid dimension format: {self.dimension}. Must match D[1-6]")
        if self.question < 1:
            raise ValueError(f"Invalid question number: {self.question}. Must be positive integer")

    def __str__(self) -> str:
        """String representation in canonical format"""
        return f"{self.policy}-{self.dimension}-Q{self.question}"

    @classmethod
    def from_string(cls, question_unique_id: str) -> 'CanonicalID':
        """
        Parse canonical ID from string format
        
        Args:
            question_unique_id: String in format P#-D#-Q#
            
        Returns:
            CanonicalID instance
            
        Raises:
            ValueError: If format is invalid
        """
        if not QUESTION_UNIQUE_ID_PATTERN.match(question_unique_id):
            raise ValueError(
                f"Invalid question unique ID format: {question_unique_id}. "
                f"Must match P(10|[1-9])-D[1-6]-Q[1-9][0-9]*"
            )
        
        parts = question_unique_id.split('-')
        policy = parts[0]
        dimension = parts[1]
        question = int(parts[2][1:])  # Remove 'Q' prefix
        
        return cls(policy=policy, dimension=dimension, question=question)

    def to_rubric_key(self) -> RubricKey:
        """
        Derive rubric key from canonical ID
        
        Example: "P4-D2-Q3" → "D2-Q3"
        
        Returns:
            RubricKey instance
        """
        return RubricKey(dimension=self.dimension, question=self.question)

    def get_policy_title(self) -> str:
        """Get the title of the policy area"""
        return PolicyArea.get_title(self.policy)

    def get_dimension_name(self) -> str:
        """Get the name of the dimension"""
        return AnalyticalDimension.get_name(self.dimension)

    def get_dimension_focus(self) -> str:
        """Get the focus description of the dimension"""
        return AnalyticalDimension.get_focus(self.dimension)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "policy": self.policy,
            "dimension": self.dimension,
            "question": self.question,
            "question_unique_id": str(self),
            "rubric_key": str(self.to_rubric_key()),
            "policy_title": self.get_policy_title(),
            "dimension_name": self.get_dimension_name()
        }


@dataclass
class EvidenceEntry:
    """
    Standard evidence entry with canonical notation and full traceability
    
    All evidence entries must follow this canonical structure for consistency
    and traceability across the evaluation system.
    
    Enhanced with:
    - PDF source traceability (page, bbox coordinates)
    - Module extraction tracking
    - Chain of custody for audit trail
    """
    evidence_id: str
    question_unique_id: str
    content: Dict[str, Any]
    confidence: float
    stage: str
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    # Enhanced traceability fields
    texto: Optional[str] = None  # Exact text extract from source
    fuente: Optional[str] = None  # Source type: "pdf", "tabla", "grafo_causal"
    pagina: Optional[int] = None  # PDF page number (1-indexed)
    bbox: Optional[Tuple[float, float, float, float]] = None  # Bounding box (x0, y0, x1, y1) in PDF coordinates
    modulo_extractor: Optional[str] = None  # Module that extracted this evidence
    chain_of_custody: Optional[str] = None  # Audit trail (e.g., "Stage 4 → AGUJA I → P1-D6-Q26")

    def __post_init__(self):
        """Validate evidence entry"""
        # Validate question_unique_id format
        canonical_id = CanonicalID.from_string(self.question_unique_id)
        
        # Validate confidence
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        
        # Validate content structure
        required_fields = {"policy", "dimension", "question", "score", "rubric_key"}
        missing_fields = required_fields - set(self.content.keys())
        if missing_fields:
            raise ValueError(f"Missing required content fields: {missing_fields}")
        
        # Validate content consistency with question_unique_id
        if self.content["policy"] != canonical_id.policy:
            raise ValueError(
                f"Content policy {self.content['policy']} doesn't match "
                f"question_unique_id policy {canonical_id.policy}"
            )
        if self.content["dimension"] != canonical_id.dimension:
            raise ValueError(
                f"Content dimension {self.content['dimension']} doesn't match "
                f"question_unique_id dimension {canonical_id.dimension}"
            )
        if self.content["question"] != canonical_id.question:
            raise ValueError(
                f"Content question {self.content['question']} doesn't match "
                f"question_unique_id question {canonical_id.question}"
            )
        
        # Validate rubric_key format and consistency
        expected_rubric_key = str(canonical_id.to_rubric_key())
        if self.content["rubric_key"] != expected_rubric_key:
            raise ValueError(
                f"Content rubric_key {self.content['rubric_key']} doesn't match "
                f"expected {expected_rubric_key}"
            )
        
        # Validate score
        if not 0.0 <= self.content["score"] <= 1.0:
            raise ValueError(f"Score must be between 0 and 1, got {self.content['score']}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with full traceability information"""
        result = {
            "evidence_id": self.evidence_id,
            "question_unique_id": self.question_unique_id,
            "content": self.content,
            "confidence": self.confidence,
            "stage": self.stage,
            "metadata": self.metadata
        }
        
        # Add enhanced traceability fields if present
        if self.texto is not None:
            result["texto"] = self.texto
        if self.fuente is not None:
            result["fuente"] = self.fuente
        if self.pagina is not None:
            result["pagina"] = self.pagina
        if self.bbox is not None:
            result["bbox"] = self.bbox
        if self.modulo_extractor is not None:
            result["modulo_extractor"] = self.modulo_extractor
        if self.chain_of_custody is not None:
            result["chain_of_custody"] = self.chain_of_custody
            
        return result

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def create(
        cls,
        policy: str,
        dimension: str,
        question: int,
        score: float,
        confidence: float,
        stage: str,
        evidence_id_prefix: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        # Enhanced traceability parameters
        texto: Optional[str] = None,
        fuente: Optional[str] = None,
        pagina: Optional[int] = None,
        bbox: Optional[tuple] = None,
        modulo_extractor: Optional[str] = None
    ) -> 'EvidenceEntry':
        """
        Create evidence entry with canonical notation and full traceability
        
        Args:
            policy: Policy ID (P1-P10)
            dimension: Dimension ID (D1-D6)
            question: Question number (positive integer)
            score: Score value (0.0-1.0)
            confidence: Confidence value (0.0-1.0)
            stage: Processing stage identifier
            evidence_id_prefix: Optional prefix for evidence_id (e.g., "toc_")
            metadata: Optional metadata dictionary
            texto: Exact text extract from source document
            fuente: Source type ("pdf", "tabla", "grafo_causal")
            pagina: PDF page number (1-indexed)
            bbox: Bounding box coordinates (x0, y0, x1, y1)
            modulo_extractor: Module that extracted this evidence
            
        Returns:
            EvidenceEntry instance with full traceability
        """
        canonical_id = CanonicalID(policy=policy, dimension=dimension, question=question)
        question_unique_id = str(canonical_id)
        rubric_key = str(canonical_id.to_rubric_key())
        
        evidence_id = f"{evidence_id_prefix}{question_unique_id}" if evidence_id_prefix else question_unique_id
        
        content = {
            "policy": policy,
            "dimension": dimension,
            "question": question,
            "score": score,
            "rubric_key": rubric_key
        }
        
        # Generate chain of custody
        chain_of_custody = f"{stage} → {modulo_extractor or 'UNKNOWN'} → {question_unique_id}"
        
        return cls(
            evidence_id=evidence_id,
            question_unique_id=question_unique_id,
            content=content,
            confidence=confidence,
            stage=stage,
            metadata=metadata or {},
            texto=texto,
            fuente=fuente,
            pagina=pagina,
            bbox=bbox,
            modulo_extractor=modulo_extractor,
            chain_of_custody=chain_of_custody
        )


class CanonicalNotationValidator:
    """Validator for canonical notation compliance"""

    @staticmethod
    def validate_question_unique_id(question_unique_id: str) -> bool:
        """
        Validate question unique ID format
        
        Args:
            question_unique_id: String to validate
            
        Returns:
            True if valid, False otherwise
        """
        return bool(QUESTION_UNIQUE_ID_PATTERN.match(question_unique_id))

    @staticmethod
    def validate_rubric_key(rubric_key: str) -> bool:
        """
        Validate rubric key format
        
        Args:
            rubric_key: String to validate
            
        Returns:
            True if valid, False otherwise
        """
        return bool(RUBRIC_KEY_PATTERN.match(rubric_key))

    @staticmethod
    def validate_policy(policy: str) -> bool:
        """Validate policy ID format"""
        return bool(POLICY_PATTERN.match(policy))

    @staticmethod
    def validate_dimension(dimension: str) -> bool:
        """Validate dimension ID format"""
        return bool(DIMENSION_PATTERN.match(dimension))

    @staticmethod
    def extract_rubric_key_from_question_id(question_unique_id: str) -> str:
        """
        Extract rubric key from question unique ID
        
        Args:
            question_unique_id: String in format P#-D#-Q#
            
        Returns:
            Rubric key in format D#-Q#
            
        Example:
            "P4-D2-Q3" → "D2-Q3"
        """
        canonical_id = CanonicalID.from_string(question_unique_id)
        return str(canonical_id.to_rubric_key())

    @staticmethod
    def migrate_legacy_id(
        legacy_id: str,
        inferred_policy: Optional[str] = None
    ) -> Optional[str]:
        """
        Migrate legacy ID format to canonical format
        
        Supports migration from:
        - Case A: D#-Q# (no policy) → P#-D#-Q# (requires policy inference)
        
        Args:
            legacy_id: Legacy identifier
            inferred_policy: Inferred policy ID (required for D#-Q# format)
            
        Returns:
            Canonical question unique ID or None if migration fails
        """
        # Case A: D#-Q# format (legacy rubric key used as question ID)
        if RUBRIC_KEY_PATTERN.match(legacy_id):
            if not inferred_policy:
                raise ValueError(
                    f"Cannot migrate legacy ID {legacy_id}: policy must be inferred from context"
                )
            if not POLICY_PATTERN.match(inferred_policy):
                raise ValueError(f"Invalid inferred policy format: {inferred_policy}")
            
            rubric_key = RubricKey.from_string(legacy_id)
            canonical_id = CanonicalID(
                policy=inferred_policy,
                dimension=rubric_key.dimension,
                question=rubric_key.question
            )
            return str(canonical_id)
        
        # Already in canonical format
        if QUESTION_UNIQUE_ID_PATTERN.match(legacy_id):
            return legacy_id
        
        return None


def generate_default_questions(max_questions_per_dimension: int = 5) -> List[CanonicalID]:
    """
    Generate default question structure
    
    By default, the system supports:
    10 policies × 6 dimensions × 5 questions = 300 total questions
    
    Args:
        max_questions_per_dimension: Maximum questions per dimension (default: 5)
        
    Returns:
        List of CanonicalID instances
    """
    questions = []
    
    for policy_num in range(1, 11):  # P1 to P10
        policy = f"P{policy_num}"
        
        for dimension_num in range(1, 7):  # D1 to D6
            dimension = f"D{dimension_num}"
            
            for question_num in range(1, max_questions_per_dimension + 1):
                canonical_id = CanonicalID(
                    policy=policy,
                    dimension=dimension,
                    question=question_num
                )
                questions.append(canonical_id)
    
    return questions


def get_system_structure_summary() -> Dict[str, Any]:
    """
    Get summary of the canonical notation system structure
    
    Returns:
        Dictionary with system structure information
    """
    return {
        "total_policies": 10,
        "total_dimensions": 6,
        "default_questions_per_dimension": 5,
        "default_total_questions": 300,
        "policies": {f"P{i}": PolicyArea.get_title(f"P{i}") for i in range(1, 11)},
        "dimensions": {f"D{i}": AnalyticalDimension.get_name(f"D{i}") for i in range(1, 7)},
        "dimension_focus": {f"D{i}": AnalyticalDimension.get_focus(f"D{i}") for i in range(1, 7)},
        "patterns": {
            "question_unique_id": r"^P(10|[1-9])-D[1-6]-Q[1-9][0-9]*$",
            "rubric_key": r"^D[1-6]-Q[1-9][0-9]*$",
            "policy": r"^P(10|[1-9])$",
            "dimension": r"^D[1-6]$"
        }
    }


# Example usage and testing
if __name__ == "__main__":
    print("=== Canonical Notation System Demo ===\n")
    
    # 1. Create canonical IDs
    print("1. Creating Canonical IDs:")
    canonical_id = CanonicalID(policy="P4", dimension="D2", question=3)
    print(f"   Canonical ID: {canonical_id}")
    print(f"   Policy Title: {canonical_id.get_policy_title()}")
    print(f"   Dimension: {canonical_id.get_dimension_name()}")
    print(f"   Rubric Key: {canonical_id.to_rubric_key()}\n")
    
    # 2. Parse from string
    print("2. Parsing from String:")
    parsed = CanonicalID.from_string("P7-D3-Q5")
    print(f"   Parsed: {parsed}")
    print(f"   Dict: {parsed.to_dict()}\n")
    
    # 3. Create evidence entry
    print("3. Creating Evidence Entry:")
    evidence = EvidenceEntry.create(
        policy="P7",
        dimension="D3",
        question=5,
        score=0.82,
        confidence=0.82,
        stage="teoria_cambio",
        evidence_id_prefix="toc_"
    )
    print(f"   Evidence ID: {evidence.evidence_id}")
    print(f"   Question ID: {evidence.question_unique_id}")
    print(f"   Rubric Key: {evidence.content['rubric_key']}")
    print(f"   JSON:\n{evidence.to_json()}\n")
    
    # 4. Validation
    print("4. Validation Examples:")
    validator = CanonicalNotationValidator()
    test_cases = [
        ("P4-D2-Q3", "Valid question ID"),
        ("P11-D2-Q3", "Invalid policy (P11)"),
        ("P4-D7-Q3", "Invalid dimension (D7)"),
        ("P4-D2-Q0", "Invalid question (Q0)"),
        ("D2-Q3", "Valid rubric key"),
    ]
    for test_id, description in test_cases:
        is_valid_q = validator.validate_question_unique_id(test_id)
        is_valid_r = validator.validate_rubric_key(test_id)
        print(f"   {test_id:15} ({description:25}): Question={is_valid_q}, Rubric={is_valid_r}")
    
    # 5. Migration
    print("\n5. Legacy ID Migration:")
    legacy_id = "D2-Q3"
    migrated = validator.migrate_legacy_id(legacy_id, inferred_policy="P4")
    print(f"   Legacy: {legacy_id} → Canonical: {migrated}")
    
    # 6. System structure
    print("\n6. System Structure Summary:")
    structure = get_system_structure_summary()
    print(f"   Total Questions: {structure['default_total_questions']}")
    print(f"   Policies: {structure['total_policies']}")
    print(f"   Dimensions: {structure['total_dimensions']}")
    print(f"   Questions per Dimension: {structure['default_questions_per_dimension']}")
    
    print("\n=== Demo Complete ===")
