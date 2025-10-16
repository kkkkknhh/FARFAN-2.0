#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canonical Questionnaire Parser for FARFAN 2.0
==============================================

Parses cuestionario_canonico as the single source of truth for all questionnaire-related
logic. Provides deterministic access to 300 questions (10 policies × 6 dimensions × 5 questions).

Design Principles:
- Single source of truth: Only reads from cuestionario_canonico file
- Deterministic: Same input always produces same parsed output
- Auditable: All questions, dimensions, and policies explicitly tracked
- Contract-based: Explicit API with type hints and validation

SIN_CARRETA Compliance:
- Canonical source enforcement (no aliases, no legacy versions)
- Deterministic parsing with reproducible results
- Complete traceability of question metadata
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Question:
    """Single question with metadata"""
    
    policy_id: str  # e.g., "P1"
    policy_name: str  # e.g., "Derechos de las mujeres e igualdad de género"
    dimension_id: str  # e.g., "D1"
    dimension_name: str  # e.g., "INSUMOS"
    question_id: str  # e.g., "Q1"
    full_id: str  # e.g., "P1-D1-Q1"
    text: str  # Full question text
    
    # Optional metadata
    rubric: Optional[str] = None
    weight: float = 1.0
    guide: Optional[str] = None


@dataclass
class Dimension:
    """Dimension with its questions"""
    
    dimension_id: str  # e.g., "D1"
    dimension_name: str  # e.g., "INSUMOS (D1)"
    full_name: str  # e.g., "Dimensión 1: INSUMOS (D1)"
    questions: List[Question] = field(default_factory=list)
    
    @property
    def question_count(self) -> int:
        return len(self.questions)


@dataclass
class Policy:
    """Policy with its dimensions"""
    
    policy_id: str  # e.g., "P1"
    policy_name: str  # e.g., "Derechos de las mujeres e igualdad de género"
    full_name: str  # e.g., "P1: Derechos de las mujeres e igualdad de género"
    dimensions: List[Dimension] = field(default_factory=list)
    
    @property
    def question_count(self) -> int:
        return sum(d.question_count for d in self.dimensions)


class QuestionnaireParser:
    """
    Canonical parser for cuestionario_canonico.
    
    Provides deterministic access to all 300 questions organized by:
    - Policy (P1-P10): 10 total
    - Dimension (D1-D6): 6 per policy
    - Question (Q1-Q30): 30 per policy, 5 per dimension
    
    Single source of truth enforcement:
    - Only parses cuestionario_canonico file
    - No aliases, no legacy versions
    - Explicit validation of structure
    """
    
    def __init__(self, cuestionario_path: Optional[Path] = None):
        """
        Initialize parser with canonical questionnaire file.
        
        Args:
            cuestionario_path: Path to cuestionario_canonico file.
                If None, uses default location in repo root.
        """
        if cuestionario_path is None:
            # Default to canonical location
            cuestionario_path = Path(__file__).parent / "cuestionario_canonico"
        
        self.cuestionario_path = Path(cuestionario_path)
        
        # Validate canonical file exists
        if not self.cuestionario_path.exists():
            raise FileNotFoundError(
                f"Canonical questionnaire not found: {self.cuestionario_path}\n"
                "Expected: cuestionario_canonico in repository root"
            )
        
        # Parsed structures (lazy loaded)
        self._policies: Optional[List[Policy]] = None
        self._question_index: Optional[Dict[str, Question]] = None
        self._dimension_index: Optional[Dict[str, List[Question]]] = None
        self._policy_index: Optional[Dict[str, Policy]] = None
    
    def _ensure_parsed(self) -> None:
        """Lazy load and parse questionnaire if not already done"""
        if self._policies is None:
            self._parse()
    
    def _parse(self) -> None:
        """Parse cuestionario_canonico file into structured data"""
        with open(self.cuestionario_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse into policies
        self._policies = []
        self._question_index = {}
        self._dimension_index = {}
        self._policy_index = {}
        
        # Split by policy markers (### **P#:)
        policy_pattern = r'### \*\*P(\d+): ([^*]+)\*\*'
        policy_matches = list(re.finditer(policy_pattern, content))
        
        for i, match in enumerate(policy_matches):
            policy_num = match.group(1)
            policy_name = match.group(2).strip()
            policy_id = f"P{policy_num}"
            
            # Extract policy content (from this match to next policy or end)
            start_pos = match.end()
            end_pos = policy_matches[i + 1].start() if i + 1 < len(policy_matches) else len(content)
            policy_content = content[start_pos:end_pos]
            
            # Create policy object
            policy = Policy(
                policy_id=policy_id,
                policy_name=policy_name,
                full_name=f"P{policy_num}: {policy_name}"
            )
            
            # Parse dimensions within this policy
            dimension_pattern = r'#### \*\*Dimensión (\d+): ([^*]+)\*\*'
            dimension_matches = list(re.finditer(dimension_pattern, policy_content))
            
            for j, dim_match in enumerate(dimension_matches):
                dim_num = dim_match.group(1)
                dim_name = dim_match.group(2).strip()
                dim_id = f"D{dim_num}"
                
                # Extract dimension content
                dim_start = dim_match.end()
                dim_end = dimension_matches[j + 1].start() if j + 1 < len(dimension_matches) else len(policy_content)
                dim_content = policy_content[dim_start:dim_end]
                
                # Create dimension object
                dimension = Dimension(
                    dimension_id=dim_id,
                    dimension_name=dim_name,
                    full_name=f"Dimensión {dim_num}: {dim_name}"
                )
                
                # Parse questions within this dimension
                # Questions format: * **P#-D#-Q#:** question text
                question_pattern = rf'\* \*\*{policy_id}-{dim_id}-Q(\d+):\*\* ([^\n]+)'
                question_matches = re.finditer(question_pattern, dim_content)
                
                for q_match in question_matches:
                    q_num = q_match.group(1)
                    q_text = q_match.group(2).strip()
                    q_id = f"Q{q_num}"
                    full_id = f"{policy_id}-{dim_id}-{q_id}"
                    
                    # Create question object
                    question = Question(
                        policy_id=policy_id,
                        policy_name=policy_name,
                        dimension_id=dim_id,
                        dimension_name=dim_name,
                        question_id=q_id,
                        full_id=full_id,
                        text=q_text
                    )
                    
                    # Add to dimension
                    dimension.questions.append(question)
                    
                    # Index question for fast lookup
                    self._question_index[full_id] = question
                    
                    # Index by dimension
                    dim_key = f"{policy_id}-{dim_id}"
                    if dim_key not in self._dimension_index:
                        self._dimension_index[dim_key] = []
                    self._dimension_index[dim_key].append(question)
                
                # Add dimension to policy
                policy.dimensions.append(dimension)
            
            # Add policy to collection and index
            self._policies.append(policy)
            self._policy_index[policy_id] = policy
    
    def get_all_policies(self) -> List[Policy]:
        """Get all 10 policies"""
        self._ensure_parsed()
        return self._policies
    
    def get_policy(self, policy_id: str) -> Optional[Policy]:
        """
        Get a specific policy by ID.
        
        Args:
            policy_id: Policy identifier (e.g., "P1")
        
        Returns:
            Policy object or None if not found
        """
        self._ensure_parsed()
        return self._policy_index.get(policy_id)
    
    def get_dimension(self, policy_id: str, dimension_id: str) -> Optional[Dimension]:
        """
        Get a specific dimension within a policy.
        
        Args:
            policy_id: Policy identifier (e.g., "P1")
            dimension_id: Dimension identifier (e.g., "D1")
        
        Returns:
            Dimension object or None if not found
        """
        policy = self.get_policy(policy_id)
        if policy:
            for dim in policy.dimensions:
                if dim.dimension_id == dimension_id:
                    return dim
        return None
    
    def get_question(self, full_id: str) -> Optional[Question]:
        """
        Get a specific question by full ID.
        
        Args:
            full_id: Full question identifier (e.g., "P1-D1-Q1")
        
        Returns:
            Question object or None if not found
        """
        self._ensure_parsed()
        return self._question_index.get(full_id)
    
    def get_questions_by_dimension(self, policy_id: str, dimension_id: str) -> List[Question]:
        """
        Get all questions for a specific dimension.
        
        Args:
            policy_id: Policy identifier (e.g., "P1")
            dimension_id: Dimension identifier (e.g., "D1")
        
        Returns:
            List of questions in that dimension
        """
        self._ensure_parsed()
        dim_key = f"{policy_id}-{dimension_id}"
        return self._dimension_index.get(dim_key, [])
    
    def get_all_questions(self) -> List[Question]:
        """Get all 300 questions"""
        self._ensure_parsed()
        return list(self._question_index.values())
    
    def get_dimension_names(self) -> Dict[str, str]:
        """
        Get mapping of dimension IDs to names.
        
        Returns:
            Dict mapping dimension IDs (D1-D6) to descriptive names
        """
        self._ensure_parsed()
        
        # Extract unique dimension names (they should be consistent across policies)
        dimension_names = {}
        if self._policies:
            # Use first policy as reference (dimensions should be consistent)
            for dim in self._policies[0].dimensions:
                dimension_names[dim.dimension_id] = dim.dimension_name
        
        return dimension_names
    
    def get_policy_names(self) -> Dict[str, str]:
        """
        Get mapping of policy IDs to names.
        
        Returns:
            Dict mapping policy IDs (P1-P10) to policy names
        """
        self._ensure_parsed()
        return {p.policy_id: p.policy_name for p in self._policies}
    
    def validate_structure(self) -> Dict[str, any]:
        """
        Validate questionnaire structure meets expected format.
        
        Returns:
            Dict with validation results including:
            - valid: bool indicating if structure is valid
            - policy_count: number of policies found
            - dimension_count: number of dimensions per policy
            - question_count: total questions found
            - errors: list of validation errors
        """
        self._ensure_parsed()
        
        errors = []
        
        # Check policy count (should be 10)
        if len(self._policies) != 10:
            errors.append(f"Expected 10 policies, found {len(self._policies)}")
        
        # Check each policy has 6 dimensions
        for policy in self._policies:
            if len(policy.dimensions) != 6:
                errors.append(
                    f"Policy {policy.policy_id} has {len(policy.dimensions)} dimensions, expected 6"
                )
            
            # Check each dimension has questions
            for dim in policy.dimensions:
                if len(dim.questions) == 0:
                    errors.append(
                        f"Dimension {policy.policy_id}-{dim.dimension_id} has no questions"
                    )
        
        # Check total question count (should be 300)
        total_questions = len(self._question_index)
        if total_questions != 300:
            errors.append(f"Expected 300 questions, found {total_questions}")
        
        return {
            "valid": len(errors) == 0,
            "policy_count": len(self._policies),
            "dimension_count": [len(p.dimensions) for p in self._policies],
            "question_count": total_questions,
            "errors": errors
        }
    
    def get_canonical_path(self) -> Path:
        """
        Get the canonical path to the questionnaire file.
        
        Returns:
            Absolute path to cuestionario_canonico
        """
        return self.cuestionario_path.resolve()


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_questionnaire_parser(cuestionario_path: Optional[Path] = None) -> QuestionnaireParser:
    """
    Factory function to create canonical questionnaire parser.
    
    Args:
        cuestionario_path: Optional path to cuestionario file.
            If None, uses canonical location.
    
    Returns:
        Initialized QuestionnaireParser
    
    Raises:
        FileNotFoundError: If canonical file not found
    """
    return QuestionnaireParser(cuestionario_path)
