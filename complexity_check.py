#!/usr/bin/env python3
import re

def count_complexity(code):
    """Simple complexity counter"""
    complexity = 1
    complexity += len(re.findall(r'\bfor\b', code))
    complexity += len(re.findall(r'\bif\b', code))
    complexity += len(re.findall(r'\b(and|or)\b', code))
    complexity += code.count('[') if 'for' in code and '[' in code else 0
    return complexity

original = """
    def _extract_point_evidence(self, text, sentences, point_code):
        pattern = self.point_patterns.get(point_code)
        if not pattern:
            return {}
        relevant_sentences = [s for s in sentences if pattern.search(s)]
        if not relevant_sentences:
            return {}
        evidence_by_dimension = {}
        for dimension, categories in self._pattern_registry.items():
            dimension_evidence = []
            for category, compiled_patterns in categories.items():
                matches = []
                positions = []
                for compiled_pattern in compiled_patterns:
                    for sentence in relevant_sentences:
                        for match in compiled_pattern.finditer(sentence):
                            matches.append(match.group(0))
                            positions.append(match.start())
                if matches:
                    confidence = self.scorer.compute_evidence_score(
                        matches, len(text), pattern_specificity=0.85
                    )
                    if confidence >= self.config.confidence_threshold:
                        bundle = EvidenceBundle(
                            dimension=dimension,
                            category=category,
                            matches=matches[:self.config.max_evidence_per_pattern],
                            confidence=confidence,
                            match_positions=positions[:self.config.max_evidence_per_pattern],
                        )
                        dimension_evidence.append(bundle.to_dict())
            if dimension_evidence:
                evidence_by_dimension[dimension.value] = dimension_evidence
        return evidence_by_dimension
"""

refactored = """
    def _extract_point_evidence(self, text, sentences, point_code):
        pattern = self.point_patterns.get(point_code)
        if not pattern:
            return {}
        relevant_sentences = [s for s in sentences if pattern.search(s)]
        if not relevant_sentences:
            return {}
        evidence_by_dimension = {}
        for dimension, categories in self._pattern_registry.items():
            dimension_evidence = []
            for category, compiled_patterns in categories.items():
                matches, positions = self._match_patterns_in_sentences(
                    compiled_patterns, relevant_sentences
                )
                if matches:
                    confidence = self._compute_evidence_confidence(
                        matches, len(text), pattern_specificity=0.85
                    )
                    if confidence >= self.config.confidence_threshold:
                        evidence_dict = self._construct_evidence_bundle(
                            dimension, category, matches, positions, confidence
                        )
                        dimension_evidence.append(evidence_dict)
            if dimension_evidence:
                evidence_by_dimension[dimension.value] = dimension_evidence
        return evidence_by_dimension
"""

print("Cyclomatic Complexity Analysis")
print("=" * 60)
print(f"Original method:    {count_complexity(original)}")
print(f"Refactored method:  {count_complexity(refactored)}")
print(f"Reduction:          {count_complexity(original) - count_complexity(refactored)}")
