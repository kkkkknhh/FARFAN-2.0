#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convergence Verification Script for FARFAN 2.0
==============================================

Verifies that cuestionario_canonico, questions_config.json, and guia_cuestionario.json
are properly aligned and use correct canonical notation (P#-D#-Q# format).

Author: AI Systems Architect
Version: 1.0.0
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
from canonical_notation import (
    CanonicalNotationValidator,
    CanonicalID,
    QUESTION_UNIQUE_ID_PATTERN,
    RUBRIC_KEY_PATTERN
)


@dataclass
class ConvergenceIssue:
    """Represents a convergence issue found during validation"""
    question_id: str
    issue_type: str
    description: str
    suggested_fix: str
    severity: str = "MEDIUM"  # LOW, MEDIUM, HIGH, CRITICAL
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "issue_type": self.issue_type,
            "description": self.description,
            "suggested_fix": self.suggested_fix,
            "severity": self.severity
        }


class ConvergenceVerifier:
    """Main convergence verification engine"""
    
    def __init__(self, repo_path: Path = None):
        self.repo_path = repo_path or Path("/home/runner/work/FARFAN-2.0/FARFAN-2.0")
        self.validator = CanonicalNotationValidator()
        self.issues: List[ConvergenceIssue] = []
        self.recommendations: List[str] = []
        
        # Expected structure
        self.expected_policies = 10  # P1-P10
        self.expected_dimensions = 6  # D1-D6
        self.expected_questions_per_dim = 5  # Q1-Q5 per dimension
        self.expected_total_questions = 300
        
        # Load configurations
        self.questions_config = self._load_questions_config()
        self.guia_cuestionario = self._load_guia_cuestionario()
        self.cuestionario_canonico_text = self._load_cuestionario_canonico()
        
    def _load_questions_config(self) -> Dict[str, Any]:
        """Load questions_config.json (handles multiple JSON objects)"""
        config_path = self.repo_path / "questions_config.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"questions_config.json not found at {config_path}")
        
        # Read the entire file
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into separate JSON objects
        json_objects = []
        current_obj = ""
        brace_count = 0
        in_string = False
        escape_next = False
        
        for char in content:
            if escape_next:
                current_obj += char
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                current_obj += char
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
            
            if not in_string:
                if char == '{':
                    if brace_count == 0:
                        current_obj = char
                    else:
                        current_obj += char
                    brace_count += 1
                elif char == '}':
                    current_obj += char
                    brace_count -= 1
                    if brace_count == 0:
                        # Complete JSON object
                        try:
                            obj = json.loads(current_obj)
                            json_objects.append(obj)
                        except json.JSONDecodeError as e:
                            print(f"Warning: Could not parse JSON object: {e}")
                        current_obj = ""
                else:
                    if brace_count > 0:
                        current_obj += char
            else:
                current_obj += char
        
        # Merge all objects
        merged = {}
        for obj in json_objects:
            for key, value in obj.items():
                if key in merged:
                    if isinstance(merged[key], dict) and isinstance(value, dict):
                        merged[key].update(value)
                    elif isinstance(merged[key], list) and isinstance(value, list):
                        merged[key].extend(value)
                    else:
                        merged[key] = value
                else:
                    merged[key] = value
        
        return merged
    
    def _load_guia_cuestionario(self) -> Dict[str, Any]:
        """Load guia_cuestionario.json"""
        guia_path = self.repo_path / "guia_cuestionario"
        
        if not guia_path.exists():
            raise FileNotFoundError(f"guia_cuestionario not found at {guia_path}")
        
        with open(guia_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Try to extract the first valid JSON object
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
            return json.loads(valid_json)
        else:
            # Fallback to regular parsing
            return json.loads(content)
    
    def _load_cuestionario_canonico(self) -> str:
        """Load cuestionario_canonico text file"""
        canonico_path = self.repo_path / "cuestionario_canonico"
        
        if not canonico_path.exists():
            raise FileNotFoundError(f"cuestionario_canonico not found at {canonico_path}")
        
        with open(canonico_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def verify_canonical_notation_usage(self) -> None:
        """Verify that all questions use correct canonical notation (P#-D#-Q#)"""
        print("=== Verifying Canonical Notation Usage ===\n")
        
        # Extract all question IDs from cuestionario_canonico
        pattern = r'\*\*\s*(P\d{1,2}-D\d-Q\d{1,2}):\*\*'
        found_ids = re.findall(pattern, self.cuestionario_canonico_text)
        
        print(f"Found {len(found_ids)} question IDs in cuestionario_canonico")
        
        # Validate each ID
        invalid_count = 0
        for qid in found_ids:
            if not self.validator.validate_question_unique_id(qid):
                self.issues.append(ConvergenceIssue(
                    question_id=qid,
                    issue_type="invalid_canonical_notation",
                    description=f"Question ID '{qid}' does not match canonical format P(10|[1-9])-D[1-6]-Q[1-9][0-9]*",
                    suggested_fix=f"Correct the question ID to follow canonical notation",
                    severity="CRITICAL"
                ))
                invalid_count += 1
        
        print(f"  Valid canonical IDs: {len(found_ids) - invalid_count}")
        print(f"  Invalid IDs: {invalid_count}")
        
        # Check for coverage (should have all 300 questions)
        unique_ids = set(found_ids)
        if len(unique_ids) != self.expected_total_questions:
            self.issues.append(ConvergenceIssue(
                question_id="SYSTEM",
                issue_type="incomplete_coverage",
                description=f"Expected {self.expected_total_questions} unique questions, found {len(unique_ids)}",
                suggested_fix="Ensure all 300 questions (10 policies × 6 dimensions × 5 questions) are present",
                severity="CRITICAL"
            ))
        
        print(f"  Unique question IDs: {len(unique_ids)}/{self.expected_total_questions}\n")
    
    def verify_scoring_consistency(self) -> None:
        """Verify that scoring rubrics are consistent across all files"""
        print("=== Verifying Scoring Consistency ===\n")
        
        # Check if guia_cuestionario has scoring system
        if 'scoring_system' in self.guia_cuestionario:
            scoring_system = self.guia_cuestionario['scoring_system']
            print("Found scoring system in guia_cuestionario")
            
            # Verify response scale
            if 'response_scale' in scoring_system:
                response_scale = scoring_system['response_scale']
                print(f"  Response scale levels: {list(response_scale.keys())}")
                
                # Verify all levels have proper ranges
                for level, config in response_scale.items():
                    if 'range' not in config:
                        self.issues.append(ConvergenceIssue(
                            question_id="SCORING",
                            issue_type="missing_score_range",
                            description=f"Response scale level '{level}' missing range specification",
                            suggested_fix=f"Add 'range' field to level '{level}'",
                            severity="HIGH"
                        ))
        
        # Check questions_config for scoring
        if 'preguntas_base' in self.questions_config:
            base_questions = self.questions_config['preguntas_base']
            print(f"\nFound {len(base_questions)} base questions in questions_config")
            
            questions_with_scoring = 0
            for q in base_questions:
                if 'scoring' in q:
                    questions_with_scoring += 1
                    # Verify scoring has all required levels
                    scoring = q['scoring']
                    expected_levels = ['excelente', 'bueno', 'aceptable', 'insuficiente']
                    for level in expected_levels:
                        if level not in scoring:
                            self.issues.append(ConvergenceIssue(
                                question_id=q.get('id', 'UNKNOWN'),
                                issue_type="incomplete_scoring",
                                description=f"Question missing '{level}' scoring level",
                                suggested_fix=f"Add '{level}' level to scoring rubric",
                                severity="MEDIUM"
                            ))
            
            print(f"  Questions with scoring: {questions_with_scoring}/{len(base_questions)}")
        
        print()
    
    def verify_dimension_mapping(self) -> None:
        """Verify that dimension mappings are consistent"""
        print("=== Verifying Dimension Mappings ===\n")
        
        # Check decalogo_dimension_mapping in guia_cuestionario
        if 'decalogo_dimension_mapping' in self.guia_cuestionario:
            mapping = self.guia_cuestionario['decalogo_dimension_mapping']
            print(f"Found dimension mapping for {len(mapping)} policies")
            
            for policy_id, policy_map in mapping.items():
                # Validate policy ID format
                if not self.validator.validate_policy(policy_id):
                    self.issues.append(ConvergenceIssue(
                        question_id=policy_id,
                        issue_type="invalid_policy_id",
                        description=f"Policy ID '{policy_id}' does not match pattern P(10|[1-9])",
                        suggested_fix=f"Correct policy ID format",
                        severity="CRITICAL"
                    ))
                
                # Check dimension weights
                dimension_weights = []
                for i in range(1, 7):
                    dim_key = f"D{i}_weight"
                    if dim_key in policy_map:
                        dimension_weights.append(policy_map[dim_key])
                
                if dimension_weights:
                    total_weight = sum(dimension_weights)
                    if abs(total_weight - 1.0) > 0.01:  # Allow small floating point error
                        self.issues.append(ConvergenceIssue(
                            question_id=policy_id,
                            issue_type="invalid_weight_sum",
                            description=f"Dimension weights sum to {total_weight:.2f}, expected 1.0",
                            suggested_fix=f"Adjust dimension weights to sum to 1.0",
                            severity="HIGH"
                        ))
        
        print()
    
    def verify_no_legacy_mapping(self) -> None:
        """Verify that there are no legacy file contributor mappings"""
        print("=== Verifying No Legacy File Mappings ===\n")
        
        # Check for any references to old file mapping patterns
        legacy_patterns = [
            r'file_contributors',
            r'archivo_contribuyente',
            r'source_files',
            r'contributing_files'
        ]
        
        files_to_check = [
            ('questions_config.json', json.dumps(self.questions_config)),
            ('guia_cuestionario', json.dumps(self.guia_cuestionario)),
            ('cuestionario_canonico', self.cuestionario_canonico_text)
        ]
        
        legacy_found = False
        for filename, content in files_to_check:
            for pattern in legacy_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    legacy_found = True
                    self.issues.append(ConvergenceIssue(
                        question_id="SYSTEM",
                        issue_type="legacy_mapping_found",
                        description=f"Found legacy mapping pattern '{pattern}' in {filename}",
                        suggested_fix=f"Remove legacy file contributor mapping from {filename}",
                        severity="HIGH"
                    ))
                    print(f"  ⚠️  Found '{pattern}' in {filename}")
        
        if not legacy_found:
            print("  ✓ No legacy file mappings found")
        
        print()
    
    def verify_module_references(self) -> None:
        """Verify that module references are correct"""
        print("=== Verifying Module References ===\n")
        
        # Expected modules in the system
        expected_modules = {
            'dnp_integration',
            'dereck_beach',
            'competencias_municipales',
            'mga_indicadores',
            'pdet_lineamientos',
            'initial_processor_causal_policy'
        }
        
        # Check questions_config for module references
        if 'preguntas_base' in self.questions_config:
            all_modules = set()
            for q in self.questions_config['preguntas_base']:
                if 'modulos_responsables' in q:
                    all_modules.update(q['modulos_responsables'])
            
            print(f"Found {len(all_modules)} unique module references:")
            for module in sorted(all_modules):
                status = "✓" if module in expected_modules else "⚠️"
                print(f"  {status} {module}")
                
                if module not in expected_modules:
                    self.issues.append(ConvergenceIssue(
                        question_id="MODULES",
                        issue_type="unknown_module_reference",
                        description=f"Reference to unknown module '{module}'",
                        suggested_fix=f"Verify module '{module}' exists or remove reference",
                        severity="MEDIUM"
                    ))
        
        print()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate the final convergence report"""
        # Calculate convergence percentage
        total_checks = self.expected_total_questions
        issues_count = len(self.issues)
        critical_issues = sum(1 for issue in self.issues if issue.severity == "CRITICAL")
        
        # Consider critical issues as blocking convergence
        if critical_issues > 0:
            percent_converged = 0.0
        else:
            percent_converged = max(0.0, (1 - issues_count / total_checks) * 100)
        
        # Generate recommendations
        self._generate_recommendations()
        
        report = {
            "convergence_issues": [issue.to_dict() for issue in self.issues],
            "recommendations": self.recommendations,
            "verification_summary": {
                "percent_questions_converged": round(percent_converged, 2),
                "issues_detected": issues_count,
                "critical_issues": critical_issues,
                "high_priority_issues": sum(1 for issue in self.issues if issue.severity == "HIGH"),
                "medium_priority_issues": sum(1 for issue in self.issues if issue.severity == "MEDIUM"),
                "low_priority_issues": sum(1 for issue in self.issues if issue.severity == "LOW"),
                "total_questions_expected": self.expected_total_questions,
                "verification_timestamp": "2025-10-14T23:57:42Z"
            }
        }
        
        return report
    
    def _generate_recommendations(self) -> None:
        """Generate actionable recommendations based on found issues"""
        self.recommendations = []
        
        # Group issues by type
        issues_by_type = {}
        for issue in self.issues:
            if issue.issue_type not in issues_by_type:
                issues_by_type[issue.issue_type] = []
            issues_by_type[issue.issue_type].append(issue)
        
        # Generate recommendations
        if 'invalid_canonical_notation' in issues_by_type:
            count = len(issues_by_type['invalid_canonical_notation'])
            self.recommendations.append(
                f"CRITICAL: Correct {count} invalid canonical notation IDs to follow P(10|[1-9])-D[1-6]-Q[1-9][0-9]* format"
            )
        
        if 'incomplete_coverage' in issues_by_type:
            self.recommendations.append(
                f"CRITICAL: Ensure all 300 questions (10 policies × 6 dimensions × 5 questions) are documented"
            )
        
        if 'incomplete_scoring' in issues_by_type:
            count = len(issues_by_type['incomplete_scoring'])
            self.recommendations.append(
                f"Add complete scoring rubrics (excelente, bueno, aceptable, insuficiente) to {count} questions"
            )
        
        if 'invalid_weight_sum' in issues_by_type:
            count = len(issues_by_type['invalid_weight_sum'])
            self.recommendations.append(
                f"Adjust dimension weights in {count} policies to sum to exactly 1.0"
            )
        
        if 'legacy_mapping_found' in issues_by_type:
            self.recommendations.append(
                "Remove all legacy file contributor mapping patterns from configuration files"
            )
        
        if 'unknown_module_reference' in issues_by_type:
            count = len(issues_by_type['unknown_module_reference'])
            self.recommendations.append(
                f"Verify or remove {count} unknown module references"
            )
        
        # General recommendations
        self.recommendations.append(
            "Alinear scoring y mapping de todas las preguntas con la guía técnica"
        )
        self.recommendations.append(
            "Verificar que todas las funciones usadas en el mapeo se correspondan con outputs esperados"
        )
        
        if len(self.issues) == 0:
            self.recommendations.append(
                "✓ Sistema completamente convergente - No se requieren acciones correctivas"
            )
    
    def run_full_verification(self) -> Dict[str, Any]:
        """Run all verification checks and generate report"""
        print("=" * 70)
        print("FARFAN 2.0 - Convergence Verification Report")
        print("=" * 70)
        print()
        
        self.verify_canonical_notation_usage()
        self.verify_scoring_consistency()
        self.verify_dimension_mapping()
        self.verify_no_legacy_mapping()
        self.verify_module_references()
        
        report = self.generate_report()
        
        print("=" * 70)
        print("Verification Summary")
        print("=" * 70)
        print(f"Convergence: {report['verification_summary']['percent_questions_converged']:.1f}%")
        print(f"Issues Found: {report['verification_summary']['issues_detected']}")
        print(f"  Critical: {report['verification_summary']['critical_issues']}")
        print(f"  High: {report['verification_summary']['high_priority_issues']}")
        print(f"  Medium: {report['verification_summary']['medium_priority_issues']}")
        print(f"  Low: {report['verification_summary']['low_priority_issues']}")
        print()
        
        return report


def main():
    """Main entry point"""
    verifier = ConvergenceVerifier()
    report = verifier.run_full_verification()
    
    # Save report
    output_path = Path("/home/runner/work/FARFAN-2.0/FARFAN-2.0/convergence_report.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"Full report saved to: {output_path}")
    print()
    
    # Print recommendations
    if report['recommendations']:
        print("=" * 70)
        print("Recommendations")
        print("=" * 70)
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")
        print()
    
    # Return exit code based on critical issues
    if report['verification_summary']['critical_issues'] > 0:
        print("❌ VERIFICATION FAILED - Critical issues must be resolved")
        return 1
    else:
        print("✓ VERIFICATION PASSED")
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
