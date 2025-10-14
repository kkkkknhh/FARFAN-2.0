#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validación y auditoría del archivo cuestionario_canonico_mapped.json
====================================================================

Este script valida la completitud, coherencia y calidad del mapeo
de preguntas a callables.
"""
import json
from pathlib import Path
from typing import Dict, List, Set, Any
from collections import defaultdict

def load_mapping():
    """Carga el archivo de mapeo"""
    with open('/home/runner/work/FARFAN-2.0/FARFAN-2.0/cuestionario_canonico_mapped.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def validate_structure(data: Dict) -> List[str]:
    """Valida la estructura del archivo"""
    errors = []
    
    # Check metadata
    if 'metadata' not in data:
        errors.append("CRITICAL: Missing 'metadata' field")
    
    # Check questions
    if 'questions' not in data:
        errors.append("CRITICAL: Missing 'questions' field")
        return errors
    
    # Validate each question
    required_fields = ['id', 'texto', 'mapping', 'resolution_mode', 'scoring', 'coverage_complete']
    for i, q in enumerate(data['questions']):
        for field in required_fields:
            if field not in q:
                errors.append(f"Question {i} ({q.get('id', 'UNKNOWN')}): Missing field '{field}'")
        
        # Validate mapping structure
        if 'mapping' in q:
            for j, m in enumerate(q['mapping']):
                required_mapping_fields = ['script', 'callable', 'contribution_type', 
                                          'question_portion_handled', 'input_needed', 'output_expected']
                for field in required_mapping_fields:
                    if field not in m:
                        errors.append(f"Question {q.get('id')}, mapping {j}: Missing field '{field}'")
        
        # Validate scoring
        if 'scoring' in q:
            required_scoring = ['cuantitativo', 'cualitativo', 'justificacion']
            for field in required_scoring:
                if field not in q['scoring']:
                    errors.append(f"Question {q.get('id')}: Missing scoring field '{field}'")
    
    return errors

def analyze_coverage(data: Dict) -> Dict[str, Any]:
    """Analiza la cobertura de preguntas y callables"""
    questions = data['questions']
    
    # Coverage by dimension
    dim_coverage = defaultdict(lambda: {'total': 0, 'complete': 0})
    for q in questions:
        # Extract dimension from ID (e.g., P1-D1-Q1 -> D1)
        dim = q['id'].split('-')[1]
        dim_coverage[dim]['total'] += 1
        if q.get('coverage_complete', False):
            dim_coverage[dim]['complete'] += 1
    
    # Coverage by policy
    policy_coverage = defaultdict(lambda: {'total': 0, 'complete': 0})
    for q in questions:
        # Extract policy from ID (e.g., P1-D1-Q1 -> P1)
        policy = q['id'].split('-')[0]
        policy_coverage[policy]['total'] += 1
        if q.get('coverage_complete', False):
            policy_coverage[policy]['complete'] += 1
    
    # Scripts usage
    script_usage = defaultdict(int)
    for q in questions:
        for m in q.get('mapping', []):
            script_usage[m['script']] += 1
    
    # Callable usage
    callable_usage = defaultdict(int)
    for q in questions:
        for m in q.get('mapping', []):
            callable_usage[f"{m['script']}::{m['callable']}"] += 1
    
    return {
        'dimension_coverage': dict(dim_coverage),
        'policy_coverage': dict(policy_coverage),
        'script_usage': dict(script_usage),
        'callable_usage': dict(callable_usage),
        'total_unique_scripts': len(script_usage),
        'total_unique_callables': len(callable_usage)
    }

def identify_orphaned_callables(data: Dict) -> List[Dict[str, Any]]:
    """Identifica callables definidos pero no utilizados"""
    # Get all callables defined in scripts_inventory
    inventory = data.get('scripts_inventory', {})
    defined_callables = set()
    
    for script_name, script_info in inventory.items():
        # Add classes
        for cls in script_info.get('classes', []):
            defined_callables.add(f"{script_name}::{cls['name']} (class)")
            # Add methods
            for method in cls.get('methods', []):
                defined_callables.add(f"{script_name}::{cls['name']}.{method}")
        
        # Add functions
        for func in script_info.get('functions', []):
            defined_callables.add(f"{script_name}::{func['name']}")
    
    # Get all used callables
    used_callables = set()
    for q in data['questions']:
        for m in q.get('mapping', []):
            used_callables.add(f"{m['script']}::{m['callable']}")
    
    # Find orphans
    orphaned = sorted(defined_callables - used_callables)
    
    return orphaned

def analyze_scoring(data: Dict) -> Dict[str, Any]:
    """Analiza el scoring de las preguntas"""
    questions = data['questions']
    
    scores = [q['scoring']['cuantitativo'] for q in questions if 'scoring' in q]
    
    # Score distribution by dimension
    dim_scores = defaultdict(list)
    for q in questions:
        if 'scoring' in q:
            dim = q['id'].split('-')[1]
            dim_scores[dim].append(q['scoring']['cuantitativo'])
    
    # Score distribution by policy
    policy_scores = defaultdict(list)
    for q in questions:
        if 'scoring' in q:
            policy = q['id'].split('-')[0]
            policy_scores[policy].append(q['scoring']['cuantitativo'])
    
    return {
        'overall': {
            'mean': sum(scores) / len(scores) if scores else 0,
            'min': min(scores) if scores else 0,
            'max': max(scores) if scores else 0
        },
        'by_dimension': {
            dim: {
                'mean': sum(scores) / len(scores),
                'count': len(scores)
            }
            for dim, scores in dim_scores.items()
        },
        'by_policy': {
            policy: {
                'mean': sum(scores) / len(scores),
                'count': len(scores)
            }
            for policy, scores in policy_scores.items()
        }
    }

def generate_audit_report(data: Dict) -> str:
    """Genera reporte de auditoría completo"""
    report = []
    report.append("="*80)
    report.append("AUDITORÍA DE cuestionario_canonico_mapped.json")
    report.append("="*80)
    report.append("")
    
    # 1. Structure validation
    report.append("1. VALIDACIÓN DE ESTRUCTURA")
    report.append("-" * 40)
    errors = validate_structure(data)
    if errors:
        report.append(f"❌ {len(errors)} errores encontrados:")
        for err in errors[:10]:  # Show first 10
            report.append(f"  - {err}")
        if len(errors) > 10:
            report.append(f"  ... y {len(errors) - 10} más")
    else:
        report.append("✅ Estructura válida - todos los campos requeridos presentes")
    report.append("")
    
    # 2. Coverage analysis
    report.append("2. ANÁLISIS DE COBERTURA")
    report.append("-" * 40)
    coverage = analyze_coverage(data)
    
    report.append(f"Total preguntas: {len(data['questions'])}")
    report.append(f"Scripts únicos utilizados: {coverage['total_unique_scripts']}")
    report.append(f"Callables únicos mapeados: {coverage['total_unique_callables']}")
    report.append("")
    
    report.append("Cobertura por dimensión:")
    for dim in sorted(coverage['dimension_coverage'].keys()):
        stats = coverage['dimension_coverage'][dim]
        pct = (stats['complete'] / stats['total'] * 100) if stats['total'] > 0 else 0
        report.append(f"  {dim}: {stats['complete']}/{stats['total']} ({pct:.1f}%)")
    report.append("")
    
    report.append("Cobertura por política:")
    for policy in sorted(coverage['policy_coverage'].keys()):
        stats = coverage['policy_coverage'][policy]
        pct = (stats['complete'] / stats['total'] * 100) if stats['total'] > 0 else 0
        report.append(f"  {policy}: {stats['complete']}/{stats['total']} ({pct:.1f}%)")
    report.append("")
    
    report.append("Scripts más utilizados:")
    top_scripts = sorted(coverage['script_usage'].items(), key=lambda x: x[1], reverse=True)[:10]
    for script, count in top_scripts:
        report.append(f"  {script}: {count} mappings")
    report.append("")
    
    # 3. Orphaned callables
    report.append("3. CALLABLES HUÉRFANOS")
    report.append("-" * 40)
    orphaned = identify_orphaned_callables(data)
    if orphaned:
        report.append(f"⚠️  {len(orphaned)} callables no mapeados:")
        for callable_name in orphaned[:20]:  # Show first 20
            report.append(f"  - {callable_name}")
        if len(orphaned) > 20:
            report.append(f"  ... y {len(orphaned) - 20} más")
        report.append("")
        report.append("NOTA: Callables privados (__init__, __str__, etc.) y de testing/demo")
        report.append("son esperados como huérfanos. Verificar solo callables públicos.")
    else:
        report.append("✅ Todos los callables están mapeados")
    report.append("")
    
    # 4. Scoring analysis
    report.append("4. ANÁLISIS DE SCORING")
    report.append("-" * 40)
    scoring = analyze_scoring(data)
    
    report.append(f"Score promedio global: {scoring['overall']['mean']:.2f}")
    report.append(f"Score mínimo: {scoring['overall']['min']:.2f}")
    report.append(f"Score máximo: {scoring['overall']['max']:.2f}")
    report.append("")
    
    report.append("Score promedio por dimensión:")
    for dim in sorted(scoring['by_dimension'].keys()):
        stats = scoring['by_dimension'][dim]
        report.append(f"  {dim}: {stats['mean']:.2f} ({stats['count']} preguntas)")
    report.append("")
    
    # 5. Unused scripts
    report.append("5. SCRIPTS NO UTILIZADOS")
    report.append("-" * 40)
    if 'unused_scripts_analysis' in data:
        unused = data['unused_scripts_analysis']
        report.append(f"Total scripts no utilizados: {unused['count']}")
        for script_info in unused['scripts']:
            report.append(f"\n  📄 {script_info['name']}")
            report.append(f"     Razón: {script_info['reason']}")
            report.append(f"     Recomendación: {script_info['recommendation']}")
    report.append("")
    
    # 6. Summary
    report.append("="*80)
    report.append("RESUMEN EJECUTIVO")
    report.append("="*80)
    
    total_complete = sum(1 for q in data['questions'] if q.get('coverage_complete', False))
    pct_complete = (total_complete / len(data['questions']) * 100) if data['questions'] else 0
    
    report.append(f"✅ Completitud: {total_complete}/{len(data['questions'])} ({pct_complete:.1f}%)")
    report.append(f"✅ Score promedio: {scoring['overall']['mean']:.2f}/10")
    report.append(f"✅ Scripts utilizados: {coverage['total_unique_scripts']}")
    report.append(f"✅ Callables mapeados: {coverage['total_unique_callables']}")
    
    if errors:
        report.append(f"⚠️  Errores de estructura: {len(errors)}")
    if orphaned:
        report.append(f"ℹ️  Callables huérfanos: {len(orphaned)}")
    
    report.append("")
    report.append("="*80)
    
    return "\n".join(report)

def main():
    """Función principal"""
    print("Cargando cuestionario_canonico_mapped.json...")
    data = load_mapping()
    
    print("Generando reporte de auditoría...")
    report = generate_audit_report(data)
    
    # Print to console
    print("\n" + report)
    
    # Save to file
    output_path = Path('/home/runner/work/FARFAN-2.0/FARFAN-2.0/VALIDATION_REPORT.md')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Reporte de Validación: cuestionario_canonico_mapped.json\n\n")
        f.write("```\n")
        f.write(report)
        f.write("\n```\n")
    
    print(f"\n✅ Reporte guardado en: {output_path}")

if __name__ == '__main__':
    main()
