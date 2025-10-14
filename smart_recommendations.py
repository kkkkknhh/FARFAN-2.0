#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SMART Recommendations Framework with AHP Prioritization
========================================================

This module implements the SMART (Specific, Measurable, Achievable, Relevant, Time-bound)
recommendations framework with Analytic Hierarchy Process (AHP) for multi-criteria prioritization.

Features:
- SMART criteria validation
- AHP-based prioritization (impact, cost, urgency, viability)
- Gantt chart generation for implementation roadmap
- Success metrics (KPIs) definition

Author: AI Systems Architect
Version: 2.0.0
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import json


class Priority(Enum):
    """Recommendation priority levels"""
    CRITICAL = "CRÍTICO"
    HIGH = "ALTO"
    MEDIUM = "MEDIO"
    LOW = "BAJO"


class ImpactLevel(Enum):
    """Expected impact levels"""
    TRANSFORMATIONAL = "Transformacional"
    HIGH = "Alto"
    MODERATE = "Moderado"
    LOW = "Bajo"


@dataclass
class SMARTCriteria:
    """
    SMART criteria for recommendations
    
    Each recommendation must satisfy all SMART criteria for validation
    """
    specific: str  # Specific action with concrete references
    measurable: str  # Quantitative, verifiable metric
    achievable: str  # Operational and budgetary conditions
    relevant: str  # Justification aligned with ODS or strategic objective
    time_bound: str  # Defined temporal horizon
    
    def validate(self) -> bool:
        """Validate that all SMART criteria are defined"""
        return all([
            self.specific and len(self.specific) > 20,
            self.measurable and len(self.measurable) > 10,
            self.achievable and len(self.achievable) > 10,
            self.relevant and len(self.relevant) > 10,
            self.time_bound and len(self.time_bound) > 5
        ])
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class AHPWeights:
    """
    Analytic Hierarchy Process (AHP) weights for prioritization
    
    Weights must sum to 1.0
    """
    impact: float = 0.4  # Weight for impact criterion
    cost: float = 0.2    # Weight for cost criterion
    urgency: float = 0.3  # Weight for urgency criterion
    viability: float = 0.1  # Weight for political viability
    
    def __post_init__(self):
        """Validate weights sum to 1.0"""
        total = self.impact + self.cost + self.urgency + self.viability
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"AHP weights must sum to 1.0, got {total}")


@dataclass
class SuccessMetric:
    """
    Success metric (KPI) for recommendation
    
    Defines measurable success criteria with baseline and target
    """
    name: str  # KPI name
    description: str  # What this KPI measures
    baseline: float  # Current value
    target: float  # Expected value after implementation
    unit: str  # Measurement unit
    measurement_method: str  # How to measure this KPI
    verification_source: str  # Source of verification data
    
    def get_expected_change(self) -> float:
        """Calculate expected percentage change"""
        if self.baseline == 0:
            return float('inf')
        return ((self.target - self.baseline) / self.baseline) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        d = asdict(self)
        d['expected_change_percent'] = self.get_expected_change()
        return d


@dataclass
class Dependency:
    """
    Dependency between recommendations
    
    Represents that one recommendation depends on another
    """
    depends_on: str  # ID of recommendation that must be completed first
    dependency_type: str  # "prerequisite", "concurrent", "sequential"
    description: str  # Description of the dependency


@dataclass
class SMARTRecommendation:
    """
    Complete SMART recommendation with AHP prioritization
    
    Represents a fully-specified, prioritized recommendation for policy improvement
    """
    id: str  # Unique recommendation ID
    title: str  # Short title
    smart_criteria: SMARTCriteria
    
    # AHP scoring (0-10 scale)
    impact_score: float  # Expected impact
    cost_score: float  # Cost (10=low cost, 1=high cost)
    urgency_score: float  # Urgency level
    viability_score: float  # Political/operational viability
    
    # Additional attributes
    priority: Priority  # Overall priority level
    impact_level: ImpactLevel  # Expected impact classification
    success_metrics: List[SuccessMetric] = field(default_factory=list)
    dependencies: List[Dependency] = field(default_factory=list)
    estimated_duration_days: int = 90  # Default 3 months
    responsible_entity: str = "Entidad Municipal"
    budget_range: Optional[Tuple[float, float]] = None  # (min, max) in COP
    ods_alignment: List[str] = field(default_factory=list)  # ODS numbers
    
    # Derived attributes
    ahp_score: float = field(init=False)  # Calculated AHP score
    
    def __post_init__(self):
        """Calculate AHP score"""
        self.ahp_score = self.calculate_ahp_score()
    
    def calculate_ahp_score(self, weights: Optional[AHPWeights] = None) -> float:
        """
        Calculate Analytic Hierarchy Process (AHP) score
        
        Args:
            weights: AHP weights (uses defaults if None)
            
        Returns:
            Weighted score (0-10 scale)
        """
        if weights is None:
            weights = AHPWeights()
        
        score = (
            self.impact_score * weights.impact +
            self.cost_score * weights.cost +
            self.urgency_score * weights.urgency +
            self.viability_score * weights.viability
        )
        return round(score, 2)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate recommendation completeness
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # Validate SMART criteria
        if not self.smart_criteria.validate():
            errors.append("SMART criteria incomplete or invalid")
        
        # Validate scores
        for score_name, score_value in [
            ("impact_score", self.impact_score),
            ("cost_score", self.cost_score),
            ("urgency_score", self.urgency_score),
            ("viability_score", self.viability_score)
        ]:
            if not (0 <= score_value <= 10):
                errors.append(f"{score_name} must be between 0 and 10, got {score_value}")
        
        # Validate success metrics
        if not self.success_metrics:
            errors.append("At least one success metric (KPI) is required")
        
        return (len(errors) == 0, errors)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "title": self.title,
            "smart_criteria": self.smart_criteria.to_dict(),
            "scoring": {
                "impact": self.impact_score,
                "cost": self.cost_score,
                "urgency": self.urgency_score,
                "viability": self.viability_score,
                "ahp_total": self.ahp_score
            },
            "priority": self.priority.value,
            "impact_level": self.impact_level.value,
            "success_metrics": [m.to_dict() for m in self.success_metrics],
            "dependencies": [asdict(d) for d in self.dependencies],
            "timeline": {
                "estimated_duration_days": self.estimated_duration_days,
                "estimated_duration_months": round(self.estimated_duration_days / 30, 1)
            },
            "responsible_entity": self.responsible_entity,
            "budget_range": self.budget_range,
            "ods_alignment": self.ods_alignment
        }
    
    def to_markdown(self) -> str:
        """Convert to markdown format"""
        md = f"""### {self.title} (Prioridad: {self.priority.value})

**ID:** {self.id}  
**Score AHP:** {self.ahp_score}/10  
**Impacto Esperado:** {self.impact_level.value}  

#### Criterios SMART

- **Específico:** {self.smart_criteria.specific}
- **Medible:** {self.smart_criteria.measurable}
- **Alcanzable:** {self.smart_criteria.achievable}
- **Relevante:** {self.smart_criteria.relevant}
- **Temporal:** {self.smart_criteria.time_bound}

#### Métricas de Éxito (KPIs)

"""
        for metric in self.success_metrics:
            change = metric.get_expected_change()
            md += f"""- **{metric.name}**: {metric.description}
  - Línea Base: {metric.baseline} {metric.unit}
  - Meta: {metric.target} {metric.unit}
  - Cambio Esperado: {change:+.1f}%
  - Verificación: {metric.verification_source}

"""
        
        if self.dependencies:
            md += "#### Dependencias\n\n"
            for dep in self.dependencies:
                md += f"- Depende de: {dep.depends_on} ({dep.dependency_type}): {dep.description}\n"
            md += "\n"
        
        if self.ods_alignment:
            md += f"#### Alineación ODS\n\n"
            md += f"ODS: {', '.join(self.ods_alignment)}\n\n"
        
        md += f"""#### Información Adicional

- **Duración Estimada:** {self.estimated_duration_days} días (~{round(self.estimated_duration_days/30, 1)} meses)
- **Entidad Responsable:** {self.responsible_entity}
"""
        
        if self.budget_range:
            md += f"- **Rango Presupuestal:** ${self.budget_range[0]:,.0f} - ${self.budget_range[1]:,.0f} COP\n"
        
        return md


class RecommendationPrioritizer:
    """
    Prioritizer for SMART recommendations using AHP
    """
    
    def __init__(self, weights: Optional[AHPWeights] = None):
        """
        Initialize prioritizer
        
        Args:
            weights: Custom AHP weights (uses defaults if None)
        """
        self.weights = weights or AHPWeights()
    
    def prioritize(self, recommendations: List[SMARTRecommendation]) -> List[SMARTRecommendation]:
        """
        Prioritize recommendations by AHP score
        
        Args:
            recommendations: List of recommendations
            
        Returns:
            Sorted list (highest priority first)
        """
        # Recalculate AHP scores with current weights
        for rec in recommendations:
            rec.ahp_score = rec.calculate_ahp_score(self.weights)
        
        # Sort by AHP score (descending)
        return sorted(recommendations, key=lambda r: r.ahp_score, reverse=True)
    
    def generate_gantt_data(self, recommendations: List[SMARTRecommendation],
                           start_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Generate Gantt chart data for recommendations
        
        Args:
            recommendations: List of prioritized recommendations
            start_date: Project start date (uses today if None)
            
        Returns:
            List of task dictionaries for Gantt chart
        """
        if start_date is None:
            start_date = datetime.now()
        
        # Build dependency graph
        dep_graph = self._build_dependency_graph(recommendations)
        
        # Calculate start dates considering dependencies
        tasks = []
        task_end_dates = {}
        
        for rec in recommendations:
            # Find earliest start date based on dependencies
            earliest_start = start_date
            for dep in rec.dependencies:
                if dep.depends_on in task_end_dates:
                    dep_end = task_end_dates[dep.depends_on]
                    if dep_end > earliest_start:
                        earliest_start = dep_end + timedelta(days=1)
            
            # Calculate end date
            end_date = earliest_start + timedelta(days=rec.estimated_duration_days)
            task_end_dates[rec.id] = end_date
            
            tasks.append({
                "id": rec.id,
                "title": rec.title,
                "start": earliest_start.isoformat(),
                "end": end_date.isoformat(),
                "duration_days": rec.estimated_duration_days,
                "priority": rec.priority.value,
                "ahp_score": rec.ahp_score,
                "dependencies": [d.depends_on for d in rec.dependencies]
            })
        
        return tasks
    
    def _build_dependency_graph(self, recommendations: List[SMARTRecommendation]) -> Dict[str, List[str]]:
        """Build dependency graph"""
        graph = {}
        for rec in recommendations:
            graph[rec.id] = [d.depends_on for d in rec.dependencies]
        return graph
    
    def generate_implementation_roadmap(self, recommendations: List[SMARTRecommendation]) -> str:
        """
        Generate implementation roadmap in Markdown format
        
        Args:
            recommendations: Prioritized recommendations
            
        Returns:
            Markdown-formatted roadmap
        """
        gantt_data = self.generate_gantt_data(recommendations)
        
        md = "# Roadmap de Implementación\n\n"
        md += f"**Fecha de Inicio:** {datetime.now().strftime('%Y-%m-%d')}\n\n"
        md += "## Cronograma de Actividades\n\n"
        
        for i, task in enumerate(gantt_data, 1):
            md += f"{i}. **{task['title']}** (Prioridad: {task['priority']})\n"
            md += f"   - Inicio: {task['start'][:10]}\n"
            md += f"   - Fin: {task['end'][:10]}\n"
            md += f"   - Duración: {task['duration_days']} días\n"
            md += f"   - Score AHP: {task['ahp_score']}/10\n"
            
            if task['dependencies']:
                md += f"   - Dependencias: {', '.join(task['dependencies'])}\n"
            
            md += "\n"
        
        return md


def create_example_recommendation() -> SMARTRecommendation:
    """Create an example SMART recommendation"""
    return SMARTRecommendation(
        id="REC-001",
        title="Fortalecer indicadores de línea base en educación",
        smart_criteria=SMARTCriteria(
            specific="Incluir indicador EDU-020 (Tasa de deserción escolar) en Meta EDU-003 con línea base actualizada del DANE",
            measurable="Reducir tasa de deserción escolar de 8.5% (línea base 2023) a 6.0% (meta 2027)",
            achievable="Requiere coordinación con Secretaría de Educación, acceso a SIMAT, presupuesto estimado $50M COP",
            relevant="Alineado con ODS 4 (Educación de Calidad) y PND 2022-2026",
            time_bound="Implementación en 6 meses (180 días), seguimiento trimestral"
        ),
        impact_score=8.5,
        cost_score=7.0,  # Relatively low cost
        urgency_score=9.0,  # High urgency
        viability_score=8.0,  # High viability
        priority=Priority.HIGH,
        impact_level=ImpactLevel.HIGH,
        success_metrics=[
            SuccessMetric(
                name="Tasa de deserción escolar",
                description="Porcentaje de estudiantes que abandonan el sistema educativo",
                baseline=8.5,
                target=6.0,
                unit="%",
                measurement_method="Reporte trimestral SIMAT",
                verification_source="Secretaría de Educación Municipal"
            ),
            SuccessMetric(
                name="Compliance Score EDU-003",
                description="Score de cumplimiento en formulación de meta educativa",
                baseline=0.55,
                target=0.85,
                unit="score",
                measurement_method="Evaluación FARFAN",
                verification_source="Sistema de evaluación municipal"
            )
        ],
        estimated_duration_days=180,
        responsible_entity="Secretaría de Educación Municipal",
        budget_range=(30_000_000, 70_000_000),
        ods_alignment=["ODS-4"]
    )


# Example usage
if __name__ == "__main__":
    print("=== SMART Recommendations Framework Demo ===\n")
    
    # Create example recommendation
    rec = create_example_recommendation()
    
    # Validate
    is_valid, errors = rec.validate()
    print(f"Recommendation valid: {is_valid}")
    if errors:
        print(f"Errors: {errors}")
    
    print(f"\nAHP Score: {rec.ahp_score}/10")
    print(f"Priority: {rec.priority.value}")
    
    # Convert to markdown
    print("\n" + rec.to_markdown())
    
    # Test prioritizer
    recommendations = [rec]
    prioritizer = RecommendationPrioritizer()
    roadmap = prioritizer.generate_implementation_roadmap(recommendations)
    print("\n" + roadmap)
