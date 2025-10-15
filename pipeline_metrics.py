#!/usr/bin/env python3
"""
Pipeline Metrics - Sistema de recolección de métricas de ejecución

Proporciona:
- Métricas por etapa (éxito/fallo, tiempo de ejecución)
- Contadores de mitigación por categoría y severidad
- Tracking de transiciones de circuit breaker
- Alertas para riesgos críticos/altos
- Traza completa de ejecución para post-mortem
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger("pipeline_metrics")


class AlertLevel(Enum):
    """Niveles de alerta"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class StageMetrics:
    """Métricas de una etapa específica"""
    stage_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = False
    execution_time_ms: float = 0.0
    error_message: Optional[str] = None
    risk_assessments: List[str] = field(default_factory=list)
    mitigation_attempts: List[str] = field(default_factory=list)
    circuit_breaker_state: str = "CLOSED"


@dataclass
class Alert:
    """Alerta generada durante ejecución"""
    level: AlertLevel
    message: str
    context: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutionTrace:
    """Traza completa de ejecución del pipeline"""
    policy_code: str
    start_time: datetime
    end_time: Optional[datetime] = None
    success: bool = False
    stages: List[StageMetrics] = field(default_factory=list)
    alerts: List[Alert] = field(default_factory=list)
    mitigation_stats: Dict[str, Any] = field(default_factory=dict)
    circuit_breaker_stats: Dict[str, Any] = field(default_factory=dict)
    total_execution_time_ms: float = 0.0


class PipelineMetrics:
    """
    Sistema de recolección y análisis de métricas del pipeline
    
    Tracks:
    - Métricas por etapa (éxito, fallos, tiempos)
    - Invocaciones de mitigación (por categoría y severidad)
    - Transiciones de circuit breaker
    - Alertas de riesgos críticos/altos
    - Traza completa para análisis post-mortem
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Current execution
        self.current_trace: Optional[ExecutionTrace] = None
        self.current_stage: Optional[StageMetrics] = None
        
        # Aggregated metrics across all executions
        self.stage_success_counts: Dict[str, int] = {}
        self.stage_failure_counts: Dict[str, int] = {}
        self.stage_total_time_ms: Dict[str, float] = {}
        
        self.mitigation_by_category: Dict[str, int] = {}
        self.mitigation_by_severity: Dict[str, int] = {}
        
        self.alerts_by_level: Dict[str, int] = {}
        
        logger.info(f"PipelineMetrics inicializado: {self.output_dir}")
    
    def start_execution(self, policy_code: str):
        """Inicia tracking de una nueva ejecución"""
        self.current_trace = ExecutionTrace(
            policy_code=policy_code,
            start_time=datetime.now()
        )
        logger.info(f"📊 Iniciando tracking de métricas para: {policy_code}")
    
    def start_stage(self, stage_name: str):
        """Inicia tracking de una etapa"""
        if not self.current_trace:
            raise RuntimeError("No hay ejecución activa")
        
        self.current_stage = StageMetrics(
            stage_name=stage_name,
            start_time=datetime.now()
        )
        logger.debug(f"Stage started: {stage_name}")
    
    def end_stage(self, success: bool, error_message: Optional[str] = None):
        """Finaliza tracking de una etapa"""
        if not self.current_stage or not self.current_trace:
            raise RuntimeError("No hay etapa activa")
        
        self.current_stage.end_time = datetime.now()
        self.current_stage.success = success
        self.current_stage.error_message = error_message
        
        # Calculate execution time
        if self.current_stage.start_time and self.current_stage.end_time:
            delta = self.current_stage.end_time - self.current_stage.start_time
            self.current_stage.execution_time_ms = delta.total_seconds() * 1000
        
        # Update aggregated metrics
        stage_name = self.current_stage.stage_name
        if success:
            self.stage_success_counts[stage_name] = self.stage_success_counts.get(stage_name, 0) + 1
        else:
            self.stage_failure_counts[stage_name] = self.stage_failure_counts.get(stage_name, 0) + 1
        
        self.stage_total_time_ms[stage_name] = (
            self.stage_total_time_ms.get(stage_name, 0.0) + self.current_stage.execution_time_ms
        )
        
        # Add to trace
        self.current_trace.stages.append(self.current_stage)
        
        status = "✓" if success else "✗"
        logger.info(f"{status} Stage completed: {stage_name} ({self.current_stage.execution_time_ms:.1f}ms)")
        
        self.current_stage = None
    
    def record_risk_assessment(self, risk_id: str):
        """Registra evaluación de riesgo"""
        if self.current_stage:
            self.current_stage.risk_assessments.append(risk_id)
    
    def record_mitigation(self, risk_id: str, category: str, severity: str):
        """Registra intento de mitigación"""
        if self.current_stage:
            self.current_stage.mitigation_attempts.append(risk_id)
        
        # Update counts
        self.mitigation_by_category[category] = self.mitigation_by_category.get(category, 0) + 1
        self.mitigation_by_severity[severity] = self.mitigation_by_severity.get(severity, 0) + 1
        
        logger.debug(f"Mitigation recorded: {risk_id} ({category}/{severity})")
    
    def record_circuit_breaker_state(self, state: str):
        """Registra estado del circuit breaker"""
        if self.current_stage:
            self.current_stage.circuit_breaker_state = state
    
    def emit_alert(self, level: AlertLevel, message: str, context: Dict[str, Any] = None):
        """Emite una alerta"""
        if not self.current_trace:
            raise RuntimeError("No hay ejecución activa")
        
        alert = Alert(
            level=level,
            message=message,
            context=context or {}
        )
        
        self.current_trace.alerts.append(alert)
        self.alerts_by_level[level.value] = self.alerts_by_level.get(level.value, 0) + 1
        
        # Log based on level
        log_message = f"🚨 {level.value}: {message}"
        if level == AlertLevel.CRITICAL:
            logger.critical(log_message)
        elif level == AlertLevel.ERROR:
            logger.error(log_message)
        elif level == AlertLevel.WARNING:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def end_execution(self, success: bool):
        """Finaliza tracking de ejecución"""
        if not self.current_trace:
            raise RuntimeError("No hay ejecución activa")
        
        self.current_trace.end_time = datetime.now()
        self.current_trace.success = success
        
        # Calculate total time
        if self.current_trace.start_time and self.current_trace.end_time:
            delta = self.current_trace.end_time - self.current_trace.start_time
            self.current_trace.total_execution_time_ms = delta.total_seconds() * 1000
        
        logger.info(f"📊 Ejecución finalizada: {self.current_trace.policy_code} ({self.current_trace.total_execution_time_ms:.1f}ms)")
    
    def export_trace(self, risk_registry=None, circuit_breaker_registry=None) -> Path:
        """
        Exporta traza completa de ejecución
        
        Args:
            risk_registry: Registry de riesgos (opcional)
            circuit_breaker_registry: Registry de circuit breakers (opcional)
        
        Returns:
            Path al archivo exportado
        """
        if not self.current_trace:
            raise RuntimeError("No hay ejecución activa")
        
        # Collect stats from registries
        if risk_registry:
            self.current_trace.mitigation_stats = risk_registry.get_mitigation_stats()
        
        if circuit_breaker_registry:
            self.current_trace.circuit_breaker_stats = circuit_breaker_registry.get_all_stats()
        
        # Export to JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"execution_trace_{self.current_trace.policy_code}_{timestamp}.json"
        output_path = self.output_dir / filename
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self._trace_to_dict(self.current_trace), f, indent=2, ensure_ascii=False)
            
            logger.info(f"✓ Traza exportada: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exportando traza: {e}")
            raise
    
    def get_stage_success_rates(self) -> Dict[str, float]:
        """Calcula tasas de éxito por etapa"""
        rates = {}
        for stage_name in set(list(self.stage_success_counts.keys()) + list(self.stage_failure_counts.keys())):
            successes = self.stage_success_counts.get(stage_name, 0)
            failures = self.stage_failure_counts.get(stage_name, 0)
            total = successes + failures
            
            if total > 0:
                rates[stage_name] = successes / total
            else:
                rates[stage_name] = 0.0
        
        return rates
    
    def get_average_stage_times(self) -> Dict[str, float]:
        """Calcula tiempos promedio por etapa"""
        avg_times = {}
        for stage_name in self.stage_total_time_ms.keys():
            total_time = self.stage_total_time_ms[stage_name]
            total_runs = self.stage_success_counts.get(stage_name, 0) + self.stage_failure_counts.get(stage_name, 0)
            
            if total_runs > 0:
                avg_times[stage_name] = total_time / total_runs
            else:
                avg_times[stage_name] = 0.0
        
        return avg_times
    
    def print_summary(self):
        """Imprime resumen de métricas"""
        if not self.current_trace:
            logger.warning("No hay traza actual para imprimir")
            return
        
        print("\n" + "="*80)
        print(f"RESUMEN DE MÉTRICAS - {self.current_trace.policy_code}")
        print("="*80)
        
        # Execution summary
        print(f"\n⏱️  Tiempo total: {self.current_trace.total_execution_time_ms:.1f}ms")
        print(f"✓  Etapas completadas: {len([s for s in self.current_trace.stages if s.success])}/{len(self.current_trace.stages)}")
        print(f"🚨 Alertas: {len(self.current_trace.alerts)}")
        
        # Stage breakdown
        print("\n📋 Etapas:")
        for stage in self.current_trace.stages:
            status = "✓" if stage.success else "✗"
            print(f"  {status} {stage.stage_name}: {stage.execution_time_ms:.1f}ms")
            if stage.risk_assessments:
                print(f"    Riesgos evaluados: {len(stage.risk_assessments)}")
            if stage.mitigation_attempts:
                print(f"    Mitigaciones: {len(stage.mitigation_attempts)}")
        
        # Mitigation stats
        if self.mitigation_by_severity:
            print("\n🛡️  Mitigaciones por severidad:")
            for severity, count in sorted(self.mitigation_by_severity.items()):
                print(f"  {severity}: {count}")
        
        # Alerts
        if self.current_trace.alerts:
            print("\n🚨 Alertas:")
            for alert in self.current_trace.alerts[-5:]:  # Last 5
                print(f"  [{alert.level.value}] {alert.message}")
        
        print("="*80 + "\n")
    
    def _trace_to_dict(self, trace: ExecutionTrace) -> dict:
        """Convierte traza a diccionario JSON-serializable"""
        return {
            'policy_code': trace.policy_code,
            'start_time': trace.start_time.isoformat(),
            'end_time': trace.end_time.isoformat() if trace.end_time else None,
            'success': trace.success,
            'total_execution_time_ms': trace.total_execution_time_ms,
            'stages': [
                {
                    'stage_name': s.stage_name,
                    'start_time': s.start_time.isoformat(),
                    'end_time': s.end_time.isoformat() if s.end_time else None,
                    'success': s.success,
                    'execution_time_ms': s.execution_time_ms,
                    'error_message': s.error_message,
                    'risk_assessments': s.risk_assessments,
                    'mitigation_attempts': s.mitigation_attempts,
                    'circuit_breaker_state': s.circuit_breaker_state
                }
                for s in trace.stages
            ],
            'alerts': [
                {
                    'level': a.level.value,
                    'message': a.message,
                    'context': a.context,
                    'timestamp': a.timestamp.isoformat()
                }
                for a in trace.alerts
            ],
            'mitigation_stats': trace.mitigation_stats,
            'circuit_breaker_stats': trace.circuit_breaker_stats,
            'aggregated_metrics': {
                'stage_success_rates': self.get_stage_success_rates(),
                'average_stage_times_ms': self.get_average_stage_times(),
                'mitigation_by_category': self.mitigation_by_category,
                'mitigation_by_severity': self.mitigation_by_severity,
                'alerts_by_level': self.alerts_by_level
            }
        }
