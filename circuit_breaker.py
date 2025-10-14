#!/usr/bin/env python3
"""
Circuit Breaker Pattern Implementation
Protección contra fallos en cascada para operaciones distribuidas

Implementa el patrón de tres estados (CLOSED/OPEN/HALF_OPEN) con:
- Ventana deslizante para cálculo de tasa de fallos
- Umbrales adaptativos según hora del día (pico/valle)
- Detección de timeouts por operación
- Métricas de salud distribuidas
- Sincronización opcional vía Redis para workers distribuidos
"""

import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from collections import deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Estados del circuit breaker"""
    CLOSED = "closed"       # Permite requests, registra fallos
    OPEN = "open"           # Bloquea requests, espera timeout
    HALF_OPEN = "half_open" # Permite requests limitados de prueba


class CircuitBreakerError(Exception):
    """Excepción lanzada cuando el circuit breaker está OPEN"""
    pass


class OperationTimeoutError(Exception):
    """Excepción lanzada cuando una operación excede su timeout"""
    pass


@dataclass
class RequestRecord:
    """Registro de una request individual en la ventana deslizante"""
    timestamp: float
    success: bool
    duration: float


@dataclass
class HealthMetrics:
    """Métricas de salud del circuit breaker"""
    total_requests: int = 0
    total_failures: int = 0
    total_successes: int = 0
    current_failure_rate: float = 0.0
    state: str = "closed"
    last_state_change: float = 0.0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    consecutive_successes: int = 0
    consecutive_failures: int = 0


class CircuitBreaker:
    """
    Circuit Breaker con ventana deslizante y umbrales adaptativos
    
    Características:
    - Ventana deslizante temporal para cálculo de failure rate
    - Tres estados: CLOSED → OPEN → HALF_OPEN → CLOSED
    - Umbrales adaptativos según hora del día (pico vs valle)
    - Detección de timeouts por operación
    - Métricas de salud detalladas
    - Sincronización Redis opcional para sistemas distribuidos
    
    Parámetros:
    - failure_threshold: Tasa de fallo base para abrir circuito (0.0-1.0)
    - window_size_seconds: Tamaño de ventana deslizante en segundos
    - timeout_duration: Tiempo en OPEN antes de pasar a HALF_OPEN
    - half_open_max_requests: Máximo de requests de prueba en HALF_OPEN
    - operation_timeout: Timeout por operación individual en segundos
    - peak_hours: Lista de horas consideradas "pico" (0-23)
    - peak_multiplier: Multiplicador de threshold en horas pico
    - off_peak_multiplier: Multiplicador de threshold en horas valle
    - redis_client: Cliente Redis opcional para estado distribuido
    - redis_key_prefix: Prefijo para keys en Redis
    """
    
    def __init__(
        self,
        failure_threshold: float = 0.5,
        window_size_seconds: int = 60,
        timeout_duration: int = 30,
        half_open_max_requests: int = 3,
        operation_timeout: float = 10.0,
        peak_hours: Optional[List[int]] = None,
        peak_multiplier: float = 1.5,
        off_peak_multiplier: float = 0.8,
        redis_client: Optional[Any] = None,
        redis_key_prefix: str = "circuit_breaker"
    ):
        self.base_failure_threshold = failure_threshold
        self.window_size_seconds = window_size_seconds
        self.timeout_duration = timeout_duration
        self.half_open_max_requests = half_open_max_requests
        self.operation_timeout = operation_timeout
        
        # Umbrales adaptativos
        self.peak_hours = peak_hours or [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
        self.peak_multiplier = peak_multiplier
        self.off_peak_multiplier = off_peak_multiplier
        
        # Redis para estado distribuido
        self.redis_client = redis_client
        self.redis_key_prefix = redis_key_prefix
        
        # Estado interno
        self._state = CircuitState.CLOSED
        self._state_changed_at = time.time()
        self._half_open_requests = 0
        
        # Ventana deslizante de requests
        self._request_window: deque = deque()
        
        # Métricas
        self._metrics = HealthMetrics(
            state=self._state.value,
            last_state_change=self._state_changed_at
        )
        
        logger.info(
            f"CircuitBreaker initialized: threshold={failure_threshold}, "
            f"window={window_size_seconds}s, timeout={timeout_duration}s, "
            f"redis={'enabled' if redis_client else 'disabled'}"
        )
    
    def _get_adaptive_threshold(self) -> float:
        """
        Calcula threshold adaptativo según hora del día
        
        Returns:
            Threshold ajustado para la hora actual
        """
        current_hour = datetime.now().hour
        
        if current_hour in self.peak_hours:
            # Horas pico: mayor tolerancia a fallos
            adjusted = self.base_failure_threshold * self.peak_multiplier
        else:
            # Horas valle: menor tolerancia
            adjusted = self.base_failure_threshold * self.off_peak_multiplier
        
        # Clamp entre 0.0 y 1.0
        return max(0.0, min(1.0, adjusted))
    
    def _clean_old_records(self):
        """Elimina registros fuera de la ventana deslizante"""
        cutoff_time = time.time() - self.window_size_seconds
        
        while self._request_window and self._request_window[0].timestamp < cutoff_time:
            self._request_window.popleft()
    
    def _calculate_failure_rate(self) -> float:
        """
        Calcula tasa de fallo actual en la ventana deslizante
        
        Returns:
            Tasa de fallo (0.0-1.0) o 0.0 si no hay datos
        """
        self._clean_old_records()
        
        if not self._request_window:
            return 0.0
        
        failures = sum(1 for r in self._request_window if not r.success)
        total = len(self._request_window)
        
        return failures / total if total > 0 else 0.0
    
    def _transition_to(self, new_state: CircuitState):
        """
        Transiciona a un nuevo estado
        
        Args:
            new_state: Nuevo estado del circuit breaker
        """
        old_state = self._state
        self._state = new_state
        self._state_changed_at = time.time()
        
        # Reset contador de requests en HALF_OPEN
        if new_state == CircuitState.HALF_OPEN:
            self._half_open_requests = 0
        
        # Actualizar métricas
        self._metrics.state = new_state.value
        self._metrics.last_state_change = self._state_changed_at
        
        logger.info(f"Circuit breaker transitioned: {old_state.value} → {new_state.value}")
        
        # Sincronizar con Redis si está habilitado
        if self.redis_client:
            self._sync_state_to_redis()
    
    def _sync_state_to_redis(self):
        """Sincroniza estado actual a Redis"""
        if not self.redis_client:
            return
        
        try:
            key = f"{self.redis_key_prefix}:state"
            state_data = {
                "state": self._state.value,
                "state_changed_at": self._state_changed_at,
                "half_open_requests": self._half_open_requests,
                "failure_rate": self._calculate_failure_rate(),
                "metrics": asdict(self._metrics)
            }
            
            # Serializar y guardar con TTL
            import json
            self.redis_client.setex(
                key,
                self.timeout_duration * 2,  # TTL doble del timeout
                json.dumps(state_data)
            )
            
            logger.debug(f"State synced to Redis: {key}")
        except Exception as e:
            logger.error(f"Failed to sync state to Redis: {e}")
    
    def _load_state_from_redis(self) -> bool:
        """
        Carga estado desde Redis
        
        Returns:
            True si se cargó estado válido, False si no existe o falló
        """
        if not self.redis_client:
            return False
        
        try:
            key = f"{self.redis_key_prefix}:state"
            data = self.redis_client.get(key)
            
            if not data:
                return False
            
            state_data = json.loads(data)
            
            # Restaurar estado
            self._state = CircuitState(state_data["state"])
            self._state_changed_at = state_data["state_changed_at"]
            self._half_open_requests = state_data["half_open_requests"]
            
            # Restaurar métricas
            metrics_data = state_data.get("metrics", {})
            for key, value in metrics_data.items():
                setattr(self._metrics, key, value)
            
            logger.info(f"State loaded from Redis: {self._state.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load state from Redis: {e}")
            return False
    
    def _record_request(self, success: bool, duration: float):
        """
        Registra resultado de una request en la ventana deslizante
        
        Args:
            success: Si la request fue exitosa
            duration: Duración de la request en segundos
        """
        record = RequestRecord(
            timestamp=time.time(),
            success=success,
            duration=duration
        )
        self._request_window.append(record)
        
        # Actualizar métricas
        self._metrics.total_requests += 1
        if success:
            self._metrics.total_successes += 1
            self._metrics.last_success_time = record.timestamp
            self._metrics.consecutive_successes += 1
            self._metrics.consecutive_failures = 0
        else:
            self._metrics.total_failures += 1
            self._metrics.last_failure_time = record.timestamp
            self._metrics.consecutive_failures += 1
            self._metrics.consecutive_successes = 0
        
        self._metrics.current_failure_rate = self._calculate_failure_rate()
    
    def _should_attempt_reset(self) -> bool:
        """
        Verifica si debe intentarse reset de OPEN a HALF_OPEN
        
        Returns:
            True si pasó el timeout y debe intentarse reset
        """
        if self._state != CircuitState.OPEN:
            return False
        
        elapsed = time.time() - self._state_changed_at
        return elapsed >= self.timeout_duration
    
    def detect_timeout(self, start_time: float, operation_name: str = "operation") -> bool:
        """
        Detecta si una operación excedió su timeout configurado
        
        Args:
            start_time: Timestamp de inicio de la operación
            operation_name: Nombre descriptivo de la operación
            
        Returns:
            True si la operación está dentro del límite de tiempo
            
        Raises:
            OperationTimeoutError: Si la operación excedió el timeout
        """
        elapsed = time.time() - start_time
        
        if elapsed > self.operation_timeout:
            logger.warning(
                f"Operation timeout detected: {operation_name} "
                f"took {elapsed:.2f}s (limit: {self.operation_timeout}s)"
            )
            raise OperationTimeoutError(
                f"{operation_name} exceeded timeout of {self.operation_timeout}s"
            )
        
        return True
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Ejecuta una función protegida por el circuit breaker
        
        Args:
            func: Función a ejecutar
            *args: Argumentos posicionales
            **kwargs: Argumentos con nombre
            
        Returns:
            Resultado de la función
            
        Raises:
            CircuitBreakerError: Si el circuito está OPEN
            OperationTimeoutError: Si la operación excede el timeout
        """
        # Cargar estado de Redis si está disponible
        if self.redis_client and self._metrics.total_requests == 0:
            self._load_state_from_redis()
        
        # Verificar si debe transicionar de OPEN a HALF_OPEN
        if self._should_attempt_reset():
            self._transition_to(CircuitState.HALF_OPEN)
        
        # Comportamiento según estado
        if self._state == CircuitState.OPEN:
            raise CircuitBreakerError(
                f"Circuit breaker is OPEN (failure rate: "
                f"{self._metrics.current_failure_rate:.2%})"
            )
        
        if self._state == CircuitState.HALF_OPEN:
            # Limitar requests en HALF_OPEN
            if self._half_open_requests >= self.half_open_max_requests:
                raise CircuitBreakerError(
                    f"Circuit breaker is HALF_OPEN and max test requests "
                    f"({self.half_open_max_requests}) reached"
                )
            self._half_open_requests += 1
        
        # Ejecutar operación con timeout monitoring
        start_time = time.time()
        success = False
        result = None
        
        try:
            result = func(*args, **kwargs)
            
            # Verificar timeout
            self.detect_timeout(start_time, func.__name__)
            
            success = True
            duration = time.time() - start_time
            
            # Registrar éxito
            self._record_request(success=True, duration=duration)
            
            # Lógica de transición de estado
            if self._state == CircuitState.HALF_OPEN:
                # Si completamos todas las pruebas exitosamente, cerrar circuito
                if self._half_open_requests >= self.half_open_max_requests:
                    self._transition_to(CircuitState.CLOSED)
                    logger.info("Circuit breaker recovered: HALF_OPEN → CLOSED")
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Registrar fallo
            self._record_request(success=False, duration=duration)
            
            # Calcular failure rate con threshold adaptativo
            failure_rate = self._calculate_failure_rate()
            adaptive_threshold = self._get_adaptive_threshold()
            
            logger.warning(
                f"Request failed: {func.__name__} - {type(e).__name__}: {e} "
                f"(failure_rate: {failure_rate:.2%}, threshold: {adaptive_threshold:.2%})"
            )
            
            # Lógica de transición de estado
            if self._state == CircuitState.HALF_OPEN:
                # Cualquier fallo en HALF_OPEN vuelve a OPEN
                self._transition_to(CircuitState.OPEN)
                logger.warning("Circuit breaker reopened: HALF_OPEN → OPEN")
            
            elif self._state == CircuitState.CLOSED:
                # Abrir si se excede threshold adaptativo
                if failure_rate >= adaptive_threshold:
                    self._transition_to(CircuitState.OPEN)
                    logger.warning(
                        f"Circuit breaker opened: failure rate {failure_rate:.2%} "
                        f">= threshold {adaptive_threshold:.2%}"
                    )
            
            raise
    
    def get_state(self) -> CircuitState:
        """
        Obtiene estado actual del circuit breaker
        
        Returns:
            Estado actual
        """
        # Verificar si debe transicionar
        if self._should_attempt_reset():
            self._transition_to(CircuitState.HALF_OPEN)
        
        return self._state
    
    def get_metrics(self) -> HealthMetrics:
        """
        Obtiene métricas de salud actuales
        
        Returns:
            Objeto con métricas detalladas
        """
        # Actualizar failure rate actual
        self._metrics.current_failure_rate = self._calculate_failure_rate()
        self._metrics.state = self._state.value
        
        return self._metrics
    
    def reset(self):
        """Resetea el circuit breaker a estado inicial CLOSED"""
        self._state = CircuitState.CLOSED
        self._state_changed_at = time.time()
        self._half_open_requests = 0
        self._request_window.clear()
        
        # Reset métricas pero mantener contadores históricos
        self._metrics.state = CircuitState.CLOSED.value
        self._metrics.last_state_change = self._state_changed_at
        self._metrics.current_failure_rate = 0.0
        self._metrics.consecutive_successes = 0
        self._metrics.consecutive_failures = 0
        
        logger.info("Circuit breaker reset to CLOSED")
        
        # Sincronizar con Redis
        if self.redis_client:
            self._sync_state_to_redis()
    
    def force_open(self):
        """Fuerza el circuit breaker a estado OPEN"""
        self._transition_to(CircuitState.OPEN)
        logger.warning("Circuit breaker forced to OPEN state")
    
    def force_close(self):
        """Fuerza el circuit breaker a estado CLOSED"""
        self._transition_to(CircuitState.CLOSED)
        logger.info("Circuit breaker forced to CLOSED state")
    
    def get_window_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas detalladas de la ventana deslizante
        
        Returns:
            Diccionario con estadísticas de la ventana actual
        """
        self._clean_old_records()
        
        if not self._request_window:
            return {
                "window_size": 0,
                "successes": 0,
                "failures": 0,
                "failure_rate": 0.0,
                "avg_duration": 0.0,
                "min_duration": 0.0,
                "max_duration": 0.0
            }
        
        successes = sum(1 for r in self._request_window if r.success)
        failures = len(self._request_window) - successes
        durations = [r.duration for r in self._request_window]
        
        return {
            "window_size": len(self._request_window),
            "successes": successes,
            "failures": failures,
            "failure_rate": failures / len(self._request_window),
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "adaptive_threshold": self._get_adaptive_threshold()
        }
