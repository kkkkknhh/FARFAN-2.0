#!/usr/bin/env python3
"""
Tests para Circuit Breaker
Valida comportamiento de estados, ventana deslizante, umbrales adaptativos,
timeouts y sincronización distribuida
"""

import time
import unittest
from unittest.mock import Mock, patch
from circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitBreakerError,
    OperationTimeoutError,
    HealthMetrics
)


class TestCircuitBreaker(unittest.TestCase):
    """Tests básicos del circuit breaker"""
    
    def setUp(self):
        """Setup antes de cada test"""
        self.cb = CircuitBreaker(
            failure_threshold=0.5,
            window_size_seconds=5,
            timeout_duration=2,
            half_open_max_requests=2,
            operation_timeout=1.0
        )
    
    def test_initial_state_closed(self):
        """Circuit breaker debe iniciar en estado CLOSED"""
        self.assertEqual(self.cb.get_state(), CircuitState.CLOSED)
    
    def test_successful_requests_stay_closed(self):
        """Requests exitosas mantienen el circuito CLOSED"""
        def success_func():
            return "ok"
        
        for _ in range(10):
            result = self.cb.call(success_func)
            self.assertEqual(result, "ok")
        
        self.assertEqual(self.cb.get_state(), CircuitState.CLOSED)
        
        metrics = self.cb.get_metrics()
        self.assertEqual(metrics.total_successes, 10)
        self.assertEqual(metrics.total_failures, 0)
    
    def test_failures_open_circuit(self):
        """Suficientes fallos deben abrir el circuito"""
        def fail_func():
            raise ValueError("test error")
        
        # Generar fallos hasta superar threshold (50%)
        for _ in range(5):
            try:
                self.cb.call(fail_func)
            except (ValueError, CircuitBreakerError):
                pass
        
        # Verificar que está OPEN
        self.assertEqual(self.cb.get_state(), CircuitState.OPEN)
        
        metrics = self.cb.get_metrics()
        self.assertGreater(metrics.total_failures, 0)
    
    def test_open_blocks_requests(self):
        """Estado OPEN debe bloquear todas las requests"""
        # Forzar a OPEN
        self.cb.force_open()
        
        def any_func():
            return "should not execute"
        
        with self.assertRaises(CircuitBreakerError):
            self.cb.call(any_func)
    
    def test_open_to_half_open_after_timeout(self):
        """Después del timeout, debe pasar de OPEN a HALF_OPEN"""
        self.cb.force_open()
        
        # Esperar el timeout
        time.sleep(self.cb.timeout_duration + 0.1)
        
        def success_func():
            return "ok"
        
        # Primera request debe transicionar a HALF_OPEN
        result = self.cb.call(success_func)
        self.assertEqual(result, "ok")
        self.assertEqual(self.cb.get_state(), CircuitState.HALF_OPEN)
    
    def test_half_open_limits_requests(self):
        """HALF_OPEN debe limitar número de requests de prueba"""
        # Crear CB con max_requests bajo y forzar OPEN
        cb = CircuitBreaker(
            failure_threshold=0.5,
            window_size_seconds=5,
            timeout_duration=1,
            half_open_max_requests=2,
            operation_timeout=1.0
        )
        cb.force_open()
        
        # Esperar timeout para pasar a HALF_OPEN
        time.sleep(1.1)
        
        def success_func():
            return "ok"
        
        # Primera request activa HALF_OPEN
        cb.call(success_func)
        
        # Segunda completa las pruebas y cierra circuito
        cb.call(success_func)
        
        # Ahora está CLOSED, no HALF_OPEN
        self.assertEqual(cb.get_state(), CircuitState.CLOSED)
    
    def test_half_open_to_closed_on_success(self):
        """Éxitos en HALF_OPEN deben cerrar el circuito"""
        self.cb._transition_to(CircuitState.HALF_OPEN)
        
        def success_func():
            return "ok"
        
        # Completar todas las requests de prueba exitosamente
        for _ in range(self.cb.half_open_max_requests):
            self.cb.call(success_func)
        
        # Debe haber transicionado a CLOSED
        self.assertEqual(self.cb.get_state(), CircuitState.CLOSED)
    
    def test_half_open_to_open_on_failure(self):
        """Fallos en HALF_OPEN deben reabrir el circuito"""
        self.cb._transition_to(CircuitState.HALF_OPEN)
        
        def fail_func():
            raise ValueError("test error")
        
        # Un solo fallo debe reabrir
        with self.assertRaises(ValueError):
            self.cb.call(fail_func)
        
        self.assertEqual(self.cb.get_state(), CircuitState.OPEN)


class TestSlidingWindow(unittest.TestCase):
    """Tests de la ventana deslizante"""
    
    def setUp(self):
        """Setup con ventana corta para tests rápidos"""
        self.cb = CircuitBreaker(
            failure_threshold=0.5,
            window_size_seconds=2,
            timeout_duration=1
        )
    
    def test_old_records_cleaned(self):
        """Registros antiguos deben ser eliminados de la ventana"""
        # Crear CB y generar mezcla de éxitos y fallos
        cb = CircuitBreaker(
            failure_threshold=0.5,
            window_size_seconds=2,
            timeout_duration=1
        )
        
        def success_func():
            return "ok"
        
        def fail_func():
            raise ValueError("error")
        
        # Mix inicial: 2 éxitos, 1 fallo = 33% failure rate (OK)
        cb.call(success_func)
        cb.call(success_func)
        try:
            cb.call(fail_func)
        except ValueError:
            pass
        
        # Verificar failure rate moderado
        initial_rate = cb._calculate_failure_rate()
        self.assertAlmostEqual(initial_rate, 0.33, places=1)
        
        # Esperar que expire la ventana
        time.sleep(cb.window_size_seconds + 0.5)
        
        # Limpiar ventana manualmente para resetear
        cb._clean_old_records()
        
        # Generar solo éxitos
        for _ in range(3):
            cb.call(success_func)
        
        # Failure rate debe ser 0 (solo cuenta ventana reciente)
        final_rate = cb._calculate_failure_rate()
        self.assertEqual(final_rate, 0.0)
    
    def test_failure_rate_calculation(self):
        """Cálculo de failure rate debe ser preciso"""
        def success_func():
            return "ok"
        
        def fail_func():
            raise ValueError("error")
        
        # 3 éxitos, 2 fallos = 40% failure rate
        for _ in range(3):
            self.cb.call(success_func)
        
        for _ in range(2):
            try:
                self.cb.call(fail_func)
            except ValueError:
                pass
        
        rate = self.cb._calculate_failure_rate()
        self.assertAlmostEqual(rate, 0.4, places=1)
    
    def test_window_stats(self):
        """Estadísticas de ventana deben ser correctas"""
        def slow_success():
            time.sleep(0.1)
            return "ok"
        
        def fast_fail():
            raise ValueError("error")
        
        # Mix de operaciones
        self.cb.call(slow_success)
        self.cb.call(slow_success)
        try:
            self.cb.call(fast_fail)
        except ValueError:
            pass
        
        stats = self.cb.get_window_stats()
        
        self.assertEqual(stats["window_size"], 3)
        self.assertEqual(stats["successes"], 2)
        self.assertEqual(stats["failures"], 1)
        self.assertAlmostEqual(stats["failure_rate"], 1/3, places=2)
        self.assertGreater(stats["avg_duration"], 0.0)


class TestAdaptiveThresholds(unittest.TestCase):
    """Tests de umbrales adaptativos según hora"""
    
    def test_peak_hours_higher_threshold(self):
        """Horas pico deben tener threshold más alto"""
        cb = CircuitBreaker(
            failure_threshold=0.5,
            peak_hours=[9, 10, 11],
            peak_multiplier=1.5,
            off_peak_multiplier=0.8
        )
        
        # Simular hora pico
        with patch('circuit_breaker.datetime') as mock_dt:
            mock_dt.now.return_value.hour = 10
            peak_threshold = cb._get_adaptive_threshold()
        
        # Simular hora valle
        with patch('circuit_breaker.datetime') as mock_dt:
            mock_dt.now.return_value.hour = 22
            off_peak_threshold = cb._get_adaptive_threshold()
        
        # Threshold pico debe ser mayor
        self.assertGreater(peak_threshold, off_peak_threshold)
        self.assertAlmostEqual(peak_threshold, 0.5 * 1.5, places=2)
        self.assertAlmostEqual(off_peak_threshold, 0.5 * 0.8, places=2)
    
    def test_adaptive_threshold_affects_opening(self):
        """Threshold adaptativo debe afectar cuándo se abre circuito"""
        cb = CircuitBreaker(
            failure_threshold=0.5,
            window_size_seconds=10,
            peak_hours=[10],
            peak_multiplier=2.0,  # Double threshold en pico
            off_peak_multiplier=0.5  # Half threshold fuera de pico
        )
        
        def fail_func():
            raise ValueError("error")
        
        def success_func():
            return "ok"
        
        # Simular hora pico: 60% failure rate no debe abrir (threshold=100%)
        with patch('circuit_breaker.datetime') as mock_dt:
            mock_dt.now.return_value.hour = 10
            
            # 3 fallos, 2 éxitos = 60% failure rate
            for _ in range(2):
                cb.call(success_func)
            for _ in range(3):
                try:
                    cb.call(fail_func)
                except ValueError:
                    pass
            
            # No debe abrir (threshold adaptativo = 100%)
            self.assertEqual(cb.get_state(), CircuitState.CLOSED)


class TestOperationTimeout(unittest.TestCase):
    """Tests de detección de timeouts"""
    
    def test_fast_operation_no_timeout(self):
        """Operaciones rápidas no deben lanzar timeout"""
        cb = CircuitBreaker(operation_timeout=1.0)
        
        def fast_func():
            time.sleep(0.1)
            return "ok"
        
        result = cb.call(fast_func)
        self.assertEqual(result, "ok")
    
    def test_slow_operation_timeout(self):
        """Operaciones lentas deben lanzar OperationTimeoutError"""
        cb = CircuitBreaker(operation_timeout=0.5)
        
        def slow_func():
            time.sleep(1.0)
            return "should not reach"
        
        with self.assertRaises(OperationTimeoutError):
            cb.call(slow_func)
    
    def test_timeout_counts_as_failure(self):
        """Timeouts deben contar como fallos"""
        cb = CircuitBreaker(
            operation_timeout=0.2,
            failure_threshold=0.5,
            window_size_seconds=10
        )
        
        def slow_func():
            time.sleep(0.5)
            return "ok"
        
        # Generar timeouts hasta que se abra
        for _ in range(5):
            try:
                cb.call(slow_func)
            except (OperationTimeoutError, CircuitBreakerError):
                pass
        
        # Debe abrir por alto failure rate
        self.assertEqual(cb.get_state(), CircuitState.OPEN)


class TestHealthMetrics(unittest.TestCase):
    """Tests de métricas de salud"""
    
    def test_metrics_track_requests(self):
        """Métricas deben trackear todas las requests"""
        cb = CircuitBreaker()
        
        def success():
            return "ok"
        
        def fail():
            raise ValueError("error")
        
        # 3 éxitos
        for _ in range(3):
            cb.call(success)
        
        # 2 fallos
        for _ in range(2):
            try:
                cb.call(fail)
            except ValueError:
                pass
        
        metrics = cb.get_metrics()
        
        self.assertEqual(metrics.total_requests, 5)
        self.assertEqual(metrics.total_successes, 3)
        self.assertEqual(metrics.total_failures, 2)
    
    def test_metrics_track_consecutive(self):
        """Métricas deben trackear éxitos/fallos consecutivos"""
        cb = CircuitBreaker(failure_threshold=1.0)  # No abrir
        
        def success():
            return "ok"
        
        def fail():
            raise ValueError("error")
        
        # 3 éxitos consecutivos
        for _ in range(3):
            cb.call(success)
        
        metrics = cb.get_metrics()
        self.assertEqual(metrics.consecutive_successes, 3)
        self.assertEqual(metrics.consecutive_failures, 0)
        
        # 2 fallos consecutivos
        for _ in range(2):
            try:
                cb.call(fail)
            except ValueError:
                pass
        
        metrics = cb.get_metrics()
        self.assertEqual(metrics.consecutive_successes, 0)
        self.assertEqual(metrics.consecutive_failures, 2)
    
    def test_metrics_track_timestamps(self):
        """Métricas deben trackear timestamps de eventos"""
        cb = CircuitBreaker()
        
        def success():
            return "ok"
        
        cb.call(success)
        
        metrics = cb.get_metrics()
        
        self.assertIsNotNone(metrics.last_success_time)
        self.assertIsNone(metrics.last_failure_time)
        self.assertGreater(metrics.last_state_change, 0)


class TestRedisIntegration(unittest.TestCase):
    """Tests de integración con Redis"""
    
    def test_sync_state_to_redis(self):
        """Estado debe sincronizarse a Redis"""
        mock_redis = Mock()
        
        cb = CircuitBreaker(
            redis_client=mock_redis,
            redis_key_prefix="test_cb"
        )
        
        # Forzar cambio de estado
        cb.force_open()
        
        # Verificar que se llamó setex
        mock_redis.setex.assert_called()
        
        # Verificar key correcta
        call_args = mock_redis.setex.call_args
        self.assertTrue(call_args[0][0].startswith("test_cb:"))
    
    def test_load_state_from_redis(self):
        """Estado debe cargarse desde Redis"""
        import json
        
        mock_redis = Mock()
        
        # Simular estado guardado en Redis
        saved_state = {
            "state": "open",
            "state_changed_at": time.time(),
            "half_open_requests": 0,
            "failure_rate": 0.6,
            "metrics": {
                "total_requests": 10,
                "total_failures": 6,
                "total_successes": 4,
                "current_failure_rate": 0.6,
                "state": "open",
                "last_state_change": time.time(),
                "consecutive_successes": 0,
                "consecutive_failures": 3
            }
        }
        
        mock_redis.get.return_value = json.dumps(saved_state)
        
        cb = CircuitBreaker(
            redis_client=mock_redis,
            redis_key_prefix="test_cb"
        )
        
        # Cargar estado
        loaded = cb._load_state_from_redis()
        
        self.assertTrue(loaded)
        self.assertEqual(cb.get_state(), CircuitState.OPEN)
        
        metrics = cb.get_metrics()
        self.assertEqual(metrics.total_requests, 10)
        self.assertEqual(metrics.total_failures, 6)
    
    def test_distributed_workers_share_state(self):
        """Workers distribuidos deben compartir estado vía Redis"""
        
        mock_redis = Mock()
        
        # Worker 1 abre el circuito
        cb1 = CircuitBreaker(
            redis_client=mock_redis,
            redis_key_prefix="shared_cb",
            failure_threshold=0.5
        )
        
        def fail():
            raise ValueError("error")
        
        # Generar fallos hasta abrir
        for _ in range(5):
            try:
                cb1.call(fail)
            except (ValueError, CircuitBreakerError):
                pass
        
        # Obtener estado serializado (última llamada a setex)
        self.assertTrue(mock_redis.setex.called)
        serialized_state = mock_redis.setex.call_args[0][2]
        
        # Worker 2 carga el estado compartido
        # Mock debe retornar bytes para simular Redis real
        mock_redis_2 = Mock()
        if isinstance(serialized_state, str):
            mock_redis_2.get.return_value = serialized_state.encode()
        else:
            mock_redis_2.get.return_value = serialized_state
        
        cb2 = CircuitBreaker(
            redis_client=mock_redis_2,
            redis_key_prefix="shared_cb"
        )
        
        cb2._load_state_from_redis()
        
        # Debe tener el mismo estado OPEN
        self.assertEqual(cb2.get_state(), CircuitState.OPEN)


class TestCircuitBreakerReset(unittest.TestCase):
    """Tests de reset del circuit breaker"""
    
    def test_reset_clears_state(self):
        """Reset debe limpiar estado y métricas"""
        cb = CircuitBreaker()
        
        # Generar actividad
        cb.force_open()
        
        def fail():
            raise ValueError("error")
        
        for _ in range(5):
            try:
                cb.call(fail)
            except (ValueError, CircuitBreakerError):
                pass
        
        # Reset
        cb.reset()
        
        # Verificar estado limpio
        self.assertEqual(cb.get_state(), CircuitState.CLOSED)
        self.assertEqual(cb._calculate_failure_rate(), 0.0)
        
        metrics = cb.get_metrics()
        self.assertEqual(metrics.consecutive_failures, 0)
        self.assertEqual(metrics.consecutive_successes, 0)
    
    def test_force_open_and_close(self):
        """Force methods deben funcionar correctamente"""
        cb = CircuitBreaker()
        
        cb.force_open()
        self.assertEqual(cb.get_state(), CircuitState.OPEN)
        
        cb.force_close()
        self.assertEqual(cb.get_state(), CircuitState.CLOSED)


def run_tests():
    """Ejecuta todos los tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestCircuitBreaker))
    suite.addTests(loader.loadTestsFromTestCase(TestSlidingWindow))
    suite.addTests(loader.loadTestsFromTestCase(TestAdaptiveThresholds))
    suite.addTests(loader.loadTestsFromTestCase(TestOperationTimeout))
    suite.addTests(loader.loadTestsFromTestCase(TestHealthMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestRedisIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestCircuitBreakerReset))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)
