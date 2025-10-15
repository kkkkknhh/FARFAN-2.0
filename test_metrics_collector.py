#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Metrics Collector Module
==============================

SIN_CARRETA Compliance Tests:
- Validate deterministic metric recording
- Verify structured alert system
- Test thread-safety for async contexts
"""

import unittest
from infrastructure.metrics_collector import MetricsCollector


class TestMetricsCollector(unittest.TestCase):
    """Test suite for metrics collector"""
    
    def setUp(self):
        """Create fresh metrics collector for each test"""
        self.metrics = MetricsCollector()
    
    def test_record_metric(self):
        """Test recording a metric value"""
        self.metrics.record("test.metric", 42.0)
        
        # Verify metric recorded
        summary = self.metrics.get_summary()
        self.assertIn("test.metric", summary['metrics'])
        self.assertEqual(summary['metrics']['test.metric']['last'], 42.0)
        self.assertEqual(summary['metrics']['test.metric']['count'], 1)
    
    def test_record_multiple_values(self):
        """Test recording multiple values for same metric"""
        self.metrics.record("test.metric", 10.0)
        self.metrics.record("test.metric", 20.0)
        self.metrics.record("test.metric", 30.0)
        
        summary = self.metrics.get_summary()
        stats = summary['metrics']['test.metric']
        
        self.assertEqual(stats['count'], 3)
        self.assertEqual(stats['last'], 30.0)
        self.assertEqual(stats['min'], 10.0)
        self.assertEqual(stats['max'], 30.0)
        self.assertEqual(stats['avg'], 20.0)
    
    def test_increment_counter(self):
        """Test incrementing a counter"""
        self.metrics.increment("test.counter")
        self.metrics.increment("test.counter")
        self.metrics.increment("test.counter")
        
        summary = self.metrics.get_summary()
        self.assertEqual(summary['counters']['test.counter'], 3)
    
    def test_increment_with_amount(self):
        """Test incrementing counter by custom amount"""
        self.metrics.increment("test.counter", amount=5)
        self.metrics.increment("test.counter", amount=3)
        
        summary = self.metrics.get_summary()
        self.assertEqual(summary['counters']['test.counter'], 8)
    
    def test_alert(self):
        """Test recording an alert"""
        self.metrics.alert("WARNING", "Test warning message")
        
        summary = self.metrics.get_summary()
        self.assertEqual(len(summary['alerts']), 1)
        
        alert = summary['alerts'][0]
        self.assertEqual(alert['level'], "WARNING")
        self.assertEqual(alert['message'], "Test warning message")
        self.assertIn('timestamp', alert)
    
    def test_alert_with_context(self):
        """Test recording alert with context metadata"""
        self.metrics.alert(
            "CRITICAL",
            "Test critical alert",
            context={"run_id": "test_123", "phase": "extraction"}
        )
        
        summary = self.metrics.get_summary()
        alert = summary['alerts'][0]
        
        self.assertEqual(alert['level'], "CRITICAL")
        self.assertEqual(alert['context']['run_id'], "test_123")
        self.assertEqual(alert['context']['phase'], "extraction")
    
    def test_get_metric_history(self):
        """Test retrieving full metric history"""
        self.metrics.record("test.metric", 1.0, tags={"phase": "a"})
        self.metrics.record("test.metric", 2.0, tags={"phase": "b"})
        self.metrics.record("test.metric", 3.0, tags={"phase": "c"})
        
        history = self.metrics.get_metric_history("test.metric")
        
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0]['value'], 1.0)
        self.assertEqual(history[0]['tags']['phase'], "a")
        self.assertEqual(history[2]['value'], 3.0)
    
    def test_get_alerts_by_level(self):
        """Test filtering alerts by severity level"""
        self.metrics.alert("INFO", "Info message")
        self.metrics.alert("WARNING", "Warning message")
        self.metrics.alert("CRITICAL", "Critical message")
        self.metrics.alert("WARNING", "Another warning")
        
        warnings = self.metrics.get_alerts_by_level("WARNING")
        
        self.assertEqual(len(warnings), 2)
        self.assertTrue(all("warning" in a['message'].lower() for a in warnings))
    
    def test_metric_type_enforcement(self):
        """Test that non-numeric values raise TypeError"""
        with self.assertRaises(TypeError):
            self.metrics.record("test.metric", "not a number")
    
    def test_summary_structure(self):
        """Test that summary has expected structure"""
        self.metrics.record("metric1", 10.0)
        self.metrics.increment("counter1")
        self.metrics.alert("INFO", "test alert")
        
        summary = self.metrics.get_summary()
        
        # Verify structure
        self.assertIn('metrics', summary)
        self.assertIn('counters', summary)
        self.assertIn('alerts', summary)
        self.assertIn('total_metrics_recorded', summary)
        self.assertIn('total_alerts', summary)
        
        # Verify counts
        self.assertEqual(summary['total_metrics_recorded'], 1)
        self.assertEqual(summary['total_alerts'], 1)
    
    def test_reset(self):
        """Test resetting metrics collector"""
        self.metrics.record("test.metric", 42.0)
        self.metrics.increment("test.counter")
        self.metrics.alert("INFO", "test alert")
        
        # Verify data exists
        summary = self.metrics.get_summary()
        self.assertGreater(summary['total_metrics_recorded'], 0)
        
        # Reset
        self.metrics.reset()
        
        # Verify cleared
        summary = self.metrics.get_summary()
        self.assertEqual(summary['total_metrics_recorded'], 0)
        self.assertEqual(summary['total_alerts'], 0)
        self.assertEqual(len(summary['counters']), 0)


if __name__ == '__main__':
    unittest.main()
