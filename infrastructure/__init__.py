#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Infrastructure Module for FARFAN 2.0
====================================

Provides dependency injection, configuration management, and robust wiring
for all CDAF components.
"""

from .di_container import (
    DeviceConfig,
    DIContainer,
    IBayesianEngine,
    ICausalBuilder,
    IExtractor,
    configure_container,
)

__all__ = [
    "DIContainer",
    "DeviceConfig",
    "configure_container",
    "IExtractor",
    "ICausalBuilder",
    "IBayesianEngine",
]
