#!/usr/bin/env python3
"""
Resilience Configuration Module for FARFAN 2.0
Defines Pydantic models for stage criticality levels, environment configurations,
and resilience settings for the orchestration pipeline.
"""

from enum import Enum
from typing import Dict, List, Optional, Set, Any, Union
from dataclasses import dataclass, field
import yaml
import json
from pathlib import Path

try:
    from pydantic import BaseModel, Field, validator, root_validator
except ImportError:
    raise ImportError(
        "Pydantic is required for resilience_config. Install with: pip install pydantic"
    )


# ============================================================================
# Enums
# ============================================================================

class StageCriticality(str, Enum):
    """
    Stage criticality levels that determine failure handling behavior.
    
    - CRITICAL: Pipeline must abort on failure; no recovery possible
    - IMPORTANT: Pipeline should retry with backoff; partial degradation acceptable
    - DEGRADABLE: Pipeline can continue with reduced functionality on failure
    """
    CRITICAL = "critical"
    IMPORTANT = "important"
    DEGRADABLE = "degradable"


class Environment(str, Enum):
    """
    Deployment environments with different resilience policies.
    
    - DEV: Permissive thresholds, verbose logging, fail-fast for debugging
    - PROD: Strict thresholds, graceful degradation, comprehensive error handling
    """
    DEV = "dev"
    PROD = "prod"


class PipelineStage(str, Enum):
    """
    FARFAN pipeline stages in canonical order.
    Must align with orchestrator.py PipelineStage enum.
    """
    PDF_PARSING = "pdf_parsing"
    TEXT_EXTRACTION = "text_extraction"
    SPACY_PROCESSING = "spacy_processing"
    DNP_API_CALLS = "dnp_api_calls"
    EMBEDDING_GENERATION = "embedding_generation"
    CAUSAL_ANALYSIS = "causal_analysis"
    MECHANISM_INFERENCE = "mechanism_inference"
    FINANCIAL_AUDIT = "financial_audit"
    DNP_VALIDATION = "dnp_validation"
    QUESTION_ANSWERING = "question_answering"
    REPORT_GENERATION = "report_generation"


# ============================================================================
# Pydantic Models
# ============================================================================

class MinimumViableState(BaseModel):
    """
    Defines minimum success criteria for a pipeline stage.
    
    Attributes:
        success_threshold: Minimum success rate (0.0-1.0) for stage to be considered viable
        required_outputs: List of output keys that must be present for stage success
        minimum_output_size: Minimum size/count thresholds for outputs (e.g., min nodes in graph)
        timeout_seconds: Maximum execution time before stage is considered failed
    """
    success_threshold: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Minimum success rate (0.0-1.0)"
    )
    required_outputs: List[str] = Field(
        default_factory=list,
        description="Required output keys for stage success"
    )
    minimum_output_size: Dict[str, int] = Field(
        default_factory=dict,
        description="Minimum size thresholds for outputs"
    )
    timeout_seconds: float = Field(
        default=300.0,
        gt=0,
        description="Maximum stage execution time"
    )
    
    class Config:
        use_enum_values = True


class StageDependency(BaseModel):
    """
    Explicit inter-stage dependency declaration.
    
    Attributes:
        stage: The dependent stage
        required_stages: List of stages that must complete before this stage
        optional_stages: List of stages that enhance this stage but aren't required
        required_outputs: Specific outputs from upstream stages that are required
    """
    stage: PipelineStage
    required_stages: List[PipelineStage] = Field(default_factory=list)
    optional_stages: List[PipelineStage] = Field(default_factory=list)
    required_outputs: Dict[PipelineStage, List[str]] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True
    
    @validator('required_stages', 'optional_stages')
    def no_self_dependency(cls, v, values):
        """Ensure stages don't depend on themselves."""
        if 'stage' in values and values['stage'] in v:
            raise ValueError(f"Stage cannot depend on itself")
        return v


class DynamicThresholdParams(BaseModel):
    """
    Parameters for dynamic threshold adjustment based on runtime conditions.
    
    Attributes:
        baseline_value: Initial threshold value
        adjustment_factor: Multiplier for threshold adjustments (e.g., 1.2 = 20% increase)
        time_window_seconds: Time window for calculating adjustment metrics
        min_samples: Minimum samples needed before adjustments are applied
        adjustment_triggers: Conditions that trigger threshold adjustments
    """
    baseline_value: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Baseline threshold value"
    )
    adjustment_factor: float = Field(
        default=1.0,
        ge=0.1,
        le=2.0,
        description="Threshold adjustment multiplier"
    )
    time_window_seconds: float = Field(
        default=3600.0,
        gt=0,
        description="Time window for metrics calculation"
    )
    min_samples: int = Field(
        default=10,
        ge=1,
        description="Minimum samples for adjustment"
    )
    adjustment_triggers: Dict[str, float] = Field(
        default_factory=dict,
        description="Trigger conditions (e.g., {'error_rate': 0.05})"
    )
    
    class Config:
        use_enum_values = True


class StageConfig(BaseModel):
    """
    Complete configuration for a single pipeline stage.
    
    Combines criticality, minimum viable state, and dynamic thresholds.
    """
    stage: PipelineStage
    criticality: StageCriticality
    minimum_viable_state: MinimumViableState
    dynamic_thresholds: Optional[DynamicThresholdParams] = None
    retry_attempts: int = Field(default=3, ge=0, description="Number of retry attempts")
    retry_backoff_seconds: float = Field(default=2.0, gt=0, description="Retry backoff time")
    circuit_breaker_threshold: int = Field(
        default=5,
        ge=1,
        description="Failures before circuit breaker opens"
    )
    
    class Config:
        use_enum_values = True


class EnvironmentOverride(BaseModel):
    """
    Environment-specific configuration overrides.
    
    Allows different thresholds and behaviors between dev and prod.
    """
    environment: Environment
    stage_overrides: Dict[PipelineStage, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Stage-specific overrides"
    )
    global_timeout_multiplier: float = Field(
        default=1.0,
        gt=0,
        description="Global timeout adjustment factor"
    )
    logging_level: str = Field(
        default="INFO",
        description="Logging level for this environment"
    )
    fail_fast: bool = Field(
        default=False,
        description="Whether to fail fast on first error"
    )
    
    class Config:
        use_enum_values = True


class ResilienceConfig(BaseModel):
    """
    Main resilience configuration schema for FARFAN pipeline.
    
    Defines:
    - Per-stage criticality mappings
    - Minimum viable state definitions
    - Inter-stage dependency declarations
    - Dynamic threshold adjustment parameters
    - Environment-specific overrides
    """
    version: str = Field(default="1.0.0", description="Config schema version")
    environment: Environment = Field(default=Environment.DEV)
    
    # Core stage configurations
    stage_configs: Dict[PipelineStage, StageConfig]
    
    # Dependency graph
    stage_dependencies: List[StageDependency] = Field(default_factory=list)
    
    # Environment-specific overrides
    environment_overrides: Dict[Environment, EnvironmentOverride] = Field(
        default_factory=dict
    )
    
    # Global settings
    global_timeout_seconds: float = Field(
        default=3600.0,
        gt=0,
        description="Global pipeline timeout"
    )
    enable_dynamic_thresholds: bool = Field(
        default=True,
        description="Enable dynamic threshold adjustments"
    )
    enable_circuit_breakers: bool = Field(
        default=True,
        description="Enable circuit breaker pattern"
    )
    
    class Config:
        use_enum_values = True
    
    @root_validator
    def validate_dependencies(cls, values):
        """Ensure all dependencies reference valid stages."""
        stage_configs = values.get('stage_configs', {})
        dependencies = values.get('stage_dependencies', [])
        
        valid_stages = set(stage_configs.keys())
        
        for dep in dependencies:
            if dep.stage not in valid_stages:
                raise ValueError(f"Dependency references invalid stage: {dep.stage}")
            
            for req_stage in dep.required_stages:
                if req_stage not in valid_stages:
                    raise ValueError(
                        f"Dependency {dep.stage} requires invalid stage: {req_stage}"
                    )
        
        return values
    
    def get_stage_config(
        self,
        stage: PipelineStage,
        apply_overrides: bool = True
    ) -> StageConfig:
        """
        Get stage configuration with optional environment overrides applied.
        
        Args:
            stage: The pipeline stage
            apply_overrides: Whether to apply environment-specific overrides
            
        Returns:
            StageConfig with overrides applied if requested
        """
        config = self.stage_configs[stage]
        
        if apply_overrides and self.environment in self.environment_overrides:
            override = self.environment_overrides[self.environment]
            if stage in override.stage_overrides:
                # Apply overrides to config
                override_dict = override.stage_overrides[stage]
                config_dict = config.dict()
                config_dict.update(override_dict)
                config = StageConfig(**config_dict)
        
        return config
    
    def get_stage_dependencies(self, stage: PipelineStage) -> Optional[StageDependency]:
        """Get dependency declaration for a stage."""
        for dep in self.stage_dependencies:
            if dep.stage == stage:
                return dep
        return None


# ============================================================================
# Default Configurations
# ============================================================================

class DefaultResilienceConfig:
    """
    Default resilience configurations for FARFAN pipeline stages.
    
    Defines sensible baseline thresholds based on stage criticality:
    - CRITICAL: 95% success, strict timeouts, no degradation
    - IMPORTANT: 85% success, moderate timeouts, retry-enabled
    - DEGRADABLE: 70% success, generous timeouts, graceful fallback
    """
    
    # Default thresholds per criticality level
    CRITICALITY_DEFAULTS = {
        StageCriticality.CRITICAL: {
            "success_threshold": 0.95,
            "timeout_seconds": 600.0,
            "retry_attempts": 1,
            "retry_backoff_seconds": 5.0,
            "circuit_breaker_threshold": 3
        },
        StageCriticality.IMPORTANT: {
            "success_threshold": 0.85,
            "timeout_seconds": 900.0,
            "retry_attempts": 3,
            "retry_backoff_seconds": 2.0,
            "circuit_breaker_threshold": 5
        },
        StageCriticality.DEGRADABLE: {
            "success_threshold": 0.70,
            "timeout_seconds": 1200.0,
            "retry_attempts": 5,
            "retry_backoff_seconds": 1.0,
            "circuit_breaker_threshold": 10
        }
    }
    
    # Stage-specific configurations
    STAGE_CONFIGS = {
        PipelineStage.PDF_PARSING: StageConfig(
            stage=PipelineStage.PDF_PARSING,
            criticality=StageCriticality.CRITICAL,
            minimum_viable_state=MinimumViableState(
                success_threshold=0.95,
                required_outputs=["raw_text", "page_count"],
                minimum_output_size={"raw_text": 100},
                timeout_seconds=300.0
            ),
            retry_attempts=1,
            retry_backoff_seconds=5.0,
            circuit_breaker_threshold=3
        ),
        PipelineStage.TEXT_EXTRACTION: StageConfig(
            stage=PipelineStage.TEXT_EXTRACTION,
            criticality=StageCriticality.CRITICAL,
            minimum_viable_state=MinimumViableState(
                success_threshold=0.95,
                required_outputs=["sections", "tables"],
                minimum_output_size={"sections": 1},
                timeout_seconds=600.0
            ),
            retry_attempts=2,
            retry_backoff_seconds=3.0,
            circuit_breaker_threshold=3
        ),
        PipelineStage.SPACY_PROCESSING: StageConfig(
            stage=PipelineStage.SPACY_PROCESSING,
            criticality=StageCriticality.IMPORTANT,
            minimum_viable_state=MinimumViableState(
                success_threshold=0.85,
                required_outputs=["semantic_chunks", "entities"],
                minimum_output_size={"semantic_chunks": 5},
                timeout_seconds=1200.0
            ),
            retry_attempts=3,
            retry_backoff_seconds=2.0,
            circuit_breaker_threshold=5
        ),
        PipelineStage.DNP_API_CALLS: StageConfig(
            stage=PipelineStage.DNP_API_CALLS,
            criticality=StageCriticality.DEGRADABLE,
            minimum_viable_state=MinimumViableState(
                success_threshold=0.70,
                required_outputs=["api_responses"],
                minimum_output_size={},
                timeout_seconds=600.0
            ),
            retry_attempts=5,
            retry_backoff_seconds=10.0,
            circuit_breaker_threshold=10
        ),
        PipelineStage.EMBEDDING_GENERATION: StageConfig(
            stage=PipelineStage.EMBEDDING_GENERATION,
            criticality=StageCriticality.IMPORTANT,
            minimum_viable_state=MinimumViableState(
                success_threshold=0.85,
                required_outputs=["embeddings", "embedding_metadata"],
                minimum_output_size={"embeddings": 10},
                timeout_seconds=900.0
            ),
            retry_attempts=3,
            retry_backoff_seconds=2.0,
            circuit_breaker_threshold=5
        ),
        PipelineStage.CAUSAL_ANALYSIS: StageConfig(
            stage=PipelineStage.CAUSAL_ANALYSIS,
            criticality=StageCriticality.CRITICAL,
            minimum_viable_state=MinimumViableState(
                success_threshold=0.95,
                required_outputs=["causal_graph", "nodes", "edges"],
                minimum_output_size={"nodes": 3, "edges": 2},
                timeout_seconds=1800.0
            ),
            retry_attempts=2,
            retry_backoff_seconds=5.0,
            circuit_breaker_threshold=3
        ),
        PipelineStage.MECHANISM_INFERENCE: StageConfig(
            stage=PipelineStage.MECHANISM_INFERENCE,
            criticality=StageCriticality.IMPORTANT,
            minimum_viable_state=MinimumViableState(
                success_threshold=0.85,
                required_outputs=["mechanism_parts", "bayesian_inferences"],
                minimum_output_size={"mechanism_parts": 1},
                timeout_seconds=1200.0
            ),
            retry_attempts=3,
            retry_backoff_seconds=3.0,
            circuit_breaker_threshold=5
        ),
        PipelineStage.FINANCIAL_AUDIT: StageConfig(
            stage=PipelineStage.FINANCIAL_AUDIT,
            criticality=StageCriticality.IMPORTANT,
            minimum_viable_state=MinimumViableState(
                success_threshold=0.85,
                required_outputs=["financial_allocations", "budget_traceability"],
                minimum_output_size={"financial_allocations": 1},
                timeout_seconds=900.0
            ),
            retry_attempts=3,
            retry_backoff_seconds=2.0,
            circuit_breaker_threshold=5
        ),
        PipelineStage.DNP_VALIDATION: StageConfig(
            stage=PipelineStage.DNP_VALIDATION,
            criticality=StageCriticality.CRITICAL,
            minimum_viable_state=MinimumViableState(
                success_threshold=0.95,
                required_outputs=["dnp_validation_results", "compliance_score"],
                minimum_output_size={"dnp_validation_results": 1},
                timeout_seconds=600.0
            ),
            retry_attempts=2,
            retry_backoff_seconds=3.0,
            circuit_breaker_threshold=3
        ),
        PipelineStage.QUESTION_ANSWERING: StageConfig(
            stage=PipelineStage.QUESTION_ANSWERING,
            criticality=StageCriticality.CRITICAL,
            minimum_viable_state=MinimumViableState(
                success_threshold=0.95,
                required_outputs=["question_responses"],
                minimum_output_size={"question_responses": 100},
                timeout_seconds=1800.0
            ),
            retry_attempts=2,
            retry_backoff_seconds=5.0,
            circuit_breaker_threshold=3
        ),
        PipelineStage.REPORT_GENERATION: StageConfig(
            stage=PipelineStage.REPORT_GENERATION,
            criticality=StageCriticality.IMPORTANT,
            minimum_viable_state=MinimumViableState(
                success_threshold=0.90,
                required_outputs=["micro_report", "meso_report", "macro_report"],
                minimum_output_size={},
                timeout_seconds=600.0
            ),
            retry_attempts=3,
            retry_backoff_seconds=2.0,
            circuit_breaker_threshold=5
        )
    }
    
    # Default dependency graph
    STAGE_DEPENDENCIES = [
        StageDependency(
            stage=PipelineStage.TEXT_EXTRACTION,
            required_stages=[PipelineStage.PDF_PARSING],
            required_outputs={
                PipelineStage.PDF_PARSING: ["raw_text"]
            }
        ),
        StageDependency(
            stage=PipelineStage.SPACY_PROCESSING,
            required_stages=[PipelineStage.TEXT_EXTRACTION],
            required_outputs={
                PipelineStage.TEXT_EXTRACTION: ["sections"]
            }
        ),
        StageDependency(
            stage=PipelineStage.EMBEDDING_GENERATION,
            required_stages=[PipelineStage.SPACY_PROCESSING],
            required_outputs={
                PipelineStage.SPACY_PROCESSING: ["semantic_chunks"]
            }
        ),
        StageDependency(
            stage=PipelineStage.CAUSAL_ANALYSIS,
            required_stages=[
                PipelineStage.SPACY_PROCESSING,
                PipelineStage.EMBEDDING_GENERATION
            ],
            required_outputs={
                PipelineStage.SPACY_PROCESSING: ["semantic_chunks"],
                PipelineStage.EMBEDDING_GENERATION: ["embeddings"]
            }
        ),
        StageDependency(
            stage=PipelineStage.MECHANISM_INFERENCE,
            required_stages=[PipelineStage.CAUSAL_ANALYSIS],
            required_outputs={
                PipelineStage.CAUSAL_ANALYSIS: ["causal_graph", "nodes"]
            }
        ),
        StageDependency(
            stage=PipelineStage.FINANCIAL_AUDIT,
            required_stages=[PipelineStage.TEXT_EXTRACTION],
            optional_stages=[PipelineStage.CAUSAL_ANALYSIS],
            required_outputs={
                PipelineStage.TEXT_EXTRACTION: ["tables"]
            }
        ),
        StageDependency(
            stage=PipelineStage.DNP_VALIDATION,
            required_stages=[
                PipelineStage.CAUSAL_ANALYSIS,
                PipelineStage.FINANCIAL_AUDIT
            ],
            optional_stages=[PipelineStage.DNP_API_CALLS],
            required_outputs={
                PipelineStage.CAUSAL_ANALYSIS: ["causal_graph"],
                PipelineStage.FINANCIAL_AUDIT: ["financial_allocations"]
            }
        ),
        StageDependency(
            stage=PipelineStage.QUESTION_ANSWERING,
            required_stages=[
                PipelineStage.SPACY_PROCESSING,
                PipelineStage.CAUSAL_ANALYSIS,
                PipelineStage.DNP_VALIDATION
            ],
            required_outputs={
                PipelineStage.SPACY_PROCESSING: ["semantic_chunks"],
                PipelineStage.CAUSAL_ANALYSIS: ["causal_graph"],
                PipelineStage.DNP_VALIDATION: ["dnp_validation_results"]
            }
        ),
        StageDependency(
            stage=PipelineStage.REPORT_GENERATION,
            required_stages=[
                PipelineStage.QUESTION_ANSWERING,
                PipelineStage.DNP_VALIDATION
            ],
            required_outputs={
                PipelineStage.QUESTION_ANSWERING: ["question_responses"],
                PipelineStage.DNP_VALIDATION: ["compliance_score"]
            }
        )
    ]
    
    # Environment-specific overrides
    DEV_OVERRIDES = EnvironmentOverride(
        environment=Environment.DEV,
        stage_overrides={
            PipelineStage.DNP_API_CALLS: {
                "retry_attempts": 1,
                "timeout_seconds": 30.0
            },
            PipelineStage.SPACY_PROCESSING: {
                "timeout_seconds": 300.0
            }
        },
        global_timeout_multiplier=0.5,
        logging_level="DEBUG",
        fail_fast=True
    )
    
    PROD_OVERRIDES = EnvironmentOverride(
        environment=Environment.PROD,
        stage_overrides={
            PipelineStage.DNP_API_CALLS: {
                "retry_attempts": 10,
                "timeout_seconds": 1200.0
            },
            PipelineStage.CAUSAL_ANALYSIS: {
                "timeout_seconds": 3600.0
            }
        },
        global_timeout_multiplier=1.5,
        logging_level="INFO",
        fail_fast=False
    )
    
    @classmethod
    def get_default_config(cls, environment: Environment = Environment.DEV) -> ResilienceConfig:
        """
        Get default resilience configuration for specified environment.
        
        Args:
            environment: Target environment (dev or prod)
            
        Returns:
            Complete ResilienceConfig with defaults
        """
        return ResilienceConfig(
            version="1.0.0",
            environment=environment,
            stage_configs=cls.STAGE_CONFIGS,
            stage_dependencies=cls.STAGE_DEPENDENCIES,
            environment_overrides={
                Environment.DEV: cls.DEV_OVERRIDES,
                Environment.PROD: cls.PROD_OVERRIDES
            },
            global_timeout_seconds=3600.0,
            enable_dynamic_thresholds=True,
            enable_circuit_breakers=True
        )


# ============================================================================
# Configuration Loader
# ============================================================================

def load_resilience_config(
    source: Union[str, Path, Dict],
    environment: Optional[Environment] = None
) -> ResilienceConfig:
    """
    Load and validate ResilienceConfig from YAML file or dictionary.
    
    Args:
        source: YAML file path, Path object, or configuration dictionary
        environment: Optional environment override
        
    Returns:
        Validated ResilienceConfig instance
        
    Raises:
        ValueError: If configuration is invalid
        FileNotFoundError: If YAML file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        
    Examples:
        >>> # Load from YAML file
        >>> config = load_resilience_config("config/resilience.yaml")
        
        >>> # Load from dictionary
        >>> config = load_resilience_config({"version": "1.0.0", ...})
        
        >>> # Use defaults with environment override
        >>> config = load_resilience_config({}, environment=Environment.PROD)
    """
    config_dict: Dict[str, Any] = {}
    
    # Load from source
    if isinstance(source, dict):
        config_dict = source
    elif isinstance(source, (str, Path)):
        file_path = Path(source)
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.suffix in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            elif file_path.suffix == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported file format: {file_path.suffix}. "
                    "Use .yaml, .yml, or .json"
                )
    else:
        raise TypeError(f"Invalid source type: {type(source)}")
    
    # Apply environment override if specified
    if environment is not None:
        config_dict['environment'] = environment.value
    
    # If config is empty or minimal, use defaults
    if not config_dict or 'stage_configs' not in config_dict:
        env = Environment(config_dict.get('environment', 'dev')) if config_dict else Environment.DEV
        return DefaultResilienceConfig.get_default_config(env)
    
    # Parse and validate with Pydantic
    try:
        config = ResilienceConfig(**config_dict)
        return config
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}") from e


def save_resilience_config(
    config: ResilienceConfig,
    output_path: Union[str, Path],
    format: str = 'yaml'
) -> None:
    """
    Save ResilienceConfig to YAML or JSON file.
    
    Args:
        config: ResilienceConfig instance to save
        output_path: Output file path
        format: Output format ('yaml' or 'json')
        
    Raises:
        ValueError: If format is unsupported
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = config.dict()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        if format == 'yaml':
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        elif format == 'json':
            json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'yaml' or 'json'")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example 1: Get default configuration for dev environment
    dev_config = DefaultResilienceConfig.get_default_config(Environment.DEV)
    print(f"Dev config loaded: {len(dev_config.stage_configs)} stages")
    
    # Example 2: Get default configuration for prod environment
    prod_config = DefaultResilienceConfig.get_default_config(Environment.PROD)
    print(f"Prod config loaded: {len(prod_config.stage_configs)} stages")
    
    # Example 3: Access stage configuration with overrides
    pdf_stage = dev_config.get_stage_config(PipelineStage.PDF_PARSING)
    print(f"PDF parsing criticality: {pdf_stage.criticality}")
    print(f"PDF parsing success threshold: {pdf_stage.minimum_viable_state.success_threshold}")
    
    # Example 4: Check stage dependencies
    qa_deps = prod_config.get_stage_dependencies(PipelineStage.QUESTION_ANSWERING)
    if qa_deps:
        print(f"QA stage requires: {[s.value for s in qa_deps.required_stages]}")
    
    # Example 5: Save default config to YAML
    save_resilience_config(dev_config, "resilience_dev_default.yaml", format='yaml')
    print("Saved default dev config to resilience_dev_default.yaml")
    
    # Example 6: Load from saved config
    loaded_config = load_resilience_config("resilience_dev_default.yaml")
    print(f"Loaded config version: {loaded_config.version}")
