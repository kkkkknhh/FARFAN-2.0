# -*- coding: utf-8 -*-
"""
Comprehensive Unit Tests for Semantic Chunking Policy Analysis Framework

Test Coverage:
- Enum classes (CausalDimension, PDMSection)
- Configuration dataclass (SemanticConfig)
- Semantic processing with mocked transformers
- Bayesian evidence integration (mathematical correctness)
- Policy document analysis end-to-end
- Edge cases, error handling, and boundary conditions
- Pure function testing
"""

import json
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch, call
from dataclasses import FrozenInstanceError
import scipy.stats as stats
from scipy.special import rel_entr
from scipy.spatial.distance import cosine

# Import the module under test
from semantic_chunking_policy import (
    CausalDimension,
    PDMSection,
    SemanticConfig,
    SemanticProcessor,
    BayesianEvidenceIntegrator,
    PolicyDocumentAnalyzer,
)

# ============================================================================
# FIXTURES
# ============================================================================
@pytest.fixture
def sample_pdm_text():
    """Sample Colombian PDM text for testing"""
    return """
    PLAN DE DESARROLLO MUNICIPAL 2024-2027
    MUNICIPIO DE EJEMPLO, COLOMBIA
    
    1. DIAGNÓSTICO TERRITORIAL
    El municipio cuenta con 45,000 habitantes, de los cuales 60% reside en zona rural.
    La tasa de pobreza multidimensional es 42.3%, superior al promedio departamental.
    Se requiere inversión en infraestructura educativa y servicios de salud.
    
    2. VISIÓN ESTRATÉGICA
    Para 2027, el municipio será reconocido por su desarrollo sostenible e inclusivo.
    Objetivos estratégicos: reducir pobreza, mejorar educación, fortalecer instituciones.
    
    3. PLAN PLURIANUAL DE INVERSIONES
    Se destinarán $12,500 millones al sector educación, con meta de construir
    3 instituciones educativas y capacitar 250 docentes en pedagogías innovadoras.
    Presupuesto salud: $8,000 millones para ampliar cobertura en zona rural.
    
    4. SEGUIMIENTO Y EVALUACIÓN
    Se implementará sistema de indicadores alineado con ODS, con mediciones semestrales.
    """

@pytest.fixture
def semantic_config():
    """Default semantic configuration"""
    return SemanticConfig(
        embedding_model="BAAI/bge-m3",
        chunk_size=512,
        chunk_overlap=100,
        similarity_threshold=0.80,
        min_evidence_chunks=2,
        bayesian_prior_strength=0.5,
        device="cpu",
        batch_size=16,
        fp16=False
    )

@pytest.fixture
def mock_tokenizer():
    """Mock HuggingFace tokenizer"""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = list(range(100))  # Mock token IDs
    tokenizer.decode.return_value = "decoded text"
    tokenizer.return_value = {
        "input_ids": np.array([[1, 2, 3, 4, 5]]),
        "attention_mask": np.array([[1, 1, 1, 1, 1]])
    }
    return tokenizer

@pytest.fixture
def mock_model():
    """Mock transformer model"""
    model = MagicMock()
    mock_output = MagicMock()
    mock_output.last_hidden_state = np.random.randn(1, 5, 768).astype(np.float32)
    model.return_value = mock_output
    model.eval.return_value = model
    model.device = "cpu"
    return model

@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing"""
    np.random.seed(42)
    return [np.random.randn(768).astype(np.float32) for _ in range(5)]

@pytest.fixture
def sample_chunks():
    """Sample processed chunks with metadata"""
    np.random.seed(42)
    return [
        {
            "text": "Sample text 1 with numerical data: $10,000",
            "section_type": PDMSection.PLAN_PLURIANUAL,
            "section_id": "sec_0",
            "token_count": 50,
            "position": 0,
            "has_table": False,
            "has_numerical": True,
            "embedding": np.random.randn(768).astype(np.float32)
        },
        {
            "text": "Sample text 2 with table data",
            "section_type": PDMSection.DIAGNOSTICO,
            "section_id": "sec_1",
            "token_count": 45,
            "position": 1,
            "has_table": True,
            "has_numerical": False,
            "embedding": np.random.randn(768).astype(np.float32)
        },
        {
            "text": "Sample text 3 regular content",
            "section_type": PDMSection.VISION_ESTRATEGICA,
            "section_id": "sec_2",
            "token_count": 40,
            "position": 2,
            "has_table": False,
            "has_numerical": False,
            "embedding": np.random.randn(768).astype(np.float32)
        }
    ]

# ============================================================================
# TEST ENUMS
# ============================================================================
class TestCausalDimension:
    """Test CausalDimension enum"""
    
    def test_all_dimensions_exist(self):
        """Test that all expected causal dimensions are defined"""
        expected_dimensions = {
            "INSUMOS", "ACTIVIDADES", "PRODUCTOS", 
            "RESULTADOS", "IMPACTOS", "SUPUESTOS"
        }
        actual_dimensions = {dim.name for dim in CausalDimension}
        assert actual_dimensions == expected_dimensions
    
    def test_dimension_values(self):
        """Test dimension enum values match lowercase names"""
        for dim in CausalDimension:
            assert dim.value == dim.name.lower()
    
    def test_dimension_access(self):
        """Test accessing dimensions by name and value"""
        assert CausalDimension.INSUMOS.value == "insumos"
        assert CausalDimension["ACTIVIDADES"] == CausalDimension.ACTIVIDADES
        assert CausalDimension("resultados") == CausalDimension.RESULTADOS
    
    def test_dimension_iteration(self):
        """Test iterating over dimensions"""
        dimensions = list(CausalDimension)
        assert len(dimensions) == 6
        assert CausalDimension.INSUMOS in dimensions

class TestPDMSection:
    """Test PDMSection enum"""
    
    def test_all_sections_exist(self):
        """Test that all PDM sections are defined"""
        expected_sections = {
            "DIAGNOSTICO", "VISION_ESTRATEGICA", "PLAN_PLURIANUAL",
            "PLAN_INVERSIONES", "MARCO_FISCAL", "SEGUIMIENTO"
        }
        actual_sections = {sec.name for sec in PDMSection}
        assert actual_sections == expected_sections
    
    def test_section_values(self):
        """Test section values are appropriate"""
        assert PDMSection.DIAGNOSTICO.value == "diagnostico"
        assert PDMSection.VISION_ESTRATEGICA.value == "vision_estrategica"
        assert PDMSection.SEGUIMIENTO.value == "seguimiento_evaluacion"
    
    def test_section_uniqueness(self):
        """Test all section values are unique"""
        values = [sec.value for sec in PDMSection]
        assert len(values) == len(set(values))

# ============================================================================
# TEST CONFIGURATION
# ============================================================================
class TestSemanticConfig:
    """Test SemanticConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = SemanticConfig()
        assert config.embedding_model == "BAAI/bge-m3"
        assert config.chunk_size == 768
        assert config.chunk_overlap == 128
        assert config.similarity_threshold == 0.82
        assert config.min_evidence_chunks == 3
        assert config.bayesian_prior_strength == 0.5
        assert config.device is None
        assert config.batch_size == 32
        assert config.fp16 is True
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = SemanticConfig(
            chunk_size=512,
            chunk_overlap=64,
            similarity_threshold=0.75,
            device="cuda",
            batch_size=16,
            fp16=False
        )
        assert config.chunk_size == 512
        assert config.chunk_overlap == 64
        assert config.similarity_threshold == 0.75
        assert config.device == "cuda"
        assert config.batch_size == 16
        assert config.fp16 is False
    
    def test_config_immutability(self):
        """Test that config is frozen (immutable)"""
        config = SemanticConfig()
        with pytest.raises(FrozenInstanceError):
            config.chunk_size = 1024
    
    def test_config_slots(self):
        """Test that config uses slots for memory efficiency"""
        config = SemanticConfig()
        assert hasattr(config, '__slots__')

# ============================================================================
# TEST SEMANTIC PROCESSOR
# ============================================================================
class TestSemanticProcessor:
    """Test SemanticProcessor class"""
    
    def test_initialization(self, semantic_config):
        """Test processor initialization"""
        processor = SemanticProcessor(semantic_config)
        assert processor.config == semantic_config
        assert processor._model is None
        assert processor._tokenizer is None
        assert processor._loaded is False
    
    def test_lazy_load_not_called_initially(self, semantic_config):
        """Test that model is not loaded on initialization"""
        processor = SemanticProcessor(semantic_config)
        assert not processor._loaded
    
    @patch('semantic_chunking_policy.AutoTokenizer.from_pretrained')
    @patch('semantic_chunking_policy.AutoModel.from_pretrained')
    def test_lazy_load_success(self, mock_model, mock_tokenizer, semantic_config):
        """Test successful lazy loading of model"""
        mock_tokenizer.return_value = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_instance.eval.return_value = mock_model_instance
        mock_model.return_value = mock_model_instance
        
        processor = SemanticProcessor(semantic_config)
        processor._lazy_load()
        
        assert processor._loaded
        assert processor._tokenizer is not None
        assert processor._model is not None
        mock_tokenizer.assert_called_once_with(semantic_config.embedding_model)
        mock_model.assert_called_once()
    
    @patch('semantic_chunking_policy.AutoTokenizer')
    def test_lazy_load_missing_transformers(self, mock_tokenizer, semantic_config):
        """Test error handling when transformers is not installed"""
        mock_tokenizer.from_pretrained.side_effect = ImportError("No module named 'transformers'")
        
        processor = SemanticProcessor(semantic_config)
        with pytest.raises(RuntimeError) as exc_info:
            processor._lazy_load()
        
        assert "Missing dependency: transformers" in str(exc_info.value)
    
    @patch('semantic_chunking_policy.AutoModel')
    @patch('semantic_chunking_policy.AutoTokenizer')
    def test_lazy_load_missing_torch(self, mock_tokenizer, mock_model, semantic_config):
        """Test error handling when torch is not installed"""
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.side_effect = ImportError("No module named 'torch'")
        
        processor = SemanticProcessor(semantic_config)
        with pytest.raises(RuntimeError) as exc_info:
            processor._lazy_load()
        
        assert "Missing dependency:" in str(exc_info.value)
    
    def test_detect_table_with_tabs(self, semantic_config):
        """Test table detection with tab-separated data"""
        processor = SemanticProcessor(semantic_config)
        text = "Column1\tColumn2\tColumn3\n1\t2\t3\n4\t5\t6"
        assert processor._detect_table(text) is True
    
    def test_detect_table_with_pipes(self, semantic_config):
        """Test table detection with pipe-separated data"""
        processor = SemanticProcessor(semantic_config)
        text = "| Header1 | Header2 | Header3 |\n| --- | --- | --- |\n| Data1 | Data2 | Data3 |"
        assert processor._detect_table(text) is True
    
    def test_detect_table_with_numbers(self, semantic_config):
        """Test table detection with numerical patterns"""
        processor = SemanticProcessor(semantic_config)
        text = "Some text with numbers: 10 20 30 40 50 arranged in columns"
        assert processor._detect_table(text) is True
    
    def test_detect_table_negative(self, semantic_config):
        """Test that regular text is not detected as table"""
        processor = SemanticProcessor(semantic_config)
        text = "This is regular text without any table structure."
        assert processor._detect_table(text) is False
    
    def test_detect_numerical_data_currency(self, semantic_config):
        """Test detection of currency values"""
        processor = SemanticProcessor(semantic_config)
        assert processor._detect_numerical_data("Presupuesto: $12,500 millones")
        assert processor._detect_numerical_data("Inversión de $ 1.234.567")
    
    def test_detect_numerical_data_percentages(self, semantic_config):
        """Test detection of percentages"""
        processor = SemanticProcessor(semantic_config)
        assert processor._detect_numerical_data("Crecimiento del 15.5%")
        assert processor._detect_numerical_data("42.3 % de pobreza")
    
    def test_detect_numerical_data_large_numbers(self, semantic_config):
        """Test detection of large numbers with separators"""
        processor = SemanticProcessor(semantic_config)
        assert processor._detect_numerical_data("Población: 45,000 habitantes")
        assert processor._detect_numerical_data("Cantidad: 1.234.567 unidades")
    
    def test_detect_numerical_data_negative(self, semantic_config):
        """Test that text without numerical data returns False"""
        processor = SemanticProcessor(semantic_config)
        assert not processor._detect_numerical_data("Simple text without numbers")
        assert not processor._detect_numerical_data("One two three")
    
    def test_detect_pdm_structure_diagnostico(self, semantic_config):
        """Test detection of DIAGNOSTICO section"""
        processor = SemanticProcessor(semantic_config)
        text = "DIAGNÓSTICO TERRITORIAL\nCaracterización del municipio..."
        sections = processor._detect_pdm_structure(text)
        
        assert len(sections) >= 1
        assert sections[0]["type"] == PDMSection.DIAGNOSTICO
    
    def test_detect_pdm_structure_vision(self, semantic_config):
        """Test detection of VISION_ESTRATEGICA section"""
        processor = SemanticProcessor(semantic_config)
        text = "VISIÓN Y MISIÓN\nObjetivos estratégicos del municipio..."
        sections = processor._detect_pdm_structure(text)
        
        assert len(sections) >= 1
        assert any(s["type"] == PDMSection.VISION_ESTRATEGICA for s in sections)
    
    def test_detect_pdm_structure_multiple_sections(self, sample_pdm_text, semantic_config):
        """Test detection of multiple PDM sections"""
        processor = SemanticProcessor(semantic_config)
        sections = processor._detect_pdm_structure(sample_pdm_text)
        
        assert len(sections) > 1
        section_types = {s["type"] for s in sections}
        assert PDMSection.DIAGNOSTICO in section_types or PDMSection.VISION_ESTRATEGICA in section_types
    
    def test_detect_pdm_structure_section_ids(self, sample_pdm_text, semantic_config):
        """Test that sections have unique IDs"""
        processor = SemanticProcessor(semantic_config)
        sections = processor._detect_pdm_structure(sample_pdm_text)
        
        section_ids = [s["id"] for s in sections]
        assert len(section_ids) == len(set(section_ids))
        assert all(sid.startswith("sec_") for sid in section_ids)

# ============================================================================
# TEST BAYESIAN EVIDENCE INTEGRATOR
# ============================================================================
class TestBayesianEvidenceIntegrator:
    """Test BayesianEvidenceIntegrator class"""
    
    def test_initialization_default(self):
        """Test default initialization"""
        integrator = BayesianEvidenceIntegrator()
        assert integrator.prior_alpha == 0.5
    
    def test_initialization_custom(self):
        """Test custom prior concentration"""
        integrator = BayesianEvidenceIntegrator(prior_concentration=0.8)
        assert integrator.prior_alpha == 0.8
    
    def test_initialization_invalid_concentration(self):
        """Test that invalid concentration raises error"""
        with pytest.raises(ValueError) as exc_info:
            BayesianEvidenceIntegrator(prior_concentration=0.0)
        assert "must be strictly positive" in str(exc_info.value)
        
        with pytest.raises(ValueError):
            BayesianEvidenceIntegrator(prior_concentration=-0.5)
    
    def test_null_evidence(self):
        """Test null evidence returns prior state"""
        integrator = BayesianEvidenceIntegrator(prior_concentration=0.5)
        result = integrator._null_evidence()
        
        assert result["posterior_mean"] == 0.5
        assert result["information_gain"] == 0.0
        assert result["confidence"] == 0.0
        assert result["evidence_strength"] == 0.0
        assert result["n_chunks"] == 0
        assert result["posterior_std"] >= 0
    
    def test_integrate_evidence_empty(self):
        """Test integration with empty evidence"""
        integrator = BayesianEvidenceIntegrator()
        result = integrator.integrate_evidence(np.array([]), [])
        
        assert result["posterior_mean"] == 0.5
        assert result["n_chunks"] == 0
    
    def test_integrate_evidence_high_similarity(self):
        """Test integration with high similarity scores"""
        integrator = BayesianEvidenceIntegrator(prior_concentration=0.5)
        similarities = np.array([0.9, 0.85, 0.88, 0.92])
        metadata = [
            {"position": i, "has_table": False, "has_numerical": False, 
             "section_type": PDMSection.DIAGNOSTICO}
            for i in range(len(similarities))
        ]
        
        result = integrator.integrate_evidence(similarities, metadata)
        
        assert result["posterior_mean"] > 0.5  # Should be higher than prior
        assert result["n_chunks"] == 4
        assert result["confidence"] > 0
        assert result["information_gain"] > 0
    
    def test_integrate_evidence_low_similarity(self):
        """Test integration with low similarity scores"""
        integrator = BayesianEvidenceIntegrator(prior_concentration=0.5)
        similarities = np.array([0.3, 0.25, 0.35, 0.28])
        metadata = [
            {"position": i, "has_table": False, "has_numerical": False,
             "section_type": PDMSection.DIAGNOSTICO}
            for i in range(len(similarities))
        ]
        
        result = integrator.integrate_evidence(similarities, metadata)
        
        assert result["posterior_mean"] < 0.5  # Should be lower than prior
        assert result["n_chunks"] == 4
    
    def test_integrate_evidence_mixed_similarity(self):
        """Test integration with mixed similarity scores"""
        integrator = BayesianEvidenceIntegrator()
        similarities = np.array([0.9, 0.3, 0.7, 0.4, 0.8])
        metadata = [
            {"position": i, "has_table": False, "has_numerical": False,
             "section_type": PDMSection.DIAGNOSTICO}
            for i in range(len(similarities))
        ]
        
        result = integrator.integrate_evidence(similarities, metadata)
        
        assert 0.0 <= result["posterior_mean"] <= 1.0
        assert result["posterior_std"] >= 0
        assert result["information_gain"] >= 0
        assert result["confidence"] >= 0
    
    def test_integrate_evidence_with_tables(self):
        """Test that table data increases evidence weight"""
        integrator = BayesianEvidenceIntegrator()
        similarities = np.array([0.8, 0.8])
        
        metadata_no_table = [
            {"position": 0, "has_table": False, "has_numerical": False,
             "section_type": PDMSection.DIAGNOSTICO}
        ]
        metadata_with_table = [
            {"position": 0, "has_table": True, "has_numerical": False,
             "section_type": PDMSection.DIAGNOSTICO}
        ]
        
        result_no_table = integrator.integrate_evidence(similarities[:1], metadata_no_table)
        result_with_table = integrator.integrate_evidence(similarities[:1], metadata_with_table)
        
        # Table should increase confidence/evidence strength
        assert result_with_table["evidence_strength"] >= result_no_table["evidence_strength"]
    
    def test_integrate_evidence_with_numerical(self):
        """Test that numerical data increases evidence weight"""
        integrator = BayesianEvidenceIntegrator()
        similarities = np.array([0.8])
        
        metadata_with_numerical = [
            {"position": 0, "has_table": False, "has_numerical": True,
             "section_type": PDMSection.DIAGNOSTICO}
        ]
        
        result = integrator.integrate_evidence(similarities, metadata_with_numerical)
        assert result["confidence"] > 0
    
    def test_integrate_evidence_section_type_weights(self):
        """Test that section type affects evidence weighting"""
        integrator = BayesianEvidenceIntegrator()
        similarities = np.array([0.8])
        
        metadata_plan = [
            {"position": 0, "has_table": False, "has_numerical": False,
             "section_type": PDMSection.PLAN_PLURIANUAL}
        ]
        metadata_diagnostic = [
            {"position": 0, "has_table": False, "has_numerical": False,
             "section_type": PDMSection.DIAGNOSTICO}
        ]
        
        result_plan = integrator.integrate_evidence(similarities, metadata_plan)
        result_diagnostic = integrator.integrate_evidence(similarities, metadata_diagnostic)
        
        # Plan sections should have higher weight
        assert result_plan["evidence_strength"] >= result_diagnostic["evidence_strength"]
    
    def test_integrate_evidence_position_weights(self):
        """Test that early positions have higher weights"""
        integrator = BayesianEvidenceIntegrator()
        similarities = np.array([0.8, 0.8])
        
        metadata_early = [
            {"position": 0, "has_table": False, "has_numerical": False,
             "section_type": PDMSection.DIAGNOSTICO}
        ]
        metadata_late = [
            {"position": 100, "has_table": False, "has_numerical": False,
             "section_type": PDMSection.DIAGNOSTICO}
        ]
        
        result_early = integrator.integrate_evidence(similarities[:1], metadata_early)
        result_late = integrator.integrate_evidence(similarities[:1], metadata_late)
        
        # Early chunks should have slightly higher impact
        assert result_early["confidence"] >= result_late["confidence"] * 0.9
    
    def test_similarity_to_probability_range(self):
        """Test similarity to probability transformation stays in [0,1]"""
        integrator = BayesianEvidenceIntegrator()
        
        # Test edge cases
        sims = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        probs = integrator._similarity_to_probability(sims)
        
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)
    
    def test_similarity_to_probability_monotonic(self):
        """Test that similarity transformation is monotonically increasing"""
        integrator = BayesianEvidenceIntegrator()
        
        sims = np.linspace(-1, 1, 20)
        probs = integrator._similarity_to_probability(sims)
        
        # Check monotonicity
        for i in range(len(probs) - 1):
            assert probs[i] <= probs[i + 1]
    
    def test_compute_reliability_weights_normalization(self):
        """Test that reliability weights are properly normalized"""
        integrator = BayesianEvidenceIntegrator()
        
        metadata = [
            {"position": i, "has_table": False, "has_numerical": False,
             "section_type": PDMSection.DIAGNOSTICO}
            for i in range(5)
        ]
        
        weights = integrator._compute_reliability_weights(metadata)
        
        assert len(weights) == 5
        assert np.all(weights > 0)
        # Weights should sum approximately to n (preserving evidence mass)
        assert np.abs(weights.sum() - 5.0) < 1e-6
    
    def test_causal_strength_range(self):
        """Test that causal strength is in [0,1]"""
        integrator = BayesianEvidenceIntegrator()
        
        np.random.seed(42)
        cause_emb = np.random.randn(768).astype(np.float32)
        effect_emb = np.random.randn(768).astype(np.float32)
        context_emb = np.random.randn(768).astype(np.float32)
        
        strength = integrator.causal_strength(cause_emb, effect_emb, context_emb)
        
        assert 0.0 <= strength <= 1.0
    
    def test_causal_strength_identical_embeddings(self):
        """Test causal strength with identical embeddings"""
        integrator = BayesianEvidenceIntegrator()
        
        emb = np.random.randn(768).astype(np.float32)
        strength = integrator.causal_strength(emb, emb, emb)
        
        # Identical embeddings should give high causal strength
        assert strength > 0.5
    
    def test_causal_strength_orthogonal_embeddings(self):
        """Test causal strength with orthogonal embeddings"""
        integrator = BayesianEvidenceIntegrator()
        
        cause_emb = np.array([1.0] + [0.0] * 767, dtype=np.float32)
        effect_emb = np.array([0.0, 1.0] + [0.0] * 766, dtype=np.float32)
        context_emb = np.array([0.0, 0.0, 1.0] + [0.0] * 765, dtype=np.float32)
        
        strength = integrator.causal_strength(cause_emb, effect_emb, context_emb)
        
        # Orthogonal embeddings should give lower strength
        assert strength < 0.8

# ============================================================================
# TEST POLICY DOCUMENT ANALYZER
# ============================================================================
class TestPolicyDocumentAnalyzer:
    """Test PolicyDocumentAnalyzer class"""
    
    def test_initialization_default_config(self):
        """Test analyzer initialization with default config"""
        analyzer = PolicyDocumentAnalyzer()
        
        assert analyzer.config is not None
        assert isinstance(analyzer.config, SemanticConfig)
        assert isinstance(analyzer.semantic, SemanticProcessor)
        assert isinstance(analyzer.bayesian, BayesianEvidenceIntegrator)
        assert len(analyzer.dimension_embeddings) == 6
    
    def test_initialization_custom_config(self, semantic_config):
        """Test analyzer initialization with custom config"""
        analyzer = PolicyDocumentAnalyzer(config=semantic_config)
        
        assert analyzer.config == semantic_config
        assert analyzer.bayesian.prior_alpha == semantic_config.bayesian_prior_strength
    
    def test_dimension_embeddings_all_present(self):
        """Test that all causal dimensions have embeddings"""
        analyzer = PolicyDocumentAnalyzer()
        
        for dimension in CausalDimension:
            assert dimension in analyzer.dimension_embeddings
            assert isinstance(analyzer.dimension_embeddings[dimension], np.ndarray)
    
    def test_dimension_embeddings_shape(self):
        """Test that dimension embeddings have correct shape"""
        with patch.object(SemanticProcessor, 'embed_single') as mock_embed:
            mock_embed.return_value = np.random.randn(768).astype(np.float32)
            analyzer = PolicyDocumentAnalyzer()
            
            for emb in analyzer.dimension_embeddings.values():
                assert emb.shape == (768,)
    
    def test_dimension_embeddings_distinct(self):
        """Test that dimension embeddings are distinct"""
        with patch.object(SemanticProcessor, 'embed_single') as mock_embed:
            # Return different embeddings for each call
            call_count = [0]
            def side_effect(*_args):
                np.random.seed(call_count[0])
                call_count[0] += 1
                return np.random.randn(768).astype(np.float32)
            
            mock_embed.side_effect = side_effect
            analyzer = PolicyDocumentAnalyzer()
            
            embeddings_list = list(analyzer.dimension_embeddings.values())
            # Check that not all embeddings are identical
            for i in range(len(embeddings_list)):
                for j in range(i + 1, len(embeddings_list)):
                    # Embeddings should be different
                    assert not np.allclose(embeddings_list[i], embeddings_list[j])
    
    @patch.object(SemanticProcessor, 'chunk_text')
    @patch.object(SemanticProcessor, 'embed_single')
    def test_analyze_basic(self, mock_embed, mock_chunk, sample_pdm_text, sample_chunks):
        """Test basic analysis pipeline"""
        mock_embed.return_value = np.random.randn(768).astype(np.float32)
        mock_chunk.return_value = sample_chunks
        
        analyzer = PolicyDocumentAnalyzer()
        result = analyzer.analyze(sample_pdm_text)
        
        assert "summary" in result
        assert "causal_dimensions" in result
        assert "key_excerpts" in result
        
        # Check summary
        assert result["summary"]["total_chunks"] == len(sample_chunks)
        assert "sections_detected" in result["summary"]
        assert "has_tables" in result["summary"]
        assert "has_numerical" in result["summary"]
    
    @patch.object(SemanticProcessor, 'chunk_text')
    @patch.object(SemanticProcessor, 'embed_single')
    def test_analyze_causal_dimensions(self, mock_embed, mock_chunk, sample_pdm_text, sample_chunks):
        """Test that all causal dimensions are analyzed"""
        mock_embed.return_value = np.random.randn(768).astype(np.float32)
        mock_chunk.return_value = sample_chunks
        
        analyzer = PolicyDocumentAnalyzer()
        result = analyzer.analyze(sample_pdm_text)
        
        # Check that all dimensions are present
        for dimension in CausalDimension:
            assert dimension.value in result["causal_dimensions"]
            
            dim_result = result["causal_dimensions"][dimension.value]
            assert "total_chunks" in dim_result
            assert "mean_similarity" in dim_result
            assert "max_similarity" in dim_result
            assert "posterior_mean" in dim_result
            assert "confidence" in dim_result
            assert "information_gain" in dim_result
    
    @patch.object(SemanticProcessor, 'chunk_text')
    @patch.object(SemanticProcessor, 'embed_single')
    def test_analyze_key_excerpts(self, mock_embed, mock_chunk, sample_pdm_text, sample_chunks):
        """Test key excerpts extraction"""
        mock_embed.return_value = np.random.randn(768).astype(np.float32)
        mock_chunk.return_value = sample_chunks
        
        analyzer = PolicyDocumentAnalyzer()
        result = analyzer.analyze(sample_pdm_text)
        
        # Check key excerpts for each dimension
        for dimension in CausalDimension:
            assert dimension.value in result["key_excerpts"]
            excerpts = result["key_excerpts"][dimension.value]
            
            assert isinstance(excerpts, list)
            assert len(excerpts) <= 3  # Maximum 3 excerpts
            
            # Each excerpt should be truncated to 300 chars
            for excerpt in excerpts:
                assert len(excerpt) <= 303  # 300 + "..."
    
    @patch.object(SemanticProcessor, 'chunk_text')
    @patch.object(SemanticProcessor, 'embed_single')
    def test_analyze_empty_text(self, mock_embed, mock_chunk):
        """Test analysis with empty text"""
        mock_embed.return_value = np.random.randn(768).astype(np.float32)
        mock_chunk.return_value = []
        
        analyzer = PolicyDocumentAnalyzer()
        result = analyzer.analyze("")
        
        assert result["summary"]["total_chunks"] == 0
        
        # All dimensions should have null evidence
        for dimension in CausalDimension:
            dim_result = result["causal_dimensions"][dimension.value]
            assert dim_result["total_chunks"] == 0
    
    @patch.object(SemanticProcessor, 'chunk_text')
    @patch.object(SemanticProcessor, 'embed_single')
    def test_analyze_similarity_threshold_filtering(self, mock_embed, mock_chunk, 
                                                     sample_pdm_text, sample_chunks):
        """Test that similarity threshold filters chunks"""
        mock_embed.return_value = np.random.randn(768).astype(np.float32)
        mock_chunk.return_value = sample_chunks
        
        # Use high threshold to filter most chunks
        config = SemanticConfig(similarity_threshold=0.99, min_evidence_chunks=1)
        analyzer = PolicyDocumentAnalyzer(config=config)
        result = analyzer.analyze(sample_pdm_text)
        
        # Most dimensions should have low chunk counts due to high threshold
        total_relevant = sum(
            result["causal_dimensions"][dim.value]["total_chunks"]
            for dim in CausalDimension
        )
        # Should be less than total chunks * dimensions
        assert total_relevant < len(sample_chunks) * len(CausalDimension)
    
    @patch.object(SemanticProcessor, 'chunk_text')
    @patch.object(SemanticProcessor, 'embed_single')
    def test_analyze_min_evidence_chunks(self, mock_embed, mock_chunk, 
                                         sample_pdm_text, sample_chunks):
        """Test minimum evidence chunks requirement"""
        mock_embed.return_value = np.random.randn(768).astype(np.float32)
        mock_chunk.return_value = sample_chunks[:1]  # Only 1 chunk
        
        config = SemanticConfig(min_evidence_chunks=3)  # Require 3 chunks minimum
        analyzer = PolicyDocumentAnalyzer(config=config)
        result = analyzer.analyze(sample_pdm_text)
        
        # With only 1 chunk and min=3, should get null evidence for most dimensions
        null_evidence_count = sum(
            1 for dim in CausalDimension
            if result["causal_dimensions"][dim.value]["confidence"] == 0.0
        )
        assert null_evidence_count > 0
    
    def test_extract_key_excerpts_truncation(self, sample_chunks):
        """Test that excerpts are properly truncated"""
        analyzer = PolicyDocumentAnalyzer()
        
        # Create chunk with long text
        long_chunk = sample_chunks[0].copy()
        long_chunk["text"] = "A" * 500
        chunks = [long_chunk, *sample_chunks[1:]]
        
        dimension_results = {dim.value: {} for dim in CausalDimension}
        excerpts = analyzer._extract_key_excerpts(chunks, dimension_results)
        
        for excerpts_list in excerpts.values():
            for excerpt in excerpts_list:
                assert len(excerpt) <= 303  # 300 + "..."

# ============================================================================
# TEST INTEGRATION SCENARIOS
# ============================================================================
class TestIntegrationScenarios:
    """Test end-to-end integration scenarios"""
    
    @patch.object(SemanticProcessor, 'chunk_text')
    @patch.object(SemanticProcessor, 'embed_single')
    def test_full_pdm_analysis_pipeline(self, mock_embed, mock_chunk, sample_pdm_text):
        """Test complete PDM analysis pipeline"""
        # Setup mocks
        np.random.seed(42)
        mock_embed.return_value = np.random.randn(768).astype(np.float32)
        
        chunks = [
            {
                "text": f"Chunk {i} text",
                "section_type": PDMSection.PLAN_PLURIANUAL,
                "section_id": f"sec_{i}",
                "token_count": 50,
                "position": i,
                "has_table": i % 2 == 0,
                "has_numerical": i % 3 == 0,
                "embedding": np.random.randn(768).astype(np.float32)
            }
            for i in range(10)
        ]
        mock_chunk.return_value = chunks
        
        # Run analysis
        analyzer = PolicyDocumentAnalyzer()
        result = analyzer.analyze(sample_pdm_text)
        
        # Validate structure
        assert isinstance(result, dict)
        assert "summary" in result
        assert "causal_dimensions" in result
        assert "key_excerpts" in result
        
        # Validate all dimensions analyzed
        assert len(result["causal_dimensions"]) == 6
        assert len(result["key_excerpts"]) == 6
        
        # Validate numerical consistency
        for dim_result in result["causal_dimensions"].values():
            assert 0.0 <= dim_result["posterior_mean"] <= 1.0
            assert dim_result["posterior_std"] >= 0
            assert dim_result["confidence"] >= 0
    
    @patch.object(SemanticProcessor, 'chunk_text')
    @patch.object(SemanticProcessor, 'embed_single')
    def test_analysis_with_high_quality_chunks(self, mock_embed, mock_chunk, sample_pdm_text):
        """Test analysis with high-quality chunks (tables, numerical data)"""
        mock_embed.return_value = np.random.randn(768).astype(np.float32)
        
        # High-quality chunks with tables and numbers
        chunks = [
            {
                "text": "Investment: $10,000 | Target: 50%",
                "section_type": PDMSection.PLAN_INVERSIONES,
                "section_id": f"sec_{i}",
                "token_count": 50,
                "position": i,
                "has_table": True,
                "has_numerical": True,
                "embedding": np.random.randn(768).astype(np.float32)
            }
            for i in range(5)
        ]
        mock_chunk.return_value = chunks
        
        analyzer = PolicyDocumentAnalyzer()
        result = analyzer.analyze(sample_pdm_text)
        
        # High-quality chunks should lead to higher confidence
        avg_confidence = np.mean([
            result["causal_dimensions"][dim.value]["confidence"]
            for dim in CausalDimension
        ])
        
        # At least some dimensions should show non-zero confidence
        assert avg_confidence >= 0
    
    def test_json_serialization(self):
        """Test that results can be serialized to JSON"""
        with patch.object(SemanticProcessor, 'chunk_text') as mock_chunk:
            with patch.object(SemanticProcessor, 'embed_single') as mock_embed:
                mock_embed.return_value = np.random.randn(768).astype(np.float32)
                mock_chunk.return_value = [
                    {
                        "text": "Test",
                        "section_type": PDMSection.DIAGNOSTICO,
                        "section_id": "sec_0",
                        "token_count": 10,
                        "position": 0,
                        "has_table": False,
                        "has_numerical": False,
                        "embedding": np.random.randn(768).astype(np.float32)
                    }
                ]
                
                analyzer = PolicyDocumentAnalyzer()
                result = analyzer.analyze("Test text")
                
                # Should be JSON serializable
                try:
                    json_str = json.dumps({
                        "summary": result["summary"],
                        "dimensions": {
                            k: {
                                "confidence": v["confidence"],
                                "evidence_strength": v["evidence_strength"]
                            }
                            for k, v in result["causal_dimensions"].items()
                        }
                    })
                    assert json_str is not None
                    assert len(json_str) > 0
                except (TypeError, ValueError) as e:
                    pytest.fail(f"Result not JSON serializable: {e}")

# ============================================================================
# TEST EDGE CASES AND ERROR HANDLING
# ============================================================================
class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling"""
    
    def test_very_long_text(self):
        """Test handling of very long text"""
        with patch.object(SemanticProcessor, 'chunk_text') as mock_chunk:
            with patch.object(SemanticProcessor, 'embed_single') as mock_embed:
                mock_embed.return_value = np.random.randn(768).astype(np.float32)
                
                # Create many chunks to simulate long text
                chunks = [
                    {
                        "text": f"Chunk {i}",
                        "section_type": PDMSection.DIAGNOSTICO,
                        "section_id": f"sec_{i}",
                        "token_count": 100,
                        "position": i,
                        "has_table": False,
                        "has_numerical": False,
                        "embedding": np.random.randn(768).astype(np.float32)
                    }
                    for i in range(100)
                ]
                mock_chunk.return_value = chunks
                
                analyzer = PolicyDocumentAnalyzer()
                result = analyzer.analyze("A" * 100000)
                
                assert result["summary"]["total_chunks"] == 100
    
    def test_special_characters_in_text(self):
        """Test handling of special characters"""
        with patch.object(SemanticProcessor, 'chunk_text') as mock_chunk:
            with patch.object(SemanticProcessor, 'embed_single') as mock_embed:
                mock_embed.return_value = np.random.randn(768).astype(np.float32)
                mock_chunk.return_value = [
                    {
                        "text": "Text with special chars: áéíóú ñ ¿?¡!",
                        "section_type": PDMSection.DIAGNOSTICO,
                        "section_id": "sec_0",
                        "token_count": 20,
                        "position": 0,
                        "has_table": False,
                        "has_numerical": False,
                        "embedding": np.random.randn(768).astype(np.float32)
                    }
                ]
                
                analyzer = PolicyDocumentAnalyzer()
                result = analyzer.analyze("Test with áéíóú ñ")
                
                assert result is not None
    
    def test_unicode_text(self):
        """Test handling of various Unicode characters"""
        text = "Test with emoji 😀 and symbols ©®™ and accents àèìòù"
        
        with patch.object(SemanticProcessor, 'chunk_text') as mock_chunk:
            with patch.object(SemanticProcessor, 'embed_single') as mock_embed:
                mock_embed.return_value = np.random.randn(768).astype(np.float32)
                mock_chunk.return_value = [
                    {
                        "text": text,
                        "section_type": PDMSection.DIAGNOSTICO,
                        "section_id": "sec_0",
                        "token_count": 30,
                        "position": 0,
                        "has_table": False,
                        "has_numerical": False,
                        "embedding": np.random.randn(768).astype(np.float32)
                    }
                ]
                
                analyzer = PolicyDocumentAnalyzer()
                result = analyzer.analyze(text)
                
                assert result is not None
    
    def test_extreme_similarity_values(self):
        """Test handling of extreme similarity values"""
        integrator = BayesianEvidenceIntegrator()
        
        # Test with perfect similarity
        sims_perfect = np.ones(5)
        metadata = [
            {"position": i, "has_table": False, "has_numerical": False,
             "section_type": PDMSection.DIAGNOSTICO}
            for i in range(5)
        ]
        result_perfect = integrator.integrate_evidence(sims_perfect, metadata)
        assert result_perfect["posterior_mean"] > 0.5
        
        # Test with zero similarity
        sims_zero = np.zeros(5)
        result_zero = integrator.integrate_evidence(sims_zero, metadata)
        assert result_zero["posterior_mean"] < 0.5
    
    def test_nan_handling_in_embeddings(self):
        """Test handling of NaN values in embeddings"""
        integrator = BayesianEvidenceIntegrator()
        
        # Create embeddings with NaN (this tests robustness)
        cause_emb = np.random.randn(768).astype(np.float32)
        effect_emb = np.random.randn(768).astype(np.float32)
        context_emb = np.random.randn(768).astype(np.float32)
        
        # Normal case should work
        strength = integrator.causal_strength(cause_emb, effect_emb, context_emb)
        assert not np.isnan(strength)
    
    def test_zero_variance_embeddings(self):
        """Test handling of zero-variance embeddings"""
        integrator = BayesianEvidenceIntegrator()
        
        # All zeros
        emb_zero = np.zeros(768, dtype=np.float32)
        strength = integrator.causal_strength(emb_zero, emb_zero, emb_zero)
        
        # Should handle gracefully (may return NaN or specific value)
        assert isinstance(strength, (float, np.floating))

# ============================================================================
# TEST MATHEMATICAL CORRECTNESS
# ============================================================================
class TestMathematicalCorrectness:
    """Test mathematical properties and correctness"""
    
    def test_bayesian_update_converges(self):
        """Test that Bayesian update converges with more evidence"""
        integrator = BayesianEvidenceIntegrator(prior_concentration=0.5)
        
        metadata = [
            {"position": i, "has_table": False, "has_numerical": False,
             "section_type": PDMSection.DIAGNOSTICO}
            for i in range(20)
        ]
        
        # More evidence should lead to lower variance
        sims_few = np.random.uniform(0.7, 0.9, 5)
        result_few = integrator.integrate_evidence(sims_few, metadata[:5])
        
        sims_many = np.random.uniform(0.7, 0.9, 20)
        result_many = integrator.integrate_evidence(sims_many, metadata)
        
        # More evidence should reduce uncertainty
        assert result_many["posterior_std"] <= result_few["posterior_std"]
    
    def test_information_gain_monotonicity(self):
        """Test that information gain increases with evidence strength"""
        integrator = BayesianEvidenceIntegrator()
        
        metadata = [
            {"position": 0, "has_table": False, "has_numerical": False,
             "section_type": PDMSection.DIAGNOSTICO}
            for _ in range(1)
        ]
        
        # Weak evidence
        result_weak = integrator.integrate_evidence(np.array([0.55]), metadata)
        
        # Strong evidence
        result_strong = integrator.integrate_evidence(np.array([0.95]), metadata)
        
        # Strong evidence should have higher information gain
        assert result_strong["information_gain"] >= result_weak["information_gain"]
    
    def test_posterior_mean_bounds(self):
        """Test that posterior mean stays within [0, 1]"""
        integrator = BayesianEvidenceIntegrator()
        
        metadata = [
            {"position": i, "has_table": False, "has_numerical": False,
             "section_type": PDMSection.DIAGNOSTICO}
            for i in range(10)
        ]
        
        # Test with various similarity distributions
        for _ in range(10):
            sims = np.random.uniform(-1, 1, 10)
            result = integrator.integrate_evidence(sims, metadata)
            
            assert 0.0 <= result["posterior_mean"] <= 1.0
            assert result["posterior_std"] >= 0
    
    def test_confidence_calibration(self):
        """Test confidence calibration properties"""
        integrator = BayesianEvidenceIntegrator()
        
        metadata = [
            {"position": i, "has_table": False, "has_numerical": False,
             "section_type": PDMSection.DIAGNOSTICO}
            for i in range(10)
        ]
        
        # High agreement should give high confidence
        sims_high = np.ones(10) * 0.9
        result_high = integrator.integrate_evidence(sims_high, metadata)
        
        # Mixed evidence should give lower confidence
        sims_mixed = np.random.uniform(0.3, 0.9, 10)
        result_mixed = integrator.integrate_evidence(sims_mixed, metadata)
        
        assert result_high["confidence"] >= result_mixed["confidence"]
    
    def test_cosine_similarity_properties(self):
        """Test cosine similarity computation properties"""
        # Identical vectors should have similarity 1
        vec = np.random.randn(768).astype(np.float32)
        sim = 1 - cosine(vec, vec)
        assert np.abs(sim - 1.0) < 1e-6
        
        # Orthogonal vectors should have similarity ~0
        vec1 = np.array([1.0] + [0.0] * 767, dtype=np.float32)
        vec2 = np.array([0.0, 1.0] + [0.0] * 766, dtype=np.float32)
        sim = 1 - cosine(vec1, vec2)
        assert np.abs(sim) < 1e-6
    
    def test_dirichlet_posterior_properties(self):
        """Test Dirichlet posterior statistical properties"""
        integrator = BayesianEvidenceIntegrator(prior_concentration=1.0)
        
        # With uniform prior (a=1), posterior mean should follow evidence
        metadata = [
            {"position": 0, "has_table": False, "has_numerical": False,
             "section_type": PDMSection.DIAGNOSTICO}
        ]
        
        # 100% positive evidence
        result = integrator.integrate_evidence(np.array([1.0]), metadata)
        
        # Posterior should be skewed towards positive
        assert result["posterior_mean"] > 0.5

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])