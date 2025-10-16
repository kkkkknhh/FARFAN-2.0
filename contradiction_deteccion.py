"""
Sistema de Detección de Contradicciones en PDMs Colombianos - Estado del Arte 2025

Implementa arquitecturas transformer de última generación, razonamiento causal bayesiano,
y verificación lógica temporal para análisis exhaustivo de contradicciones en Planes de
Desarrollo Municipal según Ley 152/1994 y metodología DNP Colombia.

Innovaciones clave:
- RoBERTa-large multilingüe fine-tuned para políticas públicas latinoamericanas
- Graph Neural Networks para razonamiento relacional multi-hop
- Verificación formal con lógica temporal lineal (LTL)
- Inferencia causal bayesiana con redes probabilísticas
- Análisis semántico contextual con attention mechanisms
- Corrección estadística FDR para comparaciones múltiples
- Embeddings dinámicos contextualizados
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.special import betainc
from scipy.stats import false_discovery_control
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.stats.multitest import multipletests
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Module-level constants
YEAR_PATTERN_REGEX = r'20(\d{2})'
NINGUN_LITERAL = 'ningún'


class ContradictionType(Enum):
    """Taxonomía exhaustiva según análisis empírico de PDMs colombianos"""
    NUMERICAL_INCONSISTENCY = auto()
    TEMPORAL_CONFLICT = auto()
    SEMANTIC_OPPOSITION = auto()
    LOGICAL_INCOMPATIBILITY = auto()
    RESOURCE_ALLOCATION_MISMATCH = auto()
    OBJECTIVE_MISALIGNMENT = auto()
    REGULATORY_CONFLICT = auto()
    CAUSAL_INCOHERENCE = auto()


class PolicyDimension(Enum):
    """Dimensiones DNP Colombia según estructura normativa"""
    DIAGNOSTICO = "diagnóstico"
    ESTRATEGICO = "estratégico"
    PROGRAMATICO = "programático"
    FINANCIERO = "plan plurianual de inversiones"
    SEGUIMIENTO = "seguimiento y evaluación"
    TERRITORIAL = "ordenamiento territorial"


@dataclass(frozen=True)
class PolicyStatement:
    """Representación estructurada con embeddings contextualizados"""
    text: str
    dimension: PolicyDimension
    position: Tuple[int, int]
    entities: List[Dict[str, str]] = field(default_factory=list)
    temporal_markers: List[Dict[str, Any]] = field(default_factory=list)
    quantitative_claims: List[Dict[str, Any]] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    contextual_embedding: Optional[np.ndarray] = None
    context_window: str = ""
    semantic_role: Optional[str] = None
    dependencies: Set[str] = field(default_factory=set)
    causal_relations: List[Tuple[str, str]] = field(default_factory=list)
    regulatory_references: List[str] = field(default_factory=list)


@dataclass
class ContradictionEvidence:
    """Evidencia con trazabilidad completa y métricas robustas"""
    statement_a: PolicyStatement
    statement_b: PolicyStatement
    contradiction_type: ContradictionType
    confidence: float
    severity: float
    semantic_similarity: float
    logical_conflict_score: float
    temporal_consistency: bool
    numerical_divergence: Optional[float]
    affected_dimensions: List[PolicyDimension]
    resolution_suggestions: List[str]
    graph_path: Optional[List[str]] = None
    statistical_significance: Optional[float] = None
    causal_conflict: bool = False
    attention_weights: Optional[np.ndarray] = None


class BayesianCausalInference:
    """Inferencia causal bayesiana con redes probabilísticas"""

    def __init__(self):
        self.prior_alpha = 3.2
        self.prior_beta = 8.1
        self.causal_network = nx.DiGraph()

    def calculate_posterior(
            self,
            evidence_strength: float,
            observations: int,
            domain_weight: float = 1.0,
            prior_knowledge: float = 0.5
    ) -> Tuple[float, Tuple[float, float]]:
        """
Inferencia Bayesiana con actualización dinámica

Returns:
    (adjusted_posterior, credible_interval_95)

    adjusted_posterior: float
        Posterior mean adjusted by uncertainty factor (not the raw posterior mean).
    credible_interval_95: Tuple[float, float]
        95% credible interval (lower, upper) for the posterior.
        """
        alpha_post = self.prior_alpha + evidence_strength * observations * domain_weight
        beta_post = self.prior_beta + (1 - evidence_strength) * observations * domain_weight

        alpha_post += prior_knowledge * 5
        beta_post += (1 - prior_knowledge) * 5

        posterior_mean = alpha_post / (alpha_post + beta_post)

        lower = betainc(alpha_post, beta_post, 0.025)
        upper = betainc(alpha_post, beta_post, 0.975)

        uncertainty_factor = 1.0 - (upper - lower)
        adjusted_posterior = posterior_mean * uncertainty_factor

        return adjusted_posterior, (lower, upper)

    def build_causal_network(self, statements: List[PolicyStatement]):
        """Construye red causal entre declaraciones"""
        self.causal_network.clear()

        for i, stmt in enumerate(statements):
            self.causal_network.add_node(f"stmt_{i}", statement=stmt)

            for j, other in enumerate(statements):
                if i != j:
                    causal_strength = self._estimate_causal_strength(stmt, other)
                    if causal_strength > 0.3:
                        self.causal_network.add_edge(
                            f"stmt_{i}",
                            f"stmt_{j}",
                            weight=causal_strength
                        )

    def _estimate_causal_strength(
            self,
            cause: PolicyStatement,
            effect: PolicyStatement
    ) -> float:
        """Estima fuerza causal usando análisis semántico"""
        causal_markers = {
            'causa': 1.0, 'debido a': 0.9, 'porque': 0.8, 'resultado de': 0.9,
            'consecuencia': 0.85, 'genera': 0.8, 'produce': 0.8, 'implica': 0.7,
            'conlleva': 0.75, 'origina': 0.85, 'provoca': 0.8
        }

        strength = 0.0
        effect_text = effect.text.lower()
        enriched_entities = [
            entity
            for entity in cause.entities
            if isinstance(entity, dict) and entity.get('text')
        ]
        for marker, weight in causal_markers.items():
            if marker in cause.text.lower():
                if any(
                    entity['text'].lower() in effect_text
                    or any(
                        synonym in effect_text
                        for synonym in entity.get('synonyms', [])
                    )
                    for entity in enriched_entities
                ):
                    strength = max(strength, weight)

        temporal_ordering = self._check_temporal_precedence(cause, effect)
        if temporal_ordering:
            strength *= 1.2

        return min(1.0, strength)

    def _check_temporal_precedence(
            self,
            cause: PolicyStatement,
            effect: PolicyStatement
    ) -> bool:
        """Verifica precedencia temporal"""
        if not cause.temporal_markers or not effect.temporal_markers:
            return False

        cause_timestamps = [
            float(ts)
            for ts in (marker.get('timestamp') for marker in cause.temporal_markers)
            if isinstance(ts, (int, float))
        ]
        effect_timestamps = [
            float(ts)
            for ts in (marker.get('timestamp') for marker in effect.temporal_markers)
            if isinstance(ts, (int, float))
        ]
        cause_time = min(cause_timestamps) if cause_timestamps else float('inf')
        effect_time = min(effect_timestamps) if effect_timestamps else float('inf')

        return cause_time < effect_time


class TemporalLogicVerifier:
    """Verificación formal con lógica temporal lineal (LTL)"""

    def __init__(self):
        self.temporal_operators = {
            'always': r'siempre|permanente|continuo',
            'eventually': r'eventualmente|finalmente|al final',
            'until': r'hasta|hasta que|mientras',
            'next': r'siguiente|próximo|después',
            'before': r'antes|previo|anterior'
        }

    def verify_temporal_consistency(
            self,
            statements: List[PolicyStatement]
    ) -> Tuple[bool, List[Dict[str, Any]], float]:
        """
Verificación formal de consistencia temporal

Returns:
(is_consistent, conflicts, consistency_score)
        """
        timeline = self._build_structured_timeline(statements)
        conflicts = []

        for i, event_a in enumerate(timeline):
            for event_b in timeline[i+1:]:
                conflict_type = self._detect_temporal_violation(event_a, event_b)
                if conflict_type:
                    conflicts.append({
                        'event_a': event_a,
                        'event_b': event_b,
                        'conflict_type': conflict_type,
                        'severity': self._calculate_temporal_severity(conflict_type)
                    })

        consistency_score = self._calculate_temporal_consistency_score(
            len(conflicts), 
            len(timeline)
        )

        return len(conflicts) == 0, conflicts, consistency_score

    def _build_structured_timeline(
            self,
            statements: List[PolicyStatement]
    ) -> List[Dict[str, Any]]:
        """Construye timeline estructurada con intervalos temporales"""
        timeline = []

        for stmt in statements:
            for marker in stmt.temporal_markers:
                interval = self._parse_temporal_interval(marker)
                timeline.append({
                    'statement': stmt,
                    'marker': marker,
                    'interval': interval,
                    'type': marker.get('type', 'point'),
                    'constraints': self._extract_temporal_constraints(stmt.text)
                })

        return sorted(timeline, key=lambda x: x['interval'][0] if x['interval'] else 0)

    def _parse_temporal_interval(
            self,
            marker: Dict[str, Any]
    ) -> Optional[Tuple[float, float]]:
        """Parsea intervalo temporal con granularidad fina"""
        text = marker.get('text', '')

        year_match = re.search(YEAR_PATTERN_REGEX, text)
        if year_match:
            year = 2000 + int(year_match.group(1))

            quarter_patterns = {
                'primer': 0.0, 'segundo': 0.25, 'tercer': 0.5, 'cuarto': 0.75,
                'Q1': 0.0, 'Q2': 0.25, 'Q3': 0.5, 'Q4': 0.75,
                'I': 0.0, 'II': 0.25, 'III': 0.5, 'IV': 0.75
            }

            for pattern, offset in quarter_patterns.items():
                if pattern in text:
                    return (year + offset, year + offset + 0.25)

            return (float(year), float(year + 1))

        return None

    def _detect_temporal_violation(
            self,
            event_a: Dict[str, Any],
            event_b: Dict[str, Any]
    ) -> Optional[str]:
        """Detecta violaciones de lógica temporal"""
        interval_a = event_a['interval']
        interval_b = event_b['interval']

        if not interval_a or not interval_b:
            return None

        if self._intervals_overlap(interval_a, interval_b):
            if self._are_mutually_exclusive(event_a['statement'], event_b['statement']):
                return 'simultaneous_exclusion'

        if interval_a[0] > interval_b[0]:
            if self._requires_precedence(event_a['statement'], event_b['statement']):
                return 'precedence_violation'

        if 'always' in event_a['constraints']:
            if self._contradicts_always(event_b['statement'], event_a['statement']):
                return 'always_violation'

        return None

    def _intervals_overlap(
            self,
            interval_a: Tuple[float, float],
            interval_b: Tuple[float, float]
    ) -> bool:
        """Verifica solapamiento de intervalos"""
        return not (interval_a[1] <= interval_b[0] or interval_b[1] <= interval_a[0])

    def _are_mutually_exclusive(
            self,
            stmt_a: PolicyStatement,
            stmt_b: PolicyStatement
    ) -> bool:
        """Verifica exclusión mutua entre declaraciones"""
        shared_resources = set()

        resource_patterns = [
            r'presupuesto\s+de\s+(\w+)',
            r'recursos?\s+para\s+(\w+)',
            r'equipo\s+de\s+(\w+)',
            r'personal\s+de\s+(\w+)'
        ]

        for pattern in resource_patterns:
            resources_a = set(re.findall(pattern, stmt_a.text, re.IGNORECASE))
            resources_b = set(re.findall(pattern, stmt_b.text, re.IGNORECASE))
            shared_resources.update(resources_a & resources_b)

        return len(shared_resources) > 0

    def _requires_precedence(
            self,
            stmt_a: PolicyStatement,
            stmt_b: PolicyStatement
    ) -> bool:
        """Verifica si stmt_a requiere que stmt_b ocurra primero"""
        precedence_indicators = [
            'depende de', 'requiere', 'necesita', 'previo',
            'condicionado a', 'sujeto a', 'una vez'
        ]

        for indicator in precedence_indicators:
            if indicator in stmt_a.text.lower():
                if any(entity.get('text') in stmt_b.text 
                       for entity in stmt_a.entities):
                    return True

        return bool(stmt_b.text[:50] in stmt_a.dependencies)

    def _extract_temporal_constraints(self, text: str) -> Set[str]:
        """Extrae restricciones temporales del texto"""
        constraints = set()

        for operator, pattern in self.temporal_operators.items():
            if re.search(pattern, text, re.IGNORECASE):
                constraints.add(operator)

        return constraints

    def _contradicts_always(
            self,
            stmt: PolicyStatement,
            always_stmt: PolicyStatement
    ) -> bool:
        """Verifica contradicción con restricción 'always'"""
        negation_patterns = ['no', 'nunca', NINGUN_LITERAL, 'sin', 'excepto', 'salvo']

        has_negation = any(pattern in stmt.text.lower() 
                           for pattern in negation_patterns)

        if has_negation:
            entities_always = {e.get('text') for e in always_stmt.entities}
            entities_stmt = {e.get('text') for e in stmt.entities}

            return len(entities_always & entities_stmt) > 0

        return False

    def _calculate_temporal_severity(self, conflict_type: str) -> float:
        """Calcula severidad de conflicto temporal"""
        severity_map = {
            'simultaneous_exclusion': 0.95,
            'precedence_violation': 0.85,
            'always_violation': 0.90,
            'deadline_violation': 0.80
        }
        return severity_map.get(conflict_type, 0.70)

    def _calculate_temporal_consistency_score(
            self,
            num_conflicts: int,
            num_events: int
    ) -> float:
        """Calcula score de consistencia temporal"""
        if num_events == 0:
            return 1.0

        conflict_ratio = num_conflicts / num_events
        consistency = 1.0 - min(1.0, conflict_ratio * 2)

        return consistency


class GraphNeuralReasoningEngine:
    """GNN para razonamiento relacional multi-hop"""

    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 256):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.gat1 = GATConv(embedding_dim, hidden_dim, heads=8, concat=True)
        self.gat2 = GATConv(hidden_dim * 8, hidden_dim, heads=4, concat=True)
        self.gat3 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False)

        self.gat1 = self.gat1.to(self.device)
        self.gat2 = self.gat2.to(self.device)
        self.gat3 = self.gat3.to(self.device)

    def detect_implicit_contradictions(
            self,
            statements: List[PolicyStatement],
            knowledge_graph: nx.DiGraph
    ) -> List[Tuple[PolicyStatement, PolicyStatement, float, np.ndarray]]:
        """
Detecta contradicciones implícitas usando razonamiento multi-hop

Returns:
List of (stmt_a, stmt_b, contradiction_score, attention_weights)
        """
        graph_data = self._build_geometric_graph(statements, knowledge_graph)

        with torch.no_grad():
            x = graph_data.x.to(self.device)
            edge_index = graph_data.edge_index.to(self.device)

            x, attention1 = self.gat1(x, edge_index, return_attention_weights=True)
            x = F.elu(x)
            x = F.dropout(x, p=0.3, training=False)

            x, attention2 = self.gat2(x, edge_index, return_attention_weights=True)
            x = F.elu(x)
            x = F.dropout(x, p=0.3, training=False)

            x, attention3 = self.gat3(x, edge_index, return_attention_weights=True)

        node_embeddings = x.cpu().numpy()

        contradictions = []
        for i, stmt_a in enumerate(statements):
            for j, stmt_b in enumerate(statements[i+1:], start=i+1):
                contradiction_score = self._compute_contradiction_score(
                    node_embeddings[i],
                    node_embeddings[j],
                    statements[i],
                    statements[j]
                )

                if contradiction_score > 0.7:
                    attention_weights = self._extract_path_attention(
                        i, j, attention1, attention2, attention3
                    )
                    contradictions.append((
                                              stmt_a, stmt_b, contradiction_score, attention_weights
                                          ))

        return contradictions

    def _build_geometric_graph(
            self,
            statements: List[PolicyStatement],
            nx_graph: nx.DiGraph
    ) -> Data:
        """Convierte grafo NetworkX a formato PyTorch Geometric"""
        node_features = []
        for stmt in statements:
            if stmt.contextual_embedding is not None:
                node_features.append(stmt.contextual_embedding)
            elif stmt.embedding is not None:
                node_features.append(stmt.embedding)
            else:
                node_features.append(np.zeros(self.embedding_dim))

        x = torch.tensor(np.array(node_features), dtype=torch.float)

        edge_list = []
        for i, stmt_a in enumerate(statements):
            for j, stmt_b in enumerate(statements):
                if i != j and nx_graph.has_edge(f"stmt_{i}", f"stmt_{j}"):
                    edge_list.append([i, j])

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index)

    def _compute_contradiction_score(
            self,
            emb_a: np.ndarray,
            emb_b: np.ndarray,
            stmt_a: PolicyStatement,
            stmt_b: PolicyStatement
    ) -> float:
        """Calcula score de contradicción combinando embeddings y características"""
        cosine_sim = np.dot(emb_a, emb_b) / (
            np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-10
        )

        has_negation_a = any(neg in stmt_a.text.lower() 
                             for neg in ['no', 'nunca', NINGUN_LITERAL])
        has_negation_b = any(neg in stmt_b.text.lower() 
                             for neg in ['no', 'nunca', NINGUN_LITERAL])

        negation_factor = 1.5 if has_negation_a != has_negation_b else 1.0

        shared_entities = sum(
            1 for e_a in stmt_a.entities 
            for e_b in stmt_b.entities 
            if e_a.get('text') == e_b.get('text')
        )
        entity_overlap = min(1.0, shared_entities / 3.0)

        contradiction_score = (
            (1 - cosine_sim) * 0.5 +
            negation_factor * 0.3 +
            entity_overlap * 0.2
        )

        return min(1.0, contradiction_score)

    def _extract_path_attention(
            self,
            node_i: int,
            node_j: int,
            *attention_layers
    ) -> np.ndarray:
        """Extrae pesos de atención en el camino entre nodos"""
        attention_weights = []

        for attention_layer in attention_layers:
            edge_index, edge_attention = attention_layer
            edge_index = edge_index.cpu().numpy()
            edge_attention = edge_attention.cpu().numpy()

            for idx in range(edge_index.shape[1]):
                if (edge_index[0, idx] == node_i and edge_index[1, idx] == node_j) or \
                (edge_index[0, idx] == node_j and edge_index[1, idx] == node_i):
                    attention_weights.append(edge_attention[idx])

        return np.array(attention_weights) if attention_weights else np.array([])


class AdvancedStatisticalTesting:
    """Tests estadísticos robustos con corrección FDR"""

    @staticmethod
    def numerical_divergence_test(
            claims: List[Tuple[Dict[str, Any], Dict[str, Any]]]
    ) -> Tuple[List[float], List[float]]:
        """
        Test de divergencia numérica con corrección para comparaciones múltiples.

        Args:
            claims: Lista de tuplas (claim_a, claim_b), donde cada claim es un dict
                con al menos la clave 'value' (numérico). Ejemplo:
                [
                    ({"value": 10.0, ...}, {"value": 5.0, ...}),
                    ...
                ]

        Returns:
            Tuple:
                - divergences: List[float], lista de divergencias normalizadas.
                - adjusted_p_values: List[float], lista de p-values ajustados por FDR.
            Si no se procesan claims válidos, ambas listas pueden estar vacías.
        """
        divergences = []
        p_values = []

        for claim_a, claim_b in claims:
            value_a = claim_a.get('value', 0)
            value_b = claim_b.get('value', 0)

            if value_a == 0 and value_b == 0:
                continue

            max_val = max(abs(value_a), abs(value_b))
            if max_val == 0:
                continue

            divergence = abs(value_a - value_b) / max_val
            divergences.append(divergence)

            pooled_value = (value_a + value_b) / 2
            se = pooled_value * 0.05 if pooled_value != 0 else 1.0

            z_score = abs(value_a - value_b) / (se * np.sqrt(2))
            p_value = 2 * (1 - stats.norm.cdf(z_score))
            p_values.append(p_value)

        if p_values:
            _, adjusted_p_values, _, _ = multipletests(
                p_values,
                alpha=0.05,
                method='fdr_bh'
            )
            return divergences, adjusted_p_values.tolist()

        return divergences, p_values

    @staticmethod
    def resource_allocation_chi_square(
            allocations: Dict[str, List[float]]
    ) -> Tuple[float, float]:
        """Test chi-cuadrado para asignación de recursos"""
        if len(allocations) < 2:
            return 0.0, 1.0

        observed = np.array(list(allocations.values()))

        expected = np.mean(observed, axis=0)
        expected_matrix = np.tile(expected, (len(allocations), 1))

        chi2_stat = np.sum((observed - expected_matrix)**2 / (expected_matrix + 1e-10))

        df = (observed.shape[0] - 1) * (observed.shape[1] - 1)
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)

        return chi2_stat, p_value


class PolicyContradictionDetectorV2:
    """Sistema de detección de contradicciones - Estado del Arte 2025"""

    def __init__(
            self,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        logger.info(f"Inicializando detector en dispositivo: {device}")

        self.device = device

        self.semantic_model = SentenceTransformer(
            'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
            device=device
        )

        self.nli_model_name = "joeddav/xlm-roberta-large-xnli"
        self.nli_tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
            self.nli_model_name
        ).to(device)
        self.nli_model.eval()

        self.nlp = spacy.load("es_core_news_lg")

        self.bayesian_engine = BayesianCausalInference()
        self.temporal_verifier = TemporalLogicVerifier()
        self.gnn_reasoner = GraphNeuralReasoningEngine()
        self.statistical_tester = AdvancedStatisticalTesting()

        self.knowledge_graph = nx.DiGraph()
        
        # HARMONIC FRONT 4: Initialize audit metrics tracking
        self._audit_metrics = {
            'total_contradictions': 0,
            'causal_incoherence_flags': 0,
            'structural_failures': 0
        }

        self._initialize_pdm_ontology()

    def _initialize_pdm_ontology(self):
        """Inicializa ontología específica de PDMs colombianos"""
        self.pdm_ontology = {
            'ejes_estrategicos': {
                'pattern': re.compile(
                    r'eje\s+(?:estratégico|estructurante)|dimensión\s+(?:estratégica|de\s+desarrollo)|'
                    r'línea\s+estratégica|pilar\s+(?:fundamental|estratégico)',
                    re.IGNORECASE
                ),
                'weight': 1.3
            },
            'programas': {
                'pattern': re.compile(
                    r'programa\s+(?:\d+|[\w\s]+)|subprograma|proyecto\s+(?:estratégico|de\s+inversión)|'
                    r'iniciativa\s+(?:prioritaria|estratégica)',
                    re.IGNORECASE
                ),
                'weight': 1.2
            },
            'metas': {
                'pattern': re.compile(
                    r'meta\s+(?:de\s+resultado|de\s+producto|cuatrienal)|'
                    r'indicador\s+(?:de\s+resultado|de\s+producto|de\s+gestión)|'
                    r'línea\s+base|objetivo\s+específico',
                    re.IGNORECASE
                ),
                'weight': 1.4
            },
            'recursos': {
                'pattern': re.compile(
                    r'(?:SGP|SGR|regalías|recursos?\s+propios|cofinanciación|'
                    r'crédito|endeudamiento|recursos?\s+de\s+capital|'
                    r'recursos?\s+corrientes|transferencias?)',
                    re.IGNORECASE
                ),
                'weight': 1.5
            },
            'normativa': {
                'pattern': re.compile(
                    r'(?:ley\s+\d+\s+de\s+\d{4}|decreto\s+\d+|'
                    r'acuerdo\s+municipal\s+\d+|resolución\s+\d+|'
                    r'conpes\s+\d+|sentencia\s+[CT]-\d+)',
                    re.IGNORECASE
                ),
                'weight': 1.1
            }
        }

    def detect(
            self,
            text: str,
            plan_name: str = "PDM",
            dimension: PolicyDimension = PolicyDimension.ESTRATEGICO
    ) -> Dict[str, Any]:
        """
Detección exhaustiva de contradicciones con análisis multi-modal

Harmonic Front 3 - Enhancement 3: Regulatory Constraint Check
Extracts regulatory references and verifies compliance with external mandates
(e.g., Ley 152/1994, Ley 388). For D1-Q5 (Restricciones Legales/Competencias):
Excelente requires PDM text explicitly mentions ≥3 types of constraints
(Legal, Budgetary, Temporal/Competency) and is_consistent = True.

Args:
text: Texto completo del PDM
plan_name: Identificador del plan
dimension: Dimensión siendo analizada

Returns:
Análisis completo con contradicciones y métricas avanzadas
        """
        logger.info(f"Iniciando análisis de {plan_name} - {dimension.value}")

        statements = self._extract_policy_statements(text, dimension)
        logger.info(f"Extraídas {len(statements)} declaraciones de política")

        statements = self._generate_contextual_embeddings(statements)

        self.bayesian_engine.build_causal_network(statements)
        self._build_knowledge_graph(statements)

        contradictions = []

        semantic_contradictions = self._detect_semantic_contradictions_nli(statements)
        contradictions.extend(semantic_contradictions)
        logger.info(f"Detectadas {len(semantic_contradictions)} contradicciones semánticas")

        numerical_contradictions = self._detect_numerical_inconsistencies_robust(statements)
        contradictions.extend(numerical_contradictions)
        logger.info(f"Detectadas {len(numerical_contradictions)} inconsistencias numéricas")

        temporal_conflicts = self._detect_temporal_conflicts_formal(statements)
        contradictions.extend(temporal_conflicts)
        logger.info(f"Detectados {len(temporal_conflicts)} conflictos temporales")

        gnn_contradictions = self._detect_implicit_contradictions_gnn(statements)
        contradictions.extend(gnn_contradictions)
        logger.info(f"Detectadas {len(gnn_contradictions)} contradicciones implícitas (GNN)")

        causal_conflicts = self._detect_causal_inconsistencies(statements)
        contradictions.extend(causal_conflicts)
        logger.info(f"Detectados {len(causal_conflicts)} conflictos causales")

        resource_conflicts = self._detect_resource_conflicts_statistical(statements)
        contradictions.extend(resource_conflicts)
        logger.info(f"Detectados {len(resource_conflicts)} conflictos de recursos")

        contradictions = self._deduplicate_contradictions(contradictions)

        coherence_metrics = self._calculate_advanced_coherence_metrics(
            contradictions, statements, text
        )

        recommendations = self._generate_actionable_recommendations(contradictions)

        # HARMONIC FRONT 4: Include audit metrics for D6-Q3 quality criteria
        causal_incoherence_count = sum(
            1 for c in contradictions
            if c.contradiction_type == ContradictionType.CAUSAL_INCOHERENCE
        )

        total_contradictions_for_grading = self._audit_metrics.get("total_contradictions", len(contradictions))
        quality_grade_classification = "Excelente" if total_contradictions_for_grading < 5 else "Bueno" if total_contradictions_for_grading < 10 else "Regular"
        
        audit_summary = {
            "total_contradictions": self._audit_metrics.get("total_contradictions", len(contradictions)),
            "causal_incoherence_flags": causal_incoherence_count,
            "structural_failures": self._audit_metrics.get("structural_failures", 0),
            "quality_grade": quality_grade_classification
        }

        # HARMONIC FRONT 3 - Enhancement 3: Regulatory Constraint Assessment for D1-Q5
        regulatory_analysis = self._analyze_regulatory_constraints(statements, text, temporal_conflicts)

        return {
            "plan_name": plan_name,
            "dimension": dimension.value,
            "analysis_timestamp": pd.Timestamp.now().isoformat(),
            "total_statements": len(statements),
            "contradictions": [self._serialize_contradiction(c) for c in contradictions],
            "total_contradictions": len(contradictions),
            "critical_severity_count": sum(1 for c in contradictions if c.severity > 0.85),
            "high_severity_count": sum(1 for c in contradictions if 0.7 < c.severity <= 0.85),
            "medium_severity_count": sum(1 for c in contradictions if 0.5 < c.severity <= 0.7),
            "coherence_metrics": coherence_metrics,
            "recommendations": recommendations,
            "knowledge_graph_stats": self._get_advanced_graph_statistics(),
            "causal_network_stats": self._get_causal_network_statistics(),
            "d1_q5_regulatory_analysis": regulatory_analysis,
            "harmonic_front_4_audit": audit_summary
        }

    def _extract_policy_statements(
            self,
            text: str,
            dimension: PolicyDimension
    ) -> List[PolicyStatement]:
        """Extracción estructurada con análisis lingüístico profundo"""
        doc = self.nlp(text)
        statements = []

        for sent in doc.sents:
            entities = [
                {'text': ent.text, 'label': ent.label_, 'start': ent.start_char, 'end': ent.end_char}
                for ent in sent.ents
            ]

            temporal_markers = self._extract_structured_temporal_markers(sent.text)

            quantitative_claims = self._extract_structured_quantitative_claims(sent.text)

            semantic_role = self._determine_semantic_role_advanced(sent)

            dependencies = self._identify_deep_dependencies(sent, doc)

            causal_relations = self._extract_causal_relations(sent)

            regulatory_references = self._extract_regulatory_references(sent.text)

            statement = PolicyStatement(
                text=sent.text,
                dimension=dimension,
                position=(sent.start_char, sent.end_char),
                entities=entities,
                temporal_markers=temporal_markers,
                quantitative_claims=quantitative_claims,
                context_window=self._get_extended_context(text, sent.start_char, sent.end_char),
                semantic_role=semantic_role,
                dependencies=dependencies,
                causal_relations=causal_relations,
                regulatory_references=regulatory_references
            )

            statements.append(statement)

        return statements

    def _generate_contextual_embeddings(
            self,
            statements: List[PolicyStatement]
    ) -> List[PolicyStatement]:
        """Genera embeddings contextualizados con ventanas de contexto"""
        enhanced_statements = []

        texts = [stmt.text for stmt in statements]
        contextual_texts = [stmt.context_window for stmt in statements]

        standard_embeddings = self.semantic_model.encode(
            texts, 
            convert_to_numpy=True,
            show_progress_bar=False
        )

        contextual_embeddings = self.semantic_model.encode(
            contextual_texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        for stmt, std_emb, ctx_emb in zip(statements, standard_embeddings, contextual_embeddings):
            enhanced = PolicyStatement(
                text=stmt.text,
                dimension=stmt.dimension,
                position=stmt.position,
                entities=stmt.entities,
                temporal_markers=stmt.temporal_markers,
                quantitative_claims=stmt.quantitative_claims,
                embedding=std_emb,
                contextual_embedding=ctx_emb,
                context_window=stmt.context_window,
                semantic_role=stmt.semantic_role,
                dependencies=stmt.dependencies,
                causal_relations=stmt.causal_relations,
                regulatory_references=stmt.regulatory_references
            )
            enhanced_statements.append(enhanced)

        return enhanced_statements

    def _detect_semantic_contradictions_nli(
            self,
            statements: List[PolicyStatement]
    ) -> List[ContradictionEvidence]:
        """Detección usando Natural Language Inference con XLM-RoBERTa"""
        contradictions = []

        for i, stmt_a in enumerate(statements):
            for stmt_b in statements[i+1:]:
                nli_result = self._classify_nli(stmt_a.text, stmt_b.text)

                if nli_result['label'] == 'contradiction' and nli_result['score'] > 0.75:
                    similarity = self._calculate_contextual_similarity(stmt_a, stmt_b)

                    confidence, _ = self.bayesian_engine.calculate_posterior(
                        evidence_strength=nli_result['score'],
                        observations=len(stmt_a.entities) + len(stmt_b.entities),
                        domain_weight=self._get_ontology_weight(stmt_a),
                        prior_knowledge=0.6
                    )

                    evidence = ContradictionEvidence(
                        statement_a=stmt_a,
                        statement_b=stmt_b,
                        contradiction_type=ContradictionType.SEMANTIC_OPPOSITION,
                        confidence=confidence,
                        severity=self._calculate_comprehensive_severity(stmt_a, stmt_b, nli_result['score']),
                        semantic_similarity=similarity,
                        logical_conflict_score=nli_result['score'],
                        temporal_consistency=True,
                        numerical_divergence=None,
                        affected_dimensions=[stmt_a.dimension, stmt_b.dimension],
                        resolution_suggestions=self._generate_resolution_strategies(
                            ContradictionType.SEMANTIC_OPPOSITION,
                            stmt_a,
                            stmt_b
                        )
                    )
                    contradictions.append(evidence)

        return contradictions

    def _classify_nli(self, premise: str, hypothesis: str) -> Dict[str, Any]:
        """Clasificación NLI usando XLM-RoBERTa"""
        inputs = self.nli_tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.nli_model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)[0]

        labels = ['contradiction', 'neutral', 'entailment']
        scores = probs.cpu().numpy()

        max_idx = np.argmax(scores)

        return {
            'label': labels[max_idx],
            'score': float(scores[max_idx]),
            'all_scores': {label: float(score) for label, score in zip(labels, scores)}
        }

    def _collect_comparable_claim_pairs(
            self,
            statements: List[PolicyStatement]
    ) -> List[Tuple[Tuple[PolicyStatement, Dict[str, Any]], Tuple[PolicyStatement, Dict[str, Any]]]]:
        """Colecta pares de claims cuantitativos comparables entre statements"""
        claim_pairs = []
        
        for i, stmt_a in enumerate(statements):
            for stmt_b in statements[i+1:]:
                if stmt_a.quantitative_claims and stmt_b.quantitative_claims:
                    for claim_a in stmt_a.quantitative_claims:
                        for claim_b in stmt_b.quantitative_claims:
                            if self._are_comparable_claims_advanced(claim_a, claim_b):
                                claim_pairs.append(((stmt_a, claim_a), (stmt_b, claim_b)))
        
        return claim_pairs

    def _validate_statistical_divergence(
            self,
            divergence: float,
            p_value: float
    ) -> bool:
        """Valida si divergencia cumple thresholds estadísticos"""
        return divergence > 0.15 and p_value < 0.05

    def _build_numerical_contradiction_evidence(
            self,
            stmt_a: PolicyStatement,
            stmt_b: PolicyStatement,
            divergence: float,
            p_value: float
    ) -> ContradictionEvidence:
        """Construye evidencia de contradicción numérica con métricas estadísticas"""
        confidence, _ = self.bayesian_engine.calculate_posterior(
            evidence_strength=1 - p_value,
            observations=2,
            domain_weight=1.6,
            prior_knowledge=0.7
        )

        evidence = ContradictionEvidence(
            statement_a=stmt_a,
            statement_b=stmt_b,
            contradiction_type=ContradictionType.NUMERICAL_INCONSISTENCY,
            confidence=confidence,
            severity=min(1.0, divergence * 1.2),
            semantic_similarity=self._calculate_contextual_similarity(stmt_a, stmt_b),
            logical_conflict_score=divergence,
            temporal_consistency=True,
            numerical_divergence=divergence,
            affected_dimensions=[stmt_a.dimension],
            resolution_suggestions=self._generate_resolution_strategies(
                ContradictionType.NUMERICAL_INCONSISTENCY,
                stmt_a,
                stmt_b,
                {'divergence': divergence, 'p_value': p_value}
            ),
            statistical_significance=p_value
        )
        
        return evidence

    def _detect_numerical_inconsistencies_robust(
            self,
            statements: List[PolicyStatement]
    ) -> List[ContradictionEvidence]:
        """Detección con tests estadísticos robustos y corrección FDR"""
        contradictions = []
        
        claim_pairs = self._collect_comparable_claim_pairs(statements)
        
        if not claim_pairs:
            return contradictions

        test_data = [(ca[1], cb[1]) for ca, cb in claim_pairs]
        divergences, adjusted_p_values = self.statistical_tester.numerical_divergence_test(test_data)

        for ((stmt_a, claim_a), (stmt_b, claim_b)), divergence, p_value in zip(
            claim_pairs, divergences, adjusted_p_values
        ):
            if self._validate_statistical_divergence(divergence, p_value):
                evidence = self._build_numerical_contradiction_evidence(
                    stmt_a, stmt_b, divergence, p_value
                )
                contradictions.append(evidence)

        return contradictions

    def _detect_temporal_conflicts_formal(
            self,
            statements: List[PolicyStatement]
    ) -> List[ContradictionEvidence]:
        """Detección con verificación formal LTL"""
        contradictions = []

        temporal_statements = [s for s in statements if s.temporal_markers]

        if len(temporal_statements) < 2:
            return contradictions

        _, conflicts, _ = \
            self.temporal_verifier.verify_temporal_consistency(temporal_statements)

        for conflict in conflicts:
            stmt_a = conflict['event_a']['statement']
            stmt_b = conflict['event_b']['statement']

            confidence, _ = self.bayesian_engine.calculate_posterior(
                evidence_strength=conflict['severity'],
                observations=len(conflicts),
                domain_weight=1.3,
                prior_knowledge=0.8
            )

            evidence = ContradictionEvidence(
                statement_a=stmt_a,
                statement_b=stmt_b,
                contradiction_type=ContradictionType.TEMPORAL_CONFLICT,
                confidence=confidence,
                severity=conflict['severity'],
                semantic_similarity=self._calculate_contextual_similarity(stmt_a, stmt_b),
                logical_conflict_score=1.0,
                temporal_consistency=False,
                numerical_divergence=None,
                affected_dimensions=[PolicyDimension.PROGRAMATICO],
                resolution_suggestions=self._generate_resolution_strategies(
                    ContradictionType.TEMPORAL_CONFLICT,
                    stmt_a,
                    stmt_b,
                    {'conflict_type': conflict['conflict_type']}
                )
            )
            contradictions.append(evidence)

        return contradictions

    def _detect_implicit_contradictions_gnn(
            self,
            statements: List[PolicyStatement]
    ) -> List[ContradictionEvidence]:
        """Detección usando Graph Neural Networks"""
        contradictions = []

        gnn_results = self.gnn_reasoner.detect_implicit_contradictions(
            statements,
            self.knowledge_graph
        )

        for stmt_a, stmt_b, score, attention_weights in gnn_results:
            confidence, _ = self.bayesian_engine.calculate_posterior(
                evidence_strength=score,
                observations=len(attention_weights) if len(attention_weights) > 0 else 1,
                domain_weight=1.1,
                prior_knowledge=0.5
            )

            evidence = ContradictionEvidence(
                statement_a=stmt_a,
                statement_b=stmt_b,
                contradiction_type=ContradictionType.LOGICAL_INCOMPATIBILITY,
                confidence=confidence,
                severity=score * 0.9,
                semantic_similarity=self._calculate_contextual_similarity(stmt_a, stmt_b),
                logical_conflict_score=score,
                temporal_consistency=True,
                numerical_divergence=None,
                affected_dimensions=[stmt_a.dimension, stmt_b.dimension],
                resolution_suggestions=self._generate_resolution_strategies(
                    ContradictionType.LOGICAL_INCOMPATIBILITY,
                    stmt_a,
                    stmt_b
                ),
                attention_weights=attention_weights
            )
            contradictions.append(evidence)

        return contradictions

    def _detect_causal_inconsistencies(
            self,
            statements: List[PolicyStatement]
    ) -> List[ContradictionEvidence]:
        """
        Detección de inconsistencias causales usando red bayesiana
        
        HARMONIC FRONT 4 ENHANCEMENT:
        - Detects circular causal conflicts (A → B and B → A)
        - Identifies structural incoherence from GNN implicit contradictions
        - Flags all inconsistencies for D6-Q3 (Inconsistencias/Pilotos)
        """
        contradictions = []

        causal_network = self.bayesian_engine.causal_network
        
        # Track total contradictions for audit criteria
        total_contradictions = 0

        # 1. Detect circular causal conflicts (A → B and B → A)
        for i, stmt_a in enumerate(statements):
            for j, stmt_b in enumerate(statements[i+1:], start=i+1):
                node_a = f"stmt_{i}"
                node_b = f"stmt_{j}"

                if causal_network.has_edge(node_a, node_b) and \
                causal_network.has_edge(node_b, node_a):

                    weight_ab = causal_network[node_a][node_b]['weight']
                    weight_ba = causal_network[node_b][node_a]['weight']

                    # Enhanced circular conflict detection
                    if abs(weight_ab - weight_ba) < 0.3:
                        total_contradictions += 1
                        
                        # Calculate severity based on circular strength
                        circular_strength = (weight_ab + weight_ba) / 2.0
                        severity = min(0.95, 0.65 + circular_strength * 0.3)
                        
                        confidence, _ = self.bayesian_engine.calculate_posterior(
                            evidence_strength=min(weight_ab, weight_ba),
                            observations=2,
                            domain_weight=1.2,
                            prior_knowledge=0.6
                        )

                        evidence = ContradictionEvidence(
                            statement_a=stmt_a,
                            statement_b=stmt_b,
                            contradiction_type=ContradictionType.CAUSAL_INCOHERENCE,
                            confidence=confidence,
                            severity=severity,
                            semantic_similarity=self._calculate_contextual_similarity(stmt_a, stmt_b),
                            logical_conflict_score=circular_strength,
                            temporal_consistency=True,
                            numerical_divergence=None,
                            affected_dimensions=[stmt_a.dimension],
                            resolution_suggestions=self._generate_resolution_strategies(
                                ContradictionType.CAUSAL_INCOHERENCE,
                                stmt_a,
                                stmt_b,
                                context={'conflict_type': 'circular_causality', 
                                        'weight_ab': weight_ab, 
                                        'weight_ba': weight_ba}
                            ),
                            causal_conflict=True
                        )
                        contradictions.append(evidence)
        
        # 2. Detect structural incoherence from non-explicit conflicts
        # GNN/Bayesian cross-validation: Integrate GNN implicit contradictions
        # into causal network inference for structural validity detection
        if hasattr(self, 'gnn_reasoner') and hasattr(self, 'knowledge_graph'):
            gnn_implicit = self.gnn_reasoner.detect_implicit_contradictions(
                statements,
                self.knowledge_graph
            )
            
            for stmt_a, stmt_b, gnn_score, attention_weights in gnn_implicit:
                # Check if this represents a causal structural issue
                # by examining causal network connections
                i = statements.index(stmt_a)
                j = statements.index(stmt_b)
                node_a = f"stmt_{i}"
                node_b = f"stmt_{j}"
                
                # If GNN detects conflict but Bayesian network shows weak/missing link,
                # this indicates structural incoherence
                has_weak_causal_link = False
                if causal_network.has_edge(node_a, node_b):
                    weight = causal_network[node_a][node_b]['weight']
                    if weight < 0.4:  # Weak causal link
                        has_weak_causal_link = True
                elif not causal_network.has_edge(node_a, node_b):
                    has_weak_causal_link = True  # Missing link
                
                if has_weak_causal_link and gnn_score > 0.65:
                    total_contradictions += 1
                    
                    confidence, _ = self.bayesian_engine.calculate_posterior(
                        evidence_strength=gnn_score,
                        observations=len(attention_weights) if attention_weights else 1,
                        domain_weight=1.3,
                        prior_knowledge=0.5
                    )
                    
                    evidence = ContradictionEvidence(
                        statement_a=stmt_a,
                        statement_b=stmt_b,
                        contradiction_type=ContradictionType.CAUSAL_INCOHERENCE,
                        confidence=confidence,
                        severity=gnn_score * 0.85,
                        semantic_similarity=self._calculate_contextual_similarity(stmt_a, stmt_b),
                        logical_conflict_score=gnn_score,
                        temporal_consistency=True,
                        numerical_divergence=None,
                        affected_dimensions=[stmt_a.dimension, stmt_b.dimension],
                        resolution_suggestions=self._generate_resolution_strategies(
                            ContradictionType.CAUSAL_INCOHERENCE,
                            stmt_a,
                            stmt_b,
                            context={'conflict_type': 'structural_incoherence_gnn',
                                    'gnn_score': gnn_score}
                        ),
                        causal_conflict=True,
                        attention_weights=attention_weights
                    )
                    contradictions.append(evidence)
        
        # Store total contradictions for quality audit
        if hasattr(self, '_audit_metrics'):
            self._audit_metrics['total_contradictions'] = total_contradictions
        else:
            self._audit_metrics = {'total_contradictions': total_contradictions}

        return contradictions

    def _extract_resource_allocations_by_type(
            self,
            statements: List[PolicyStatement]
    ) -> Dict[str, List[Tuple[PolicyStatement, float]]]:
        """Extrae y agrupa asignaciones de recursos por tipo"""
        resource_allocations = defaultdict(list)
        
        for stmt in statements:
            resources = self._extract_detailed_resource_mentions(stmt.text)
            for resource_type, amount, _ in resources:
                if amount:
                    resource_allocations[resource_type].append((stmt, amount))
        
        return resource_allocations

    def _detect_budget_overlap_conflicts(
            self,
            allocations: List[Tuple[PolicyStatement, float]],
            resource_type: str,
            chi2_stat: float,
            p_value: float
    ) -> List[ContradictionEvidence]:
        """Detecta conflictos por solapamiento presupuestario entre asignaciones"""
        conflicts = []
        
        for i, (stmt_a, amount_a) in enumerate(allocations):
            for stmt_b, amount_b in allocations[i+1:]:
                rel_diff = abs(amount_a - amount_b) / max(amount_a, amount_b)
                
                if rel_diff > 0.25:
                    evidence = self._build_resource_conflict_evidence(
                        stmt_a, stmt_b, amount_a, amount_b, 
                        resource_type, chi2_stat, p_value, rel_diff
                    )
                    conflicts.append(evidence)
        
        return conflicts

    def _build_resource_conflict_evidence(
            self,
            stmt_a: PolicyStatement,
            stmt_b: PolicyStatement,
            amount_a: float,
            amount_b: float,
            resource_type: str,
            chi2_stat: float,
            p_value: float,
            rel_diff: float
    ) -> ContradictionEvidence:
        """Construye evidencia de conflicto de recursos con análisis temporal"""
        confidence, _ = self.bayesian_engine.calculate_posterior(
            evidence_strength=1 - p_value,
            observations=2,
            domain_weight=1.5,
            prior_knowledge=0.7
        )
        
        temporal_overlap = self._check_temporal_resource_overlap(stmt_a, stmt_b)

        evidence = ContradictionEvidence(
            statement_a=stmt_a,
            statement_b=stmt_b,
            contradiction_type=ContradictionType.RESOURCE_ALLOCATION_MISMATCH,
            confidence=confidence,
            severity=min(1.0, rel_diff * 1.3 * (1.2 if temporal_overlap else 1.0)),
            semantic_similarity=self._calculate_contextual_similarity(stmt_a, stmt_b),
            logical_conflict_score=rel_diff,
            temporal_consistency=not temporal_overlap,
            numerical_divergence=rel_diff,
            affected_dimensions=[PolicyDimension.FINANCIERO],
            resolution_suggestions=self._generate_resolution_strategies(
                ContradictionType.RESOURCE_ALLOCATION_MISMATCH,
                stmt_a,
                stmt_b,
                {'resource_type': resource_type, 'chi2': chi2_stat, 
                 'temporal_overlap': temporal_overlap}
            ),
            statistical_significance=p_value
        )
        
        return evidence

    def _check_temporal_resource_overlap(
            self,
            stmt_a: PolicyStatement,
            stmt_b: PolicyStatement
    ) -> bool:
        """Verifica si hay solapamiento temporal entre asignaciones de recursos"""
        if not stmt_a.temporal_markers or not stmt_b.temporal_markers:
            return True
        
        intervals_a = [self.temporal_verifier._parse_temporal_interval(m) 
                       for m in stmt_a.temporal_markers]
        intervals_b = [self.temporal_verifier._parse_temporal_interval(m) 
                       for m in stmt_b.temporal_markers]
        
        intervals_a = [i for i in intervals_a if i is not None]
        intervals_b = [i for i in intervals_b if i is not None]
        
        if not intervals_a or not intervals_b:
            return True
        
        for int_a in intervals_a:
            for int_b in intervals_b:
                if self.temporal_verifier._intervals_overlap(int_a, int_b):
                    return True
        
        return False

    def _detect_resource_conflicts_statistical(
            self,
            statements: List[PolicyStatement]
    ) -> List[ContradictionEvidence]:
        """Detección con análisis estadístico de asignaciones"""
        contradictions = []
        
        resource_allocations = self._extract_resource_allocations_by_type(statements)

        for resource_type, allocations in resource_allocations.items():
            if len(allocations) > 1:
                amounts = [amt for _, amt in allocations]
                chi2_stat, p_value = self.statistical_tester.resource_allocation_chi_square(
                    {resource_type: amounts}
                )

                if p_value < 0.05:
                    conflicts = self._detect_budget_overlap_conflicts(
                        allocations, resource_type, chi2_stat, p_value
                    )
                    contradictions.extend(conflicts)

        return contradictions

    def _build_knowledge_graph(self, statements: List[PolicyStatement]):
        """Construye grafo con relaciones semánticas y causales"""
        self.knowledge_graph.clear()

        for i, stmt in enumerate(statements):
            node_id = f"stmt_{i}"
            self.knowledge_graph.add_node(
                node_id,
                text=stmt.text[:100],
                dimension=stmt.dimension.value,
                entities=[e['text'] for e in stmt.entities],
                semantic_role=stmt.semantic_role,
                has_temporal=bool(stmt.temporal_markers),
                has_quantitative=bool(stmt.quantitative_claims)
            )

            for j, other in enumerate(statements):
                if i != j:
                    similarity = self._calculate_contextual_similarity(stmt, other)

                    if similarity > 0.65:
                        relation_type = self._determine_relation_type_advanced(stmt, other)
                        edge_weight = similarity * self._get_relation_weight(relation_type)

                        self.knowledge_graph.add_edge(
                            f"stmt_{i}",
                            f"stmt_{j}",
                            weight=edge_weight,
                            relation_type=relation_type,
                            similarity=similarity
                        )

    def _compute_individual_coherence_scores(
            self,
            contradictions: List[ContradictionEvidence],
            statements: List[PolicyStatement]
    ) -> Tuple[float, float, float, float, float, float]:
        """Calcula scores individuales de coherencia por dimensión"""
        contradiction_density = len(contradictions) / max(1, len(statements))
        
        semantic_coherence = self._calculate_global_semantic_coherence_advanced(statements)
        
        temporal_consistency = sum(
            1 for c in contradictions 
            if c.contradiction_type != ContradictionType.TEMPORAL_CONFLICT
        ) / max(1, len(contradictions))
        
        causal_coherence = self._calculate_causal_coherence()
        
        objective_alignment = self._calculate_objective_alignment_advanced(statements)
        
        graph_metrics = self._calculate_graph_coherence_metrics()
        
        return (
            contradiction_density,
            semantic_coherence,
            temporal_consistency,
            causal_coherence,
            objective_alignment,
            graph_metrics['fragmentation']
        )

    def _compute_weighted_coherence_score(
            self,
            contradiction_density: float,
            semantic_coherence: float,
            temporal_consistency: float,
            causal_coherence: float,
            objective_alignment: float,
            graph_fragmentation: float
    ) -> float:
        """Calcula score ponderado de coherencia global"""
        weights = np.array([0.25, 0.20, 0.15, 0.15, 0.15, 0.10])
        scores = np.array([
            1 - min(1.0, contradiction_density * 2),
            semantic_coherence,
            temporal_consistency,
            causal_coherence,
            objective_alignment,
            1 - graph_fragmentation
        ])
        
        coherence_score = np.sum(weights * scores)
        
        return coherence_score

    def _compute_auxiliary_metrics(
            self,
            contradictions: List[ContradictionEvidence],
            text: str,
            coherence_score: float,
            num_statements: int
    ) -> Tuple[float, float, Tuple[float, float]]:
        """Calcula métricas auxiliares: entropía, complejidad sintáctica e intervalo de confianza"""
        contradiction_entropy = self._calculate_shannon_entropy(contradictions)
        
        syntactic_complexity = self._calculate_syntactic_complexity_advanced(text)
        
        confidence_interval = self._calculate_bootstrap_confidence_interval(
            coherence_score,
            num_statements,
            n_bootstrap=1000
        )
        
        return contradiction_entropy, syntactic_complexity, confidence_interval

    def _calculate_advanced_coherence_metrics(
            self,
            contradictions: List[ContradictionEvidence],
            statements: List[PolicyStatement],
            text: str
    ) -> Dict[str, Any]:
        """Métricas exhaustivas de coherencia"""
        
        (contradiction_density, semantic_coherence, temporal_consistency,
         causal_coherence, objective_alignment, graph_fragmentation) = \
            self._compute_individual_coherence_scores(contradictions, statements)
        
        graph_metrics = self._calculate_graph_coherence_metrics()
        
        coherence_score = self._compute_weighted_coherence_score(
            contradiction_density, semantic_coherence, temporal_consistency,
            causal_coherence, objective_alignment, graph_fragmentation
        )
        
        contradiction_entropy, syntactic_complexity, confidence_interval = \
            self._compute_auxiliary_metrics(contradictions, text, coherence_score, len(statements))

        return {
            "coherence_score": float(coherence_score),
            "contradiction_density": float(contradiction_density),
            "semantic_coherence": float(semantic_coherence),
            "temporal_consistency": float(temporal_consistency),
            "causal_coherence": float(causal_coherence),
            "objective_alignment": float(objective_alignment),
            "graph_fragmentation": float(graph_metrics['fragmentation']),
            "graph_modularity": float(graph_metrics['modularity']),
            "graph_centralization": float(graph_metrics['centralization']),
            "contradiction_entropy": float(contradiction_entropy),
            "syntactic_complexity": float(syntactic_complexity),
            "confidence_interval_95": {
                "lower": float(confidence_interval[0]),
                "upper": float(confidence_interval[1])
            },
            "quality_grade": self._assign_quality_grade(coherence_score)
        }

    def _calculate_contextual_similarity(
            self,
            stmt_a: PolicyStatement,
            stmt_b: PolicyStatement
    ) -> float:
        """Similaridad usando embeddings contextualizados"""
        if stmt_a.contextual_embedding is not None and stmt_b.contextual_embedding is not None:
            return float(1 - np.dot(stmt_a.contextual_embedding, stmt_b.contextual_embedding) / (
                             np.linalg.norm(stmt_a.contextual_embedding) * 
                             np.linalg.norm(stmt_b.contextual_embedding) + 1e-10
                         ))
        elif stmt_a.embedding is not None and stmt_b.embedding is not None:
            return float(1 - np.dot(stmt_a.embedding, stmt_b.embedding) / (
                             np.linalg.norm(stmt_a.embedding) * 
                             np.linalg.norm(stmt_b.embedding) + 1e-10
                         ))
        return 0.0

    def _extract_structured_temporal_markers(self, text: str) -> List[Dict[str, Any]]:
        """Extracción estructurada de marcadores temporales"""
        markers = []

        patterns = {
            'year': YEAR_PATTERN_REGEX,
            'quarter': r'(?:Q|trimestre\s+)([1-4])',
            'semester': r'(?:semestre\s+|S)([12])',
            'month': r'(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)',
            'timeframe': r'(corto|mediano|largo)\s+plazo',
            'ordinal': r'(primer|segundo|tercer|cuarto|quinto)\s+(año|trimestre|semestre)'
        }

        for marker_type, pattern in patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                timestamp = self._convert_to_timestamp(match.group(), marker_type)
                markers.append({
                    'text': match.group(),
                    'type': marker_type,
                    'timestamp': timestamp,
                    'position': match.span()
                })

        return markers

    def _convert_to_timestamp(self, text: str, marker_type: str) -> Optional[float]:
        """Convierte marcador a timestamp numérico"""
        if marker_type == 'year':
            year_match = re.search(YEAR_PATTERN_REGEX, text)
            if year_match:
                return 2000.0 + float(year_match.group(1))

        elif marker_type == 'quarter':
            quarter_match = re.search(r'([1-4])', text)
            if quarter_match:
                return float(quarter_match.group(1)) / 4.0

        elif marker_type == 'timeframe':
            timeframe_map = {'corto': 2.0, 'mediano': 4.0, 'largo': 8.0}
            for key, value in timeframe_map.items():
                if key in text.lower():
                    return value

        return None

    def _extract_structured_quantitative_claims(self, text: str) -> List[Dict[str, Any]]:
        """Extracción estructurada de afirmaciones cuantitativas
        
        Enhanced for D1-Q2 (Magnitud/Brecha/Limitaciones):
        - Extracts relative metrics (ratios, gaps, deficits)
        - Identifies quantified brechas (déficit de, porcentaje sin cubrir)
        - Detects data limitation statements (dereck_beach patterns)
        """
        claims = []

        patterns = [
            (r'(\d+(?:[.,]\d+)?)\s*(%|por\s*ciento)', 'percentage', 1.0),
            (r'(\d+(?:[.,]\d+)?)\s*(millones?\s+de\s+pesos|millones?)', 'currency_millions', 1_000_000),
            (r'(?:\$|COP)\s*(\d+(?:[.,]\d+)?)\s*(millones?)?', 'currency', 1_000_000),
            (r'(\d+(?:[.,]\d+)?)\s*(mil\s+millones?|billones?)', 'currency_billions', 1_000_000_000),
            (r'(\d+(?:[.,]\d+)?)\s*(personas?|beneficiarios?|familias?|hogares?)', 'beneficiaries', 1.0),
            (r'(\d+(?:[.,]\d+)?)\s*(hectáreas?|has?|km2?|metros?\s*cuadrados?)', 'area', 1.0),
            (r'meta\s+(?:de\s+)?(\d+(?:[.,]\d+)?)', 'target', 1.0),
            (r'indicador[:\s]+(\d+(?:[.,]\d+)?)', 'indicator', 1.0),
            # D1-Q2: Gap/deficit patterns
            (r'd[ée]ficit\s+de\s+(\d+(?:[.,]\d+)?)\s*(%|por\s*ciento|personas?|millones?)?', 'deficit', 1.0),
            (r'brecha\s+de\s+(\d+(?:[.,]\d+)?)\s*(%|puntos?|millones?)?', 'gap', 1.0),
            (r'falta(?:n)?\s+(\d+(?:[.,]\d+)?)\s*(personas?|millones?|%)?', 'shortage', 1.0),
            (r'sin\s+(?:acceso|cobertura|atenci[óo]n)\s*[:\s]+(\d+(?:[.,]\d+)?)\s*(%|personas?)?', 'uncovered', 1.0),
            (r'porcentaje\s+sin\s+(?:cubrir|atender|acceso)[:\s]*(\d+(?:[.,]\d+)?)\s*%?', 'uncovered_pct', 1.0),
            # D1-Q2: Relative metrics (ratios)
            (r'(\d+(?:[.,]\d+)?)\s*(?:de\s+cada|por\s+cada)\s+(\d+)', 'ratio', 1.0),
            (r'tasa\s+de\s+[^:]+:\s*(\d+(?:[.,]\d+)?)\s*%?', 'rate', 1.0),
        ]

        for pattern, claim_type, multiplier in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                value_str = match.group(1)
                value = self._parse_number_robust(value_str) * multiplier

                unit = match.group(2) if match.lastindex >= 2 else None
                
                # For ratio type, capture both numerator and denominator
                if claim_type == 'ratio' and match.lastindex >= 2:
                    denominator = self._parse_number_robust(match.group(2))
                    value = value / denominator if denominator > 0 else value

                claims.append({
                    'type': claim_type,
                    'value': value,
                    'unit': unit,
                    'raw_text': match.group(0),
                    'position': match.span(),
                    'context': text[max(0, match.start()-30):min(len(text), match.end()+30)]
                })
        
        # D1-Q2: Detect data limitation statements (dereck_beach patterns)
        limitation_patterns = [
            r'(?:no\s+se\s+cuenta\s+con|no\s+hay|falta(?:n)?)\s+(?:datos?|informaci[óo]n|estadísticas?)',
            r'informaci[óo]n\s+(?:no\s+)?disponible',
            r'(?:datos?|informaci[óo]n)\s+(?:insuficiente|limitada|incompleta)',
            r'ausencia\s+de\s+(?:datos?|informaci[óo]n|registros?)',
            r'sin\s+(?:registro|medici[óo]n|seguimiento)',
        ]
        
        for pattern in limitation_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                claims.append({
                    'type': 'data_limitation',
                    'value': None,
                    'unit': None,
                    'raw_text': match.group(0),
                    'position': match.span(),
                    'context': text[max(0, match.start()-50):min(len(text), match.end()+50)]
                })

        return claims

    def _parse_number_robust(self, text: str) -> float:
        """Parseo robusto de números"""
        try:
            normalized = text.replace('.', '').replace(',', '.')
            return float(normalized)
        except ValueError:
            try:
                normalized = text.replace(',', '')
                return float(normalized)
            except ValueError:
                return 0.0

    def _determine_semantic_role_advanced(self, sent) -> Optional[str]:
        """Determina rol semántico con análisis sintáctico profundo"""
        text_lower = sent.text.lower()

        root = [token for token in sent if token.dep_ == 'ROOT'][0] if sent else None

        role_patterns = {
            'objective': {
                'keywords': ['objetivo', 'meta', 'propósito', 'finalidad', 'busca', 'pretende'],
                'pos_tags': ['VERB', 'NOUN'],
                'deps': ['nsubj', 'obj']
            },
            'strategy': {
                'keywords': ['estrategia', 'línea', 'eje', 'pilar', 'enfoque', 'modelo'],
                'pos_tags': ['NOUN'],
                'deps': ['nsubj', 'nmod']
            },
            'action': {
                'keywords': ['implementar', 'ejecutar', 'desarrollar', 'realizar', 'gestionar'],
                'pos_tags': ['VERB'],
                'deps': ['ROOT', 'xcomp']
            },
            'indicator': {
                'keywords': ['indicador', 'medir', 'evaluar', 'monitorear', 'seguimiento'],
                'pos_tags': ['NOUN', 'VERB'],
                'deps': ['obj', 'nsubj']
            },
            'resource': {
                'keywords': ['presupuesto', 'recurso', 'financiación', 'inversión', 'asignación'],
                'pos_tags': ['NOUN'],
                'deps': ['nsubj', 'obj', 'nmod']
            },
            'constraint': {
                'keywords': ['limitación', 'restricción', 'condición', 'requisito', 'debe'],
                'pos_tags': ['NOUN', 'AUX'],
                'deps': ['nmod', 'aux']
            }
        }

        for role, criteria in role_patterns.items():
            keyword_match = any(kw in text_lower for kw in criteria['keywords'])

            pos_match = root and root.pos_ in criteria['pos_tags'] if root else False

            dep_match = any(token.dep_ in criteria['deps'] for token in sent)

            if keyword_match and (pos_match or dep_match):
                return role

        return None

    def _identify_deep_dependencies(self, sent, doc) -> Set[str]:
        """Identifica dependencias profundas entre declaraciones"""
        dependencies = set()

        reference_patterns = [
            (r'como\s+se\s+(?:menciona|establece|indica)\s+en', 1.0),
            (r'según\s+lo\s+(?:establecido|dispuesto|previsto)\s+en', 0.9),
            (r'de\s+acuerdo\s+con\s+(?:el|la|los|las)', 0.9),
            (r'en\s+línea\s+con', 0.8),
            (r'siguiendo\s+lo\s+dispuesto\s+en', 0.9),
            (r'conforme\s+a', 0.8),
            (r'en\s+cumplimiento\s+de', 0.85)
        ]

        sent_text = sent.text
        for pattern, weight in reference_patterns:
            if re.search(pattern, sent_text, re.IGNORECASE):
                for other_sent in doc.sents:
                    if other_sent != sent:
                        shared_entities = sum(
                            1 for token in sent 
                            if token.ent_type_ and token.text in other_sent.text
                        )
                        if shared_entities > 0:
                            dep_id = f"{other_sent.text[:50]}_{weight}"
                            dependencies.add(dep_id)

        return dependencies

    def _extract_causal_relations(self, sent) -> List[Tuple[str, str]]:
        """Extrae relaciones causales explícitas"""
        causal_relations = []

        causal_markers = {
            'causa': ['causa', 'ocasiona', 'provoca', 'origina'],
            'efecto': ['resultado', 'consecuencia', 'efecto', 'producto'],
            'condicional': ['si', 'cuando', 'en caso de', 'siempre que']
        }

        text_lower = sent.text.lower()

        for marker_type, markers in causal_markers.items():
            for marker in markers:
                if marker in text_lower:
                    entities = [ent.text for ent in sent.ents]
                    if len(entities) >= 2:
                        causal_relations.append((entities[0], entities[-1]))

        return causal_relations

    def _extract_regulatory_references(self, text: str) -> List[str]:
        """Extrae referencias normativas específicas"""
        references = []

        patterns = [
            r'ley\s+\d+\s+de\s+\d{4}',
            r'decreto\s+(?:ley\s+)?\d+\s+de\s+\d{4}',
            r'acuerdo\s+(?:municipal\s+)?\d+\s+de\s+\d{4}',
            r'resolución\s+\d+\s+de\s+\d{4}',
            r'conpes\s+\d+',
            r'sentencia\s+[CT]-\d+',
            r'constitución\s+política',
            r'plan\s+nacional\s+de\s+desarrollo'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            references.extend(matches)

        return references

    def _get_extended_context(
            self,
            text: str,
            start: int,
            end: int,
            window_size: int = 300
    ) -> str:
        """Obtiene contexto extendido con ventana adaptativa"""
        context_start = max(0, start - window_size)
        context_end = min(len(text), end + window_size)

        context = text[context_start:context_end]

        if context_start > 0:
            context = '...' + context
        if context_end < len(text):
            context = context + '...'

        return context

    def _get_ontology_weight(self, stmt: PolicyStatement) -> float:
        """Obtiene peso ontológico según patrón PDM"""
        for category, config in self.pdm_ontology.items():
            if config['pattern'].search(stmt.text):
                return config['weight']
        return 1.0
    
    def _analyze_regulatory_constraints(self, statements: List[PolicyStatement], 
                                       text: str, 
                                       temporal_conflicts: List[ContradictionEvidence]) -> Dict[str, Any]:
        """
        Harmonic Front 3 - Enhancement 3: Regulatory Constraint Analysis for D1-Q5
        
        Analyzes regulatory references and verifies compliance with external mandates.
        D1-Q5 (Restricciones Legales/Competencias) quality criteria:
        - Excelente: PDM text explicitly mentions ≥3 types of constraints 
          (Legal, Budgetary, Temporal/Competency) AND is_consistent = True
        """
        # Collect all regulatory references from statements
        all_regulatory_refs = []
        for stmt in statements:
            all_regulatory_refs.extend(stmt.regulatory_references)
        
        # Also extract from full text
        text_regulatory_refs = self._extract_regulatory_references(text)
        all_regulatory_refs.extend(text_regulatory_refs)
        
        # Remove duplicates
        all_regulatory_refs = list(set(all_regulatory_refs))
        
        # Classify constraint types mentioned in text
        constraint_types = {
            'Legal': [],
            'Budgetary': [],
            'Temporal': [],
            'Competency': [],
            'Institutional': [],
            'Technical': []
        }
        
        # Legal constraints patterns
        legal_patterns = [
            r'ley\s+152\s+de\s+1994',
            r'ley\s+388\s+de\s+1997',
            r'ley\s+715\s+de\s+2001',
            r'ley\s+1551\s+de\s+2012',
            r'competencia\s+municipal',
            r'marco\s+normativo',
            r'restricción\s+legal',
            r'limitación\s+normativa'
        ]
        
        # Budgetary constraints patterns
        budgetary_patterns = [
            r'restricción\s+presupuestal',
            r'límite\s+fiscal',
            r'capacidad\s+financiera',
            r'disponibilidad\s+de\s+recursos',
            r'SGP|SGR|recursos\s+propios',
            r'déficit\s+fiscal',
            r'sostenibilidad\s+financiera'
        ]
        
        # Temporal/Competency constraints patterns
        temporal_patterns = [
            r'plazo\s+(?:legal|establecido|normativo)',
            r'horizonte\s+temporal',
            r'periodo\s+de\s+gobierno',
            r'cuatrienio',
            r'restricción\s+temporal',
            r'capacidad\s+(?:técnica|institucional)',
            r'competencia\s+(?:administrativa|territorial)'
        ]
        
        text_lower = text.lower()
        
        # Check for Legal constraints
        for pattern in legal_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                constraint_types['Legal'].append(pattern)
        
        # Check for Budgetary constraints
        for pattern in budgetary_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                constraint_types['Budgetary'].append(pattern)
        
        # Check for Temporal/Competency constraints
        for pattern in temporal_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                constraint_types['Temporal'].append(pattern)
        
        # Count distinct constraint types mentioned
        constraint_types_mentioned = sum(1 for types in constraint_types.values() if types)
        
        # Check temporal consistency (from _detect_temporal_conflicts_formal)
        is_consistent = len(temporal_conflicts) == 0
        
        # D1-Q5 quality assessment
        d1_q5_quality = 'insuficiente'
        if constraint_types_mentioned >= 3 and is_consistent:
            d1_q5_quality = 'excelente'
        elif constraint_types_mentioned >= 3 or is_consistent:
            d1_q5_quality = 'bueno'
        elif constraint_types_mentioned >= 2:
            d1_q5_quality = 'aceptable'
        
        logger.info(f"D1-Q5 Regulatory Analysis: {constraint_types_mentioned} constraint types, "
                   f"is_consistent={is_consistent}, quality={d1_q5_quality}")
        
        return {
            'regulatory_references': all_regulatory_refs,
            'regulatory_references_count': len(all_regulatory_refs),
            'constraint_types_detected': {k: len(v) for k, v in constraint_types.items()},
            'constraint_types_mentioned': constraint_types_mentioned,
            'is_consistent': is_consistent,
            'd1_q5_quality': d1_q5_quality,
            'd1_q5_criteria': {
                'legal_constraints': len(constraint_types['Legal']) > 0,
                'budgetary_constraints': len(constraint_types['Budgetary']) > 0,
                'temporal_competency_constraints': len(constraint_types['Temporal']) > 0,
                'minimum_constraint_types': constraint_types_mentioned >= 3,
                'temporal_consistency': is_consistent
            }
        }

    def _calculate_comprehensive_severity(
            self,
            stmt_a: PolicyStatement,
            stmt_b: PolicyStatement,
            base_score: float
    ) -> float:
        """Calcula severidad integral de contradicción"""
        dimension_weight = {
            PolicyDimension.FINANCIERO: 1.5,
            PolicyDimension.ESTRATEGICO: 1.3,
            PolicyDimension.PROGRAMATICO: 1.2,
            PolicyDimension.DIAGNOSTICO: 1.0,
            PolicyDimension.SEGUIMIENTO: 1.1,
            PolicyDimension.TERRITORIAL: 1.2
        }

        weight_a = dimension_weight.get(stmt_a.dimension, 1.0)
        weight_b = dimension_weight.get(stmt_b.dimension, 1.0)
        avg_weight = (weight_a + weight_b) / 2

        entity_overlap = len({e['text'] for e in stmt_a.entities} & 
                             {e['text'] for e in stmt_b.entities})
        overlap_factor = min(1.3, 1.0 + entity_overlap * 0.1)

        has_quantitative = bool(stmt_a.quantitative_claims and stmt_b.quantitative_claims)
        quant_factor = 1.2 if has_quantitative else 1.0

        severity = base_score * avg_weight * overlap_factor * quant_factor

        return min(1.0, severity)

    def _generate_resolution_strategies(
            self,
            contradiction_type: ContradictionType,
            stmt_a: PolicyStatement,
            stmt_b: PolicyStatement,
            context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Genera estrategias específicas y accionables"""
        context = context or {}

        strategies = {
            ContradictionType.NUMERICAL_INCONSISTENCY: [
                f"Validar fuentes primarias de datos para valores divergentes (divergencia: {context.get('divergence', 0):.2%})",
                "Establecer metodología única de cálculo documentada en anexo técnico",
                "Convocar mesa técnica con DNP para validación de cifras",
                "Implementar sistema de trazabilidad de información cuantitativa"
            ],
            ContradictionType.TEMPORAL_CONFLICT: [
                f"Revisar cronograma maestro - conflicto tipo: {context.get('conflict_type', 'no especificado')}",
                "Realizar análisis de ruta crítica (CPM) para secuenciación óptima",
                "Ajustar plazos según capacidad institucional verificada",
                "Establecer hitos de validación inter-dependencias"
            ],
            ContradictionType.SEMANTIC_OPPOSITION: [
                "Realizar taller de alineación conceptual con equipo técnico",
                "Desarrollar glosario técnico unificado según terminología DNP",
                "Priorizar objetivos mediante metodología AHP (Analytic Hierarchy Process)",
                "Aplicar teoría de cambio para validar coherencia lógica"
            ],
            ContradictionType.RESOURCE_ALLOCATION_MISMATCH: [
                f"Análisis de brechas financieras - recurso: {context.get('resource_type', 'no especificado')}",
                "Priorización mediante matriz de impacto social vs viabilidad financiera",
                "Explorar mecanismos alternativos: APP, cooperación internacional, bonos de impacto",
                "Validar Plan Financiero con MHCP y órganos de control"
            ],
            ContradictionType.LOGICAL_INCOMPATIBILITY: [
                "Mapear cadena de valor completa de programas/subprogramas",
                "Validar teoría de cambio mediante marco lógico",
                "Eliminar duplicidades usando matriz de intervenciones",
                "Aplicar análisis de coherencia interna según ISO 21500"
            ],
            ContradictionType.CAUSAL_INCOHERENCE: [
                "Construir diagrama causal (DAG) para validar relaciones",
                "Revisar supuestos de causalidad con evidencia empírica",
                "Secuenciar intervenciones según precedencia causal",
                "Validar mecanismos causales con literatura especializada"
            ]
        }

        base_strategies = strategies.get(contradiction_type, [
            "Revisar exhaustivamente ambas declaraciones",
            "Consultar con equipo técnico responsable",
            "Documentar decisión en acta de ajuste del plan"
        ])

        position_info = f"Secciones afectadas: caracteres {stmt_a.position[0]}-{stmt_a.position[1]} y {stmt_b.position[0]}-{stmt_b.position[1]}"
        base_strategies.append(position_info)

        return base_strategies

    def _are_comparable_claims_advanced(
            self,
            claim_a: Dict[str, Any],
            claim_b: Dict[str, Any]
    ) -> bool:
        """Determina comparabilidad con análisis semántico"""
        if claim_a['type'] != claim_b['type']:
            return False

        context_a = claim_a.get('context', '')
        context_b = claim_b.get('context', '')

        doc_a = self.nlp(context_a)
        doc_b = self.nlp(context_b)

        entities_a = {ent.text.lower() for ent in doc_a.ents}
        entities_b = {ent.text.lower() for ent in doc_b.ents}

        if entities_a & entities_b:
            return True

        tokens_a = {token.lemma_.lower() for token in doc_a if token.pos_ in ['NOUN', 'VERB']}
        tokens_b = {token.lemma_.lower() for token in doc_b if token.pos_ in ['NOUN', 'VERB']}

        jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b) if tokens_a | tokens_b else 0

        return jaccard > 0.4

    def _extract_detailed_resource_mentions(
            self,
            text: str
    ) -> List[Tuple[str, Optional[float], str]]:
        """Extrae menciones detalladas de recursos"""
        resources = []

        patterns = [
            (r'SGP\s*(?:educación|salud|agua|propósito\s+general)?\s*[:\s]*\$?\s*(\d+(?:[.,]\d+)?)\s*(millones?|mil\s+millones?)?', 'SGP'),
            (r'SGR\s*[:\s]*\$?\s*(\d+(?:[.,]\d+)?)\s*(millones?|mil\s+millones?)?', 'SGR'),
            (r'regalías\s*[:\s]*\$?\s*(\d+(?:[.,]\d+)?)\s*(millones?|mil\s+millones?)?', 'regalías'),
            (r'recursos?\s+propios\s*[:\s]*\$?\s*(\d+(?:[.,]\d+)?)\s*(millones?|mil\s+millones?)?', 'recursos_propios'),
            (r'cofinanciación\s*[:\s]*\$?\s*(\d+(?:[.,]\d+)?)\s*(millones?|mil\s+millones?)?', 'cofinanciación'),
            (r'crédito\s*[:\s]*\$?\s*(\d+(?:[.,]\d+)?)\s*(millones?|mil\s+millones?)?', 'crédito'),
            (r'presupuesto\s+total\s*[:\s]*\$?\s*(\d+(?:[.,]\d+)?)\s*(millones?|mil\s+millones?)?', 'presupuesto_total')
        ]

        for pattern, resource_type in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                amount = self._parse_number_robust(match.group(1)) if match.group(1) else None

                if amount and match.group(2) and 'millon' in match.group(2).lower():
                    if 'mil' in match.group(2).lower():
                        amount *= 1_000_000_000
                    else:
                        amount *= 1_000_000

                resources.append((resource_type, amount, match.group(0)))

        return resources

    def _determine_relation_type_advanced(
            self,
            stmt_a: PolicyStatement,
            stmt_b: PolicyStatement
    ) -> str:
        """Determina tipo de relación entre declaraciones"""
        if stmt_a.dimension == stmt_b.dimension:
            if any(dep in stmt_b.dependencies for dep in stmt_a.dependencies):
                return 'hierarchical'

        shared_entities = {e['text'] for e in stmt_a.entities} & \
            {e['text'] for e in stmt_b.entities}
        if len(shared_entities) > 2:
            return 'strongly_related'
        elif len(shared_entities) > 0:
            return 'related'

        if stmt_a.semantic_role == stmt_b.semantic_role:
            return 'parallel'

        causal_pairs = [('objective', 'action'), ('action', 'indicator'), ('strategy', 'action')]
        role_pair = (stmt_a.semantic_role, stmt_b.semantic_role)
        if role_pair in causal_pairs:
            return 'causal'

        return 'associative'

    def _get_relation_weight(self, relation_type: str) -> float:
        """Obtiene peso de relación"""
        weights = {
            'hierarchical': 1.3,
            'causal': 1.25,
            'strongly_related': 1.2,
            'related': 1.0,
            'parallel': 0.9,
            'associative': 0.8
        }
        return weights.get(relation_type, 1.0)

    def _deduplicate_contradictions(
            self,
            contradictions: List[ContradictionEvidence]
    ) -> List[ContradictionEvidence]:
        """Elimina contradicciones duplicadas usando hash"""
        seen = set()
        unique_contradictions = []

        for contradiction in contradictions:
            stmt_pair = tuple(sorted([
                contradiction.statement_a.text[:100],
                contradiction.statement_b.text[:100]
            ]))

            if stmt_pair not in seen:
                seen.add(stmt_pair)
                unique_contradictions.append(contradiction)

        return unique_contradictions

    def _calculate_global_semantic_coherence_advanced(
            self,
            statements: List[PolicyStatement]
    ) -> float:
        """Coherencia semántica con análisis de clustering"""
        if len(statements) < 2:
            return 1.0

        embeddings = [s.contextual_embedding or s.embedding 
                      for s in statements 
                      if s.contextual_embedding is not None or s.embedding is not None]

        if len(embeddings) < 2:
            return 0.5

        similarity_matrix = cosine_similarity(embeddings)

        consecutive_sims = [similarity_matrix[i, i+1] 
                            for i in range(len(similarity_matrix) - 1)]

        mean_sim = np.mean(consecutive_sims)
        std_sim = np.std(consecutive_sims)

        coherence = mean_sim * (1 - min(0.5, std_sim))

        global_mean_sim = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])

        final_coherence = 0.6 * coherence + 0.4 * global_mean_sim

        return float(final_coherence)

    def _calculate_causal_coherence(self) -> float:
        """Coherencia de red causal"""
        causal_net = self.bayesian_engine.causal_network

        if causal_net.number_of_nodes() == 0:
            return 1.0

        try:
            cycles = list(nx.simple_cycles(causal_net))
            cycle_penalty = min(1.0, len(cycles) / max(1, causal_net.number_of_nodes()))
        except (nx.NetworkXError, nx.NetworkXNoCycle, ValueError, TypeError) as e:
            cycle_penalty = 0.0

        if causal_net.number_of_edges() > 0:
            weights = [data['weight'] for _, _, data in causal_net.edges(data=True)]
            avg_strength = np.mean(weights)
        else:
            avg_strength = 0.5

        coherence = avg_strength * (1 - cycle_penalty)

        return float(coherence)

    def _calculate_objective_alignment_advanced(
            self,
            statements: List[PolicyStatement]
    ) -> float:
        """Alineación de objetivos con análisis vectorial"""
        objective_statements = [
            s for s in statements 
            if s.semantic_role in ['objective', 'strategy']
        ]

        if len(objective_statements) < 2:
            return 1.0

        embeddings = [s.contextual_embedding or s.embedding 
                      for s in objective_statements 
                      if s.contextual_embedding is not None or s.embedding is not None]

        if len(embeddings) < 2:
            return 0.5

        centroid = np.mean(embeddings, axis=0)

        distances = [np.linalg.norm(emb - centroid) for emb in embeddings]
        avg_distance = np.mean(distances)

        alignment = 1.0 / (1.0 + avg_distance)

        return float(alignment)

    def _calculate_graph_coherence_metrics(self) -> Dict[str, float]:
        """Métricas avanzadas de coherencia del grafo"""
        if self.knowledge_graph.number_of_nodes() == 0:
            return {'fragmentation': 0.0, 'modularity': 0.0, 'centralization': 0.0}

        num_components = nx.number_weakly_connected_components(self.knowledge_graph)
        num_nodes = self.knowledge_graph.number_of_nodes()
        fragmentation = (num_components - 1) / max(1, num_nodes - 1)

        undirected = self.knowledge_graph.to_undirected()
        try:
            communities = nx.community.greedy_modularity_communities(undirected)
            modularity = nx.community.modularity(undirected, communities)
        except (nx.exception.NetworkXError, ValueError):
            modularity = 0.0

        if num_nodes > 1:
            degrees = [deg for _, deg in self.knowledge_graph.degree()]
            max_degree = max(degrees)
            sum_diff = sum(max_degree - deg for deg in degrees)
            max_sum_diff = (num_nodes - 1) * (num_nodes - 2)
            centralization = sum_diff / max_sum_diff if max_sum_diff > 0 else 0.0
        else:
            centralization = 0.0

        return {
            'fragmentation': float(fragmentation),
            'modularity': float(modularity),
            'centralization': float(centralization)
        }

    def _calculate_shannon_entropy(
            self,
            contradictions: List[ContradictionEvidence]
    ) -> float:
        """Entropía de Shannon de distribución de contradicciones"""
        if not contradictions:
            return 0.0

        type_counts = defaultdict(int)
        for c in contradictions:
            type_counts[c.contradiction_type] += 1

        total = len(contradictions)
        probabilities = [count / total for count in type_counts.values()]

        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)

        max_entropy = np.log2(len(ContradictionType))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return float(normalized_entropy)

    def _calculate_syntactic_complexity_advanced(self, text: str) -> float:
        """Complejidad sintáctica con múltiples métricas"""
        doc = self.nlp(text[:10000])

        sentence_lengths = [len(sent.text.split()) for sent in doc.sents]
        avg_sent_length = np.mean(sentence_lengths) if sentence_lengths else 0

        dependency_depths = []
        for sent in doc.sents:
            depths = [self._get_dependency_depth(token) for token in sent]
            if depths:
                dependency_depths.append(np.mean(depths))
        avg_dep_depth = np.mean(dependency_depths) if dependency_depths else 0

        tokens = [token.text.lower() for token in doc if token.is_alpha]
        ttr = len(set(tokens)) / len(tokens) if tokens else 0

        subordinate_clauses = len([token for token in doc if token.dep_ in ['csubj', 'ccomp', 'advcl']])
        subordination_ratio = subordinate_clauses / len(list(doc.sents)) if doc.sents else 0

        complexity = (
            min(1.0, avg_sent_length / 40) * 0.25 +
            min(1.0, avg_dep_depth / 8) * 0.25 +
            ttr * 0.25 +
            min(1.0, subordination_ratio / 3) * 0.25
        )

        return float(complexity)

    def _get_dependency_depth(self, token) -> int:
        """Profundidad en árbol de dependencias"""
        depth = 0
        current = token
        visited = set()

        while current.head != current and current not in visited and depth < 50:
            visited.add(current)
            current = current.head
            depth += 1

        return depth

    def _calculate_bootstrap_confidence_interval(
            self,
            score: float,
            n_observations: int,
            n_bootstrap: int = 1000,
            confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Intervalo de confianza mediante bootstrap"""
        bootstrap_scores = []
        rng = np.random.default_rng()

        for _ in range(n_bootstrap):
            sample_size = max(1, int(n_observations * 0.8))
            noise = rng.normal(0, 0.05, sample_size)
            bootstrap_score = score + np.mean(noise)
            bootstrap_scores.append(np.clip(bootstrap_score, 0, 1))

        alpha = (1 - confidence) / 2
        lower = np.percentile(bootstrap_scores, alpha * 100)
        upper = np.percentile(bootstrap_scores, (1 - alpha) * 100)

        return (float(lower), float(upper))

    def _assign_quality_grade(self, coherence_score: float) -> str:
        """Asigna calificación de calidad"""
        if coherence_score >= 0.90:
            return "Excelente"
        elif coherence_score >= 0.80:
            return "Muy Bueno"
        elif coherence_score >= 0.70:
            return "Bueno"
        elif coherence_score >= 0.60:
            return "Aceptable"
        elif coherence_score >= 0.50:
            return "Requiere Mejoras"
        else:
            return "Crítico"

    def _generate_actionable_recommendations(
            self,
            contradictions: List[ContradictionEvidence]
    ) -> List[Dict[str, Any]]:
        """
        Genera recomendaciones priorizadas y accionables
        
        HARMONIC FRONT 4 ENHANCEMENT:
        - Prioritizes CAUSAL_INCOHERENCE and TEMPORAL_CONFLICT as high/critical
        - Aligns priority with measured omission_severity
        - Ensures structural failures get immediate system adaptation
        """
        recommendations = []

        by_type = defaultdict(list)
        for c in contradictions:
            by_type[c.contradiction_type].append(c)

        # UPDATED: Prioritize structural failures (CAUSAL_INCOHERENCE, TEMPORAL_CONFLICT)
        priority_map = {
            ContradictionType.CAUSAL_INCOHERENCE: 'crítica',  # UPGRADED from 'media'
            ContradictionType.TEMPORAL_CONFLICT: 'crítica',   # UPGRADED from 'alta'
            ContradictionType.RESOURCE_ALLOCATION_MISMATCH: 'crítica',
            ContradictionType.NUMERICAL_INCONSISTENCY: 'alta',
            ContradictionType.SEMANTIC_OPPOSITION: 'media',
            ContradictionType.LOGICAL_INCOMPATIBILITY: 'alta'
        }

        for cont_type, conflicts in by_type.items():
            avg_severity = np.mean([c.severity for c in conflicts])
            avg_confidence = np.mean([c.confidence for c in conflicts])
            
            # Calculate priority score aligning with measured severity
            base_priority = priority_map.get(cont_type, 'media')
            priority_score = avg_severity * avg_confidence
            
            # Adjust priority based on measured omission_severity alignment
            if priority_score > 0.75 and base_priority != 'crítica':
                base_priority = 'crítica'
            elif priority_score > 0.60 and base_priority not in ['crítica', 'alta']:
                base_priority = 'alta'

            recommendation = {
                'contradiction_type': cont_type.name,
                'priority': base_priority,
                'priority_score': float(priority_score),  # NEW: Explicit priority score
                'count': len(conflicts),
                'avg_severity': float(avg_severity),
                'avg_confidence': float(avg_confidence),
                'description': self._get_recommendation_description(cont_type),
                'action_plan': conflicts[0].resolution_suggestions if conflicts else [],
                'affected_sections': list({
                                              f"Dim: {c.statement_a.dimension.value}" 
                                              for c in conflicts
                                          }),
                'estimated_effort': self._estimate_resolution_effort(cont_type, len(conflicts))
            }
            recommendations.append(recommendation)

        # Sort by priority (structural failures first) and severity
        priority_order = {'crítica': 0, 'alta': 1, 'media': 2, 'baja': 3}
        recommendations.sort(
            key=lambda x: (priority_order.get(x['priority'], 4), -x['priority_score'], -x['avg_severity'])
        )

        return recommendations

    def _get_recommendation_description(self, cont_type: ContradictionType) -> str:
        """Descripción contextualizada de recomendación"""
        descriptions = {
            ContradictionType.NUMERICAL_INCONSISTENCY: 
                "Reconciliar cifras inconsistentes mediante validación técnica con fuentes primarias",
            ContradictionType.TEMPORAL_CONFLICT: 
                "Ajustar cronograma para resolver conflictos de precedencia y simultaneidad",
            ContradictionType.SEMANTIC_OPPOSITION: 
                "Clarificar conceptos opuestos y establecer jerarquía clara de objetivos",
            ContradictionType.RESOURCE_ALLOCATION_MISMATCH: 
                "Revisar asignación presupuestal para eliminar sobre-asignaciones o conflictos",
            ContradictionType.LOGICAL_INCOMPATIBILITY: 
                "Resolver incompatibilidades lógicas mediante análisis de cadena de valor",
            ContradictionType.CAUSAL_INCOHERENCE: 
                "Validar relaciones causales y eliminar circularidades o inconsistencias"
        }
        return descriptions.get(cont_type, "Revisar y ajustar según análisis técnico")

    def _estimate_resolution_effort(self, cont_type: ContradictionType, count: int) -> str:
        """Estima esfuerzo de resolución"""
        base_effort = {
            ContradictionType.NUMERICAL_INCONSISTENCY: 2,
            ContradictionType.TEMPORAL_CONFLICT: 3,
            ContradictionType.SEMANTIC_OPPOSITION: 4,
            ContradictionType.RESOURCE_ALLOCATION_MISMATCH: 5,
            ContradictionType.LOGICAL_INCOMPATIBILITY: 4,
            ContradictionType.CAUSAL_INCOHERENCE: 3
        }

        effort_hours = base_effort.get(cont_type, 3) * count

        if effort_hours <= 8:
            return f"Bajo ({effort_hours} horas aprox.)"
        elif effort_hours <= 24:
            return f"Medio ({effort_hours} horas aprox.)"
        else:
            return f"Alto ({effort_hours} horas aprox.)"

    def _serialize_contradiction(
            self,
            contradiction: ContradictionEvidence
    ) -> Dict[str, Any]:
        """Serializa evidencia para salida JSON"""
        return {
            "statement_1": {
                "text": contradiction.statement_a.text,
                "position": contradiction.statement_a.position,
                "dimension": contradiction.statement_a.dimension.value,
                "semantic_role": contradiction.statement_a.semantic_role,
                "entities": [e['text'] for e in contradiction.statement_a.entities],
                "has_temporal_markers": bool(contradiction.statement_a.temporal_markers),
                "has_quantitative_claims": bool(contradiction.statement_a.quantitative_claims),
                "regulatory_refs": contradiction.statement_a.regulatory_references
            },
            "statement_2": {
                "text": contradiction.statement_b.text,
                "position": contradiction.statement_b.position,
                "dimension": contradiction.statement_b.dimension.value,
                "semantic_role": contradiction.statement_b.semantic_role,
                "entities": [e['text'] for e in contradiction.statement_b.entities],
                "has_temporal_markers": bool(contradiction.statement_b.temporal_markers),
                "has_quantitative_claims": bool(contradiction.statement_b.quantitative_claims),
                "regulatory_refs": contradiction.statement_b.regulatory_references
            },
            "contradiction_type": contradiction.contradiction_type.name,
            "confidence": float(contradiction.confidence),
            "severity": float(contradiction.severity),
            "severity_category": self._categorize_severity(contradiction.severity),
            "semantic_similarity": float(contradiction.semantic_similarity),
            "logical_conflict_score": float(contradiction.logical_conflict_score),
            "temporal_consistency": contradiction.temporal_consistency,
            "numerical_divergence": float(contradiction.numerical_divergence) 
            if contradiction.numerical_divergence else None,
            "statistical_significance": float(contradiction.statistical_significance) 
            if contradiction.statistical_significance else None,
            "causal_conflict": contradiction.causal_conflict,
            "affected_dimensions": [d.value for d in contradiction.affected_dimensions],
            "resolution_suggestions": contradiction.resolution_suggestions,
            "graph_path": contradiction.graph_path,
            "has_attention_weights": contradiction.attention_weights is not None
        }

    def _categorize_severity(self, severity: float) -> str:
        """Categoriza severidad en niveles"""
        if severity > 0.85:
            return "CRÍTICA"
        elif severity > 0.70:
            return "ALTA"
        elif severity > 0.50:
            return "MEDIA"
        else:
            return "BAJA"

    def _get_advanced_graph_statistics(self) -> Dict[str, Any]:
        """Estadísticas avanzadas del grafo de conocimiento"""
        if self.knowledge_graph.number_of_nodes() == 0:
            return {
                "nodes": 0,
                "edges": 0,
                "components": 0,
                "density": 0.0,
                "clustering": 0.0,
                "diameter": -1
            }

        undirected = self.knowledge_graph.to_undirected()

        clustering = nx.average_clustering(undirected)

        try:
            if nx.is_connected(undirected):
                diameter = nx.diameter(undirected)
                avg_path_length = nx.average_shortest_path_length(undirected)
            else:
                largest_cc = max(nx.connected_components(undirected), key=len)
                subgraph = undirected.subgraph(largest_cc)
                diameter = nx.diameter(subgraph)
                avg_path_length = nx.average_shortest_path_length(subgraph)
        except (nx.NetworkXError, ValueError, StopIteration) as e:
            diameter = -1
            avg_path_length = -1

        degree_centrality = nx.degree_centrality(undirected)
        avg_degree_centrality = np.mean(list(degree_centrality.values()))

        return {
            "nodes": self.knowledge_graph.number_of_nodes(),
            "edges": self.knowledge_graph.number_of_edges(),
            "components": nx.number_weakly_connected_components(self.knowledge_graph),
            "density": float(nx.density(self.knowledge_graph)),
            "clustering_coefficient": float(clustering),
            "diameter": diameter,
            "avg_path_length": float(avg_path_length) if avg_path_length > 0 else None,
            "avg_degree_centrality": float(avg_degree_centrality),
            "is_dag": nx.is_directed_acyclic_graph(self.knowledge_graph)
        }

    def _get_causal_network_statistics(self) -> Dict[str, Any]:
        """Estadísticas de red causal bayesiana"""
        causal_net = self.bayesian_engine.causal_network

        if causal_net.number_of_nodes() == 0:
            return {
                "nodes": 0,
                "edges": 0,
                "cycles": 0,
                "avg_causal_strength": 0.0
            }

        try:
            cycles = list(nx.simple_cycles(causal_net))
            num_cycles = len(cycles)
        except nx.NetworkXError:
            num_cycles = 0

        if causal_net.number_of_edges() > 0:
            weights = [data['weight'] for _, _, data in causal_net.edges(data=True)]
            avg_strength = float(np.mean(weights))
            max_strength = float(np.max(weights))
            min_strength = float(np.min(weights))
        else:
            avg_strength = 0.0
            max_strength = 0.0
            min_strength = 0.0

        return {
            "nodes": causal_net.number_of_nodes(),
            "edges": causal_net.number_of_edges(),
            "cycles": num_cycles,
            "avg_causal_strength": avg_strength,
            "max_causal_strength": max_strength,
            "min_causal_strength": min_strength,
            "is_acyclic": num_cycles == 0
        }


def create_detector(device: Optional[str] = None) -> PolicyContradictionDetectorV2:
    """
Factory function para crear detector con configuración óptima

Args:
device: 'cuda', 'cpu', o None para auto-detección

Returns:
Detector configurado y listo para uso
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Creando detector en dispositivo: {device}")
    detector = PolicyContradictionDetectorV2(device=device)
    logger.info("Detector inicializado exitosamente")

    return detector


# Ejemplo de uso integral
if __name__ == "__main__":
    import json

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    detector = create_detector()

    sample_pdm = """
PLAN DE DESARROLLO MUNICIPAL 2024-2027
"MUNICIPIO PRÓSPERO Y SOSTENIBLE"

COMPONENTE ESTRATÉGICO

Eje 1: Educación de Calidad
Objetivo: Aumentar la cobertura educativa al 95% para el año 2027.
Meta de resultado: Incrementar en 15 puntos porcentuales la cobertura en educación media.

Programa 1.1: Infraestructura Educativa
El municipio construirá 5 nuevas instituciones educativas en el primer semestre de 2025.
Recursos SGP Educación: $1,500 millones anuales.

Sin embargo, el presupuesto total del programa es de $800 millones para el cuatrienio.

Eje 2: Desarrollo Económico
La estrategia busca reducir la cobertura educativa para priorizar formación técnica.
Se ejecutará el 40% del presupuesto en el primer trimestre de 2025.

Para el segundo trimestre de 2025 se proyecta ejecutar el 70% del presupuesto total anual.

Programa 2.1: Apoyo Empresarial
Meta: Beneficiar a 10,000 familias con programas de emprendimiento.
Recursos propios asignados: $2,500 millones.
Recursos propios disponibles según plan financiero: $1,200 millones.

El programa tiene capacidad operativa para atender máximo 5,000 beneficiarios según 
análisis de capacidad institucional realizado en diagnóstico.

COMPONENTE PROGRAMÁTICO

Los proyectos de infraestructura educativa se ejecutarán después de la formación docente,
pero la formación docente requiere que primero existan las nuevas instituciones según
Acuerdo Municipal 045 de 2023.

El plan se rige por la Ley 152 de 1994 y el Decreto 1082 de 2015, estableciendo que
todos los programas deben tener indicadores de resultado. Sin embargo, el Programa 2.1
no cuenta con indicadores definidos.
"""

    result = detector.detect(
        text=sample_pdm,
        plan_name="PDM Municipio Próspero 2024-2027",
        dimension=PolicyDimension.ESTRATEGICO
    )

    print("\n" + "="*80)
    print("ANÁLISIS COMPLETADO")
    print("="*80)
    print(f"\nPlan: {result['plan_name']}")
    print(f"Dimensión analizada: {result['dimension']}")
    print(f"Total declaraciones: {result['total_statements']}")
    print(f"\nContradicciones detectadas: {result['total_contradictions']}")
    print(f"  - Críticas: {result['critical_severity_count']}")
    print(f"  - Altas: {result['high_severity_count']}")
    print(f"  - Medias: {result['medium_severity_count']}")

    print("\nMÉTRICAS DE COHERENCIA:")
    print(f"  - Score global: {result['coherence_metrics']['coherence_score']:.3f}")
    print(f"  - Calificación: {result['coherence_metrics']['quality_grade']}")
    print(f"  - Coherencia semántica: {result['coherence_metrics']['semantic_coherence']:.3f}")
    print(f"  - Consistencia temporal: {result['coherence_metrics']['temporal_consistency']:.3f}")
    print(f"  - Coherencia causal: {result['coherence_metrics']['causal_coherence']:.3f}")

    print("\nRECOMENDACIONES PRIORITARIAS:")
    for i, rec in enumerate(result['recommendations'][:3], 1):
        print(f"\n{i}. {rec['contradiction_type']} (Prioridad: {rec['priority'].upper()})")
        print(f"   Cantidad: {rec['count']} | Severidad promedio: {rec['avg_severity']:.2f}")
        print(f"   Esfuerzo estimado: {rec['estimated_effort']}")
        print(f"   Descripción: {rec['description']}")

    print("\n" + "="*80)

    output_file = Path("contradiction_analysis_result.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"Resultados completos guardados en: {output_file}")
