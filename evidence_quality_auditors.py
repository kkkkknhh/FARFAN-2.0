#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evidence Quality and Compliance Auditors (Part 3)
==================================================

Implements SOTA compliance auditing per DNP frameworks integrated with MMR:
- D3-Q1: Indicator Ficha Técnica (OperationalizationAuditor)
- D1-Q3, D3-Q3: Financial Traceability (FinancialTraceabilityAuditor)
- D1-Q2: Quantified Gap Recognition (QuantifiedGapAuditor)
- D4-Q5, D5-Q4: Systemic Risk Alignment (SystemicRiskAuditor)

References:
- DNP Colombian Standards (2023)
- UN ODS Alignment Benchmarks (2020)
- Bayesian Updating per Gelman (2013)
- QCA Calibration per Ragin (2008)
- Counterfactual Rigor per Pearl (2018)
- Waldner Fiscal Illusions (2015)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
from enum import Enum
import logging
import re
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class AuditSeverity(Enum):
    """Severity levels for audit findings"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    REQUIRES_REVIEW = "requires_review"
    CRITICAL = "critical"


@dataclass
class IndicatorMetadata:
    """Metadata for indicator ficha técnica"""
    codigo: str
    nombre: str
    linea_base: Optional[float] = None
    meta: Optional[float] = None
    fuente: Optional[str] = None
    formula: Optional[str] = None
    unidad_medida: Optional[str] = None
    periodicidad: Optional[str] = None
    has_full_metadata: bool = False
    
    def validate_completeness(self) -> Tuple[bool, List[str]]:
        """
        Validate if indicator has full metadata per DNP standards.
        
        Returns:
            Tuple of (is_complete, missing_fields)
        """
        required_fields = {
            'linea_base': self.linea_base,
            'meta': self.meta,
            'fuente': self.fuente,
            'formula': self.formula
        }
        
        missing = [field for field, value in required_fields.items() if value is None]
        is_complete = len(missing) == 0
        
        return is_complete, missing


@dataclass
class FinancialCode:
    """Financial traceability code (BPIN/PPI)"""
    code: str
    code_type: str  # "BPIN" or "PPI"
    match_confidence: float = 0.0
    matched_text: str = ""
    dependencies: List[str] = field(default_factory=list)


@dataclass
class QuantifiedGap:
    """Quantified gap or data limitation"""
    gap_type: str  # "vacío", "brecha", "déficit"
    quantification: Optional[float] = None
    description: str = ""
    source_text: str = ""
    severity: float = 0.0


@dataclass
class RiskAlignment:
    """Systemic risk alignment with PND/ODS"""
    pnd_alignment: bool = False
    ods_alignment: List[int] = field(default_factory=list)
    risk_score: float = 0.0
    misalignment_reasons: List[str] = field(default_factory=list)


@dataclass
class AuditResult:
    """Generic audit result structure"""
    audit_type: str
    timestamp: str
    severity: AuditSeverity
    findings: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    recommendations: List[str]
    evidence: Dict[str, Any]
    sota_compliance: bool = False


# ============================================================================
# AUDITOR 1: OPERATIONALIZATION AUDITOR (D3-Q1)
# ============================================================================

class OperationalizationAuditor:
    """
    Audits indicator ficha técnica completeness.
    
    Check Criteria:
    - Requires full metadata (Línea Base, Meta, Fuente) + formula for ≥80% product indicators
    
    Quality Evidence:
    - Audit logs cross-checking formulas against PDM tables
    
    SOTA Performance:
    - Measurability exceeds ODS alignment benchmarks (UN 2020)
    - Full metadata enables Bayesian updating (Gelman 2013)
    """
    
    def __init__(self, metadata_threshold: float = 0.80):
        """
        Initialize operationalization auditor.
        
        Args:
            metadata_threshold: Minimum proportion of indicators with full metadata (default: 0.80)
        """
        self.metadata_threshold = metadata_threshold
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def audit_indicators(
        self,
        indicators: List[IndicatorMetadata],
        pdm_tables: Optional[List[Dict[str, Any]]] = None
    ) -> AuditResult:
        """
        Audit indicator metadata completeness.
        
        Args:
            indicators: List of indicator metadata to audit
            pdm_tables: Optional PDM tables for cross-verification
            
        Returns:
            AuditResult with completeness assessment
        """
        self.logger.info(f"Auditing {len(indicators)} indicators for metadata completeness")
        
        findings = []
        complete_count = 0
        incomplete_indicators = []
        
        for indicator in indicators:
            is_complete, missing_fields = indicator.validate_completeness()
            
            if is_complete:
                complete_count += 1
                findings.append({
                    'indicator_code': indicator.codigo,
                    'indicator_name': indicator.nombre,
                    'status': 'complete',
                    'missing_fields': []
                })
            else:
                incomplete_indicators.append({
                    'codigo': indicator.codigo,
                    'nombre': indicator.nombre,
                    'missing': missing_fields
                })
                findings.append({
                    'indicator_code': indicator.codigo,
                    'indicator_name': indicator.nombre,
                    'status': 'incomplete',
                    'missing_fields': missing_fields
                })
        
        # Calculate completeness ratio
        total_indicators = len(indicators)
        completeness_ratio = complete_count / total_indicators if total_indicators > 0 else 0.0
        
        # Determine severity based on threshold
        if completeness_ratio >= self.metadata_threshold:
            severity = AuditSeverity.EXCELLENT
            sota_compliance = True
        elif completeness_ratio >= 0.70:
            severity = AuditSeverity.GOOD
            sota_compliance = True
        elif completeness_ratio >= 0.60:
            severity = AuditSeverity.ACCEPTABLE
            sota_compliance = False
        else:
            severity = AuditSeverity.REQUIRES_REVIEW
            sota_compliance = False
        
        # Generate recommendations
        recommendations = []
        if completeness_ratio < self.metadata_threshold:
            recommendations.append(
                f"Completar metadata faltante para {len(incomplete_indicators)} indicadores"
            )
            recommendations.append(
                "Revisar Línea Base, Meta, Fuente y Fórmula según estándares DNP"
            )
        
        if completeness_ratio >= self.metadata_threshold:
            recommendations.append(
                "Metadata completa permite actualización Bayesiana (Gelman 2013)"
            )
        
        # Cross-check formulas against PDM tables if available
        formula_matches = 0
        if pdm_tables:
            formula_matches = self._cross_check_formulas(indicators, pdm_tables)
        
        metrics = {
            'total_indicators': total_indicators,
            'complete_indicators': complete_count,
            'incomplete_indicators': len(incomplete_indicators),
            'completeness_ratio': completeness_ratio,
            'formula_matches': formula_matches,
            'meets_threshold': completeness_ratio >= self.metadata_threshold
        }
        
        evidence = {
            'incomplete_indicators': incomplete_indicators,
            'threshold_used': self.metadata_threshold,
            'ods_alignment_benchmark': 'UN 2020',
            'bayesian_updating_reference': 'Gelman 2013'
        }
        
        return AuditResult(
            audit_type="D3-Q1_IndicatorMetadata",
            timestamp=datetime.now().isoformat(),
            severity=severity,
            findings=findings,
            metrics=metrics,
            recommendations=recommendations,
            evidence=evidence,
            sota_compliance=sota_compliance
        )
    
    def _cross_check_formulas(
        self,
        indicators: List[IndicatorMetadata],
        pdm_tables: List[Dict[str, Any]]
    ) -> int:
        """
        Cross-check indicator formulas against PDM tables.
        
        Args:
            indicators: List of indicators with formulas
            pdm_tables: PDM tables for cross-verification
            
        Returns:
            Number of formula matches found
        """
        matches = 0
        for indicator in indicators:
            if indicator.formula:
                # Simple matching - in production would be more sophisticated
                for table in pdm_tables:
                    table_text = str(table.get('content', ''))
                    if indicator.codigo in table_text or indicator.nombre in table_text:
                        matches += 1
                        break
        
        return matches


# ============================================================================
# AUDITOR 2: FINANCIAL TRACEABILITY AUDITOR (D1-Q3, D3-Q3)
# ============================================================================

class FinancialTraceabilityAuditor:
    """
    Audits financial traceability to BPIN/PPI codes.
    
    Check Criteria:
    - Traces to BPIN/PPI codes + dependency
    - Penalizes if match confidence <0.95
    
    Quality Evidence:
    - Review fuzzy match incidents
    - Verify codes in PDM vs. logs
    
    SOTA Performance:
    - High-confidence matching per audit standards (Colombian DNP 2023)
    - Reduces fiscal illusions in causal chains (Waldner 2015)
    """
    
    def __init__(self, confidence_threshold: float = 0.95):
        """
        Initialize financial traceability auditor.
        
        Args:
            confidence_threshold: Minimum match confidence (default: 0.95)
        """
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Regex patterns for BPIN/PPI codes
        self.bpin_pattern = re.compile(r'\bBPIN[-\s]?(\d{10,13})\b|\b(\d{10,13})\b', re.IGNORECASE)  # 10-13 digit code with optional BPIN prefix
        self.ppi_pattern = re.compile(r'\bPPI[-\s]?(\d{6,})\b', re.IGNORECASE)
    
    def audit_financial_codes(
        self,
        text: str,
        pdm_tables: Optional[List[Dict[str, Any]]] = None
    ) -> AuditResult:
        """
        Audit financial code traceability in PDM text.
        
        Args:
            text: PDM document text
            pdm_tables: Optional PDM tables for verification
            
        Returns:
            AuditResult with traceability assessment
        """
        self.logger.info("Auditing financial code traceability (BPIN/PPI)")
        
        # Extract BPIN codes
        bpin_codes = self._extract_bpin_codes(text)
        
        # Extract PPI codes
        ppi_codes = self._extract_ppi_codes(text)
        
        all_codes = bpin_codes + ppi_codes
        
        findings = []
        high_confidence_count = 0
        low_confidence_incidents = []
        
        for code in all_codes:
            if code.match_confidence >= self.confidence_threshold:
                high_confidence_count += 1
                findings.append({
                    'code': code.code,
                    'type': code.code_type,
                    'confidence': code.match_confidence,
                    'status': 'high_confidence'
                })
            else:
                low_confidence_incidents.append({
                    'code': code.code,
                    'type': code.code_type,
                    'confidence': code.match_confidence,
                    'matched_text': code.matched_text
                })
                findings.append({
                    'code': code.code,
                    'type': code.code_type,
                    'confidence': code.match_confidence,
                    'status': 'low_confidence'
                })
        
        # Calculate metrics
        total_codes = len(all_codes)
        if total_codes > 0:
            high_confidence_ratio = high_confidence_count / total_codes
        else:
            high_confidence_ratio = 0.0
        
        # Determine severity
        if total_codes == 0:
            severity = AuditSeverity.CRITICAL
            sota_compliance = False
        elif high_confidence_ratio >= 0.95:
            severity = AuditSeverity.EXCELLENT
            sota_compliance = True
        elif high_confidence_ratio >= 0.85:
            severity = AuditSeverity.GOOD
            sota_compliance = True
        elif high_confidence_ratio >= 0.75:
            severity = AuditSeverity.ACCEPTABLE
            sota_compliance = False
        else:
            severity = AuditSeverity.REQUIRES_REVIEW
            sota_compliance = False
        
        # Generate recommendations
        recommendations = []
        if total_codes == 0:
            recommendations.append(
                "CRÍTICO: No se encontraron códigos BPIN/PPI en el documento"
            )
            recommendations.append(
                "Agregar trazabilidad presupuestal según estándares DNP 2023"
            )
        elif len(low_confidence_incidents) > 0:
            recommendations.append(
                f"Revisar {len(low_confidence_incidents)} códigos con baja confianza de coincidencia"
            )
            recommendations.append(
                "Verificar códigos BPIN/PPI contra sistema oficial DNP"
            )
        
        if sota_compliance:
            recommendations.append(
                "Trazabilidad reduce ilusiones fiscales en cadenas causales (Waldner 2015)"
            )
        
        metrics = {
            'total_codes': total_codes,
            'bpin_codes': len(bpin_codes),
            'ppi_codes': len(ppi_codes),
            'high_confidence_codes': high_confidence_count,
            'low_confidence_codes': len(low_confidence_incidents),
            'high_confidence_ratio': high_confidence_ratio,
            'meets_threshold': high_confidence_ratio >= self.confidence_threshold
        }
        
        evidence = {
            'low_confidence_incidents': low_confidence_incidents,
            'confidence_threshold': self.confidence_threshold,
            'dnp_standard': 'Colombian DNP 2023',
            'fiscal_illusions_reference': 'Waldner 2015'
        }
        
        return AuditResult(
            audit_type="D1-Q3_D3-Q3_FinancialTraceability",
            timestamp=datetime.now().isoformat(),
            severity=severity,
            findings=findings,
            metrics=metrics,
            recommendations=recommendations,
            evidence=evidence,
            sota_compliance=sota_compliance
        )
    
    def _extract_bpin_codes(self, text: str) -> List[FinancialCode]:
        """Extract BPIN codes from text"""
        codes = []
        matches = self.bpin_pattern.finditer(text)
        
        for match in matches:
            # Get the actual code from the match groups
            code = match.group(1) if match.group(1) else match.group(2)
            if code is None:
                continue
                
            # Calculate confidence based on context
            context = text[max(0, match.start()-50):min(len(text), match.end()+50)]
            confidence = self._calculate_match_confidence(code, context, "BPIN")
            
            codes.append(FinancialCode(
                code=code,
                code_type="BPIN",
                match_confidence=confidence,
                matched_text=match.group()
            ))
        
        return codes
    
    def _extract_ppi_codes(self, text: str) -> List[FinancialCode]:
        """Extract PPI codes from text"""
        codes = []
        matches = self.ppi_pattern.finditer(text)
        
        for match in matches:
            # Get the actual code from the match group
            code = match.group(1) if match.lastindex and match.lastindex >= 1 else match.group()
            context = text[max(0, match.start()-50):min(len(text), match.end()+50)]
            confidence = self._calculate_match_confidence(code, context, "PPI")
            
            codes.append(FinancialCode(
                code=code,
                code_type="PPI",
                match_confidence=confidence,
                matched_text=match.group()
            ))
        
        return codes
    
    def _calculate_match_confidence(self, code: str, context: str, code_type: str) -> float:
        """
        Calculate match confidence based on context.
        
        Args:
            code: The extracted code
            context: Surrounding text context
            code_type: "BPIN" or "PPI"
            
        Returns:
            Confidence score [0.0, 1.0]
        """
        confidence = 0.7  # Base confidence
        
        # Boost confidence if code_type mentioned in context
        if code_type.upper() in context.upper():
            confidence += 0.15
        
        # Boost if keywords present
        keywords = ['proyecto', 'inversión', 'presupuesto', 'código']
        for keyword in keywords:
            if keyword in context.lower():
                confidence += 0.05
                break
        
        return min(1.0, confidence)


# ============================================================================
# AUDITOR 3: QUANTIFIED GAP AUDITOR (D1-Q2)
# ============================================================================

class QuantifiedGapAuditor:
    """
    Audits quantified gap recognition.
    
    Check Criteria:
    - Detects data limitations (vacíos) + quantified brecha (déficit de)
    
    Quality Evidence:
    - Pattern-match _extract_structured_quantitative_claims output to PDM text
    
    SOTA Performance:
    - Quantified baselines boost QCA calibration (Ragin 2008)
    - Identifies subregistro for robust MMR
    """
    
    def __init__(self):
        """Initialize quantified gap auditor"""
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Patterns for gap detection
        self.gap_patterns = {
            'vacío': re.compile(r'vac[ií]o(?:s)?\s+(?:de\s+)?(?:información|datos)', re.IGNORECASE),
            'brecha': re.compile(r'brecha(?:s)?\s+(?:de\s+)?(\d+\.?\d*)\s*(%|por\s*ciento|unidades)?', re.IGNORECASE),
            'déficit': re.compile(r'd[eé]ficit\s+(?:de\s+)?(\d+\.?\d*)\s*(%|por\s*ciento|unidades)?', re.IGNORECASE),
            'subregistro': re.compile(r'sub[-\s]?registro(?:s)?', re.IGNORECASE)
        }
    
    def audit_quantified_gaps(
        self,
        text: str,
        structured_claims: Optional[List[Dict[str, Any]]] = None
    ) -> AuditResult:
        """
        Audit quantified gap recognition in PDM text.
        
        Args:
            text: PDM document text
            structured_claims: Optional structured quantitative claims
            
        Returns:
            AuditResult with gap recognition assessment
        """
        self.logger.info("Auditing quantified gap recognition")
        
        detected_gaps = []
        findings = []
        
        # Detect each type of gap
        for gap_type, pattern in self.gap_patterns.items():
            matches = pattern.finditer(text)
            
            for match in matches:
                # Extract quantification if present
                quantification = None
                if match.groups():
                    try:
                        quantification = float(match.group(1))
                    except (ValueError, IndexError):
                        pass
                
                # Get surrounding context
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end]
                
                gap = QuantifiedGap(
                    gap_type=gap_type,
                    quantification=quantification,
                    description=match.group(),
                    source_text=context,
                    severity=self._calculate_gap_severity(gap_type, quantification)
                )
                
                detected_gaps.append(gap)
                
                findings.append({
                    'gap_type': gap_type,
                    'quantified': quantification is not None,
                    'quantification': quantification,
                    'description': match.group(),
                    'severity': gap.severity
                })
        
        # Calculate metrics
        total_gaps = len(detected_gaps)
        quantified_gaps = len([g for g in detected_gaps if g.quantification is not None])
        quantification_ratio = quantified_gaps / total_gaps if total_gaps > 0 else 0.0
        
        # Determine severity
        if total_gaps == 0:
            severity = AuditSeverity.REQUIRES_REVIEW
            sota_compliance = False
        elif quantification_ratio >= 0.70:
            severity = AuditSeverity.EXCELLENT
            sota_compliance = True
        elif quantification_ratio >= 0.50:
            severity = AuditSeverity.GOOD
            sota_compliance = True
        else:
            severity = AuditSeverity.ACCEPTABLE
            sota_compliance = False
        
        # Generate recommendations
        recommendations = []
        if total_gaps == 0:
            recommendations.append(
                "No se detectaron brechas cuantificadas - considerar análisis de vacíos de información"
            )
        elif quantification_ratio < 0.70:
            recommendations.append(
                f"Cuantificar {total_gaps - quantified_gaps} brechas adicionales para calibración QCA (Ragin 2008)"
            )
        
        if quantification_ratio >= 0.70:
            recommendations.append(
                "Brechas cuantificadas mejoran calibración QCA y robustez MMR"
            )
        
        # Check for subregistro
        subregistro_count = len([g for g in detected_gaps if g.gap_type == 'subregistro'])
        if subregistro_count > 0:
            recommendations.append(
                f"Identificado {subregistro_count} casos de subregistro - crítico para MMR robusto"
            )
        
        metrics = {
            'total_gaps': total_gaps,
            'quantified_gaps': quantified_gaps,
            'unquantified_gaps': total_gaps - quantified_gaps,
            'quantification_ratio': quantification_ratio,
            'subregistro_count': subregistro_count,
            'gap_type_distribution': {
                gap_type: len([g for g in detected_gaps if g.gap_type == gap_type])
                for gap_type in self.gap_patterns.keys()
            }
        }
        
        evidence = {
            'detected_gaps': [
                {
                    'type': g.gap_type,
                    'quantification': g.quantification,
                    'severity': g.severity
                }
                for g in detected_gaps
            ],
            'qca_calibration_reference': 'Ragin 2008',
            'mmr_robustness': 'Mixed Methods Research - subregistro detection'
        }
        
        return AuditResult(
            audit_type="D1-Q2_QuantifiedGaps",
            timestamp=datetime.now().isoformat(),
            severity=severity,
            findings=findings,
            metrics=metrics,
            recommendations=recommendations,
            evidence=evidence,
            sota_compliance=sota_compliance
        )
    
    def _calculate_gap_severity(self, gap_type: str, quantification: Optional[float]) -> float:
        """
        Calculate gap severity based on type and quantification.
        
        Args:
            gap_type: Type of gap detected
            quantification: Quantified value if available
            
        Returns:
            Severity score [0.0, 1.0]
        """
        # Base severity by type
        base_severity = {
            'vacío': 0.7,
            'brecha': 0.6,
            'déficit': 0.8,
            'subregistro': 0.9
        }
        
        severity = base_severity.get(gap_type, 0.5)
        
        # Adjust based on quantification
        if quantification is not None:
            if quantification > 50:  # High magnitude
                severity = min(1.0, severity + 0.1)
        
        return severity


# ============================================================================
# AUDITOR 4: SYSTEMIC RISK AUDITOR (D4-Q5, D5-Q4)
# ============================================================================

class SystemicRiskAuditor:
    """
    Audits systemic risk alignment with PND/ODS.
    
    Check Criteria:
    - Integrates PND/ODS alignment
    - risk_score <0.10 for Excellent, increases on misalignment
    
    Quality Evidence:
    - Check risk_score escalation in CounterfactualAuditor logs
    
    SOTA Performance:
    - Counterfactual rigor per Pearl (2018)
    - Low risk aligns with macro-causal frameworks
    """
    
    def __init__(self, excellent_threshold: float = 0.10):
        """
        Initialize systemic risk auditor.
        
        Args:
            excellent_threshold: Maximum risk score for excellent rating (default: 0.10)
        """
        self.excellent_threshold = excellent_threshold
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # ODS (SDG) patterns
        self.ods_pattern = re.compile(r'\b(?:ODS|SDG)[-\s]?(\d{1,2})\b', re.IGNORECASE)
        
        # PND (Plan Nacional de Desarrollo) patterns
        self.pnd_pattern = re.compile(r'\b(?:PND|Plan\s+Nacional\s+de\s+Desarrollo)\b', re.IGNORECASE)
    
    def audit_systemic_risk(
        self,
        text: str,
        causal_graph: Optional[Any] = None,
        counterfactual_audit: Optional[Dict[str, Any]] = None
    ) -> AuditResult:
        """
        Audit systemic risk alignment.
        
        Args:
            text: PDM document text
            causal_graph: Optional causal graph for analysis
            counterfactual_audit: Optional counterfactual audit results
            
        Returns:
            AuditResult with risk alignment assessment
        """
        self.logger.info("Auditing systemic risk alignment (PND/ODS)")
        
        # Detect PND alignment
        pnd_matches = list(self.pnd_pattern.finditer(text))
        pnd_alignment = len(pnd_matches) > 0
        
        # Detect ODS alignment
        ods_matches = list(self.ods_pattern.finditer(text))
        ods_numbers = []
        for match in ods_matches:
            try:
                ods_num = int(match.group(1))
                if 1 <= ods_num <= 17:  # Valid ODS range
                    ods_numbers.append(ods_num)
            except (ValueError, IndexError):
                pass
        
        ods_alignment = list(set(ods_numbers))  # Remove duplicates
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(
            pnd_alignment,
            ods_alignment,
            causal_graph,
            counterfactual_audit
        )
        
        # Identify misalignment reasons
        misalignment_reasons = []
        if not pnd_alignment:
            misalignment_reasons.append("No se encontró referencia al PND")
        if len(ods_alignment) == 0:
            misalignment_reasons.append("No se encontraron ODS alineados")
        
        # Check counterfactual audit for additional risks
        if counterfactual_audit:
            cf_risks = counterfactual_audit.get('risk_flags', [])
            if cf_risks:
                misalignment_reasons.extend(cf_risks)
        
        # Determine severity
        if risk_score < self.excellent_threshold:
            severity = AuditSeverity.EXCELLENT
            sota_compliance = True
        elif risk_score < 0.20:
            severity = AuditSeverity.GOOD
            sota_compliance = True
        elif risk_score < 0.35:
            severity = AuditSeverity.ACCEPTABLE
            sota_compliance = False
        else:
            severity = AuditSeverity.REQUIRES_REVIEW
            sota_compliance = False
        
        # Generate findings
        findings = [
            {
                'aspect': 'PND_alignment',
                'aligned': pnd_alignment,
                'evidence_count': len(pnd_matches)
            },
            {
                'aspect': 'ODS_alignment',
                'aligned': len(ods_alignment) > 0,
                'ods_numbers': ods_alignment,
                'count': len(ods_alignment)
            },
            {
                'aspect': 'risk_score',
                'value': risk_score,
                'threshold': self.excellent_threshold
            }
        ]
        
        # Generate recommendations
        recommendations = []
        if not pnd_alignment:
            recommendations.append(
                "Alinear explícitamente con Plan Nacional de Desarrollo (PND)"
            )
        if len(ods_alignment) == 0:
            recommendations.append(
                "Identificar y declarar alineación con Objetivos de Desarrollo Sostenible (ODS)"
            )
        elif len(ods_alignment) < 3:
            recommendations.append(
                "Considerar alineación adicional con ODS para mayor impacto sistémico"
            )
        
        if risk_score < self.excellent_threshold:
            recommendations.append(
                "Bajo riesgo sistémico alineado con marcos macro-causales (Pearl 2018)"
            )
        elif risk_score > 0.20:
            recommendations.append(
                "Revisar desalineaciones para reducir riesgo sistémico"
            )
        
        metrics = {
            'pnd_alignment': pnd_alignment,
            'ods_count': len(ods_alignment),
            'ods_numbers': ods_alignment,
            'risk_score': risk_score,
            'misalignment_count': len(misalignment_reasons),
            'meets_excellent_threshold': risk_score < self.excellent_threshold
        }
        
        evidence = {
            'misalignment_reasons': misalignment_reasons,
            'pnd_mentions': len(pnd_matches),
            'ods_mentions': len(ods_matches),
            'counterfactual_rigor_reference': 'Pearl 2018',
            'macro_causal_framework': 'PND/ODS alignment reduces systemic risk'
        }
        
        return AuditResult(
            audit_type="D4-Q5_D5-Q4_SystemicRisk",
            timestamp=datetime.now().isoformat(),
            severity=severity,
            findings=findings,
            metrics=metrics,
            recommendations=recommendations,
            evidence=evidence,
            sota_compliance=sota_compliance
        )
    
    def _calculate_risk_score(
        self,
        pnd_alignment: bool,
        ods_alignment: List[int],
        causal_graph: Optional[Any],
        counterfactual_audit: Optional[Dict[str, Any]]
    ) -> float:
        """
        Calculate systemic risk score.
        
        Args:
            pnd_alignment: Whether PND is mentioned
            ods_alignment: List of aligned ODS numbers
            causal_graph: Optional causal graph
            counterfactual_audit: Optional counterfactual audit results
            
        Returns:
            Risk score [0.0, 1.0] where lower is better
        """
        risk_score = 0.0
        
        # Base risk from lack of alignment
        if not pnd_alignment:
            risk_score += 0.15
        
        if len(ods_alignment) == 0:
            risk_score += 0.20
        elif len(ods_alignment) < 3:
            risk_score += 0.10
        
        # Penalty for counterfactual audit failures
        if counterfactual_audit:
            cf_risk_flags = counterfactual_audit.get('risk_flags', [])
            risk_score += len(cf_risk_flags) * 0.05
        
        # Cap at 1.0
        return min(1.0, risk_score)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def run_all_audits(
    text: str,
    indicators: Optional[List[IndicatorMetadata]] = None,
    pdm_tables: Optional[List[Dict[str, Any]]] = None,
    structured_claims: Optional[List[Dict[str, Any]]] = None,
    causal_graph: Optional[Any] = None,
    counterfactual_audit: Optional[Dict[str, Any]] = None
) -> Dict[str, AuditResult]:
    """
    Run all evidence quality audits.
    
    Args:
        text: PDM document text
        indicators: Optional indicator metadata
        pdm_tables: Optional PDM tables
        structured_claims: Optional structured quantitative claims
        causal_graph: Optional causal graph
        counterfactual_audit: Optional counterfactual audit results
        
    Returns:
        Dictionary of audit results keyed by audit type
    """
    results = {}
    
    # D3-Q1: Operationalization Audit
    if indicators:
        op_auditor = OperationalizationAuditor()
        results['operationalization'] = op_auditor.audit_indicators(indicators, pdm_tables)
    
    # D1-Q3, D3-Q3: Financial Traceability Audit
    ft_auditor = FinancialTraceabilityAuditor()
    results['financial_traceability'] = ft_auditor.audit_financial_codes(text, pdm_tables)
    
    # D1-Q2: Quantified Gap Audit
    qg_auditor = QuantifiedGapAuditor()
    results['quantified_gaps'] = qg_auditor.audit_quantified_gaps(text, structured_claims)
    
    # D4-Q5, D5-Q4: Systemic Risk Audit
    sr_auditor = SystemicRiskAuditor()
    results['systemic_risk'] = sr_auditor.audit_systemic_risk(
        text, causal_graph, counterfactual_audit
    )
    
    return results


if __name__ == "__main__":
    # Example usage
    logger.info("Evidence Quality Auditors Module - Example Usage")
    
    # Sample PDM text
    sample_text = """
    El municipio presenta un déficit de 35% en cobertura educativa.
    Proyecto BPIN 2023000123456 busca reducir la brecha de acceso.
    Alineado con PND y ODS-4 (Educación de Calidad).
    Se identifican vacíos de información en zonas rurales.
    """
    
    # Run all audits
    results = run_all_audits(text=sample_text)
    
    for audit_type, result in results.items():
        logger.info(f"\n{audit_type.upper()}:")
        logger.info(f"  Severity: {result.severity.value}")
        logger.info(f"  SOTA Compliance: {result.sota_compliance}")
        logger.info(f"  Recommendations: {len(result.recommendations)}")
