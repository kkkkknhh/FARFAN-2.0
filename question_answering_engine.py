#!/usr/bin/env python3
"""
Question Answering Engine for FARFAN 2.0
Sistema de respuesta a las 300 preguntas del cuestionario de evaluación causal

Este módulo:
1. Carga el cuestionario de 300 preguntas (30 base × 10 áreas)
2. Mapea qué módulos/funciones responden cada pregunta
3. Orquesta la respuesta coordinando todos los módulos
4. Genera respuestas estructuradas (respuesta + argumento + nota cuantitativa)
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger("question_answering_engine")


class DimensionCausal(Enum):
    """Dimensiones analíticas del Marco Lógico"""
    D1_INSUMOS = "D1"  # Diagnóstico y Líneas Base
    D2_ACTIVIDADES = "D2"  # Actividades Formalizadas
    D3_PRODUCTOS = "D3"  # Productos Verificables
    D4_RESULTADOS = "D4"  # Resultados Medibles
    D5_IMPACTOS = "D5"  # Impactos de Largo Plazo
    D6_CAUSALIDAD = "D6"  # Teoría de Cambio Explícita


class PuntoDecalogo(Enum):
    """10 áreas temáticas del decálogo"""
    P1_SEGURIDAD = "P1"
    P2_ALERTAS_TEMPRANAS = "P2"
    P3_AMBIENTE = "P3"
    P4_DERECHOS_BASICOS = "P4"
    P5_VICTIMAS = "P5"
    P6_NINEZ_JUVENTUD = "P6"
    P7_RURAL = "P7"
    P8_LIDERES_SOCIALES = "P8"
    P9_CARCEL = "P9"
    P10_MIGRACION = "P10"


@dataclass
class PreguntaBase:
    """Pregunta base que se replica en las 10 áreas"""
    id_base: str  # D1-Q1, D1-Q2, etc.
    dimension: DimensionCausal
    numero: int
    texto_template: str
    criterios_evaluacion: Dict[str, Any]
    scoring: Dict[str, Any]
    modulos_responsables: List[str]  # Qué módulos la responden


@dataclass
class RespuestaPregunta:
    """Respuesta estructurada a una pregunta"""
    pregunta_id: str  # ej: "P1-D1-Q1"
    respuesta_texto: str  # La respuesta directa
    argumento: str  # Mínimo 2 párrafos, nivel doctoral
    nota_cuantitativa: float  # 0.0 a 1.0
    evidencia: List[str]  # Referencias al documento
    modulos_utilizados: List[str]  # Qué módulos contribuyeron
    nivel_confianza: float  # 0.0 a 1.0
    

class QuestionAnsweringEngine:
    """
    Motor de respuesta a preguntas que coordina todos los módulos
    para generar respuestas estructuradas y fundamentadas
    """
    
    def __init__(self, cdaf=None, dnp_validator=None, competencias=None, 
                 mga_catalog=None, pdet_lineamientos=None):
        """
        Initialize with all available modules
        
        Args:
            cdaf: CDAFFramework instance
            dnp_validator: ValidadorDNP instance
            competencias: CatalogoCompetenciasMunicipales instance
            mga_catalog: CatalogoIndicadoresMGA instance
            pdet_lineamientos: LineamientosPDET instance
        """
        self.cdaf = cdaf
        self.dnp_validator = dnp_validator
        self.competencias = competencias
        self.mga_catalog = mga_catalog
        self.pdet_lineamientos = pdet_lineamientos
        
        # Load question templates
        self.preguntas_base = self._load_question_templates()
        
        logger.info(f"QuestionAnsweringEngine inicializado con {len(self.preguntas_base)} preguntas base")
    
    def _load_question_templates(self) -> Dict[str, PreguntaBase]:
        """Carga las 30 preguntas base desde la configuración"""
        
        # Preguntas base hardcoded (en producción, cargar de JSON)
        preguntas = {}
        
        # D1: Diagnóstico y Líneas Base (5 preguntas)
        preguntas["D1-Q1"] = PreguntaBase(
            id_base="D1-Q1",
            dimension=DimensionCausal.D1_INSUMOS,
            numero=1,
            texto_template="¿El plan presenta líneas base cuantitativas con fuentes oficiales para el área de {}?",
            criterios_evaluacion={
                "indicadores_cuantitativos": 3,
                "fuentes_oficiales": 2,
                "series_temporales_años": 3
            },
            scoring={
                "excellent": {"min_score": 0.85, "criteria": "5+ indicadores, fuentes oficiales, 5+ años"},
                "good": {"min_score": 0.70, "criteria": "3-4 indicadores, 2+ fuentes, 3-4 años"},
                "acceptable": {"min_score": 0.55, "criteria": "2 indicadores, 1-2 fuentes"},
                "poor": {"min_score": 0.0, "criteria": "< 2 indicadores o sin fuentes"}
            },
            modulos_responsables=["initial_processor_causal_policy", "dereck_beach"]
        )
        
        preguntas["D1-Q2"] = PreguntaBase(
            id_base="D1-Q2",
            dimension=DimensionCausal.D1_INSUMOS,
            numero=2,
            texto_template="¿Se cuantifica la magnitud del problema y se reconocen vacíos de información en {}?",
            criterios_evaluacion={
                "cuantificacion_brecha": True,
                "vacios_explicitos": True,
                "sesgos_reconocidos": True
            },
            scoring={
                "excellent": {"min_score": 0.85},
                "good": {"min_score": 0.70},
                "acceptable": {"min_score": 0.55},
                "poor": {"min_score": 0.0}
            },
            modulos_responsables=["dereck_beach", "initial_processor_causal_policy"]
        )
        
        preguntas["D1-Q3"] = PreguntaBase(
            id_base="D1-Q3",
            dimension=DimensionCausal.D1_INSUMOS,
            numero=3,
            texto_template="¿Los recursos asignados son trazables y suficientes para {} según el diagnóstico?",
            criterios_evaluacion={
                "trazabilidad_ppi_bpin": True,
                "suficiencia_justificada": True
            },
            scoring={
                "excellent": {"min_score": 0.85},
                "good": {"min_score": 0.70},
                "acceptable": {"min_score": 0.55},
                "poor": {"min_score": 0.0}
            },
            modulos_responsables=["dereck_beach", "dnp_integration"]
        )
        
        preguntas["D1-Q4"] = PreguntaBase(
            id_base="D1-Q4",
            dimension=DimensionCausal.D1_INSUMOS,
            numero=4,
            texto_template="¿Se identifican capacidades institucionales y cuellos de botella para {}?",
            criterios_evaluacion={
                "capacidades_descritas": 4,
                "cuellos_botella_identificados": True
            },
            scoring={
                "excellent": {"min_score": 0.85},
                "good": {"min_score": 0.70},
                "acceptable": {"min_score": 0.55},
                "poor": {"min_score": 0.0}
            },
            modulos_responsables=["dereck_beach", "competencias_municipales"]
        )
        
        preguntas["D1-Q5"] = PreguntaBase(
            id_base="D1-Q5",
            dimension=DimensionCausal.D1_INSUMOS,
            numero=5,
            texto_template="¿Hay coherencia entre objetivos y recursos considerando restricciones legales, presupuestales y temporales en {}?",
            criterios_evaluacion={
                "restricciones_modeladas": 3,
                "coherencia_demostrada": True
            },
            scoring={
                "excellent": {"min_score": 0.85},
                "good": {"min_score": 0.70},
                "acceptable": {"min_score": 0.55},
                "poor": {"min_score": 0.0}
            },
            modulos_responsables=["dnp_integration", "dereck_beach"]
        )
        
        # D2: Actividades Formalizadas (5 preguntas)
        preguntas["D2-Q6"] = PreguntaBase(
            id_base="D2-Q6",
            dimension=DimensionCausal.D2_ACTIVIDADES,
            numero=6,
            texto_template="¿Las actividades de {} están formalizadas en estructura tabular con responsables, insumos, productos, cronograma y costos?",
            criterios_evaluacion={
                "estructura_tabular": True,
                "columnas_completas": 6
            },
            scoring={
                "excellent": {"min_score": 0.85},
                "good": {"min_score": 0.70},
                "acceptable": {"min_score": 0.55},
                "poor": {"min_score": 0.0}
            },
            modulos_responsables=["dereck_beach"]
        )
        
        # Continue with D2-Q7 through D6-Q30...
        # For brevity, adding a few more key questions
        
        preguntas["D3-Q11"] = PreguntaBase(
            id_base="D3-Q11",
            dimension=DimensionCausal.D3_PRODUCTOS,
            numero=11,
            texto_template="¿Los productos de {} tienen indicadores con fórmula, fuente, línea base y meta?",
            criterios_evaluacion={
                "formula_indicador": True,
                "fuente_verificacion": True,
                "linea_base": True,
                "meta": True
            },
            scoring={
                "excellent": {"min_score": 0.85},
                "good": {"min_score": 0.70},
                "acceptable": {"min_score": 0.55},
                "poor": {"min_score": 0.0}
            },
            modulos_responsables=["mga_indicadores", "dereck_beach"]
        )
        
        preguntas["D4-Q16"] = PreguntaBase(
            id_base="D4-Q16",
            dimension=DimensionCausal.D4_RESULTADOS,
            numero=16,
            texto_template="¿Los resultados de {} están definidos con métrica, línea base, meta y ventana de maduración?",
            criterios_evaluacion={
                "metrica_outcome": True,
                "linea_base": True,
                "meta": True,
                "ventana_maduracion": True
            },
            scoring={
                "excellent": {"min_score": 0.85},
                "good": {"min_score": 0.70},
                "acceptable": {"min_score": 0.55},
                "poor": {"min_score": 0.0}
            },
            modulos_responsables=["dereck_beach", "mga_indicadores"]
        )
        
        preguntas["D6-Q26"] = PreguntaBase(
            id_base="D6-Q26",
            dimension=DimensionCausal.D6_CAUSALIDAD,
            numero=26,
            texto_template="¿Existe una teoría de cambio explícita para {} con causas, mediadores, moderadores y supuestos verificables?",
            criterios_evaluacion={
                "diagrama_explicito": True,
                "causas": True,
                "mediadores": True,
                "moderadores": True,
                "supuestos_verificables": True
            },
            scoring={
                "excellent": {"min_score": 0.85},
                "good": {"min_score": 0.70},
                "acceptable": {"min_score": 0.55},
                "poor": {"min_score": 0.0}
            },
            modulos_responsables=["dereck_beach", "initial_processor_causal_policy"]
        )
        
        # TODO: Add all 30 preguntas base
        
        return preguntas
    
    def answer_all_questions(self, pipeline_context) -> Dict[str, RespuestaPregunta]:
        """
        Responde las 300 preguntas (30 base × 10 áreas de política)
        
        Args:
            pipeline_context: Contexto del pipeline con datos extraídos
        
        Returns:
            Dict mapping question_id to RespuestaPregunta
        """
        logger.info("Respondiendo 300 preguntas del cuestionario...")
        
        respuestas = {}
        
        # Para cada punto del decálogo (10 áreas)
        for punto in PuntoDecalogo:
            area_nombre = self._get_area_nombre(punto)
            
            # Para cada pregunta base (30 preguntas)
            for pregunta_id, pregunta_base in self.preguntas_base.items():
                # Generate question ID: P1-D1-Q1, P1-D1-Q2, etc.
                question_full_id = f"{punto.value}-{pregunta_id}"
                
                # Answer the question
                respuesta = self._answer_single_question(
                    pregunta_base, punto, area_nombre, pipeline_context
                )
                
                respuestas[question_full_id] = respuesta
        
        logger.info(f"✓ {len(respuestas)} preguntas respondidas")
        
        return respuestas
    
    def _answer_single_question(self, pregunta: PreguntaBase, punto: PuntoDecalogo,
                                area_nombre: str, ctx) -> RespuestaPregunta:
        """
        Responde una pregunta específica utilizando los módulos correspondientes
        
        Este es el corazón del sistema: coordina los módulos para generar
        una respuesta fundamentada.
        """
        # Instantiate the question text
        pregunta_texto = pregunta.texto_template.format(area_nombre)
        
        # Collect evidence from different modules
        evidencia = []
        modulos_usados = []
        
        # Use CDAF for causal analysis
        if self.cdaf and "dereck_beach" in pregunta.modulos_responsables:
            # Extract relevant nodes for this area
            area_nodes = self._filter_nodes_by_area(ctx.nodes, area_nombre)
            evidencia.extend([f"Nodo {nid}: {node.text[:100]}..." 
                            for nid, node in list(area_nodes.items())[:3]])
            modulos_usados.append("dereck_beach")
        
        # Use DNP Validator
        if self.dnp_validator and "dnp_integration" in pregunta.modulos_responsables:
            # Check DNP compliance for this area
            area_validations = [v for v in ctx.dnp_validation_results 
                               if self._matches_area(v, area_nombre)]
            if area_validations:
                avg_score = sum(v['resultado'].score_total for v in area_validations) / len(area_validations)
                evidencia.append(f"Score DNP promedio: {avg_score:.1f}/100")
                modulos_usados.append("dnp_integration")
        
        # Use MGA Catalog
        if self.mga_catalog and "mga_indicadores" in pregunta.modulos_responsables:
            # Find relevant MGA indicators
            sector_key = self._map_area_to_sector(area_nombre)
            indicators = self.mga_catalog.buscar_por_sector(sector_key)
            if indicators:
                evidencia.append(f"Indicadores MGA disponibles: {len(indicators)}")
                modulos_usados.append("mga_indicadores")
        
        # Use Competencias
        if self.competencias and "competencias_municipales" in pregunta.modulos_responsables:
            # Check municipal competencies
            sector_key = self._map_area_to_sector(area_nombre)
            competencia = self.competencias.validar_competencia_municipal(sector_key)
            if competencia:
                evidencia.append(f"Competencia municipal: {competencia['tipo']}")
                modulos_usados.append("competencias_municipales")
        
        # Generate score based on criteria and evidence
        nota = self._calculate_score(pregunta, evidencia, ctx)
        
        # Generate response text
        respuesta_texto = self._generate_response_text(pregunta, evidencia, nota)
        
        # Generate argument (minimum 2 paragraphs, doctoral level)
        argumento = self._generate_argument(pregunta, evidencia, nota, area_nombre, ctx)
        
        return RespuestaPregunta(
            pregunta_id=f"{punto.value}-{pregunta.id_base}",
            respuesta_texto=respuesta_texto,
            argumento=argumento,
            nota_cuantitativa=nota,
            evidencia=evidencia,
            modulos_utilizados=modulos_usados,
            nivel_confianza=min(0.9, len(evidencia) / 5.0)  # More evidence = higher confidence
        )
    
    def _filter_nodes_by_area(self, nodes: Dict, area_nombre: str) -> Dict:
        """Filtra nodos relevantes para un área temática"""
        # Simplified: filter by keyword matching
        keywords = self._get_area_keywords(area_nombre)
        
        filtered = {}
        for nid, node in nodes.items():
            if any(kw in node.text.lower() for kw in keywords):
                filtered[nid] = node
        
        return filtered
    
    def _get_area_keywords(self, area_nombre: str) -> List[str]:
        """Obtiene palabras clave para filtrar por área"""
        keywords_map = {
            "Seguridad y Convivencia": ["seguridad", "policía", "convivencia", "violencia"],
            "Alertas Tempranas": ["alerta", "prevención", "riesgo", "defensoría"],
            "Ambiente y Recursos Naturales": ["ambiente", "agua", "reforestación", "residuos"],
            "Derechos Básicos": ["educación", "salud", "vivienda", "servicios"],
            "Víctimas": ["víctima", "reparación", "restitución", "desplazamiento"],
            "Niñez y Juventud": ["niño", "niña", "joven", "adolescente", "educación"],
            "Desarrollo Rural": ["rural", "campesino", "agropecuario", "tierra"],
            "Líderes Sociales": ["líder", "defensor", "protección", "amenaza"],
            "Sistema Carcelario": ["cárcel", "reclusión", "ppl", "privados de libertad"],
            "Migración": ["migrante", "venezolano", "extranjero", "frontera"]
        }
        return keywords_map.get(area_nombre, [area_nombre.lower()])
    
    def _matches_area(self, validation: Dict, area_nombre: str) -> bool:
        """Verifica si una validación corresponde al área"""
        # Simplified matching
        return True  # TODO: implement proper matching
    
    def _map_area_to_sector(self, area_nombre: str) -> str:
        """Mapea área temática a sector para consultas"""
        sector_map = {
            "Seguridad y Convivencia": "seguridad_convivencia",
            "Alertas Tempranas": "seguridad_convivencia",
            "Ambiente y Recursos Naturales": "medio_ambiente",
            "Derechos Básicos": "educacion",  # Multiple sectors
            "Víctimas": "atencion_grupos_vulnerables",
            "Niñez y Juventud": "educacion",
            "Desarrollo Rural": "desarrollo_rural",
            "Líderes Sociales": "seguridad_convivencia",
            "Sistema Carcelario": "justicia_seguridad",
            "Migración": "atencion_grupos_vulnerables"
        }
        return sector_map.get(area_nombre, "general")
    
    def _calculate_score(self, pregunta: PreguntaBase, evidencia: List[str], ctx) -> float:
        """Calcula la nota cuantitativa basada en criterios y evidencia"""
        # Base score on amount and quality of evidence
        base_score = min(0.5, len(evidencia) / 10.0)
        
        # Adjust based on scoring criteria
        if len(evidencia) >= 5:
            return 0.85  # Excellent
        elif len(evidencia) >= 3:
            return 0.70  # Good
        elif len(evidencia) >= 1:
            return 0.55  # Acceptable
        else:
            return 0.30  # Poor
    
    def _generate_response_text(self, pregunta: PreguntaBase, 
                                evidencia: List[str], nota: float) -> str:
        """Genera el texto de respuesta directa"""
        if nota >= 0.85:
            return f"Sí, el plan cumple excelentemente con este criterio. Se encontraron {len(evidencia)} elementos de evidencia que lo respaldan."
        elif nota >= 0.70:
            return f"Sí, el plan cumple adecuadamente con este criterio. Se identificaron {len(evidencia)} elementos de soporte."
        elif nota >= 0.55:
            return f"Parcialmente. El plan presenta algunos elementos ({len(evidencia)} identificados) pero requiere fortalecimiento."
        else:
            return f"No, el plan no cumple adecuadamente con este criterio. Evidencia insuficiente."
    
    def _generate_argument(self, pregunta: PreguntaBase, evidencia: List[str], 
                           nota: float, area_nombre: str, ctx) -> str:
        """
        Genera argumento de nivel doctoral (mínimo 2 párrafos)
        
        El argumento debe:
        1. Contextualizar la pregunta en el marco teórico de evaluación causal
        2. Presentar la evidencia encontrada de manera sistemática
        3. Analizar las implicaciones para la efectividad del plan
        4. Proponer recomendaciones específicas
        """
        # Paragraph 1: Theoretical framing and findings
        parrafo_1 = f"""
La evaluación de {area_nombre} en el marco de la dimensión {pregunta.dimension.value} 
requiere un análisis riguroso de la evidencia documental disponible. Desde la perspectiva
de la teoría de cambio y el marco lógico, esta pregunta indaga sobre elementos fundamentales
que determinan la viabilidad y trazabilidad de las intervenciones propuestas. En el análisis
del documento se identificaron {len(evidencia)} elementos de evidencia que permiten
fundamentar la evaluación. La nota cuantitativa de {nota:.2f} refleja el nivel de cumplimiento
observado, considerando los criterios de evaluación establecidos por los estándares DNP y
las mejores prácticas en formulación de políticas públicas territoriales.
"""
        
        # Paragraph 2: Specific evidence and recommendations
        evidencia_text = "; ".join(evidencia[:3]) if evidencia else "No se encontró evidencia documental"
        
        if nota >= 0.70:
            parrafo_2 = f"""
La evidencia encontrada ({evidencia_text}) sugiere un nivel de cumplimiento satisfactorio
que permite anticipar una implementación efectiva de las intervenciones en {area_nombre}.
Los elementos identificados demuestran coherencia con los principios de causalidad,
medibilidad y trazabilidad que requiere la gestión basada en resultados. Sin embargo,
se recomienda fortalecer la documentación explícita de supuestos y condiciones habilitantes,
así como la especificación de mecanismos de seguimiento y evaluación que permitan validar
la teoría de cambio subyacente. La integración con los estándares DNP (competencias municipales,
indicadores MGA y lineamientos PDET donde apliquen) debe ser verificada para garantizar
la alineación institucional y la sostenibilidad de los resultados esperados.
"""
        else:
            parrafo_2 = f"""
La evidencia limitada ({evidencia_text}) sugiere debilidades significativas en la formulación
que podrían comprometer la efectividad de las intervenciones en {area_nombre}. Es imperativo
que el plan fortalezca los elementos faltantes identificados en esta evaluación, particularmente
en lo relacionado con {pregunta.texto_template[:50]}... Se recomienda una reformulación
que incorpore de manera explícita los criterios de evaluación establecidos, incluyendo
líneas base cuantitativas, fuentes de verificación, especificación de mecanismos causales,
y alineación con estándares técnicos DNP. Sin estas correcciones, existe un riesgo alto
de que las intervenciones no logren los resultados esperados o que su seguimiento y evaluación
sean inadecuados para la rendición de cuentas y el aprendizaje institucional.
"""
        
        return parrafo_1.strip() + "\n\n" + parrafo_2.strip()
    
    def _get_area_nombre(self, punto: PuntoDecalogo) -> str:
        """Obtiene el nombre completo del área temática"""
        nombres = {
            PuntoDecalogo.P1_SEGURIDAD: "Seguridad y Convivencia",
            PuntoDecalogo.P2_ALERTAS_TEMPRANAS: "Alertas Tempranas",
            PuntoDecalogo.P3_AMBIENTE: "Ambiente y Recursos Naturales",
            PuntoDecalogo.P4_DERECHOS_BASICOS: "Derechos Básicos",
            PuntoDecalogo.P5_VICTIMAS: "Víctimas",
            PuntoDecalogo.P6_NINEZ_JUVENTUD: "Niñez y Juventud",
            PuntoDecalogo.P7_RURAL: "Desarrollo Rural",
            PuntoDecalogo.P8_LIDERES_SOCIALES: "Líderes Sociales",
            PuntoDecalogo.P9_CARCEL: "Sistema Carcelario",
            PuntoDecalogo.P10_MIGRACION: "Migración"
        }
        return nombres[punto]
