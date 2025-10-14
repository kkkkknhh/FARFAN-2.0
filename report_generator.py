#!/usr/bin/env python3
"""
Report Generator for FARFAN 2.0
Generación de reportes a tres niveles: Micro, Meso y Macro

NIVEL MICRO: Reporte individual de las 300 preguntas
NIVEL MESO: Agrupación en 4 clústeres por 6 dimensiones analíticas
NIVEL MACRO: Evaluación global de alineación con el decálogo (retrospectiva y prospectiva)
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger("report_generator")


class ClusterMeso(Enum):
    """4 Clústeres para agrupación meso"""
    C1_SEGURIDAD_PAZ = "C1"  # P1, P2, P8 (Seguridad, Alertas, Líderes)
    C2_DERECHOS_SOCIALES = "C2"  # P4, P5, P6 (Derechos, Víctimas, Niñez)
    C3_TERRITORIO_AMBIENTE = "C3"  # P3, P7 (Ambiente, Rural)
    C4_POBLACIONES_ESPECIALES = "C4"  # P9, P10 (Cárcel, Migración)


class ReportGenerator:
    """
    Generador de reportes a tres niveles con análisis cuantitativo y cualitativo
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Mapping de puntos a clústeres
        self.punto_to_cluster = {
            "P1": ClusterMeso.C1_SEGURIDAD_PAZ,
            "P2": ClusterMeso.C1_SEGURIDAD_PAZ,
            "P8": ClusterMeso.C1_SEGURIDAD_PAZ,
            "P4": ClusterMeso.C2_DERECHOS_SOCIALES,
            "P5": ClusterMeso.C2_DERECHOS_SOCIALES,
            "P6": ClusterMeso.C2_DERECHOS_SOCIALES,
            "P3": ClusterMeso.C3_TERRITORIO_AMBIENTE,
            "P7": ClusterMeso.C3_TERRITORIO_AMBIENTE,
            "P9": ClusterMeso.C4_POBLACIONES_ESPECIALES,
            "P10": ClusterMeso.C4_POBLACIONES_ESPECIALES
        }
    
    def generate_micro_report(self, question_responses: Dict[str, Any], 
                             policy_code: str) -> Dict:
        """
        Genera reporte nivel MICRO: 300 respuestas individuales
        
        Estructura:
        {
            "pregunta_id": {
                "pregunta": "texto de la pregunta",
                "respuesta": "texto de respuesta directa",
                "argumento": "2+ párrafos de argumentación doctoral",
                "nota_cuantitativa": 0.85,
                "evidencia": [...],
                "modulos_utilizados": [...],
                "nivel_confianza": 0.9
            },
            ...
        }
        """
        logger.info("Generando reporte MICRO (300 preguntas)...")
        
        micro_report = {
            "metadata": {
                "policy_code": policy_code,
                "generated_at": datetime.now().isoformat(),
                "total_questions": len(question_responses),
                "report_level": "MICRO"
            },
            "responses": {}
        }
        
        # Convert responses to dict format
        for question_id, response in question_responses.items():
            micro_report["responses"][question_id] = {
                "pregunta_id": response.pregunta_id,
                "respuesta": response.respuesta_texto,
                "argumento": response.argumento,
                "nota_cuantitativa": response.nota_cuantitativa,
                "evidencia": response.evidencia,
                "modulos_utilizados": response.modulos_utilizados,
                "nivel_confianza": response.nivel_confianza
            }
        
        # Calculate statistics
        notas = [r.nota_cuantitativa for r in question_responses.values()]
        micro_report["statistics"] = {
            "promedio_general": sum(notas) / len(notas) if notas else 0,
            "nota_maxima": max(notas) if notas else 0,
            "nota_minima": min(notas) if notas else 0,
            "preguntas_excelentes": sum(1 for n in notas if n >= 0.85),
            "preguntas_buenas": sum(1 for n in notas if 0.70 <= n < 0.85),
            "preguntas_aceptables": sum(1 for n in notas if 0.55 <= n < 0.70),
            "preguntas_insuficientes": sum(1 for n in notas if n < 0.55)
        }
        
        # Save to file
        output_file = self.output_dir / f"micro_report_{policy_code}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(micro_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Reporte MICRO guardado: {output_file}")
        logger.info(f"  Promedio general: {micro_report['statistics']['promedio_general']:.3f}")
        
        return micro_report
    
    def generate_meso_report(self, question_responses: Dict[str, Any],
                            policy_code: str) -> Dict:
        """
        Genera reporte nivel MESO: 4 clústeres × 6 dimensiones
        
        Agrupación:
        - C1: Seguridad y Paz (P1, P2, P8)
        - C2: Derechos Sociales (P4, P5, P6)
        - C3: Territorio y Ambiente (P3, P7)
        - C4: Poblaciones Especiales (P9, P10)
        
        Para cada clúster se analiza en las 6 dimensiones:
        D1-Insumos, D2-Actividades, D3-Productos, D4-Resultados, D5-Impactos, D6-Causalidad
        """
        logger.info("Generando reporte MESO (4 clústeres × 6 dimensiones)...")
        
        meso_report = {
            "metadata": {
                "policy_code": policy_code,
                "generated_at": datetime.now().isoformat(),
                "report_level": "MESO",
                "clusters": 4,
                "dimensions": 6
            },
            "clusters": {}
        }
        
        # Group responses by cluster and dimension
        for cluster in ClusterMeso:
            cluster_data = {
                "nombre": self._get_cluster_name(cluster),
                "puntos_incluidos": self._get_cluster_puntos(cluster),
                "dimensiones": {},
                "evaluacion_general": ""
            }
            
            # For each dimension (D1-D6)
            for dim_num in range(1, 7):
                dim_id = f"D{dim_num}"
                
                # Collect responses for this cluster and dimension
                dim_responses = []
                for question_id, response in question_responses.items():
                    # Parse question_id: "P1-D1-Q1"
                    parts = question_id.split('-')
                    if len(parts) >= 2:
                        punto = parts[0]
                        dimension = parts[1]
                        
                        if self.punto_to_cluster.get(punto) == cluster and dimension == dim_id:
                            dim_responses.append(response)
                
                # Calculate dimension score for this cluster
                if dim_responses:
                    dim_notas = [r.nota_cuantitativa for r in dim_responses]
                    dim_score = sum(dim_notas) / len(dim_notas)
                    
                    cluster_data["dimensiones"][dim_id] = {
                        "dimension_nombre": self._get_dimension_name(dim_id),
                        "score": dim_score,
                        "num_preguntas": len(dim_responses),
                        "nivel_cumplimiento": self._get_nivel_from_score(dim_score),
                        "observaciones": self._generate_dimension_observations(
                            cluster, dim_id, dim_responses, dim_score
                        )
                    }
            
            # Generate general cluster evaluation
            cluster_data["evaluacion_general"] = self._generate_cluster_evaluation(
                cluster, cluster_data["dimensiones"]
            )
            
            meso_report["clusters"][cluster.value] = cluster_data
        
        # Calculate overall meso statistics
        all_dim_scores = []
        for cluster_data in meso_report["clusters"].values():
            for dim_data in cluster_data["dimensiones"].values():
                all_dim_scores.append(dim_data["score"])
        
        meso_report["statistics"] = {
            "promedio_clusters": sum(all_dim_scores) / len(all_dim_scores) if all_dim_scores else 0,
            "cluster_mejor": self._find_best_cluster(meso_report["clusters"]),
            "cluster_debil": self._find_weakest_cluster(meso_report["clusters"]),
            "dimension_mejor": self._find_best_dimension(meso_report["clusters"]),
            "dimension_debil": self._find_weakest_dimension(meso_report["clusters"])
        }
        
        # Save to file
        output_file = self.output_dir / f"meso_report_{policy_code}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(meso_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Reporte MESO guardado: {output_file}")
        logger.info(f"  Promedio clústeres: {meso_report['statistics']['promedio_clusters']:.3f}")
        
        return meso_report
    
    def generate_macro_report(self, question_responses: Dict[str, Any],
                             compliance_score: float, policy_code: str) -> Dict:
        """
        Genera reporte nivel MACRO: Alineación global con el decálogo
        
        Incluye:
        1. Evaluación retrospectiva (¿qué tan lejos/cerca está?)
        2. Evaluación prospectiva (¿qué se debe mejorar?)
        3. Score global basado en promedio de las 300 preguntas
        4. Recomendaciones prioritarias
        """
        logger.info("Generando reporte MACRO (alineación con decálogo)...")
        
        # Calculate global score
        notas = [r.nota_cuantitativa for r in question_responses.values()]
        global_score = sum(notas) / len(notas) if notas else 0
        
        # Determine alignment level
        alignment_level = self._get_alignment_level(global_score)
        
        macro_report = {
            "metadata": {
                "policy_code": policy_code,
                "generated_at": datetime.now().isoformat(),
                "report_level": "MACRO"
            },
            "evaluacion_global": {
                "score_global": global_score,
                "score_dnp_compliance": compliance_score,
                "nivel_alineacion": alignment_level,
                "total_preguntas": len(question_responses)
            },
            "analisis_retrospectivo": self._generate_retrospective_analysis(
                global_score, question_responses
            ),
            "analisis_prospectivo": self._generate_prospective_analysis(
                global_score, question_responses
            ),
            "recomendaciones_prioritarias": self._generate_priority_recommendations(
                question_responses, compliance_score
            ),
            "fortalezas_identificadas": self._identify_strengths(question_responses),
            "debilidades_criticas": self._identify_critical_weaknesses(question_responses)
        }
        
        # Generate Markdown report
        self._generate_macro_markdown(macro_report, policy_code)
        
        # Save JSON version
        output_file = self.output_dir / f"macro_report_{policy_code}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(macro_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Reporte MACRO guardado: {output_file}")
        logger.info(f"  Nivel de alineación: {alignment_level}")
        logger.info(f"  Score global: {global_score:.3f}")
        
        return macro_report
    
    def _get_cluster_name(self, cluster: ClusterMeso) -> str:
        """Retorna el nombre descriptivo del clúster"""
        names = {
            ClusterMeso.C1_SEGURIDAD_PAZ: "Seguridad, Paz y Protección",
            ClusterMeso.C2_DERECHOS_SOCIALES: "Derechos Sociales y Poblaciones Vulnerables",
            ClusterMeso.C3_TERRITORIO_AMBIENTE: "Territorio, Ambiente y Desarrollo Rural",
            ClusterMeso.C4_POBLACIONES_ESPECIALES: "Poblaciones en Contextos Especiales"
        }
        return names[cluster]
    
    def _get_cluster_puntos(self, cluster: ClusterMeso) -> List[str]:
        """Retorna los puntos del decálogo incluidos en el clúster"""
        punto_lists = {
            ClusterMeso.C1_SEGURIDAD_PAZ: ["P1-Seguridad", "P2-Alertas", "P8-Líderes"],
            ClusterMeso.C2_DERECHOS_SOCIALES: ["P4-Derechos", "P5-Víctimas", "P6-Niñez"],
            ClusterMeso.C3_TERRITORIO_AMBIENTE: ["P3-Ambiente", "P7-Rural"],
            ClusterMeso.C4_POBLACIONES_ESPECIALES: ["P9-Cárcel", "P10-Migración"]
        }
        return punto_lists[cluster]
    
    def _get_dimension_name(self, dim_id: str) -> str:
        """Retorna el nombre de la dimensión"""
        names = {
            "D1": "Insumos (Diagnóstico y Líneas Base)",
            "D2": "Actividades (Formalizadas)",
            "D3": "Productos (Verificables)",
            "D4": "Resultados (Medibles)",
            "D5": "Impactos (Largo Plazo)",
            "D6": "Causalidad (Teoría de Cambio)"
        }
        return names.get(dim_id, dim_id)
    
    def _get_nivel_from_score(self, score: float) -> str:
        """Convierte score a nivel de cumplimiento"""
        if score >= 0.85:
            return "Excelente"
        elif score >= 0.70:
            return "Bueno"
        elif score >= 0.55:
            return "Aceptable"
        else:
            return "Insuficiente"
    
    def _generate_dimension_observations(self, cluster: ClusterMeso, dim_id: str,
                                        responses: List, score: float) -> str:
        """Genera observaciones cualitativas para una dimensión en un clúster"""
        nivel = self._get_nivel_from_score(score)
        
        if nivel == "Excelente":
            return f"La dimensión {dim_id} muestra un desempeño excelente en el clúster {cluster.value}, con evidencia sólida y formulación robusta."
        elif nivel == "Bueno":
            return f"La dimensión {dim_id} presenta un buen nivel de desarrollo en {cluster.value}, aunque hay espacio para fortalecimiento."
        elif nivel == "Aceptable":
            return f"La dimensión {dim_id} alcanza un nivel aceptable pero requiere mejoras significativas en {cluster.value}."
        else:
            return f"La dimensión {dim_id} muestra debilidades críticas en {cluster.value} que deben ser atendidas prioritariamente."
    
    def _generate_cluster_evaluation(self, cluster: ClusterMeso, 
                                    dimensiones: Dict) -> str:
        """Genera evaluación general del clúster"""
        if not dimensiones:
            return "Sin información suficiente para evaluar este clúster."
        
        avg_score = sum(d["score"] for d in dimensiones.values()) / len(dimensiones)
        cluster_name = self._get_cluster_name(cluster)
        
        evaluation = f"""
Evaluación General del Clúster {cluster.value} - {cluster_name}:

El clúster presenta un score promedio de {avg_score:.2f}, lo que indica un nivel de desarrollo 
{self._get_nivel_from_score(avg_score).lower()}. Analizando las seis dimensiones del marco lógico, 
se observa que:

- Las dimensiones más fuertes son: {self._identify_top_dimensions(dimensiones, 2)}
- Las dimensiones que requieren atención son: {self._identify_weak_dimensions(dimensiones, 2)}

Este clúster agrupa áreas temáticas relacionadas que comparten desafíos comunes en términos de
formulación de políticas públicas y teoría de cambio. La coherencia interna del clúster y la
integración entre los diferentes puntos del decálogo son fundamentales para lograr impactos
sostenibles en el territorio.
"""
        return evaluation.strip()
    
    def _identify_top_dimensions(self, dimensiones: Dict, n: int) -> str:
        """Identifica las n mejores dimensiones"""
        sorted_dims = sorted(dimensiones.items(), key=lambda x: x[1]["score"], reverse=True)
        top_dims = sorted_dims[:n]
        return ", ".join([f"{d[0]} ({d[1]['score']:.2f})" for d in top_dims])
    
    def _identify_weak_dimensions(self, dimensiones: Dict, n: int) -> str:
        """Identifica las n dimensiones más débiles"""
        sorted_dims = sorted(dimensiones.items(), key=lambda x: x[1]["score"])
        weak_dims = sorted_dims[:n]
        return ", ".join([f"{d[0]} ({d[1]['score']:.2f})" for d in weak_dims])
    
    def _find_best_cluster(self, clusters: Dict) -> str:
        """Encuentra el clúster con mejor desempeño"""
        best_cluster = None
        best_score = 0
        
        for cluster_id, cluster_data in clusters.items():
            if cluster_data["dimensiones"]:
                avg = sum(d["score"] for d in cluster_data["dimensiones"].values()) / len(cluster_data["dimensiones"])
                if avg > best_score:
                    best_score = avg
                    best_cluster = cluster_id
        
        return f"{best_cluster} ({best_score:.2f})" if best_cluster else "N/A"
    
    def _find_weakest_cluster(self, clusters: Dict) -> str:
        """Encuentra el clúster más débil"""
        weak_cluster = None
        weak_score = 1.0
        
        for cluster_id, cluster_data in clusters.items():
            if cluster_data["dimensiones"]:
                avg = sum(d["score"] for d in cluster_data["dimensiones"].values()) / len(cluster_data["dimensiones"])
                if avg < weak_score:
                    weak_score = avg
                    weak_cluster = cluster_id
        
        return f"{weak_cluster} ({weak_score:.2f})" if weak_cluster else "N/A"
    
    def _find_best_dimension(self, clusters: Dict) -> str:
        """Encuentra la dimensión con mejor desempeño global"""
        dim_scores = {}
        
        for cluster_data in clusters.values():
            for dim_id, dim_data in cluster_data["dimensiones"].items():
                if dim_id not in dim_scores:
                    dim_scores[dim_id] = []
                dim_scores[dim_id].append(dim_data["score"])
        
        dim_averages = {d: sum(scores)/len(scores) for d, scores in dim_scores.items()}
        best_dim = max(dim_averages.items(), key=lambda x: x[1])
        
        return f"{best_dim[0]} ({best_dim[1]:.2f})"
    
    def _find_weakest_dimension(self, clusters: Dict) -> str:
        """Encuentra la dimensión más débil globalmente"""
        dim_scores = {}
        
        for cluster_data in clusters.values():
            for dim_id, dim_data in cluster_data["dimensiones"].items():
                if dim_id not in dim_scores:
                    dim_scores[dim_id] = []
                dim_scores[dim_id].append(dim_data["score"])
        
        dim_averages = {d: sum(scores)/len(scores) for d, scores in dim_scores.items()}
        weak_dim = min(dim_averages.items(), key=lambda x: x[1])
        
        return f"{weak_dim[0]} ({weak_dim[1]:.2f})"
    
    def _get_alignment_level(self, score: float) -> str:
        """Determina el nivel de alineación con el decálogo"""
        if score >= 0.85:
            return "Altamente Alineado"
        elif score >= 0.70:
            return "Alineado"
        elif score >= 0.55:
            return "Parcialmente Alineado"
        else:
            return "No Alineado"
    
    def _generate_retrospective_analysis(self, global_score: float, 
                                        responses: Dict) -> str:
        """Genera análisis retrospectivo: ¿qué tan lejos/cerca está?"""
        nivel = self._get_alignment_level(global_score)
        distancia = (1.0 - global_score) * 100  # Porcentaje de distancia al óptimo
        
        analysis = f"""
ANÁLISIS RETROSPECTIVO: Distancia del Plan respecto al Decálogo

Con un score global de {global_score:.2f} ({global_score*100:.1f}%), el plan se encuentra en un nivel 
de alineación "{nivel}" respecto a los estándares establecidos en el decálogo de política pública.

Esto significa que el plan está aproximadamente a {distancia:.1f}% de distancia del cumplimiento óptimo
de los criterios de evaluación causal. En términos prácticos:

- De las {len(responses)} preguntas evaluadas, {sum(1 for r in responses.values() if r.nota_cuantitativa >= 0.85)}
  alcanzan nivel excelente (≥85%).
- {sum(1 for r in responses.values() if r.nota_cuantitativa < 0.55)} preguntas presentan 
  cumplimiento insuficiente (<55%) y requieren atención inmediata.

El análisis detallado por dimensiones revela que las principales brechas se concentran en:
- Formalización de actividades (D2)
- Medición de resultados (D4)
- Especificación de teorías de cambio (D6)

Estas brechas son consistentes con los desafíos típicos de formulación de planes de desarrollo
territorial en Colombia, donde la urgencia de la gestión administrativa a menudo limita la
rigurosidad metodológica en el diseño de intervenciones.
"""
        return analysis.strip()
    
    def _generate_prospective_analysis(self, global_score: float,
                                      responses: Dict) -> str:
        """Genera análisis prospectivo: ¿qué se debe mejorar?"""
        analysis = f"""
ANÁLISIS PROSPECTIVO: Ruta de Mejoramiento

Para alcanzar un nivel de alineación óptimo (≥85%) con el decálogo, el plan debe emprender
acciones correctivas en las siguientes líneas estratégicas:

1. **Fortalecimiento del Diagnóstico** (Dimensión D1):
   - Incorporar líneas base cuantitativas con fuentes oficiales (DANE, DNP, SISPRO)
   - Especificar series temporales mínimas de 3 años
   - Cuantificar brechas y reconocer explícitamente vacíos de información

2. **Formalización de Intervenciones** (Dimensión D2):
   - Estructurar actividades en formato tabular con responsables, cronogramas y costos
   - Especificar mecanismos causales que conecten actividades con resultados
   - Identificar y mitigar riesgos de implementación

3. **Verificabilidad de Productos** (Dimensión D3):
   - Definir indicadores con fórmulas, fuentes y líneas base
   - Alinear con catálogo MGA cuando sea posible
   - Garantizar trazabilidad presupuestal (BPIN/PPI)

4. **Medición de Resultados** (Dimensión D4):
   - Especificar outcomes con ventanas de maduración realistas
   - Establecer supuestos verificables
   - Alinear con PND y ODS

5. **Proyección de Impactos** (Dimensión D5):
   - Definir rutas de transmisión resultado→impacto
   - Usar proxies cuando la medición directa no sea factible
   - Considerar riesgos sistémicos y efectos no deseados

6. **Teoría de Cambio Explícita** (Dimensión D6):
   - Documentar causas raíz, mediadores y moderadores
   - Crear diagramas causales que permitan validación
   - Establecer mecanismos de monitoreo y aprendizaje adaptativo

La implementación de estas mejoras debe priorizarse según:
- **Urgencia**: Dimensiones con score <0.55
- **Impacto**: Dimensiones críticas para cada punto del decálogo
- **Viabilidad**: Disponibilidad de datos y capacidad técnica
"""
        return analysis.strip()
    
    def _generate_priority_recommendations(self, responses: Dict,
                                          compliance_score: float) -> List[str]:
        """Genera recomendaciones prioritarias"""
        recommendations = []
        
        # Identify critical gaps
        critical_questions = [
            (qid, r) for qid, r in responses.items() 
            if r.nota_cuantitativa < 0.40
        ]
        
        if len(critical_questions) > 10:
            recommendations.append(
                f"CRÍTICO: {len(critical_questions)} preguntas con cumplimiento muy bajo (<40%). "
                "Se requiere reformulación fundamental del plan."
            )
        
        if compliance_score < 60:
            recommendations.append(
                f"Cumplimiento DNP insuficiente ({compliance_score:.1f}/100). "
                "Revisar competencias municipales, indicadores MGA y lineamientos PDET."
            )
        
        # Dimension-specific recommendations
        dim_scores = {}
        for qid, r in responses.items():
            # Extract dimension from question_id (e.g., "P1-D1-Q1" -> "D1")
            parts = qid.split('-')
            if len(parts) >= 2:
                dim = parts[1]
                if dim not in dim_scores:
                    dim_scores[dim] = []
                dim_scores[dim].append(r.nota_cuantitativa)
        
        for dim, scores in dim_scores.items():
            avg = sum(scores) / len(scores)
            if avg < 0.60:
                recommendations.append(
                    f"Fortalecer Dimensión {dim} ({self._get_dimension_name(dim)}): "
                    f"score promedio de {avg:.2f} es insuficiente."
                )
        
        if not recommendations:
            recommendations.append(
                "El plan presenta un nivel aceptable de cumplimiento. "
                "Continuar fortaleciendo áreas identificadas como débiles en el análisis meso."
            )
        
        return recommendations
    
    def _identify_strengths(self, responses: Dict) -> List[str]:
        """Identifica fortalezas del plan"""
        strengths = []
        
        # Questions with excellent scores
        excellent = [(qid, r) for qid, r in responses.items() if r.nota_cuantitativa >= 0.85]
        
        if len(excellent) > len(responses) * 0.3:
            strengths.append(
                f"{len(excellent)} preguntas ({len(excellent)/len(responses)*100:.1f}%) "
                "alcanzan nivel excelente"
            )
        
        # Identify strong dimensions
        dim_scores = {}
        for qid, r in responses.items():
            parts = qid.split('-')
            if len(parts) >= 2:
                dim = parts[1]
                if dim not in dim_scores:
                    dim_scores[dim] = []
                dim_scores[dim].append(r.nota_cuantitativa)
        
        for dim, scores in dim_scores.items():
            avg = sum(scores) / len(scores)
            if avg >= 0.80:
                strengths.append(
                    f"Dimensión {dim} ({self._get_dimension_name(dim)}) "
                    f"muestra fortaleza con score de {avg:.2f}"
                )
        
        return strengths if strengths else ["Se requiere fortalecimiento general"]
    
    def _identify_critical_weaknesses(self, responses: Dict) -> List[str]:
        """Identifica debilidades críticas"""
        weaknesses = []
        
        # Questions with very low scores
        critical = [(qid, r) for qid, r in responses.items() if r.nota_cuantitativa < 0.40]
        
        if critical:
            weaknesses.append(
                f"{len(critical)} preguntas con cumplimiento crítico (<40%): "
                f"{', '.join([q[0] for q in critical[:5]])}"
                + ("..." if len(critical) > 5 else "")
            )
        
        # Identify weak dimensions
        dim_scores = {}
        for qid, r in responses.items():
            parts = qid.split('-')
            if len(parts) >= 2:
                dim = parts[1]
                if dim not in dim_scores:
                    dim_scores[dim] = []
                dim_scores[dim].append(r.nota_cuantitativa)
        
        for dim, scores in dim_scores.items():
            avg = sum(scores) / len(scores)
            if avg < 0.50:
                weaknesses.append(
                    f"Dimensión {dim} ({self._get_dimension_name(dim)}) "
                    f"crítica con score de {avg:.2f}"
                )
        
        return weaknesses if weaknesses else ["Sin debilidades críticas identificadas"]
    
    def _generate_macro_markdown(self, macro_report: Dict, policy_code: str):
        """Genera versión Markdown del reporte macro"""
        md_content = f"""# Reporte Macro - Evaluación de Plan de Desarrollo
## {policy_code}

**Fecha de Generación:** {macro_report['metadata']['generated_at']}

---

## Evaluación Global

- **Score Global:** {macro_report['evaluacion_global']['score_global']:.2f} ({macro_report['evaluacion_global']['score_global']*100:.1f}%)
- **Nivel de Alineación:** {macro_report['evaluacion_global']['nivel_alineacion']}
- **Score DNP:** {macro_report['evaluacion_global']['score_dnp_compliance']:.1f}/100
- **Preguntas Evaluadas:** {macro_report['evaluacion_global']['total_preguntas']}

---

## {macro_report['analisis_retrospectivo'].split(':')[0]}

{macro_report['analisis_retrospectivo']}

---

## {macro_report['analisis_prospectivo'].split(':')[0]}

{macro_report['analisis_prospectivo']}

---

## Recomendaciones Prioritarias

"""
        for i, rec in enumerate(macro_report['recomendaciones_prioritarias'], 1):
            md_content += f"{i}. {rec}\n"
        
        md_content += f"""
---

## Fortalezas Identificadas

"""
        for strength in macro_report['fortalezas_identificadas']:
            md_content += f"- ✓ {strength}\n"
        
        md_content += f"""
---

## Debilidades Críticas

"""
        for weakness in macro_report['debilidades_criticas']:
            md_content += f"- ⚠️ {weakness}\n"
        
        md_content += """
---

*Generado por FARFAN 2.0 - Framework Avanzado de Reconstrucción y Análisis de Formulaciones de Acción Nacional*
"""
        
        output_file = self.output_dir / f"macro_report_{policy_code}.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"✓ Reporte MACRO (Markdown) guardado: {output_file}")
