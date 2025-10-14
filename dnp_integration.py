#!/usr/bin/env python3
"""
DNP Standards Integration Module
Integrates competencias municipales, MGA indicators, and PDET guidelines
into the existing CDAF framework for comprehensive compliance validation
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Import our new modules
try:
    from competencias_municipales import (
        CATALOGO_COMPETENCIAS, 
        CompetenciaMunicipal,
        SectorCompetencia,
        TipoCompetencia
    )
    from mga_indicadores import (
        CATALOGO_MGA,
        IndicadorMGA,
        TipoIndicadorMGA
    )
    from pdet_lineamientos import (
        LINEAMIENTOS_PDET,
        LineamientoPDET,
        PilarPDET,
        RequisitosPDET
    )
except ImportError as e:
    logging.error(f"Error importando módulos DNP: {e}")
    logging.error("Asegúrese de que competencias_municipales.py, mga_indicadores.py y pdet_lineamientos.py estén en el mismo directorio")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dnp_integration")


class NivelCumplimiento(Enum):
    """Compliance level with DNP standards"""
    EXCELENTE = "excelente"  # >90%
    BUENO = "bueno"  # 75-90%
    ACEPTABLE = "aceptable"  # 60-75%
    INSUFICIENTE = "insuficiente"  # <60%


@dataclass
class ResultadoValidacionDNP:
    """Comprehensive validation result for DNP standards"""
    cumple_competencias: bool = False
    cumple_mga: bool = False
    cumple_pdet: bool = False  # Only for PDET municipalities
    nivel_cumplimiento: NivelCumplimiento = NivelCumplimiento.INSUFICIENTE
    score_total: float = 0.0
    
    # Detailed breakdowns
    competencias_validadas: List[str] = field(default_factory=list)
    competencias_fuera_alcance: List[str] = field(default_factory=list)
    indicadores_mga_usados: List[str] = field(default_factory=list)
    indicadores_mga_faltantes: List[str] = field(default_factory=list)
    lineamientos_pdet_cumplidos: List[str] = field(default_factory=list)
    lineamientos_pdet_pendientes: List[str] = field(default_factory=list)
    
    # Recommendations
    recomendaciones: List[str] = field(default_factory=list)
    alertas_criticas: List[str] = field(default_factory=list)
    
    # Metadata
    es_municipio_pdet: bool = False
    sectores_intervenidos: List[str] = field(default_factory=list)


class ValidadorDNP:
    """
    Comprehensive validator for DNP standards compliance
    Validates municipal competencies, MGA indicators, and PDET guidelines
    """
    
    def __init__(self, es_municipio_pdet: bool = False):
        self.catalogo_competencias = CATALOGO_COMPETENCIAS
        self.catalogo_mga = CATALOGO_MGA
        self.lineamientos_pdet = LINEAMIENTOS_PDET
        self.es_municipio_pdet = es_municipio_pdet
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validar_proyecto_integral(self,
                                  sector: str,
                                  descripcion: str,
                                  indicadores_propuestos: List[str],
                                  presupuesto: float = 0.0,
                                  es_rural: bool = False,
                                  poblacion_victimas: bool = False) -> ResultadoValidacionDNP:
        """
        Comprehensive validation of a project/program against DNP standards
        
        Args:
            sector: Project sector
            descripcion: Project description
            indicadores_propuestos: List of proposed indicator codes
            presupuesto: Budget allocation
            es_rural: Whether project targets rural areas
            poblacion_victimas: Whether project serves conflict victims
        """
        resultado = ResultadoValidacionDNP(es_municipio_pdet=self.es_municipio_pdet)
        
        # 1. Validate municipal competencies
        self.logger.info("Validando competencias municipales...")
        competencias_result = self._validar_competencias(sector, descripcion)
        resultado.cumple_competencias = competencias_result["valido"]
        resultado.competencias_validadas = competencias_result.get("competencias_aplicables", [])
        resultado.sectores_intervenidos = [competencias_result.get("sector", sector)]
        
        if not competencias_result["valido"]:
            resultado.alertas_criticas.append(
                "CRÍTICO: Proyecto fuera de competencias municipales según normativa colombiana"
            )
            resultado.recomendaciones.append(
                "Revisar competencias municipales (Ley 136/1994, Ley 715/2001, Ley 1551/2012)"
            )
        
        # 2. Validate MGA indicators
        self.logger.info("Validando indicadores MGA...")
        mga_result = self._validar_indicadores_mga(indicadores_propuestos, sector)
        resultado.cumple_mga = mga_result["cumple_mga"]
        resultado.indicadores_mga_usados = mga_result["indicadores_validos"]
        resultado.indicadores_mga_faltantes = mga_result["recomendados_faltantes"]
        
        if not mga_result["cumple_mga"]:
            resultado.alertas_criticas.append(
                "CRÍTICO: Falta alineación con catálogo de indicadores MGA del DNP"
            )
            resultado.recomendaciones.append(
                "Incorporar indicadores MGA estándar para el sector de intervención"
            )
        
        # 3. Validate PDET requirements (if applicable)
        if self.es_municipio_pdet:
            self.logger.info("Validando lineamientos PDET...")
            pdet_result = self._validar_lineamientos_pdet(
                sector, es_rural, poblacion_victimas
            )
            resultado.cumple_pdet = pdet_result["cumple"]
            resultado.lineamientos_pdet_cumplidos = pdet_result["cumplidos"]
            resultado.lineamientos_pdet_pendientes = pdet_result["pendientes"]
            
            if not pdet_result["cumple"]:
                resultado.alertas_criticas.append(
                    "CRÍTICO: Municipio PDET debe cumplir lineamientos especiales (Decreto 893/2017)"
                )
                resultado.recomendaciones.extend(pdet_result["recomendaciones"])
        
        # 4. Calculate overall compliance score
        score_competencias = 40.0 if resultado.cumple_competencias else 0.0
        score_mga = 40.0 if resultado.cumple_mga else (
            20.0 if len(resultado.indicadores_mga_usados) > 0 else 0.0
        )
        score_pdet = 0.0
        if self.es_municipio_pdet:
            score_pdet = 20.0 if resultado.cumple_pdet else (
                10.0 if len(resultado.lineamientos_pdet_cumplidos) > 0 else 0.0
            )
        else:
            # If not PDET, redistribute weight
            score_competencias = 50.0 if resultado.cumple_competencias else 0.0
            score_mga = 50.0 if resultado.cumple_mga else (
                25.0 if len(resultado.indicadores_mga_usados) > 0 else 0.0
            )
        
        resultado.score_total = score_competencias + score_mga + score_pdet
        
        # 5. Determine compliance level
        if resultado.score_total >= 90:
            resultado.nivel_cumplimiento = NivelCumplimiento.EXCELENTE
        elif resultado.score_total >= 75:
            resultado.nivel_cumplimiento = NivelCumplimiento.BUENO
        elif resultado.score_total >= 60:
            resultado.nivel_cumplimiento = NivelCumplimiento.ACEPTABLE
        else:
            resultado.nivel_cumplimiento = NivelCumplimiento.INSUFICIENTE
        
        # 6. Generate general recommendations
        self._generar_recomendaciones_generales(resultado)
        
        return resultado
    
    def _validar_competencias(self, sector: str, descripcion: str) -> Dict[str, Any]:
        """Validate municipal competencies"""
        return self.catalogo_competencias.validar_competencia_municipal(sector, descripcion)
    
    def _validar_indicadores_mga(self, 
                                 indicadores_propuestos: List[str],
                                 sector: str) -> Dict[str, Any]:
        """Validate MGA indicators"""
        indicadores_validos = []
        indicadores_invalidos = []
        
        for codigo in indicadores_propuestos:
            indicador = self.catalogo_mga.get_indicador(codigo)
            if indicador:
                indicadores_validos.append(codigo)
            else:
                indicadores_invalidos.append(codigo)
        
        # Get recommended indicators for sector
        indicadores_sector = self.catalogo_mga.get_by_sector(sector)
        codigos_recomendados = [ind.codigo for ind in indicadores_sector]
        
        # Find missing recommended indicators
        recomendados_faltantes = [
            cod for cod in codigos_recomendados[:3]  # Top 3 recommendations
            if cod not in indicadores_validos
        ]
        
        # MGA compliance requires at least 1 valid product and 1 result indicator
        tiene_producto = any(
            self.catalogo_mga.get_indicador(cod).tipo == TipoIndicadorMGA.PRODUCTO
            for cod in indicadores_validos
        )
        tiene_resultado = any(
            self.catalogo_mga.get_indicador(cod).tipo == TipoIndicadorMGA.RESULTADO
            for cod in indicadores_validos
        )
        
        cumple_mga = tiene_producto and tiene_resultado
        
        return {
            "cumple_mga": cumple_mga,
            "indicadores_validos": indicadores_validos,
            "indicadores_invalidos": indicadores_invalidos,
            "recomendados_faltantes": recomendados_faltantes,
            "tiene_producto": tiene_producto,
            "tiene_resultado": tiene_resultado
        }
    
    def _validar_lineamientos_pdet(self,
                                   sector: str,
                                   es_rural: bool,
                                   poblacion_victimas: bool,
                                   lineamientos_cumplidos: List[str]) -> Dict[str, Any]:
        """Validate PDET guidelines compliance"""
        lineamientos_recomendados = self.lineamientos_pdet.recomendar_lineamientos(
            sector, es_rural, poblacion_victimas
        )
        
        # Evaluate actual compliance based on provided fulfilled guidelines
        cumplidos = [lin.codigo for lin in lineamientos_recomendados if lin.codigo in lineamientos_cumplidos]
        pendientes = [lin.codigo for lin in lineamientos_recomendados if lin.codigo not in lineamientos_cumplidos]
        
        cumple = len(pendientes) == 0  # True if all recommended guidelines are fulfilled
        
        recomendaciones = []
        if not cumple:
            if es_rural:
                recomendaciones.append("Fortalecer enfoque de desarrollo rural con enfoque territorial")
            if poblacion_victimas:
                recomendaciones.append("Incluir componentes de reconciliación y reparación para víctimas")
            recomendaciones.append("Alinear intervenciones con los 8 pilares del PDET")
        
        return {
            "cumple": cumple,
            "cumplidos": cumplidos,
            "pendientes": pendientes,
            "recomendaciones": recomendaciones
        }
    
    def _generar_recomendaciones_generales(self, resultado: ResultadoValidacionDNP):
        """Generate general recommendations based on validation results"""
        if resultado.nivel_cumplimiento == NivelCumplimiento.INSUFICIENTE:
            resultado.recomendaciones.insert(0, 
                "URGENTE: Revisar formulación del proyecto para cumplir estándares mínimos DNP"
            )
        
        if not resultado.indicadores_mga_usados:
            resultado.recomendaciones.append(
                "Incorporar indicadores del catálogo MGA para permitir seguimiento y reporte en SPI"
            )
        
        if resultado.es_municipio_pdet and not resultado.cumple_pdet:
            resultado.recomendaciones.append(
                "Consultar PATR (Plan de Acción para la Transformación Regional) de su subregión"
            )
            resultado.recomendaciones.append(
                "Articular con iniciativas de la Agencia de Renovación del Territorio (ART)"
            )
    
    def generar_reporte_cumplimiento(self, resultado: ResultadoValidacionDNP) -> str:
        """Generate comprehensive compliance report"""
        lineas = []
        lineas.append("=" * 80)
        lineas.append("REPORTE DE CUMPLIMIENTO - ESTÁNDARES DNP")
        lineas.append("=" * 80)
        lineas.append("")
        
        # Overall score
        lineas.append(f"SCORE TOTAL: {resultado.score_total:.1f}/100")
        lineas.append(f"NIVEL DE CUMPLIMIENTO: {resultado.nivel_cumplimiento.value.upper()}")
        lineas.append("")
        
        # Competencies
        lineas.append("1. COMPETENCIAS MUNICIPALES")
        lineas.append(f"   Estado: {'✓ CUMPLE' if resultado.cumple_competencias else '✗ NO CUMPLE'}")
        if resultado.competencias_validadas:
            lineas.append(f"   Competencias aplicables: {', '.join(resultado.competencias_validadas[:3])}")
        if resultado.competencias_fuera_alcance:
            lineas.append(f"   Fuera de alcance: {', '.join(resultado.competencias_fuera_alcance)}")
        lineas.append("")
        
        # MGA Indicators
        lineas.append("2. INDICADORES MGA")
        lineas.append(f"   Estado: {'✓ CUMPLE' if resultado.cumple_mga else '✗ NO CUMPLE'}")
        if resultado.indicadores_mga_usados:
            lineas.append(f"   Indicadores válidos: {', '.join(resultado.indicadores_mga_usados)}")
        if resultado.indicadores_mga_faltantes:
            lineas.append(f"   Recomendados: {', '.join(resultado.indicadores_mga_faltantes)}")
        lineas.append("")
        
        # PDET (if applicable)
        if resultado.es_municipio_pdet:
            lineas.append("3. LINEAMIENTOS PDET")
            lineas.append(f"   Estado: {'✓ CUMPLE' if resultado.cumple_pdet else '✗ NO CUMPLE'}")
            if resultado.lineamientos_pdet_cumplidos:
                lineas.append(f"   Cumplidos: {len(resultado.lineamientos_pdet_cumplidos)} lineamientos")
            if resultado.lineamientos_pdet_pendientes:
                lineas.append(f"   Pendientes: {len(resultado.lineamientos_pdet_pendientes)} lineamientos")
            lineas.append("")
        
        # Critical alerts
        if resultado.alertas_criticas:
            lineas.append("ALERTAS CRÍTICAS:")
            for alerta in resultado.alertas_criticas:
                lineas.append(f"  ⚠ {alerta}")
            lineas.append("")
        
        # Recommendations
        if resultado.recomendaciones:
            lineas.append("RECOMENDACIONES:")
            for i, rec in enumerate(resultado.recomendaciones, 1):
                lineas.append(f"  {i}. {rec}")
            lineas.append("")
        
        lineas.append("=" * 80)
        
        return "\n".join(lineas)


def validar_plan_desarrollo_completo(
    proyectos: List[Dict[str, Any]],
    es_municipio_pdet: bool = False,
    presupuesto_total: float = 0.0,
    presupuesto_rural: float = 0.0,
    participacion: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Validate complete municipal development plan
    
    Args:
        proyectos: List of projects/programs
        es_municipio_pdet: Whether municipality is PDET
        presupuesto_total: Total budget
        presupuesto_rural: Rural budget allocation
        participacion: Participation percentages
    """
    validador = ValidadorDNP(es_municipio_pdet=es_municipio_pdet)
    
    resultados_proyectos = []
    for proyecto in proyectos:
        resultado = validador.validar_proyecto_integral(
            sector=proyecto.get("sector", ""),
            descripcion=proyecto.get("descripcion", ""),
            indicadores_propuestos=proyecto.get("indicadores", []),
            presupuesto=proyecto.get("presupuesto", 0.0),
            es_rural=proyecto.get("es_rural", False),
            poblacion_victimas=proyecto.get("poblacion_victimas", False)
        )
        resultados_proyectos.append({
            "proyecto": proyecto.get("nombre", "Sin nombre"),
            "resultado": resultado
        })
    
    # Aggregate results
    total_proyectos = len(proyectos)
    proyectos_cumplen = sum(1 for r in resultados_proyectos 
                           if r["resultado"].nivel_cumplimiento in 
                           [NivelCumplimiento.EXCELENTE, NivelCumplimiento.BUENO])
    
    score_promedio = sum(r["resultado"].score_total for r in resultados_proyectos) / total_proyectos if total_proyectos > 0 else 0
    
    # PDET-specific validation
    cumplimiento_pdet = None
    if es_municipio_pdet and participacion:
        cumplimiento_pdet = LINEAMIENTOS_PDET.validar_cumplimiento_pdet(
            participacion=participacion,
            presupuesto_rural=presupuesto_rural,
            presupuesto_total=presupuesto_total,
            alineacion_patr=80.0  # Default, should be calculated
        )
    
    return {
        "total_proyectos": total_proyectos,
        "proyectos_cumplen": proyectos_cumplen,
        "tasa_cumplimiento": (proyectos_cumplen / total_proyectos * 100) if total_proyectos > 0 else 0,
        "score_promedio": score_promedio,
        "resultados_proyectos": resultados_proyectos,
        "cumplimiento_pdet": cumplimiento_pdet,
        "es_municipio_pdet": es_municipio_pdet
    }


if __name__ == "__main__":
    # Demo validation
    print("=== Validador DNP - Demo ===\n")
    
    validador = ValidadorDNP(es_municipio_pdet=True)
    
    # Example project
    resultado = validador.validar_proyecto_integral(
        sector="educacion",
        descripcion="Construcción de 5 sedes educativas en zona rural",
        indicadores_propuestos=["EDU-020", "EDU-021", "EDU-002"],
        presupuesto=2_000_000_000,
        es_rural=True,
        poblacion_victimas=True
    )
    
    print(validador.generar_reporte_cumplimiento(resultado))
