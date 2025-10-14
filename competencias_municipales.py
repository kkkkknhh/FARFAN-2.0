#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Competencias Municipales - Colombian Municipal Competencies Framework
Comprehensive catalog of own and concurrent competencies for Colombian municipalities
Based on:
- Constitución Política de Colombia (1991)
- Ley 136 de 1994 (Municipal Organization)
- Ley 715 de 2001 (Resources and Competencies)
- Ley 1551 de 2012 (Municipal Modernization)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("competencias_municipales")


class TipoCompetencia(Enum):
    """Types of municipal competencies"""
    PROPIA = "propia"  # Own exclusive competency
    CONCURRENTE = "concurrente"  # Concurrent with department/nation
    COMPLEMENTARIA = "complementaria"  # Complementary support role


class SectorCompetencia(Enum):
    """Competency sectors aligned with DNP guidelines"""
    EDUCACION = "educacion"
    SALUD = "salud"
    AGUA_POTABLE_SANEAMIENTO = "agua_potable_saneamiento"
    VIVIENDA = "vivienda"
    VIAS_TRANSPORTE = "vias_transporte"
    DESARROLLO_AGROPECUARIO = "desarrollo_agropecuario"
    AMBIENTE = "ambiente"
    CULTURA_DEPORTE_RECREACION = "cultura_deporte_recreacion"
    DESARROLLO_ECONOMICO = "desarrollo_economico"
    ATENCION_GRUPOS_VULNERABLES = "atencion_grupos_vulnerables"
    JUSTICIA_SEGURIDAD = "justicia_seguridad"
    ORDENAMIENTO_TERRITORIAL = "ordenamiento_territorial"
    EQUIPAMIENTO_MUNICIPAL = "equipamiento_municipal"
    PREVENCION_ATENCION_DESASTRES = "prevencion_atencion_desastres"


@dataclass
class CompetenciaMunicipal:
    """Represents a specific municipal competency"""
    codigo: str
    sector: SectorCompetencia
    tipo: TipoCompetencia
    descripcion: str
    base_legal: List[str]
    indicadores_mga_aplicables: List[str] = field(default_factory=list)
    restricciones: List[str] = field(default_factory=list)
    coordinacion_requerida: List[str] = field(default_factory=list)
    aplicable_pdet: bool = False
    prioridad_pdet: Optional[str] = None


class CatalogoCompetenciasMunicipales:
    """
    Complete catalog of Colombian municipal competencies
    Ensures strict compliance with national legal framework
    """
    
    def __init__(self):
        self.competencias: Dict[str, CompetenciaMunicipal] = {}
        self._initialize_competencias()
    
    def _initialize_competencias(self):
        """Initialize comprehensive competencies catalog"""
        
        # EDUCACIÓN (Ley 715/2001)
        self.add_competencia(CompetenciaMunicipal(
            codigo="EDU-001",
            sector=SectorCompetencia.EDUCACION,
            tipo=TipoCompetencia.CONCURRENTE,
            descripcion="Administrar, financiar y garantizar el servicio educativo en los niveles de preescolar, básica y media",
            base_legal=["Ley 715/2001 Art. 7", "Ley 115/1994"],
            indicadores_mga_aplicables=["EDU-001", "EDU-002", "EDU-003", "EDU-010"],
            coordinacion_requerida=["Ministerio de Educación Nacional", "Secretaría Departamental de Educación"],
            aplicable_pdet=True,
            prioridad_pdet="alta"
        ))
        
        self.add_competencia(CompetenciaMunicipal(
            codigo="EDU-002",
            sector=SectorCompetencia.EDUCACION,
            tipo=TipoCompetencia.PROPIA,
            descripcion="Construir, ampliar y realizar mantenimiento de infraestructura educativa",
            base_legal=["Ley 715/2001 Art. 76.7", "Decreto 4791/2008"],
            indicadores_mga_aplicables=["EDU-020", "EDU-021"],
            aplicable_pdet=True,
            prioridad_pdet="alta"
        ))
        
        # SALUD (Ley 715/2001)
        self.add_competencia(CompetenciaMunicipal(
            codigo="SAL-001",
            sector=SectorCompetencia.SALUD,
            tipo=TipoCompetencia.CONCURRENTE,
            descripcion="Gestión de la salud pública y prestación de servicios de salud del primer nivel",
            base_legal=["Ley 715/2001 Art. 44", "Ley 1438/2011"],
            indicadores_mga_aplicables=["SAL-001", "SAL-002", "SAL-010", "SAL-015"],
            coordinacion_requerida=["Ministerio de Salud", "Secretaría Departamental de Salud"],
            aplicable_pdet=True,
            prioridad_pdet="alta"
        ))
        
        self.add_competencia(CompetenciaMunicipal(
            codigo="SAL-002",
            sector=SectorCompetencia.SALUD,
            tipo=TipoCompetencia.PROPIA,
            descripcion="Construcción, dotación, adecuación y mantenimiento de centros de salud y puestos de salud",
            base_legal=["Ley 715/2001 Art. 76.4"],
            indicadores_mga_aplicables=["SAL-020", "SAL-021"],
            aplicable_pdet=True,
            prioridad_pdet="alta"
        ))
        
        # AGUA POTABLE Y SANEAMIENTO BÁSICO (Ley 142/1994, Ley 715/2001)
        self.add_competencia(CompetenciaMunicipal(
            codigo="APS-001",
            sector=SectorCompetencia.AGUA_POTABLE_SANEAMIENTO,
            tipo=TipoCompetencia.PROPIA,
            descripcion="Prestar directa o indirectamente los servicios de acueducto, alcantarillado y aseo",
            base_legal=["Ley 142/1994", "Ley 715/2001 Art. 76.5"],
            indicadores_mga_aplicables=["APS-001", "APS-002", "APS-003", "APS-010"],
            aplicable_pdet=True,
            prioridad_pdet="alta"
        ))
        
        self.add_competencia(CompetenciaMunicipal(
            codigo="APS-002",
            sector=SectorCompetencia.AGUA_POTABLE_SANEAMIENTO,
            tipo=TipoCompetencia.PROPIA,
            descripcion="Diseñar, construir, operar y mantener sistemas de acueducto y alcantarillado",
            base_legal=["Ley 142/1994", "Decreto 1575/2007"],
            indicadores_mga_aplicables=["APS-020", "APS-021", "APS-030"],
            aplicable_pdet=True,
            prioridad_pdet="alta"
        ))
        
        # VIVIENDA (Ley 715/2001)
        self.add_competencia(CompetenciaMunicipal(
            codigo="VIV-001",
            sector=SectorCompetencia.VIVIENDA,
            tipo=TipoCompetencia.CONCURRENTE,
            descripcion="Promover y apoyar programas de vivienda de interés social y prioritario",
            base_legal=["Ley 715/2001 Art. 76.8", "Ley 1537/2012"],
            indicadores_mga_aplicables=["VIV-001", "VIV-002", "VIV-010"],
            coordinacion_requerida=["Ministerio de Vivienda"],
            aplicable_pdet=True,
            prioridad_pdet="media"
        ))
        
        # VÍAS Y TRANSPORTE (Ley 1228/2008, Ley 105/1993)
        self.add_competencia(CompetenciaMunicipal(
            codigo="VIA-001",
            sector=SectorCompetencia.VIAS_TRANSPORTE,
            tipo=TipoCompetencia.PROPIA,
            descripcion="Construcción, mantenimiento y mejoramiento de vías urbanas y rurales municipales",
            base_legal=["Ley 105/1993", "Ley 1228/2008"],
            indicadores_mga_aplicables=["VIA-001", "VIA-002", "VIA-010"],
            aplicable_pdet=True,
            prioridad_pdet="alta"
        ))
        
        # DESARROLLO AGROPECUARIO (Ley 101/1993)
        self.add_competencia(CompetenciaMunicipal(
            codigo="AGR-001",
            sector=SectorCompetencia.DESARROLLO_AGROPECUARIO,
            tipo=TipoCompetencia.COMPLEMENTARIA,
            descripcion="Promover, cofinanciar y ejecutar programas de desarrollo rural y asistencia técnica agropecuaria",
            base_legal=["Ley 101/1993", "Ley 607/2000"],
            indicadores_mga_aplicables=["AGR-001", "AGR-002", "AGR-010"],
            coordinacion_requerida=["Ministerio de Agricultura"],
            aplicable_pdet=True,
            prioridad_pdet="alta"
        ))
        
        # MEDIO AMBIENTE (Ley 99/1993)
        self.add_competencia(CompetenciaMunicipal(
            codigo="AMB-001",
            sector=SectorCompetencia.AMBIENTE,
            tipo=TipoCompetencia.CONCURRENTE,
            descripcion="Protección del medio ambiente y recursos naturales renovables",
            base_legal=["Ley 99/1993 Art. 65", "Ley 388/1997"],
            indicadores_mga_aplicables=["AMB-001", "AMB-002", "AMB-010"],
            coordinacion_requerida=["Corporaciones Autónomas Regionales", "Ministerio de Ambiente"],
            aplicable_pdet=True,
            prioridad_pdet="alta"
        ))
        
        # CULTURA, DEPORTE Y RECREACIÓN (Ley 181/1995, Ley 1185/2008)
        self.add_competencia(CompetenciaMunicipal(
            codigo="CUL-001",
            sector=SectorCompetencia.CULTURA_DEPORTE_RECREACION,
            tipo=TipoCompetencia.PROPIA,
            descripcion="Fomentar el acceso a la cultura, deporte y recreación, construir y mantener infraestructura",
            base_legal=["Ley 181/1995", "Ley 1185/2008"],
            indicadores_mga_aplicables=["CUL-001", "CUL-002", "DEP-001", "DEP-002"],
            aplicable_pdet=True,
            prioridad_pdet="media"
        ))
        
        # DESARROLLO ECONÓMICO (Ley 1551/2012)
        self.add_competencia(CompetenciaMunicipal(
            codigo="ECO-001",
            sector=SectorCompetencia.DESARROLLO_ECONOMICO,
            tipo=TipoCompetencia.PROPIA,
            descripcion="Promover el desarrollo económico local, empleo y emprendimiento",
            base_legal=["Ley 1551/2012 Art. 3", "Ley 590/2000"],
            indicadores_mga_aplicables=["ECO-001", "ECO-002", "ECO-010"],
            aplicable_pdet=True,
            prioridad_pdet="alta"
        ))
        
        # ATENCIÓN A GRUPOS VULNERABLES (Ley 1098/2006, Ley 1251/2008)
        self.add_competencia(CompetenciaMunicipal(
            codigo="SOC-001",
            sector=SectorCompetencia.ATENCION_GRUPOS_VULNERABLES,
            tipo=TipoCompetencia.CONCURRENTE,
            descripcion="Atención integral a niñez, adolescencia, adultos mayores y población vulnerable",
            base_legal=["Ley 1098/2006", "Ley 1251/2008", "Ley 1257/2008"],
            indicadores_mga_aplicables=["SOC-001", "SOC-002", "SOC-010"],
            coordinacion_requerida=["ICBF", "Ministerio de Inclusión Social"],
            aplicable_pdet=True,
            prioridad_pdet="alta"
        ))
        
        # JUSTICIA Y SEGURIDAD (Ley 62/1993)
        self.add_competencia(CompetenciaMunicipal(
            codigo="SEG-001",
            sector=SectorCompetencia.JUSTICIA_SEGURIDAD,
            tipo=TipoCompetencia.CONCURRENTE,
            descripcion="Seguridad ciudadana, convivencia y orden público municipal",
            base_legal=["Ley 62/1993", "Ley 1421/2010"],
            indicadores_mga_aplicables=["SEG-001", "SEG-002", "SEG-010"],
            coordinacion_requerida=["Policía Nacional", "Ministerio de Defensa"],
            aplicable_pdet=True,
            prioridad_pdet="alta"
        ))
        
        # ORDENAMIENTO TERRITORIAL (Ley 388/1997)
        self.add_competencia(CompetenciaMunicipal(
            codigo="ORD-001",
            sector=SectorCompetencia.ORDENAMIENTO_TERRITORIAL,
            tipo=TipoCompetencia.PROPIA,
            descripcion="Formular, adoptar y ejecutar el Plan de Ordenamiento Territorial (POT/PBOT/EOT)",
            base_legal=["Ley 388/1997", "Decreto 1077/2015"],
            indicadores_mga_aplicables=["ORD-001", "ORD-002"],
            aplicable_pdet=True,
            prioridad_pdet="alta"
        ))
        
        # PREVENCIÓN Y ATENCIÓN DE DESASTRES (Ley 1523/2012)
        self.add_competencia(CompetenciaMunicipal(
            codigo="DES-001",
            sector=SectorCompetencia.PREVENCION_ATENCION_DESASTRES,
            tipo=TipoCompetencia.CONCURRENTE,
            descripcion="Gestión del riesgo de desastres: conocimiento, reducción y manejo",
            base_legal=["Ley 1523/2012"],
            indicadores_mga_aplicables=["DES-001", "DES-002", "DES-010"],
            coordinacion_requerida=["UNGRD", "Gobernación"],
            aplicable_pdet=True,
            prioridad_pdet="alta"
        ))
        
        # EQUIPAMIENTO MUNICIPAL
        self.add_competencia(CompetenciaMunicipal(
            codigo="EQU-001",
            sector=SectorCompetencia.EQUIPAMIENTO_MUNICIPAL,
            tipo=TipoCompetencia.PROPIA,
            descripcion="Construcción y mantenimiento de infraestructura institucional y equipamiento municipal",
            base_legal=["Ley 136/1994", "Ley 1551/2012"],
            indicadores_mga_aplicables=["EQU-001", "EQU-002"],
            aplicable_pdet=True,
            prioridad_pdet="media"
        ))
    
    def add_competencia(self, competencia: CompetenciaMunicipal):
        """Add competency to catalog"""
        self.competencias[competencia.codigo] = competencia
        logger.debug(f"Competencia agregada: {competencia.codigo}")
    
    def get_competencia(self, codigo: str) -> Optional[CompetenciaMunicipal]:
        """Get competency by code"""
        return self.competencias.get(codigo)
    
    def get_by_sector(self, sector: SectorCompetencia) -> List[CompetenciaMunicipal]:
        """Get all competencies for a sector"""
        return [c for c in self.competencias.values() if c.sector == sector]
    
    def get_by_tipo(self, tipo: TipoCompetencia) -> List[CompetenciaMunicipal]:
        """Get all competencies by type"""
        return [c for c in self.competencias.values() if c.tipo == tipo]
    
    def get_pdet_prioritarias(self) -> List[CompetenciaMunicipal]:
        """Get high-priority competencies for PDET municipalities"""
        return [c for c in self.competencias.values() 
                if c.aplicable_pdet and c.prioridad_pdet == "alta"]
    
    def validar_competencia_municipal(self, sector: str, descripcion: str) -> Dict[str, Any]:
        """
        Validate if a project/goal is within municipal competencies
        Returns validation result with competency alignment
        """
        sector_match = None
        for s in SectorCompetencia:
            if s.value in sector.lower():
                sector_match = s
                break
        
        if not sector_match:
            return {
                "valido": False,
                "mensaje": "Sector no identificado en competencias municipales",
                "competencias_aplicables": []
            }
        
        competencias_sector = self.get_by_sector(sector_match)
        
        return {
            "valido": True,
            "sector": sector_match.value,
            "competencias_aplicables": [c.codigo for c in competencias_sector],
            "tipo_competencias": [c.tipo.value for c in competencias_sector],
            "base_legal": list(set([bl for c in competencias_sector for bl in c.base_legal])),
            "coordinacion_requerida": list(set([coord for c in competencias_sector 
                                                for coord in c.coordinacion_requerida]))
        }
    
    def generar_matriz_competencias(self) -> Dict[str, List[str]]:
        """Generate competency matrix for reporting"""
        matriz = {}
        for sector in SectorCompetencia:
            competencias_sector = self.get_by_sector(sector)
            matriz[sector.value] = [f"{c.codigo}: {c.descripcion[:80]}..." 
                                   for c in competencias_sector]
        return matriz


# Singleton instance for easy access
CATALOGO_COMPETENCIAS = CatalogoCompetenciasMunicipales()


if __name__ == "__main__":
    # Demo usage
    catalogo = CatalogoCompetenciasMunicipales()
    
    print("=== Competencias Municipales - Demo ===\n")
    
    print(f"Total competencias: {len(catalogo.competencias)}")
    print(f"Competencias PDET prioritarias: {len(catalogo.get_pdet_prioritarias())}\n")
    
    # Example validation
    validacion = catalogo.validar_competencia_municipal("educacion", "construcción de escuelas")
    print(f"Validación educación: {validacion}\n")
    
    # Show PDET priorities
    print("Competencias prioritarias PDET:")
    for comp in catalogo.get_pdet_prioritarias()[:5]:
        print(f"  - {comp.codigo}: {comp.descripcion[:60]}...")
