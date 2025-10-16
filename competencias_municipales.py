#!/usr/bin/env python3
"""
Competencias Municipales - Catálogo de Competencias Municipales
Complete catalog of municipal competencies based on Colombian legislation
for validation of municipal development plans and projects.

Reference: Ley 136/1994, Ley 715/2001, Ley 1551/2012
Last updated: 2024
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("competencias_municipales")


class TipoCompetencia(Enum):
    """Type of municipal competency"""
    EXCLUSIVA = "exclusiva"  # Exclusive municipal competency
    CONCURRENTE = "concurrente"  # Shared with other levels
    SUBSIDIARIA = "subsidiaria"  # Subsidiary competency
    COMPLEMENTARIA = "complementaria"  # Complementary competency


class SectorCompetencia(Enum):
    """Sector classification for competencies"""
    EDUCACION = "educacion"
    SALUD = "salud"
    AGUA_SANEAMIENTO = "agua_saneamiento"
    VIVIENDA = "vivienda"
    TRANSPORTE = "transporte"
    AGRICULTURA = "agricultura"
    AMBIENTE = "ambiente"
    CULTURA = "cultura"
    DEPORTE = "deporte"
    DESARROLLO_ECONOMICO = "desarrollo_economico"
    INCLUSION_SOCIAL = "inclusion_social"
    SEGURIDAD = "seguridad"
    ORDENAMIENTO_TERRITORIAL = "ordenamiento_territorial"
    GESTION_RIESGO = "gestion_riesgo"
    EQUIPAMIENTO = "equipamiento"
    SERVICIOS_PUBLICOS = "servicios_publicos"


@dataclass
class CompetenciaMunicipal:
    """Represents a municipal competency"""
    codigo: str
    nombre: str
    sector: SectorCompetencia
    tipo: TipoCompetencia
    descripcion: str
    base_normativa: List[str]
    responsabilidades: List[str]
    limitaciones: List[str] = field(default_factory=list)
    requisitos: List[str] = field(default_factory=list)
    nivel_aplicacion: str = "Municipal"


class CatalogoCompetenciasMunicipales:
    """
    Official catalog of municipal competencies
    Validates projects against Colombian municipal legal framework
    """
    
    def __init__(self):
        self.competencias: Dict[str, CompetenciaMunicipal] = {}
        self._initialize_competencias()
    
    def _initialize_competencias(self):
        """Initialize comprehensive municipal competencies catalog"""
        
        # ===== EDUCACIÓN =====
        self.add_competencia(CompetenciaMunicipal(
            codigo="COMP-EDU-001",
            nombre="Administración del servicio educativo preescolar, básica y media",
            sector=SectorCompetencia.EDUCACION,
            tipo=TipoCompetencia.CONCURRENTE,
            descripcion="Administrar, financiar y cofinanciar el servicio educativo en los niveles de preescolar, básica y media",
            base_normativa=["Ley 715/2001 Art. 7", "Ley 1551/2012"],
            responsabilidades=[
                "Administrar establecimientos educativos",
                "Mantener infraestructura educativa",
                "Proveer dotación escolar",
                "Garantizar gratuidad educativa"
            ],
            limitaciones=[
                "No incluye pago de docentes (competencia nacional)",
                "Aplica solo para establecimientos oficiales"
            ]
        ))
        
        self.add_competencia(CompetenciaMunicipal(
            codigo="COMP-EDU-002",
            nombre="Construcción y mantenimiento de infraestructura educativa",
            sector=SectorCompetencia.EDUCACION,
            tipo=TipoCompetencia.EXCLUSIVA,
            descripcion="Construcción, ampliación, mantenimiento y dotación de establecimientos educativos",
            base_normativa=["Ley 715/2001", "Ley 136/1994"],
            responsabilidades=[
                "Construir nuevas sedes educativas",
                "Mantener infraestructura existente",
                "Proveer mobiliario y equipamiento"
            ]
        ))
        
        # ===== SALUD =====
        self.add_competencia(CompetenciaMunicipal(
            codigo="COMP-SAL-001",
            nombre="Prestación de servicios de salud",
            sector=SectorCompetencia.SALUD,
            tipo=TipoCompetencia.CONCURRENTE,
            descripcion="Prestar servicios de salud de primer nivel y financiar acciones de salud pública",
            base_normativa=["Ley 715/2001 Art. 44", "Ley 1438/2011"],
            responsabilidades=[
                "Operar ESE de primer nivel",
                "Implementar Plan de Salud Pública",
                "Vigilancia epidemiológica",
                "Atención materno-infantil"
            ]
        ))
        
        self.add_competencia(CompetenciaMunicipal(
            codigo="COMP-SAL-002",
            nombre="Infraestructura de salud",
            sector=SectorCompetencia.SALUD,
            tipo=TipoCompetencia.EXCLUSIVA,
            descripcion="Construcción y mantenimiento de centros y puestos de salud",
            base_normativa=["Ley 715/2001"],
            responsabilidades=[
                "Construir centros de salud",
                "Mantener infraestructura sanitaria",
                "Dotar establecimientos de salud"
            ]
        ))
        
        # ===== AGUA Y SANEAMIENTO =====
        self.add_competencia(CompetenciaMunicipal(
            codigo="COMP-AGUA-001",
            nombre="Prestación de servicios de agua potable y saneamiento básico",
            sector=SectorCompetencia.AGUA_SANEAMIENTO,
            tipo=TipoCompetencia.EXCLUSIVA,
            descripcion="Garantizar prestación de servicios de acueducto, alcantarillado y aseo",
            base_normativa=["Ley 142/1994", "Ley 715/2001 Art. 76"],
            responsabilidades=[
                "Asegurar prestación del servicio",
                "Mantener infraestructura de acueducto",
                "Operar sistemas de alcantarillado",
                "Gestionar residuos sólidos"
            ]
        ))
        
        # ===== VIVIENDA =====
        self.add_competencia(CompetenciaMunicipal(
            codigo="COMP-VIV-001",
            nombre="Vivienda de interés social",
            sector=SectorCompetencia.VIVIENDA,
            tipo=TipoCompetencia.CONCURRENTE,
            descripcion="Promover y apoyar programas de vivienda de interés social",
            base_normativa=["Ley 388/1997", "Ley 1537/2012"],
            responsabilidades=[
                "Promover construcción de VIS",
                "Apoyar mejoramiento de vivienda",
                "Facilitar acceso a subsidios"
            ]
        ))
        
        # ===== TRANSPORTE =====
        self.add_competencia(CompetenciaMunicipal(
            codigo="COMP-TRA-001",
            nombre="Infraestructura vial municipal",
            sector=SectorCompetencia.TRANSPORTE,
            tipo=TipoCompetencia.EXCLUSIVA,
            descripcion="Construcción y mantenimiento de vías urbanas y rurales municipales",
            base_normativa=["Ley 1228/2008", "Ley 105/1993"],
            responsabilidades=[
                "Construir vías urbanas",
                "Mantener red vial rural",
                "Señalización vial"
            ]
        ))
        
        # ===== AGRICULTURA =====
        self.add_competencia(CompetenciaMunicipal(
            codigo="COMP-AGR-001",
            nombre="Desarrollo agropecuario",
            sector=SectorCompetencia.AGRICULTURA,
            tipo=TipoCompetencia.COMPLEMENTARIA,
            descripcion="Promover y apoyar desarrollo agropecuario local",
            base_normativa=["Ley 101/1993", "Ley 607/2000"],
            responsabilidades=[
                "Asistencia técnica agropecuaria",
                "Apoyo a comercialización",
                "Desarrollo rural integral"
            ]
        ))
        
        # ===== AMBIENTE =====
        self.add_competencia(CompetenciaMunicipal(
            codigo="COMP-AMB-001",
            nombre="Gestión ambiental municipal",
            sector=SectorCompetencia.AMBIENTE,
            tipo=TipoCompetencia.CONCURRENTE,
            descripcion="Promover y ejecutar acciones de protección ambiental",
            base_normativa=["Ley 99/1993", "Ley 1333/2009"],
            responsabilidades=[
                "Reforestación y conservación",
                "Gestión de residuos sólidos",
                "Educación ambiental"
            ]
        ))
        
        # ===== CULTURA =====
        self.add_competencia(CompetenciaMunicipal(
            codigo="COMP-CUL-001",
            nombre="Fomento cultural",
            sector=SectorCompetencia.CULTURA,
            tipo=TipoCompetencia.EXCLUSIVA,
            descripcion="Promover y financiar actividades culturales",
            base_normativa=["Ley 397/1997", "Ley 1185/2008"],
            responsabilidades=[
                "Operar bibliotecas y casas de cultura",
                "Programas de formación artística",
                "Protección patrimonio cultural"
            ]
        ))
        
        # ===== DEPORTE =====
        self.add_competencia(CompetenciaMunicipal(
            codigo="COMP-DEP-001",
            nombre="Recreación y deporte",
            sector=SectorCompetencia.DEPORTE,
            tipo=TipoCompetencia.EXCLUSIVA,
            descripcion="Promover y financiar programas de recreación y deporte",
            base_normativa=["Ley 181/1995"],
            responsabilidades=[
                "Construir y mantener escenarios deportivos",
                "Programas de recreación",
                "Apoyo a deportistas"
            ]
        ))
        
        # ===== ORDENAMIENTO TERRITORIAL =====
        self.add_competencia(CompetenciaMunicipal(
            codigo="COMP-ORD-001",
            nombre="Ordenamiento territorial",
            sector=SectorCompetencia.ORDENAMIENTO_TERRITORIAL,
            tipo=TipoCompetencia.EXCLUSIVA,
            descripcion="Formular y adoptar Plan de Ordenamiento Territorial",
            base_normativa=["Ley 388/1997"],
            responsabilidades=[
                "Adoptar POT/PBOT/EOT",
                "Control urbanístico",
                "Licencias de construcción"
            ]
        ))
        
        # ===== GESTIÓN DEL RIESGO =====
        self.add_competencia(CompetenciaMunicipal(
            codigo="COMP-RIE-001",
            nombre="Gestión del riesgo de desastres",
            sector=SectorCompetencia.GESTION_RIESGO,
            tipo=TipoCompetencia.CONCURRENTE,
            descripcion="Conocimiento, reducción y manejo del riesgo de desastres",
            base_normativa=["Ley 1523/2012"],
            responsabilidades=[
                "Elaborar Plan de Gestión del Riesgo",
                "Prevención de desastres",
                "Atención de emergencias"
            ]
        ))
    
    def add_competencia(self, competencia: CompetenciaMunicipal):
        """Add competency to catalog"""
        self.competencias[competencia.codigo] = competencia
    
    def get_competencia(self, codigo: str) -> Optional[CompetenciaMunicipal]:
        """Get competency by code"""
        return self.competencias.get(codigo)
    
    def get_by_sector(self, sector: str) -> List[CompetenciaMunicipal]:
        """Get all competencies for a specific sector"""
        # Normalize sector name
        sector_lower = sector.lower().strip()
        
        # Map common sector names to enum values
        sector_mapping = {
            "educacion": SectorCompetencia.EDUCACION,
            "educación": SectorCompetencia.EDUCACION,
            "salud": SectorCompetencia.SALUD,
            "agua": SectorCompetencia.AGUA_SANEAMIENTO,
            "agua_potable": SectorCompetencia.AGUA_SANEAMIENTO,
            "saneamiento": SectorCompetencia.AGUA_SANEAMIENTO,
            "vivienda": SectorCompetencia.VIVIENDA,
            "transporte": SectorCompetencia.TRANSPORTE,
            "agricultura": SectorCompetencia.AGRICULTURA,
            "ambiente": SectorCompetencia.AMBIENTE,
            "cultura": SectorCompetencia.CULTURA,
            "deporte": SectorCompetencia.DEPORTE,
            "desarrollo_economico": SectorCompetencia.DESARROLLO_ECONOMICO,
            "inclusion_social": SectorCompetencia.INCLUSION_SOCIAL,
            "seguridad": SectorCompetencia.SEGURIDAD,
            "ordenamiento_territorial": SectorCompetencia.ORDENAMIENTO_TERRITORIAL,
            "gestion_riesgo": SectorCompetencia.GESTION_RIESGO,
            "equipamiento": SectorCompetencia.EQUIPAMIENTO,
        }
        
        sector_enum = sector_mapping.get(sector_lower)
        if not sector_enum:
            return []
        
        return [c for c in self.competencias.values() if c.sector == sector_enum]
    
    def get_by_tipo(self, tipo: TipoCompetencia) -> List[CompetenciaMunicipal]:
        """Get all competencies of a specific type"""
        return [c for c in self.competencias.values() if c.tipo == tipo]
    
    def validar_competencia_municipal(self, sector: str, descripcion: str) -> Dict[str, Any]:
        """
        Validate if a project/program is within municipal competencies
        
        Args:
            sector: Project sector
            descripcion: Project description
            
        Returns:
            Validation result with competencies and recommendations
        """
        competencias_sector = self.get_by_sector(sector)
        
        if not competencias_sector:
            return {
                "valido": False,
                "sector": sector,
                "razon": f"Sector '{sector}' no reconocido o sin competencias municipales definidas",
                "competencias_aplicables": [],
                "recomendaciones": [
                    "Verificar que el sector esté dentro del ámbito de competencias municipales",
                    "Consultar marco normativo (Ley 136/1994, Ley 715/2001, Ley 1551/2012)"
                ]
            }
        
        # Project is valid if there are applicable competencies for the sector
        return {
            "valido": True,
            "sector": sector,
            "competencias_aplicables": [c.codigo for c in competencias_sector],
            "competencias_detalle": [
                {
                    "codigo": c.codigo,
                    "nombre": c.nombre,
                    "tipo": c.tipo.value,
                    "base_normativa": c.base_normativa
                }
                for c in competencias_sector
            ],
            "recomendaciones": [
                f"Alinear con competencia {c.codigo}: {c.nombre}"
                for c in competencias_sector[:2]  # Top 2 recommendations
            ]
        }
    
    def generar_reporte_competencias(self) -> Dict[str, Any]:
        """Generate comprehensive competencies report"""
        return {
            "total_competencias": len(self.competencias),
            "por_sector": {
                sector.value: len(self.get_by_sector(sector.value))
                for sector in SectorCompetencia
            },
            "por_tipo": {
                tipo.value: len(self.get_by_tipo(tipo))
                for tipo in TipoCompetencia
            }
        }


# Singleton instance
CATALOGO_COMPETENCIAS = CatalogoCompetenciasMunicipales()


if __name__ == "__main__":
    # Demo usage
    catalogo = CatalogoCompetenciasMunicipales()
    
    print("=== Catálogo Competencias Municipales - Demo ===\n")
    
    reporte = catalogo.generar_reporte_competencias()
    print(f"Total competencias: {reporte['total_competencias']}")
    print(f"Competencias por tipo:")
    for tipo, count in reporte['por_tipo'].items():
        print(f"  - {tipo}: {count}")
    
    print("\nValidación de proyecto educativo:")
    resultado = catalogo.validar_competencia_municipal(
        "educacion",
        "Construcción de escuela rural"
    )
    print(f"  Válido: {resultado['valido']}")
    print(f"  Competencias aplicables: {len(resultado['competencias_aplicables'])}")
