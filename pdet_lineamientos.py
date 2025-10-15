#!/usr/bin/env python3
"""
PDET Lineamientos - Programas de Desarrollo con Enfoque Territorial
Special planning guidelines for PDET municipalities in Colombia

Based on:
- Decreto 893 de 2017 (Creación PDET)
- Acuerdo Final de Paz (2016)
- Lineamientos DNP para formulación de Planes de Desarrollo en municipios PDET
- Resolución 0464 de 2020 - ART (Agencia de Renovación del Territorio)

170 PDET municipalities across 19 subregions in Colombia
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Set, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdet_lineamientos")


class SubregionPDET(Enum):
    """19 PDET subregions"""
    ALTO_PATIA = "Alto Patía - Norte del Cauca"
    ARAUCA = "Arauca"
    BAJO_CAUCA = "Bajo Cauca y Nordeste Antioqueño"
    CATATUMBO = "Catatumbo"
    CHOCO = "Chocó"
    CUENCA_CAGUÁN = "Cuenca del Caguán y Piedemonte Caqueteño"
    MACARENA_GUAVIARE = "Macarena - Guaviare"
    MONTES_MARIA = "Montes de María"
    PACIFICO_MEDIO = "Pacífico Medio"
    PACIFICO_SUR = "Pacífico y Frontera Nariñense"
    PUTUMAYO = "Putumayo"
    SIERRA_NEVADA = "Sierra Nevada - Perijá - Zona Bananera"
    SUR_BOLIVAR = "Sur de Bolívar"
    SUR_CORDOBA = "Sur de Córdoba"
    SUR_TOLIMA = "Sur del Tolima"
    URABÁ_ANTIOQUEÑO = "Urabá Antioqueño"
    PACÍFICO_MEDIO_NARIÑO = "Pacífico Medio"


class PilarPDET(Enum):
    """8 pillars of PDET - Acuerdo Final"""
    ORDENAMIENTO_TERRITORIAL = 1  # Social and productive land use
    SALUD_RURAL = 2
    EDUCACION_RURAL = 3
    VIVIENDA_AGUA_SANEAMIENTO = 4
    REACTIVACION_ECONOMICA = 5
    RECONCILIACION_CONVIVENCIA = 6
    SISTEMA_ALIMENTACION = 7
    INFRAESTRUCTURA_CONECTIVIDAD = 8


class EnfoquePDET(Enum):
    """PDET planning approaches"""
    TERRITORIAL = "territorial"  # Bottom-up territorial planning
    PARTICIPATIVO = "participativo"  # Participatory with communities
    DIFERENCIAL = "diferencial"  # Differential approach (ethnic, gender, etc.)
    TRANSFORMADOR = "transformador"  # Structural transformation focus
    REPARADOR = "reparador"  # Reparative for conflict victims


@dataclass
class LineamientoPDET:
    """Represents a specific PDET planning guideline"""
    codigo: str
    pilar: PilarPDET
    titulo: str
    descripcion: str
    criterios_priorizacion: List[str]
    enfoque_requerido: List[EnfoquePDET]
    indicadores_especificos: List[str]
    articulacion_institucional: List[str]
    presupuesto_minimo_recomendado: Optional[float] = None
    base_normativa: List[str] = field(default_factory=list)


@dataclass
class RequisitosPDET:
    """Special requirements for PDET municipal development plans"""
    participacion_comunitaria_minima: float = 70.0  # % communities engaged
    junta_accion_comunal_participacion: float = 60.0  # % JAC participation
    consejo_comunitario_participacion: float = 50.0  # % for afro communities
    cabildo_indigena_participacion: float = 50.0  # % for indigenous
    victimas_participacion: float = 40.0  # % conflict victims participation
    mujeres_participacion: float = 30.0  # % women in planning
    jovenes_participacion: float = 20.0  # % youth participation
    presupuesto_inversion_rural: float = 60.0  # % budget for rural areas
    alineacion_patr_minima: float = 80.0  # % alignment with PATR
    

class LineamientosPDET:
    """
    Comprehensive PDET planning guidelines
    Ensures compliance with peace agreement and territorial development
    """
    
    def __init__(self):
        self.lineamientos: Dict[str, LineamientoPDET] = {}
        self.requisitos = RequisitosPDET()
        self._initialize_lineamientos()
    
    def _initialize_lineamientos(self):
        """Initialize PDET planning guidelines"""
        
        # PILAR 1: ORDENAMIENTO SOCIAL DE LA PROPIEDAD Y USO DEL SUELO
        self.add_lineamiento(LineamientoPDET(
            codigo="PDET-ORD-001",
            pilar=PilarPDET.ORDENAMIENTO_TERRITORIAL,
            titulo="Formalización de la propiedad rural",
            descripcion="Titulación y formalización de predios rurales de pequeños campesinos",
            criterios_priorizacion=[
                "Prioridad a víctimas del conflicto",
                "Predios en zonas de reserva campesina",
                "Territorialidad étnica",
                "Mujeres cabeza de hogar"
            ],
            enfoque_requerido=[EnfoquePDET.TERRITORIAL, EnfoquePDET.DIFERENCIAL, EnfoquePDET.REPARADOR],
            indicadores_especificos=[
                "Número de predios formalizados",
                "Hectáreas tituladas",
                "Familias beneficiadas con títulos",
                "% predios de mujeres tituladas"
            ],
            articulacion_institucional=["ANT", "IGAC", "Notarías", "Oficina de Registro"],
            base_normativa=["Decreto 902/2017", "Ley 160/1994"]
        ))
        
        self.add_lineamiento(LineamientoPDET(
            codigo="PDET-ORD-002",
            pilar=PilarPDET.ORDENAMIENTO_TERRITORIAL,
            titulo="Zonas de Reserva Campesina (ZRC)",
            descripcion="Constitución y fortalecimiento de Zonas de Reserva Campesina",
            criterios_priorizacion=[
                "Presencia histórica campesina",
                "Conflictos de uso del suelo",
                "Economía campesina consolidada"
            ],
            enfoque_requerido=[EnfoquePDET.TERRITORIAL, EnfoquePDET.PARTICIPATIVO],
            indicadores_especificos=[
                "Número de ZRC constituidas",
                "Hectáreas bajo figura ZRC",
                "Familias beneficiadas en ZRC"
            ],
            articulacion_institucional=["ANT", "INCODER", "Ministerio de Agricultura"],
            base_normativa=["Ley 160/1994"]
        ))
        
        # PILAR 2: SALUD RURAL
        self.add_lineamiento(LineamientoPDET(
            codigo="PDET-SAL-001",
            pilar=PilarPDET.SALUD_RURAL,
            titulo="Red de salud rural integrada",
            descripcion="Fortalecimiento de la red de prestación de servicios de salud en zonas rurales",
            criterios_priorizacion=[
                "Veredas sin acceso a salud (>2 horas)",
                "Alta mortalidad materno-infantil",
                "Presencia de enfermedades tropicales"
            ],
            enfoque_requerido=[EnfoquePDET.TERRITORIAL, EnfoquePDET.DIFERENCIAL],
            indicadores_especificos=[
                "SAL-001",  # MGA indicators
                "SAL-002",
                "Tiempo promedio acceso a servicios salud",
                "% veredas con puesto de salud a <1 hora"
            ],
            articulacion_institucional=["Ministerio de Salud", "EPS", "IPS rurales"],
            base_normativa=["Ley 1438/2011", "Resolución 3280/2018"]
        ))
        
        self.add_lineamiento(LineamientoPDET(
            codigo="PDET-SAL-002",
            pilar=PilarPDET.SALUD_RURAL,
            titulo="Brigadas de salud móviles",
            descripcion="Implementación de equipos móviles de salud para zonas dispersas",
            criterios_priorizacion=[
                "Veredas de muy difícil acceso",
                "Población indígena y afro",
                "Zonas con presencia de minas antipersonal"
            ],
            enfoque_requerido=[EnfoquePDET.TERRITORIAL, EnfoquePDET.DIFERENCIAL],
            indicadores_especificos=[
                "Número de brigadas realizadas",
                "Personas atendidas en brigadas",
                "% cobertura veredas dispersas"
            ],
            articulacion_institucional=["Secretaría de Salud", "Hospital", "Ejército Nacional"],
            base_normativa=["Decreto 893/2017"]
        ))
        
        # PILAR 3: EDUCACIÓN RURAL
        self.add_lineamiento(LineamientoPDET(
            codigo="PDET-EDU-001",
            pilar=PilarPDET.EDUCACION_RURAL,
            titulo="Infraestructura educativa rural de calidad",
            descripcion="Construcción y mejoramiento de sedes educativas en zonas rurales",
            criterios_priorizacion=[
                "Sedes en estado crítico",
                "Veredas sin sede educativa",
                "Alta deserción por infraestructura deficiente"
            ],
            enfoque_requerido=[EnfoquePDET.TERRITORIAL, EnfoquePDET.TRANSFORMADOR],
            indicadores_especificos=[
                "EDU-020",  # MGA
                "EDU-021",
                "% sedes rurales en buen estado",
                "Estudiantes beneficiados"
            ],
            articulacion_institucional=["Ministerio de Educación", "Secretaría de Educación"],
            base_normativa=["Decreto 1075/2015"]
        ))
        
        self.add_lineamiento(LineamientoPDET(
            codigo="PDET-EDU-002",
            pilar=PilarPDET.EDUCACION_RURAL,
            titulo="Modelos educativos flexibles rurales",
            descripcion="Implementación de modelos educativos pertinentes para el contexto rural",
            criterios_priorizacion=[
                "Baja cobertura en secundaria rural",
                "Alta dispersión poblacional",
                "Vocación agropecuaria del territorio"
            ],
            enfoque_requerido=[EnfoquePDET.TERRITORIAL, EnfoquePDET.DIFERENCIAL],
            indicadores_especificos=[
                "Número de estudiantes en modelos flexibles",
                "Cobertura secundaria rural",
                "Tasa de deserción en modelos flexibles"
            ],
            articulacion_institucional=["Ministerio de Educación", "SENA"],
            base_normativa=["Decreto 1851/2015"]
        ))
        
        # PILAR 4: VIVIENDA, AGUA Y SANEAMIENTO
        self.add_lineamiento(LineamientoPDET(
            codigo="PDET-VAS-001",
            pilar=PilarPDET.VIVIENDA_AGUA_SANEAMIENTO,
            titulo="Agua potable y saneamiento básico rural",
            descripcion="Soluciones de acueducto, alcantarillado y manejo de residuos en zonas rurales",
            criterios_priorizacion=[
                "Veredas sin acceso a agua potable",
                "IRCA alto o sin sistema",
                "Enfermedades de origen hídrico"
            ],
            enfoque_requerido=[EnfoquePDET.TERRITORIAL, EnfoquePDET.DIFERENCIAL],
            indicadores_especificos=[
                "APS-001",  # MGA
                "APS-002",
                "APS-030",
                "% cobertura agua rural",
                "Calidad del agua (IRCA)"
            ],
            articulacion_institucional=["Ministerio de Vivienda", "PAP - Planes Departamentales de Agua"],
            presupuesto_minimo_recomendado=500_000_000,  # COP
            base_normativa=["Decreto 1077/2015"]
        ))
        
        self.add_lineamiento(LineamientoPDET(
            codigo="PDET-VAS-002",
            pilar=PilarPDET.VIVIENDA_AGUA_SANEAMIENTO,
            titulo="Vivienda rural digna",
            descripcion="Mejoramiento y construcción de vivienda rural",
            criterios_priorizacion=[
                "Víctimas del conflicto",
                "Viviendas en estado crítico",
                "Hacinamiento crítico"
            ],
            enfoque_requerido=[EnfoquePDET.REPARADOR, EnfoquePDET.DIFERENCIAL],
            indicadores_especificos=[
                "VIV-010",  # MGA
                "Número de viviendas rurales mejoradas",
                "Déficit cualitativo reducido"
            ],
            articulacion_institucional=["Ministerio de Vivienda", "Banco Agrario"],
            base_normativa=["Ley 1537/2012"]
        ))
        
        # PILAR 5: REACTIVACIÓN ECONÓMICA Y PRODUCCIÓN AGROPECUARIA
        self.add_lineamiento(LineamientoPDET(
            codigo="PDET-ECO-001",
            pilar=PilarPDET.REACTIVACION_ECONOMICA,
            titulo="Asistencia técnica agropecuaria integral",
            descripcion="Servicios de asistencia técnica directa rural con enfoque agroecológico",
            criterios_priorizacion=[
                "Pequeños productores campesinos",
                "Economía campesina familiar",
                "Sustitución de cultivos ilícitos"
            ],
            enfoque_requerido=[EnfoquePDET.TERRITORIAL, EnfoquePDET.TRANSFORMADOR],
            indicadores_especificos=[
                "AGR-002",  # MGA
                "Productores con asistencia técnica",
                "Incremento productividad agrícola",
                "Familias con seguridad alimentaria"
            ],
            articulacion_institucional=["MADR", "ADR", "EPSAGRO municipales"],
            base_normativa=["Ley 1876/2017"]
        ))
        
        self.add_lineamiento(LineamientoPDET(
            codigo="PDET-ECO-002",
            pilar=PilarPDET.REACTIVACION_ECONOMICA,
            titulo="Proyectos productivos y economía solidaria",
            descripcion="Apoyo a proyectos productivos asociativos y cooperativas rurales",
            criterios_priorizacion=[
                "Asociaciones de víctimas",
                "Cooperativas campesinas",
                "Proyectos de mujeres rurales"
            ],
            enfoque_requerido=[EnfoquePDET.DIFERENCIAL, EnfoquePDET.TRANSFORMADOR],
            indicadores_especificos=[
                "Número de proyectos productivos apoyados",
                "Familias beneficiadas",
                "Incremento de ingresos rurales"
            ],
            articulacion_institucional=["MADR", "ADR", "Banco Agrario"],
            base_normativa=["Ley 454/1998"]
        ))
        
        self.add_lineamiento(LineamientoPDET(
            codigo="PDET-ECO-003",
            pilar=PilarPDET.REACTIVACION_ECONOMICA,
            titulo="Sustitución de cultivos ilícitos",
            descripcion="Programas de sustitución voluntaria y alternativas productivas legales",
            criterios_priorizacion=[
                "Veredas con coca/marihuana/amapola",
                "Familias en PNIS",
                "Zonas de frontera agrícola"
            ],
            enfoque_requerido=[EnfoquePDET.TERRITORIAL, EnfoquePDET.TRANSFORMADOR, EnfoquePDET.REPARADOR],
            indicadores_especificos=[
                "Hectáreas de cultivos ilícitos erradicadas",
                "Familias en sustitución voluntaria",
                "Proyectos productivos alternativos implementados"
            ],
            articulacion_institucional=["PNIS", "ADR", "Fuerza Pública"],
            base_normativa=["Decreto 896/2017"]
        ))
        
        # PILAR 6: RECONCILIACIÓN, CONVIVENCIA Y CONSTRUCCIÓN DE PAZ
        self.add_lineamiento(LineamientoPDET(
            codigo="PDET-PAZ-001",
            pilar=PilarPDET.RECONCILIACION_CONVIVENCIA,
            titulo="Programas de reconciliación y memoria histórica",
            descripcion="Espacios de diálogo, reconciliación y construcción de memoria",
            criterios_priorizacion=[
                "Víctimas del conflicto",
                "Comunidades receptoras de reincorporados",
                "Sitios de memoria"
            ],
            enfoque_requerido=[EnfoquePDET.REPARADOR, EnfoquePDET.PARTICIPATIVO],
            indicadores_especificos=[
                "Número de iniciativas de reconciliación",
                "Personas participantes en espacios de paz",
                "Sitios de memoria establecidos"
            ],
            articulacion_institucional=["Unidad de Víctimas", "Centro Nacional de Memoria", "ARN"],
            base_normativa=["Ley 1448/2011"]
        ))
        
        self.add_lineamiento(LineamientoPDET(
            codigo="PDET-PAZ-002",
            pilar=PilarPDET.RECONCILIACION_CONVIVENCIA,
            titulo="Fortalecimiento de tejido social comunitario",
            descripcion="Apoyo a organizaciones comunitarias, JAC, consejos comunitarios",
            criterios_priorizacion=[
                "JAC debilitadas",
                "Comunidades étnicas",
                "Mujeres lideresas"
            ],
            enfoque_requerido=[EnfoquePDET.PARTICIPATIVO, EnfoquePDET.DIFERENCIAL],
            indicadores_especificos=[
                "Número de JAC fortalecidas",
                "Líderes formados en paz y convivencia",
                "Iniciativas comunitarias apoyadas"
            ],
            articulacion_institucional=["Ministerio del Interior", "Gobernación"],
            base_normativa=["Ley 743/2002"]
        ))
        
        # PILAR 7: SISTEMA DE ALIMENTACIÓN Y NUTRICIÓN
        self.add_lineamiento(LineamientoPDET(
            codigo="PDET-ALI-001",
            pilar=PilarPDET.SISTEMA_ALIMENTACION,
            titulo="Seguridad alimentaria y nutricional rural",
            descripcion="Programas de agricultura familiar y huertas comunitarias",
            criterios_priorizacion=[
                "Desnutrición infantil alta",
                "Inseguridad alimentaria severa",
                "Familias con monocultivo"
            ],
            enfoque_requerido=[EnfoquePDET.TERRITORIAL, EnfoquePDET.DIFERENCIAL],
            indicadores_especificos=[
                "Familias con huerta casera",
                "Tasa de desnutrición infantil",
                "Diversidad alimentaria mejorada"
            ],
            articulacion_institucional=["ICBF", "MADR", "Secretaría de Salud"],
            base_normativa=["CONPES 113/2008"]
        ))
        
        # PILAR 8: INFRAESTRUCTURA Y CONECTIVIDAD RURAL
        self.add_lineamiento(LineamientoPDET(
            codigo="PDET-INF-001",
            pilar=PilarPDET.INFRAESTRUCTURA_CONECTIVIDAD,
            titulo="Vías terciarias y caminos rurales",
            descripcion="Construcción y mantenimiento de vías terciarias para conectividad rural",
            criterios_priorizacion=[
                "Veredas sin vía de acceso",
                "Vías intransitables en invierno",
                "Zonas productivas aisladas"
            ],
            enfoque_requerido=[EnfoquePDET.TERRITORIAL, EnfoquePDET.TRANSFORMADOR],
            indicadores_especificos=[
                "VIA-001",  # MGA
                "VIA-002",
                "VIA-010",
                "Km de vías terciarias mejoradas",
                "% veredas con acceso vehicular"
            ],
            articulacion_institucional=["INVIAS", "ANI", "Gobernación"],
            presupuesto_minimo_recomendado=1_000_000_000,  # COP
            base_normativa=["Ley 1228/2008"]
        ))
        
        self.add_lineamiento(LineamientoPDET(
            codigo="PDET-INF-002",
            pilar=PilarPDET.INFRAESTRUCTURA_CONECTIVIDAD,
            titulo="Conectividad digital rural",
            descripcion="Ampliación de cobertura de internet y telefonía móvil en zonas rurales",
            criterios_priorizacion=[
                "Veredas sin cobertura",
                "Centros educativos sin internet",
                "Centros de salud sin conectividad"
            ],
            enfoque_requerido=[EnfoquePDET.TERRITORIAL, EnfoquePDET.TRANSFORMADOR],
            indicadores_especificos=[
                "Número de veredas con cobertura internet",
                "% población rural con acceso digital",
                "Instituciones públicas conectadas"
            ],
            articulacion_institucional=["MinTIC", "Operadores de telecomunicaciones"],
            base_normativa=["Ley 1341/2009"]
        ))
        
        self.add_lineamiento(LineamientoPDET(
            codigo="PDET-INF-003",
            pilar=PilarPDET.INFRAESTRUCTURA_CONECTIVIDAD,
            titulo="Electrificación rural",
            descripcion="Ampliación de cobertura de energía eléctrica en zonas rurales",
            criterios_priorizacion=[
                "Veredas sin servicio eléctrico",
                "Soluciones con paneles solares",
                "Centros educativos y salud sin energía"
            ],
            enfoque_requerido=[EnfoquePDET.TERRITORIAL],
            indicadores_especificos=[
                "Número de viviendas rurales electrificadas",
                "% cobertura eléctrica rural",
                "Soluciones de energías alternativas"
            ],
            articulacion_institucional=["IPSE", "Operador de red", "MinMinas"],
            base_normativa=["Ley 143/1994"]
        ))
    
    def add_lineamiento(self, lineamiento: LineamientoPDET):
        """Add PDET guideline to catalog"""
        self.lineamientos[lineamiento.codigo] = lineamiento
        logger.debug(f"Lineamiento PDET agregado: {lineamiento.codigo}")
    
    def get_lineamiento(self, codigo: str) -> Optional[LineamientoPDET]:
        """Get guideline by code"""
        return self.lineamientos.get(codigo)
    
    def get_by_pilar(self, pilar: PilarPDET) -> List[LineamientoPDET]:
        """Get all guidelines for a pillar"""
        return [lin for lin in self.lineamientos.values() if lin.pilar == pilar]
    
    def validar_cumplimiento_pdet(self, 
                                  participacion: Dict[str, float],
                                  presupuesto_rural: float,
                                  presupuesto_total: float,
                                  alineacion_patr: float) -> Dict[str, any]:
        """
        Validate PDET compliance for a municipal development plan
        
        Args:
            participacion: Dict with participation percentages
            presupuesto_rural: Budget allocated to rural areas
            presupuesto_total: Total municipal budget
            alineacion_patr: Alignment percentage with PATR
        """
        resultados = {
            "cumple_requisitos": True,
            "validaciones": [],
            "alertas": [],
            "recomendaciones": []
        }
        
        # Check participation requirements
        if participacion.get("comunitaria", 0) < self.requisitos.participacion_comunitaria_minima:
            resultados["cumple_requisitos"] = False
            resultados["alertas"].append(
                f"Participación comunitaria insuficiente: {participacion.get('comunitaria', 0)}% "
                f"(mínimo requerido: {self.requisitos.participacion_comunitaria_minima}%)"
            )
        
        if participacion.get("victimas", 0) < self.requisitos.victimas_participacion:
            resultados["alertas"].append(
                f"Participación de víctimas baja: {participacion.get('victimas', 0)}% "
                f"(recomendado: {self.requisitos.victimas_participacion}%)"
            )
        
        if participacion.get("mujeres", 0) < self.requisitos.mujeres_participacion:
            resultados["alertas"].append(
                f"Participación de mujeres baja: {participacion.get('mujeres', 0)}% "
                f"(recomendado: {self.requisitos.mujeres_participacion}%)"
            )
        
        # Check rural budget allocation
        porcentaje_rural = (presupuesto_rural / presupuesto_total * 100) if presupuesto_total > 0 else 0
        if porcentaje_rural < self.requisitos.presupuesto_inversion_rural:
            resultados["cumple_requisitos"] = False
            resultados["alertas"].append(
                f"Presupuesto para inversión rural insuficiente: {porcentaje_rural:.1f}% "
                f"(mínimo requerido: {self.requisitos.presupuesto_inversion_rural}%)"
            )
        else:
            resultados["validaciones"].append(
                f"✓ Presupuesto rural adecuado: {porcentaje_rural:.1f}%"
            )
        
        # Check PATR alignment
        if alineacion_patr < self.requisitos.alineacion_patr_minima:
            resultados["cumple_requisitos"] = False
            resultados["alertas"].append(
                f"Alineación con PATR insuficiente: {alineacion_patr}% "
                f"(mínimo requerido: {self.requisitos.alineacion_patr_minima}%)"
            )
        else:
            resultados["validaciones"].append(
                f"✓ Alineación con PATR adecuada: {alineacion_patr}%"
            )
        
        # Generate recommendations
        if not resultados["cumple_requisitos"]:
            resultados["recomendaciones"].append(
                "Fortalecer procesos participativos con enfoque diferencial"
            )
            resultados["recomendaciones"].append(
                "Incrementar asignación presupuestal para zona rural"
            )
            resultados["recomendaciones"].append(
                "Alinear metas e indicadores con el PATR subregional"
            )
        
        return resultados
    
    def generar_matriz_priorizacion_pdet(self) -> Dict[PilarPDET, List[str]]:
        """Generate prioritization matrix by PDET pillar"""
        matriz = {}
        for pilar in PilarPDET:
            lineamientos_pilar = self.get_by_pilar(pilar)
            matriz[pilar] = [f"{lin.codigo}: {lin.titulo}" for lin in lineamientos_pilar]
        return matriz
    
    def recomendar_lineamientos(self, 
                                sector: str, 
                                poblacion_victimas: bool = False) -> List[LineamientoPDET]:
        """Recommend relevant PDET guidelines based on project characteristics"""
        recomendaciones = []
        
        sector_lower = sector.lower()
        
        # Map sectors to PDET pillars
        if "educaci" in sector_lower:
            recomendaciones.extend(self.get_by_pilar(PilarPDET.EDUCACION_RURAL))
        elif "salud" in sector_lower:
            recomendaciones.extend(self.get_by_pilar(PilarPDET.SALUD_RURAL))
        elif "agua" in sector_lower or "saneamiento" in sector_lower or "vivienda" in sector_lower:
            recomendaciones.extend(self.get_by_pilar(PilarPDET.VIVIENDA_AGUA_SANEAMIENTO))
        elif "v" in sector_lower or "transporte" in sector_lower or "conectividad" in sector_lower:
            recomendaciones.extend(self.get_by_pilar(PilarPDET.INFRAESTRUCTURA_CONECTIVIDAD))
        elif "agr" in sector_lower or "producc" in sector_lower or "econom" in sector_lower:
            recomendaciones.extend(self.get_by_pilar(PilarPDET.REACTIVACION_ECONOMICA))
        
        # Add reconciliation if victims involved
        if poblacion_victimas:
            recomendaciones.extend(self.get_by_pilar(PilarPDET.RECONCILIACION_CONVIVENCIA))
        
        return recomendaciones


# Singleton instance
LINEAMIENTOS_PDET = LineamientosPDET()


# List of 170 PDET municipalities (representative sample)
MUNICIPIOS_PDET = {
    "Alto Patía - Norte del Cauca": ["Buenos Aires", "Caloto", "Corinto", "Miranda", "Santander de Quilichao", "Suárez", "Toribío"],
    "Arauca": ["Arauquita", "Fortul", "Saravena", "Tame"],
    "Bajo Cauca y Nordeste Antioqueño": ["Cáceres", "Caucasia", "El Bagre", "Nechí", "Tarazá", "Zaragoza", "Anorí", "Remedios", "Segovia"],
    "Catatumbo": ["Convención", "El Carmen", "El Tarra", "Hacarí", "San Calixto", "Sardinata", "Teorama", "Tibú"],
    # ... (More municipalities for demo, full list would be extensive)
}


if __name__ == "__main__":
    # Demo usage
    lineamientos = LineamientosPDET()
    
    print("=== Lineamientos PDET - Demo ===\n")
    
    print(f"Total lineamientos: {len(lineamientos.lineamientos)}")
    print(f"Pilares PDET: {len(PilarPDET)}\n")
    
    # Example validation
    participacion = {
        "comunitaria": 75.0,
        "victimas": 45.0,
        "mujeres": 35.0
    }
    
    validacion = lineamientos.validar_cumplimiento_pdet(
        participacion=participacion,
        presupuesto_rural=6_000_000_000,
        presupuesto_total=10_000_000_000,
        alineacion_patr=85.0
    )
    
    print("Validación cumplimiento PDET:")
    print(f"  Cumple: {validacion['cumple_requisitos']}")
    print(f"  Alertas: {len(validacion['alertas'])}")
    for alerta in validacion['alertas']:
        print(f"    - {alerta}")
