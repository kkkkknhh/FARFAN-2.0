#!/usr/bin/env python3
"""
MGA Indicators - Catálogo de Indicadores de Producto y Resultado
Complete catalog of MGA (Metodología General Ajustada) indicators
Based on DNP's official MGA indicator catalog for project formulation

Reference: DNP - Sistema de Seguimiento a Proyectos de Inversión (SPI)
Last updated: 2024
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Set, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mga_indicadores")


# Sector constants
SECTOR_EDUCACION = "Educación"
SECTOR_SALUD = "Salud"
SECTOR_AGUA_POTABLE_SANEAMIENTO = "Agua Potable y Saneamiento"
SECTOR_VIVIENDA = "Vivienda"
SECTOR_TRANSPORTE = "Transporte"
SECTOR_AGRICULTURA = "Agricultura"
SECTOR_AMBIENTE = "Ambiente"
SECTOR_CULTURA = "Cultura"
SECTOR_DEPORTE = "Deporte"
SECTOR_DESARROLLO_ECONOMICO = "Desarrollo Económico"
SECTOR_INCLUSION_SOCIAL = "Inclusión Social"
SECTOR_SEGURIDAD = "Seguridad"
SECTOR_ORDENAMIENTO_TERRITORIAL = "Ordenamiento Territorial"
SECTOR_GESTION_RIESGO = "Gestión del Riesgo"
SECTOR_EQUIPAMIENTO = "Equipamiento"


class TipoIndicadorMGA(Enum):
    """MGA indicator types"""
    PRODUCTO = "producto"  # Product/output indicator
    RESULTADO = "resultado"  # Outcome/result indicator
    GESTION = "gestion"  # Management indicator


class UnidadMedida(Enum):
    """Standard measurement units for MGA indicators"""
    NUMERO = "numero"
    PORCENTAJE = "porcentaje"
    TASA = "tasa"
    INDICE = "indice"
    RAZON = "razon"
    PROPORCION = "proporcion"
    KILOMETROS = "kilometros"
    METROS = "metros"
    HECTAREAS = "hectareas"
    PERSONAS = "personas"
    FAMILIAS = "familias"
    HOGARES = "hogares"
    INSTITUCIONES = "instituciones"
    UNIDADES = "unidades"


@dataclass
class IndicadorMGA:
    """Represents an official MGA indicator"""
    codigo: str
    nombre: str
    tipo: TipoIndicadorMGA
    sector: str
    formula: str
    unidad_medida: UnidadMedida
    definicion: str
    fuente_informacion: List[str]
    periodicidad: str
    desagregaciones: List[str] = field(default_factory=list)
    referencias_normativas: List[str] = field(default_factory=list)
    ods_relacionados: List[int] = field(default_factory=list)  # SDG alignment
    nivel_aplicacion: List[str] = field(default_factory=list)  # Municipal, Departamental, Nacional
    

class CatalogoIndicadoresMGA:
    """
    Official MGA indicators catalog
    Ensures alignment with DNP project formulation standards
    """
    
    def __init__(self):
        self.indicadores: Dict[str, IndicadorMGA] = {}
        self._initialize_indicadores()
    
    def _initialize_indicadores(self):
        """Initialize comprehensive MGA indicators catalog"""
        
        # ===== EDUCACIÓN =====
        
        self.add_indicador(IndicadorMGA(
            codigo="EDU-001",
            nombre="Tasa de cobertura neta en educación preescolar",
            tipo=TipoIndicadorMGA.RESULTADO,
            sector=SECTOR_EDUCACION,
            formula="(Matrícula oficial preescolar edad 5 años / Población 5 años) * 100",
            unidad_medida=UnidadMedida.PORCENTAJE,
            definicion="Porcentaje de niños de 5 años matriculados en preescolar sobre el total de población de 5 años",
            fuente_informacion=["SIMAT", "DANE - Proyecciones de población"],
            periodicidad="Anual",
            desagregaciones=["Sexo", "Zona (urbana/rural)", "Etnia"],
            referencias_normativas=["Ley 115/1994", "Ley 715/2001"],
            ods_relacionados=[4],
            nivel_aplicacion=["Municipal", "Departamental"]
        ))
        
        self.add_indicador(IndicadorMGA(
            codigo="EDU-002",
            nombre="Tasa de cobertura neta en educación básica primaria",
            tipo=TipoIndicadorMGA.RESULTADO,
            sector=SECTOR_EDUCACION,
            formula="(Matrícula oficial primaria edad 6-10 años / Población 6-10 años) * 100",
            unidad_medida=UnidadMedida.PORCENTAJE,
            definicion="Porcentaje de niños de 6 a 10 años matriculados en primaria",
            fuente_informacion=["SIMAT", "DANE"],
            periodicidad="Anual",
            desagregaciones=["Sexo", "Zona", "Etnia"],
            referencias_normativas=["Ley 115/1994"],
            ods_relacionados=[4],
            nivel_aplicacion=["Municipal", "Departamental"]
        ))
        
        self.add_indicador(IndicadorMGA(
            codigo="EDU-003",
            nombre="Tasa de cobertura neta en educación básica secundaria",
            tipo=TipoIndicadorMGA.RESULTADO,
            sector=SECTOR_EDUCACION,
            formula="(Matrícula oficial secundaria edad 11-14 años / Población 11-14 años) * 100",
            unidad_medida=UnidadMedida.PORCENTAJE,
            definicion="Porcentaje de adolescentes de 11 a 14 años matriculados en secundaria",
            fuente_informacion=["SIMAT", "DANE"],
            periodicidad="Anual",
            desagregaciones=["Sexo", "Zona", "Etnia"],
            referencias_normativas=["Ley 115/1994"],
            ods_relacionados=[4],
            nivel_aplicacion=["Municipal", "Departamental"]
        ))
        
        self.add_indicador(IndicadorMGA(
            codigo="EDU-010",
            nombre="Tasa de deserción escolar",
            tipo=TipoIndicadorMGA.RESULTADO,
            sector=SECTOR_EDUCACION,
            formula="(Estudiantes desertores en el año / Total matriculados) * 100",
            unidad_medida=UnidadMedida.PORCENTAJE,
            definicion="Porcentaje de estudiantes que abandonan el sistema educativo durante el año lectivo",
            fuente_informacion=["SIMAT"],
            periodicidad="Anual",
            desagregaciones=["Nivel educativo", "Sexo", "Zona"],
            referencias_normativas=["Ley 715/2001"],
            ods_relacionados=[4],
            nivel_aplicacion=["Municipal", "Departamental"]
        ))
        
        self.add_indicador(IndicadorMGA(
            codigo="EDU-020",
            nombre="Número de sedes educativas construidas o mejoradas",
            tipo=TipoIndicadorMGA.PRODUCTO,
            sector=SECTOR_EDUCACION,
            formula="Número de sedes construidas + Número de sedes mejoradas",
            unidad_medida=UnidadMedida.NUMERO,
            definicion="Cantidad de infraestructura educativa nueva o mejorada",
            fuente_informacion=["Secretaría de Educación", "Registro fotográfico", "Actas de entrega"],
            periodicidad="Anual",
            desagregaciones=["Zona", "Tipo de intervención"],
            referencias_normativas=["Decreto 4791/2008"],
            ods_relacionados=[4],
            nivel_aplicacion=["Municipal"]
        ))
        
        self.add_indicador(IndicadorMGA(
            codigo="EDU-021",
            nombre="Número de aulas escolares dotadas",
            tipo=TipoIndicadorMGA.PRODUCTO,
            sector=SECTOR_EDUCACION,
            formula="Sumatoria de aulas con dotación completa entregada",
            unidad_medida=UnidadMedida.NUMERO,
            definicion="Aulas que reciben dotación de mobiliario y material pedagógico",
            fuente_informacion=["Inventarios", "Actas de entrega"],
            periodicidad="Anual",
            nivel_aplicacion=["Municipal"]
        ))
        
        # ===== SALUD =====
        
        self.add_indicador(IndicadorMGA(
            codigo="SAL-001",
            nombre="Cobertura de vacunación DPT en menores de 1 año",
            tipo=TipoIndicadorMGA.RESULTADO,
            sector=SECTOR_SALUD,
            formula="(Niños < 1 año con esquema DPT completo / Población < 1 año) * 100",
            unidad_medida=UnidadMedida.PORCENTAJE,
            definicion="Porcentaje de menores de un año con esquema completo de vacunación DPT",
            fuente_informacion=["PAI - Programa Ampliado de Inmunizaciones", "SISPRO"],
            periodicidad="Trimestral",
            desagregaciones=["Zona"],
            referencias_normativas=["Resolución 518/2015"],
            ods_relacionados=[3],
            nivel_aplicacion=["Municipal", "Departamental"]
        ))
        
        self.add_indicador(IndicadorMGA(
            codigo="SAL-002",
            nombre="Tasa de mortalidad infantil",
            tipo=TipoIndicadorMGA.RESULTADO,
            sector=SECTOR_SALUD,
            formula="(Defunciones de menores de 1 año / Nacidos vivos) * 1000",
            unidad_medida=UnidadMedida.TASA,
            definicion="Número de muertes de menores de un año por cada 1000 nacidos vivos",
            fuente_informacion=["RUAF", "Certificados de defunción", "DANE"],
            periodicidad="Anual",
            desagregaciones=["Sexo", "Causa de muerte", "Zona"],
            referencias_normativas=["Ley 1438/2011"],
            ods_relacionados=[3],
            nivel_aplicacion=["Municipal", "Departamental"]
        ))
        
        self.add_indicador(IndicadorMGA(
            codigo="SAL-010",
            nombre="Cobertura de afiliación al régimen subsidiado",
            tipo=TipoIndicadorMGA.RESULTADO,
            sector=SECTOR_SALUD,
            formula="(Afiliados régimen subsidiado / Población SISBEN A, B, C) * 100",
            unidad_medida=UnidadMedida.PORCENTAJE,
            definicion="Porcentaje de población vulnerable afiliada al régimen subsidiado",
            fuente_informacion=["BDUA", "SISBEN"],
            periodicidad="Trimestral",
            referencias_normativas=["Ley 1438/2011"],
            ods_relacionados=[3],
            nivel_aplicacion=["Municipal"]
        ))
        
        self.add_indicador(IndicadorMGA(
            codigo="SAL-015",
            nombre="Cobertura de atención prenatal",
            tipo=TipoIndicadorMGA.RESULTADO,
            sector=SECTOR_SALUD,
            formula="(Gestantes con 4+ controles prenatales / Total gestantes) * 100",
            unidad_medida=UnidadMedida.PORCENTAJE,
            definicion="Porcentaje de gestantes con al menos 4 controles prenatales",
            fuente_informacion=["SISPRO", "RIPS"],
            periodicidad="Trimestral",
            referencias_normativas=["Resolución 412/2000"],
            ods_relacionados=[3],
            nivel_aplicacion=["Municipal", "Departamental"]
        ))
        
        self.add_indicador(IndicadorMGA(
            codigo="SAL-020",
            nombre="Número de centros de salud construidos o mejorados",
            tipo=TipoIndicadorMGA.PRODUCTO,
            sector=SECTOR_SALUD,
            formula="Número de centros construidos + Número de centros mejorados",
            unidad_medida=UnidadMedida.NUMERO,
            definicion="Infraestructura de salud del primer nivel nueva o mejorada",
            fuente_informacion=["Secretaría de Salud", "Actas de obra"],
            periodicidad="Anual",
            nivel_aplicacion=["Municipal"]
        ))
        
        self.add_indicador(IndicadorMGA(
            codigo="SAL-021",
            nombre="Número de centros de salud dotados",
            tipo=TipoIndicadorMGA.PRODUCTO,
            sector=SECTOR_SALUD,
            formula="Sumatoria de centros con dotación biomédica entregada",
            unidad_medida=UnidadMedida.NUMERO,
            definicion="Centros de salud que reciben dotación de equipos e insumos",
            fuente_informacion=["Inventarios", "Actas de entrega"],
            periodicidad="Anual",
            nivel_aplicacion=["Municipal"]
        ))
        
        # ===== AGUA POTABLE Y SANEAMIENTO =====
        
        self.add_indicador(IndicadorMGA(
            codigo="APS-001",
            nombre="Cobertura de acueducto",
            tipo=TipoIndicadorMGA.RESULTADO,
            sector=SECTOR_AGUA_POTABLE_SANEAMIENTO,
            formula="(Viviendas con servicio de acueducto / Total viviendas) * 100",
            unidad_medida=UnidadMedida.PORCENTAJE,
            definicion="Porcentaje de viviendas con acceso al servicio de acueducto",
            fuente_informacion=["SUI", "DANE - Censo"],
            periodicidad="Anual",
            desagregaciones=["Zona urbana/rural"],
            referencias_normativas=["Ley 142/1994"],
            ods_relacionados=[6],
            nivel_aplicacion=["Municipal"]
        ))
        
        self.add_indicador(IndicadorMGA(
            codigo="APS-002",
            nombre="Cobertura de alcantarillado",
            tipo=TipoIndicadorMGA.RESULTADO,
            sector=SECTOR_AGUA_POTABLE_SANEAMIENTO,
            formula="(Viviendas con alcantarillado / Total viviendas) * 100",
            unidad_medida=UnidadMedida.PORCENTAJE,
            definicion="Porcentaje de viviendas con acceso al servicio de alcantarillado",
            fuente_informacion=["SUI", "DANE - Censo"],
            periodicidad="Anual",
            desagregaciones=["Zona urbana/rural"],
            referencias_normativas=["Ley 142/1994"],
            ods_relacionados=[6],
            nivel_aplicacion=["Municipal"]
        ))
        
        self.add_indicador(IndicadorMGA(
            codigo="APS-003",
            nombre="Índice de Riesgo de la Calidad del Agua (IRCA)",
            tipo=TipoIndicadorMGA.RESULTADO,
            sector=SECTOR_AGUA_POTABLE_SANEAMIENTO,
            formula="Ponderación de características fisicoquímicas y microbiológicas",
            unidad_medida=UnidadMedida.INDICE,
            definicion="Grado de riesgo de la calidad del agua para consumo humano",
            fuente_informacion=["SIVICAP", "Laboratorios departamentales"],
            periodicidad="Mensual",
            referencias_normativas=["Resolución 2115/2007"],
            ods_relacionados=[6],
            nivel_aplicacion=["Municipal"]
        ))
        
        self.add_indicador(IndicadorMGA(
            codigo="APS-010",
            nombre="Continuidad del servicio de acueducto",
            tipo=TipoIndicadorMGA.RESULTADO,
            sector=SECTOR_AGUA_POTABLE_SANEAMIENTO,
            formula="(Horas de servicio promedio día / 24 horas) * 100",
            unidad_medida=UnidadMedida.PORCENTAJE,
            definicion="Porcentaje de horas al día con servicio de acueducto",
            fuente_informacion=["SUI", "Prestador del servicio"],
            periodicidad="Trimestral",
            referencias_normativas=["Ley 142/1994"],
            ods_relacionados=[6],
            nivel_aplicacion=["Municipal"]
        ))
        
        self.add_indicador(IndicadorMGA(
            codigo="APS-020",
            nombre="Kilómetros de red de acueducto construidos o rehabilitados",
            tipo=TipoIndicadorMGA.PRODUCTO,
            sector=SECTOR_AGUA_POTABLE_SANEAMIENTO,
            formula="Km red nueva + Km red rehabilitada",
            unidad_medida=UnidadMedida.KILOMETROS,
            definicion="Extensión de red de acueducto nueva o rehabilitada",
            fuente_informacion=["Actas de obra", "Planos as-built"],
            periodicidad="Anual",
            nivel_aplicacion=["Municipal"]
        ))
        
        self.add_indicador(IndicadorMGA(
            codigo="APS-021",
            nombre="Kilómetros de red de alcantarillado construidos o rehabilitados",
            tipo=TipoIndicadorMGA.PRODUCTO,
            sector=SECTOR_AGUA_POTABLE_SANEAMIENTO,
            formula="Km red nueva + Km red rehabilitada",
            unidad_medida=UnidadMedida.KILOMETROS,
            definicion="Extensión de red de alcantarillado nueva o rehabilitada",
            fuente_informacion=["Actas de obra", "Planos as-built"],
            periodicidad="Anual",
            nivel_aplicacion=["Municipal"]
        ))
        
        self.add_indicador(IndicadorMGA(
            codigo="APS-030",
            nombre="Número de soluciones individuales de agua y saneamiento implementadas",
            tipo=TipoIndicadorMGA.PRODUCTO,
            sector=SECTOR_AGUA_POTABLE_SANEAMIENTO,
            formula="Suma de pozos, aljibes, filtros, letrinas instalados",
            unidad_medida=UnidadMedida.NUMERO,
            definicion="Soluciones individuales de agua y saneamiento en zonas rurales dispersas",
            fuente_informacion=["Actas de entrega", "Registro fotográfico"],
            periodicidad="Anual",
            nivel_aplicacion=["Municipal"]
        ))
        
        # ===== VIVIENDA =====
        
        self.add_indicador(IndicadorMGA(
            codigo="VIV-001",
            nombre="Déficit cuantitativo de vivienda",
            tipo=TipoIndicadorMGA.RESULTADO,
            sector=SECTOR_VIVIENDA,
            formula="(Hogares - Viviendas adecuadas) / Total hogares * 100",
            unidad_medida=UnidadMedida.PORCENTAJE,
            definicion="Porcentaje de hogares sin vivienda propia o en cohabitación",
            fuente_informacion=["DANE - Censo", "Encuestas de calidad de vida"],
            periodicidad="Anual",
            referencias_normativas=["Ley 1537/2012"],
            ods_relacionados=[11],
            nivel_aplicacion=["Municipal", "Departamental"]
        ))
        
        self.add_indicador(IndicadorMGA(
            codigo="VIV-002",
            nombre="Déficit cualitativo de vivienda",
            tipo=TipoIndicadorMGA.RESULTADO,
            sector=SECTOR_VIVIENDA,
            formula="(Viviendas con carencias / Total viviendas) * 100",
            unidad_medida=UnidadMedida.PORCENTAJE,
            definicion="Porcentaje de viviendas con deficiencias en estructura, espacio o servicios",
            fuente_informacion=["DANE - Censo"],
            periodicidad="Anual",
            referencias_normativas=["Ley 1537/2012"],
            ods_relacionados=[11],
            nivel_aplicacion=["Municipal"]
        ))
        
        self.add_indicador(IndicadorMGA(
            codigo="VIV-010",
            nombre="Número de viviendas de interés social construidas",
            tipo=TipoIndicadorMGA.PRODUCTO,
            sector=SECTOR_VIVIENDA,
            formula="Sumatoria de VIS entregadas",
            unidad_medida=UnidadMedida.NUMERO,
            definicion="Viviendas de interés social nuevas entregadas a beneficiarios",
            fuente_informacion=["Actas de entrega", "Escrituras"],
            periodicidad="Anual",
            nivel_aplicacion=["Municipal"]
        ))
        
        # ===== VÍAS Y TRANSPORTE =====
        
        self.add_indicador(IndicadorMGA(
            codigo="VIA-001",
            nombre="Porcentaje de vías terciarias en buen estado",
            tipo=TipoIndicadorMGA.RESULTADO,
            sector=SECTOR_TRANSPORTE,
            formula="(Km vías en buen estado / Total km red terciaria) * 100",
            unidad_medida=UnidadMedida.PORCENTAJE,
            definicion="Porcentaje de la red vial terciaria en estado bueno",
            fuente_informacion=["INVIAS", "Inventario vial municipal"],
            periodicidad="Anual",
            referencias_normativas=["Ley 1228/2008"],
            ods_relacionados=[9],
            nivel_aplicacion=["Municipal"]
        ))
        
        self.add_indicador(IndicadorMGA(
            codigo="VIA-002",
            nombre="Índice de accesibilidad vial rural",
            tipo=TipoIndicadorMGA.RESULTADO,
            sector=SECTOR_TRANSPORTE,
            formula="(Veredas con acceso vial / Total veredas) * 100",
            unidad_medida=UnidadMedida.PORCENTAJE,
            definicion="Porcentaje de veredas con acceso por vía transitable todo el año",
            fuente_informacion=["Inventario vial municipal"],
            periodicidad="Anual",
            ods_relacionados=[9],
            nivel_aplicacion=["Municipal"]
        ))
        
        self.add_indicador(IndicadorMGA(
            codigo="VIA-010",
            nombre="Kilómetros de vías terciarias construidos o mejorados",
            tipo=TipoIndicadorMGA.PRODUCTO,
            sector=SECTOR_TRANSPORTE,
            formula="Km nuevos + Km mejorados",
            unidad_medida=UnidadMedida.KILOMETROS,
            definicion="Extensión de red vial terciaria nueva o mejorada",
            fuente_informacion=["Actas de obra", "Informes técnicos"],
            periodicidad="Anual",
            nivel_aplicacion=["Municipal"]
        ))
        
        # ===== DESARROLLO AGROPECUARIO =====
        
        self.add_indicador(IndicadorMGA(
            codigo="AGR-001",
            nombre="Productividad agrícola por hectárea",
            tipo=TipoIndicadorMGA.RESULTADO,
            sector=SECTOR_AGRICULTURA,
            formula="Toneladas producidas / Hectáreas cultivadas",
            unidad_medida=UnidadMedida.RAZON,
            definicion="Rendimiento promedio de cultivos principales",
            fuente_informacion=["MADR", "EVA", "UPRA"],
            periodicidad="Anual",
            referencias_normativas=["Ley 101/1993"],
            ods_relacionados=[2],
            nivel_aplicacion=["Municipal"]
        ))
        
        self.add_indicador(IndicadorMGA(
            codigo="AGR-002",
            nombre="Número de productores agropecuarios asistidos técnicamente",
            tipo=TipoIndicadorMGA.PRODUCTO,
            sector=SECTOR_AGRICULTURA,
            formula="Suma de productores con asistencia técnica directa",
            unidad_medida=UnidadMedida.PERSONAS,
            definicion="Productores que reciben asistencia técnica agropecuaria",
            fuente_informacion=["EPSAGRO", "Secretaría de Agricultura"],
            periodicidad="Anual",
            referencias_normativas=["Ley 607/2000"],
            nivel_aplicacion=["Municipal"]
        ))
        
        self.add_indicador(IndicadorMGA(
            codigo="AGR-010",
            nombre="Hectáreas de distritos de riego construidas o rehabilitadas",
            tipo=TipoIndicadorMGA.PRODUCTO,
            sector=SECTOR_AGRICULTURA,
            formula="Ha nuevas + Ha rehabilitadas",
            unidad_medida=UnidadMedida.HECTAREAS,
            definicion="Área beneficiada con sistemas de riego",
            fuente_informacion=["MADR", "Actas de obra"],
            periodicidad="Anual",
            nivel_aplicacion=["Municipal", "Departamental"]
        ))
        
        # ===== MEDIO AMBIENTE =====
        
        self.add_indicador(IndicadorMGA(
            codigo="AMB-001",
            nombre="Porcentaje de áreas protegidas del territorio municipal",
            tipo=TipoIndicadorMGA.RESULTADO,
            sector=SECTOR_AMBIENTE,
            formula="(Hectáreas protegidas / Área total municipal) * 100",
            unidad_medida=UnidadMedida.PORCENTAJE,
            definicion="Porcentaje del territorio bajo alguna figura de protección ambiental",
            fuente_informacion=["RUNAP", "POT", "CAR"],
            periodicidad="Anual",
            referencias_normativas=["Ley 99/1993"],
            ods_relacionados=[15],
            nivel_aplicacion=["Municipal"]
        ))
        
        self.add_indicador(IndicadorMGA(
            codigo="AMB-002",
            nombre="Hectáreas de ecosistemas restaurados",
            tipo=TipoIndicadorMGA.PRODUCTO,
            sector=SECTOR_AMBIENTE,
            formula="Suma de hectáreas con acciones de restauración",
            unidad_medida=UnidadMedida.HECTAREAS,
            definicion="Área con procesos de restauración ecológica implementados",
            fuente_informacion=["CAR", "Informes técnicos"],
            periodicidad="Anual",
            referencias_normativas=["Ley 99/1993"],
            ods_relacionados=[15],
            nivel_aplicacion=["Municipal"]
        ))
        
        self.add_indicador(IndicadorMGA(
            codigo="AMB-010",
            nombre="Toneladas de residuos sólidos aprovechados",
            tipo=TipoIndicadorMGA.PRODUCTO,
            sector=SECTOR_AMBIENTE,
            formula="Suma de toneladas de residuos reciclados o compostados",
            unidad_medida=UnidadMedida.NUMERO,
            definicion="Residuos sólidos que entran a procesos de aprovechamiento",
            fuente_informacion=["PGIRS", "Prestador del servicio"],
            periodicidad="Mensual",
            ods_relacionados=[12],
            nivel_aplicacion=["Municipal"]
        ))
        
        # ===== CULTURA, DEPORTE Y RECREACIÓN =====
        
        self.add_indicador(IndicadorMGA(
            codigo="CUL-001",
            nombre="Número de personas beneficiadas con programas culturales",
            tipo=TipoIndicadorMGA.PRODUCTO,
            sector=SECTOR_CULTURA,
            formula="Suma de participantes en actividades culturales",
            unidad_medida=UnidadMedida.PERSONAS,
            definicion="Personas que participan en programas, talleres o eventos culturales",
            fuente_informacion=["Secretaría de Cultura", "Listados de asistencia"],
            periodicidad="Anual",
            referencias_normativas=["Ley 1185/2008"],
            ods_relacionados=[11],
            nivel_aplicacion=["Municipal"]
        ))
        
        self.add_indicador(IndicadorMGA(
            codigo="CUL-002",
            nombre="Número de bienes de interés cultural protegidos",
            tipo=TipoIndicadorMGA.PRODUCTO,
            sector=SECTOR_CULTURA,
            formula="Suma de BIC con planes de manejo aprobados",
            unidad_medida=UnidadMedida.NUMERO,
            definicion="Bienes de interés cultural con protección efectiva",
            fuente_informacion=["Ministerio de Cultura", "Lista de BIC"],
            periodicidad="Anual",
            nivel_aplicacion=["Municipal"]
        ))
        
        self.add_indicador(IndicadorMGA(
            codigo="DEP-001",
            nombre="Número de personas beneficiadas con programas deportivos",
            tipo=TipoIndicadorMGA.PRODUCTO,
            sector=SECTOR_DEPORTE,
            formula="Suma de participantes en escuelas deportivas y eventos",
            unidad_medida=UnidadMedida.PERSONAS,
            definicion="Personas que participan en programas deportivos municipales",
            fuente_informacion=["Instituto de deportes", "Listados"],
            periodicidad="Anual",
            referencias_normativas=["Ley 181/1995"],
            ods_relacionados=[3],
            nivel_aplicacion=["Municipal"]
        ))
        
        self.add_indicador(IndicadorMGA(
            codigo="DEP-002",
            nombre="Número de escenarios deportivos construidos o mejorados",
            tipo=TipoIndicadorMGA.PRODUCTO,
            sector=SECTOR_DEPORTE,
            formula="Escenarios nuevos + Escenarios mejorados",
            unidad_medida=UnidadMedida.NUMERO,
            definicion="Infraestructura deportiva nueva o mejorada",
            fuente_informacion=["Actas de obra"],
            periodicidad="Anual",
            nivel_aplicacion=["Municipal"]
        ))
        
        # ===== DESARROLLO ECONÓMICO =====
        
        self.add_indicador(IndicadorMGA(
            codigo="ECO-001",
            nombre="Número de empresas nuevas creadas",
            tipo=TipoIndicadorMGA.RESULTADO,
            sector=SECTOR_DESARROLLO_ECONOMICO,
            formula="Suma de empresas registradas en Cámara de Comercio",
            unidad_medida=UnidadMedida.NUMERO,
            definicion="Nuevas empresas formalizadas en el municipio",
            fuente_informacion=["Cámara de Comercio", "RUES"],
            periodicidad="Anual",
            referencias_normativas=["Ley 590/2000"],
            ods_relacionados=[8],
            nivel_aplicacion=["Municipal"]
        ))
        
        self.add_indicador(IndicadorMGA(
            codigo="ECO-002",
            nombre="Número de emprendedores capacitados",
            tipo=TipoIndicadorMGA.PRODUCTO,
            sector=SECTOR_DESARROLLO_ECONOMICO,
            formula="Suma de personas en formación empresarial",
            unidad_medida=UnidadMedida.PERSONAS,
            definicion="Personas que completan programas de formación en emprendimiento",
            fuente_informacion=["Certificados de asistencia", "Listados"],
            periodicidad="Anual",
            nivel_aplicacion=["Municipal"]
        ))
        
        self.add_indicador(IndicadorMGA(
            codigo="ECO-010",
            nombre="Tasa de desempleo municipal",
            tipo=TipoIndicadorMGA.RESULTADO,
            sector=SECTOR_DESARROLLO_ECONOMICO,
            formula="(Personas desempleadas / Población económicamente activa) * 100",
            unidad_medida=UnidadMedida.PORCENTAJE,
            definicion="Porcentaje de desempleo en el municipio",
            fuente_informacion=["DANE - GEIH"],
            periodicidad="Trimestral",
            ods_relacionados=[8],
            nivel_aplicacion=["Municipal"]
        ))
        
        # ===== ATENCIÓN A GRUPOS VULNERABLES =====
        
        self.add_indicador(IndicadorMGA(
            codigo="SOC-001",
            nombre="Número de niños, niñas y adolescentes atendidos integralmente",
            tipo=TipoIndicadorMGA.PRODUCTO,
            sector=SECTOR_INCLUSION_SOCIAL,
            formula="Suma de NNA en programas de atención integral",
            unidad_medida=UnidadMedida.PERSONAS,
            definicion="Niños, niñas y adolescentes en programas de protección y desarrollo",
            fuente_informacion=["ICBF", "Comisaría de Familia", "SIM"],
            periodicidad="Trimestral",
            referencias_normativas=["Ley 1098/2006"],
            ods_relacionados=[1, 2, 3],
            nivel_aplicacion=["Municipal"]
        ))
        
        self.add_indicador(IndicadorMGA(
            codigo="SOC-002",
            nombre="Número de adultos mayores atendidos",
            tipo=TipoIndicadorMGA.PRODUCTO,
            sector=SECTOR_INCLUSION_SOCIAL,
            formula="Suma de adultos mayores en programas municipales",
            unidad_medida=UnidadMedida.PERSONAS,
            definicion="Adultos mayores que reciben atención en centros vida o subsidios",
            fuente_informacion=["Secretaría Social", "Listados"],
            periodicidad="Trimestral",
            referencias_normativas=["Ley 1251/2008"],
            ods_relacionados=[1, 3],
            nivel_aplicacion=["Municipal"]
        ))
        
        self.add_indicador(IndicadorMGA(
            codigo="SOC-010",
            nombre="Número de familias en pobreza extrema con acompañamiento",
            tipo=TipoIndicadorMGA.PRODUCTO,
            sector=SECTOR_INCLUSION_SOCIAL,
            formula="Suma de familias en programas de superación de pobreza",
            unidad_medida=UnidadMedida.FAMILIAS,
            definicion="Familias en pobreza extrema con acompañamiento psicosocial",
            fuente_informacion=["DNP", "UNIDOS", "SISBEN"],
            periodicidad="Semestral",
            ods_relacionados=[1],
            nivel_aplicacion=["Municipal"]
        ))
        
        # ===== JUSTICIA Y SEGURIDAD =====
        
        self.add_indicador(IndicadorMGA(
            codigo="SEG-001",
            nombre="Tasa de homicidios por cada 100.000 habitantes",
            tipo=TipoIndicadorMGA.RESULTADO,
            sector=SECTOR_SEGURIDAD,
            formula="(Número de homicidios / Población total) * 100000",
            unidad_medida=UnidadMedida.TASA,
            definicion="Tasa de homicidios en el municipio",
            fuente_informacion=["Policía Nacional", "Medicina Legal"],
            periodicidad="Mensual",
            referencias_normativas=["Ley 62/1993"],
            ods_relacionados=[16],
            nivel_aplicacion=["Municipal"]
        ))
        
        self.add_indicador(IndicadorMGA(
            codigo="SEG-002",
            nombre="Número de cámaras de seguridad instaladas",
            tipo=TipoIndicadorMGA.PRODUCTO,
            sector=SECTOR_SEGURIDAD,
            formula="Suma de cámaras en operación",
            unidad_medida=UnidadMedida.NUMERO,
            definicion="Cámaras de videovigilancia instaladas y operativas",
            fuente_informacion=["Secretaría de Gobierno", "Inventario"],
            periodicidad="Anual",
            nivel_aplicacion=["Municipal"]
        ))
        
        self.add_indicador(IndicadorMGA(
            codigo="SEG-010",
            nombre="Tasa de hurtos por cada 100.000 habitantes",
            tipo=TipoIndicadorMGA.RESULTADO,
            sector=SECTOR_SEGURIDAD,
            formula="(Número de hurtos / Población total) * 100000",
            unidad_medida=UnidadMedida.TASA,
            definicion="Tasa de hurtos en el municipio",
            fuente_informacion=["Policía Nacional"],
            periodicidad="Mensual",
            ods_relacionados=[16],
            nivel_aplicacion=["Municipal"]
        ))
        
        # ===== ORDENAMIENTO TERRITORIAL =====
        
        self.add_indicador(IndicadorMGA(
            codigo="ORD-001",
            nombre="Porcentaje de avance en actualización del POT",
            tipo=TipoIndicadorMGA.PRODUCTO,
            sector=SECTOR_ORDENAMIENTO_TERRITORIAL,
            formula="(Fases completadas / Total fases) * 100",
            unidad_medida=UnidadMedida.PORCENTAJE,
            definicion="Porcentaje de avance en la actualización del Plan de Ordenamiento",
            fuente_informacion=["Secretaría de Planeación"],
            periodicidad="Trimestral",
            referencias_normativas=["Ley 388/1997"],
            ods_relacionados=[11],
            nivel_aplicacion=["Municipal"]
        ))
        
        self.add_indicador(IndicadorMGA(
            codigo="ORD-002",
            nombre="Número de instrumentos de planificación territorial adoptados",
            tipo=TipoIndicadorMGA.PRODUCTO,
            sector=SECTOR_ORDENAMIENTO_TERRITORIAL,
            formula="Suma de planes parciales, UPR, POMCA adoptados",
            unidad_medida=UnidadMedida.NUMERO,
            definicion="Instrumentos complementarios de ordenamiento adoptados",
            fuente_informacion=["Acuerdos municipales"],
            periodicidad="Anual",
            nivel_aplicacion=["Municipal"]
        ))
        
        # ===== PREVENCIÓN Y ATENCIÓN DE DESASTRES =====
        
        self.add_indicador(IndicadorMGA(
            codigo="DES-001",
            nombre="Porcentaje de avance en el Plan Municipal de Gestión del Riesgo",
            tipo=TipoIndicadorMGA.PRODUCTO,
            sector=SECTOR_GESTION_RIESGO,
            formula="(Componentes implementados / Total componentes) * 100",
            unidad_medida=UnidadMedida.PORCENTAJE,
            definicion="Avance en la implementación del PMGRD",
            fuente_informacion=["CMGRD", "Informes de gestión"],
            periodicidad="Semestral",
            referencias_normativas=["Ley 1523/2012"],
            ods_relacionados=[11, 13],
            nivel_aplicacion=["Municipal"]
        ))
        
        self.add_indicador(IndicadorMGA(
            codigo="DES-002",
            nombre="Número de familias con reducción de riesgo",
            tipo=TipoIndicadorMGA.RESULTADO,
            sector=SECTOR_GESTION_RIESGO,
            formula="Suma de familias beneficiadas con obras de mitigación",
            unidad_medida=UnidadMedida.FAMILIAS,
            definicion="Familias con riesgo reducido por obras de mitigación",
            fuente_informacion=["CMGRD", "Actas de obra"],
            periodicidad="Anual",
            ods_relacionados=[11],
            nivel_aplicacion=["Municipal"]
        ))
        
        self.add_indicador(IndicadorMGA(
            codigo="DES-010",
            nombre="Número de personas capacitadas en gestión del riesgo",
            tipo=TipoIndicadorMGA.PRODUCTO,
            sector=SECTOR_GESTION_RIESGO,
            formula="Suma de personas en capacitaciones sobre gestión del riesgo",
            unidad_medida=UnidadMedida.PERSONAS,
            definicion="Personas capacitadas en prevención y atención de emergencias",
            fuente_informacion=["CMGRD", "Listados de asistencia"],
            periodicidad="Anual",
            nivel_aplicacion=["Municipal"]
        ))
        
        # ===== EQUIPAMIENTO MUNICIPAL =====
        
        self.add_indicador(IndicadorMGA(
            codigo="EQU-001",
            nombre="Número de edificaciones institucionales construidas o mejoradas",
            tipo=TipoIndicadorMGA.PRODUCTO,
            sector=SECTOR_EQUIPAMIENTO,
            formula="Edificios nuevos + Edificios mejorados",
            unidad_medida=UnidadMedida.NUMERO,
            definicion="Infraestructura institucional municipal nueva o mejorada",
            fuente_informacion=["Secretaría de Obras", "Actas"],
            periodicidad="Anual",
            nivel_aplicacion=["Municipal"]
        ))
        
        self.add_indicador(IndicadorMGA(
            codigo="EQU-002",
            nombre="Número de parques y espacios públicos recuperados",
            tipo=TipoIndicadorMGA.PRODUCTO,
            sector=SECTOR_EQUIPAMIENTO,
            formula="Suma de parques intervenidos",
            unidad_medida=UnidadMedida.NUMERO,
            definicion="Parques y espacios públicos recuperados o mejorados",
            fuente_informacion=["Secretaría de Obras"],
            periodicidad="Anual",
            ods_relacionados=[11],
            nivel_aplicacion=["Municipal"]
        ))
    
    def add_indicador(self, indicador: IndicadorMGA):
        """Add indicator to catalog"""
        self.indicadores[indicador.codigo] = indicador
        logger.debug(f"Indicador agregado: {indicador.codigo}")
    
    def get_indicador(self, codigo: str) -> Optional[IndicadorMGA]:
        """Get indicator by code"""
        return self.indicadores.get(codigo)
    
    def get_by_sector(self, sector: str) -> List[IndicadorMGA]:
        """Get all indicators for a sector"""
        return [ind for ind in self.indicadores.values() 
                if sector.lower() in ind.sector.lower()]
    
    def get_by_tipo(self, tipo: TipoIndicadorMGA) -> List[IndicadorMGA]:
        """Get all indicators by type"""
        return [ind for ind in self.indicadores.values() if ind.tipo == tipo]
    
    def get_indicadores_producto(self) -> List[IndicadorMGA]:
        """Get all product indicators"""
        return self.get_by_tipo(TipoIndicadorMGA.PRODUCTO)
    
    def get_indicadores_resultado(self) -> List[IndicadorMGA]:
        """Get all result indicators"""
        return self.get_by_tipo(TipoIndicadorMGA.RESULTADO)
    
    def buscar_indicador_por_descripcion(self, terminos: str) -> List[IndicadorMGA]:
        """Search indicators by description keywords"""
        terminos_lower = terminos.lower()
        resultados = []
        for ind in self.indicadores.values():
            if (terminos_lower in ind.nombre.lower() or 
                terminos_lower in ind.definicion.lower()):
                resultados.append(ind)
        return resultados
    
    def generar_reporte_alineacion(self, codigos_usados: List[str]) -> Dict[str, Any]:
        """Generate alignment report for indicators used in a project"""
        indicadores_usados = [self.get_indicador(cod) for cod in codigos_usados 
                             if self.get_indicador(cod)]
        
        return {
            "total_indicadores_usados": len(indicadores_usados),
            "productos": len([i for i in indicadores_usados if i.tipo == TipoIndicadorMGA.PRODUCTO]),
            "resultados": len([i for i in indicadores_usados if i.tipo == TipoIndicadorMGA.RESULTADO]),
            "sectores_cubiertos": list({i.sector for i in indicadores_usados}),
            "ods_relacionados": list({ods for i in indicadores_usados for ods in i.ods_relacionados}),
            "cumple_mga": len(indicadores_usados) > 0
        }


# Singleton instance
CATALOGO_MGA = CatalogoIndicadoresMGA()


if __name__ == "__main__":
    # Demo usage
    catalogo = CatalogoIndicadoresMGA()
    
    print("=== Catálogo MGA Indicadores - Demo ===\n")
    
    print(f"Total indicadores: {len(catalogo.indicadores)}")
    print(f"Indicadores de producto: {len(catalogo.get_indicadores_producto())}")
    print(f"Indicadores de resultado: {len(catalogo.get_indicadores_resultado())}\n")
    
    # Search example
    resultados = catalogo.buscar_indicador_por_descripcion("educación")
    print(f"Indicadores relacionados con 'educación': {len(resultados)}")
    for ind in resultados[:3]:
        print(f"  - {ind.codigo}: {ind.nombre}")
