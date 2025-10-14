#!/usr/bin/env python3
"""
Ejemplo completo de validación DNP
Demuestra el uso de todos los módulos de estándares DNP
"""

from competencias_municipales import CATALOGO_COMPETENCIAS, SectorCompetencia
from mga_indicadores import CATALOGO_MGA, TipoIndicadorMGA
from pdet_lineamientos import LINEAMIENTOS_PDET, PilarPDET
from dnp_integration import ValidadorDNP, validar_plan_desarrollo_completo

def ejemplo_1_competencias():
    """Ejemplo 1: Validación de Competencias Municipales"""
    print("=" * 80)
    print("EJEMPLO 1: VALIDACIÓN DE COMPETENCIAS MUNICIPALES")
    print("=" * 80)
    print()
    
    # Validar si un proyecto está dentro de competencias
    validacion = CATALOGO_COMPETENCIAS.validar_competencia_municipal(
        sector="educacion",
        descripcion="Construcción y mejoramiento de infraestructura educativa"
    )
    
    print("Proyecto: Construcción de sedes educativas")
    print(f"¿Es competencia municipal válida? {validacion['valido']}")
    print(f"Sector: {validacion['sector']}")
    print(f"Competencias aplicables: {', '.join(validacion['competencias_aplicables'])}")
    print(f"Base legal: {validacion['base_legal'][0]}")
    print()
    
    # Mostrar competencias PDET prioritarias
    competencias_pdet = CATALOGO_COMPETENCIAS.get_pdet_prioritarias()
    print(f"Total competencias prioritarias PDET: {len(competencias_pdet)}")
    print("Top 5 competencias PDET:")
    for comp in competencias_pdet[:5]:
        print(f"  • {comp.codigo}: {comp.descripcion[:60]}...")
    print()


def ejemplo_2_indicadores_mga():
    """Ejemplo 2: Uso de Indicadores MGA"""
    print("=" * 80)
    print("EJEMPLO 2: CATÁLOGO DE INDICADORES MGA")
    print("=" * 80)
    print()
    
    # Obtener indicadores de educación
    indicadores_edu = CATALOGO_MGA.get_by_sector("educacion")
    print(f"Indicadores de educación: {len(indicadores_edu)}")
    
    # Mostrar un indicador de producto
    ind_producto = CATALOGO_MGA.get_indicador("EDU-020")
    if ind_producto:
        print(f"\nIndicador de Producto: {ind_producto.codigo}")
        print(f"Nombre: {ind_producto.nombre}")
        print(f"Fórmula: {ind_producto.formula}")
        print(f"Unidad: {ind_producto.unidad_medida.value}")
        print(f"Fuente: {', '.join(ind_producto.fuente_informacion)}")
    
    # Mostrar un indicador de resultado
    ind_resultado = CATALOGO_MGA.get_indicador("EDU-001")
    if ind_resultado:
        print(f"\nIndicador de Resultado: {ind_resultado.codigo}")
        print(f"Nombre: {ind_resultado.nombre}")
        print(f"Fórmula: {ind_resultado.formula}")
        print(f"ODS relacionados: {ind_resultado.ods_relacionados}")
    
    # Generar reporte de alineación
    print("\n" + "-" * 80)
    reporte = CATALOGO_MGA.generar_reporte_alineacion(["EDU-020", "EDU-021", "EDU-001"])
    print(f"Reporte de alineación MGA:")
    print(f"  Total indicadores: {reporte['total_indicadores_usados']}")
    print(f"  Productos: {reporte['productos']}")
    print(f"  Resultados: {reporte['resultados']}")
    print(f"  Cumple MGA: {reporte['cumple_mga']}")
    print()


def ejemplo_3_lineamientos_pdet():
    """Ejemplo 3: Lineamientos PDET"""
    print("=" * 80)
    print("EJEMPLO 3: LINEAMIENTOS PDET")
    print("=" * 80)
    print()
    
    # Obtener lineamientos por pilar
    lineamientos_educacion = LINEAMIENTOS_PDET.get_by_pilar(PilarPDET.EDUCACION_RURAL)
    print(f"Lineamientos para Educación Rural: {len(lineamientos_educacion)}")
    for lin in lineamientos_educacion:
        print(f"  • {lin.codigo}: {lin.titulo}")
        print(f"    Enfoques: {', '.join([e.value for e in lin.enfoque_requerido])}")
    
    # Validar cumplimiento PDET
    print("\n" + "-" * 80)
    print("Validación de cumplimiento PDET:")
    validacion = LINEAMIENTOS_PDET.validar_cumplimiento_pdet(
        participacion={
            "comunitaria": 75.0,
            "victimas": 45.0,
            "mujeres": 35.0,
            "jovenes": 25.0
        },
        presupuesto_rural=6_000_000_000,
        presupuesto_total=10_000_000_000,
        alineacion_patr=85.0
    )
    
    print(f"  ¿Cumple requisitos PDET? {validacion['cumple_requisitos']}")
    print(f"  Validaciones exitosas: {len(validacion['validaciones'])}")
    for val in validacion['validaciones']:
        print(f"    {val}")
    
    if validacion['alertas']:
        print(f"  Alertas: {len(validacion['alertas'])}")
        for alerta in validacion['alertas']:
            print(f"    ⚠ {alerta}")
    
    print()


def ejemplo_4_validacion_integral():
    """Ejemplo 4: Validación Integral de Proyecto"""
    print("=" * 80)
    print("EJEMPLO 4: VALIDACIÓN INTEGRAL DNP")
    print("=" * 80)
    print()
    
    # Crear validador
    validador = ValidadorDNP(es_municipio_pdet=True)
    
    # Validar proyecto
    resultado = validador.validar_proyecto_integral(
        sector="educacion",
        descripcion="Construcción de 5 sedes educativas rurales con dotación completa",
        indicadores_propuestos=["EDU-020", "EDU-021", "EDU-002"],
        presupuesto=2_000_000_000,
        es_rural=True,
        poblacion_victimas=True
    )
    
    # Generar y mostrar reporte
    reporte = validador.generar_reporte_cumplimiento(resultado)
    print(reporte)
    print()


def ejemplo_5_plan_completo():
    """Ejemplo 5: Validación de Plan de Desarrollo Completo"""
    print("=" * 80)
    print("EJEMPLO 5: VALIDACIÓN DE PLAN DE DESARROLLO COMPLETO")
    print("=" * 80)
    print()
    
    # Definir proyectos del plan
    proyectos = [
        {
            "nombre": "Construcción escuelas rurales",
            "sector": "educacion",
            "descripcion": "Construcción de 5 sedes educativas en zona rural",
            "indicadores": ["EDU-020", "EDU-021", "EDU-002"],
            "presupuesto": 2_000_000_000,
            "es_rural": True,
            "poblacion_victimas": True
        },
        {
            "nombre": "Mejoramiento centros de salud",
            "sector": "salud",
            "descripcion": "Mejoramiento y dotación de 3 centros de salud rurales",
            "indicadores": ["SAL-020", "SAL-021", "SAL-001"],
            "presupuesto": 1_500_000_000,
            "es_rural": True,
            "poblacion_victimas": False
        },
        {
            "nombre": "Acueductos veredales",
            "sector": "agua_potable_saneamiento",
            "descripcion": "Construcción de 10 acueductos veredales",
            "indicadores": ["APS-020", "APS-001", "APS-003"],
            "presupuesto": 3_000_000_000,
            "es_rural": True,
            "poblacion_victimas": True
        },
        {
            "nombre": "Vías terciarias",
            "sector": "vias_transporte",
            "descripcion": "Mejoramiento de 50 km de vías terciarias",
            "indicadores": ["VIA-010", "VIA-001"],
            "presupuesto": 4_000_000_000,
            "es_rural": True,
            "poblacion_victimas": False
        },
        {
            "nombre": "Asistencia técnica agropecuaria",
            "sector": "desarrollo_agropecuario",
            "descripcion": "Programa de asistencia técnica a 500 productores",
            "indicadores": ["AGR-002", "AGR-001"],
            "presupuesto": 800_000_000,
            "es_rural": True,
            "poblacion_victimas": True
        }
    ]
    
    # Validar plan completo
    resultado = validar_plan_desarrollo_completo(
        proyectos=proyectos,
        es_municipio_pdet=True,
        presupuesto_total=20_000_000_000,
        presupuesto_rural=15_000_000_000,
        participacion={
            "comunitaria": 75.0,
            "jac": 65.0,
            "victimas": 45.0,
            "mujeres": 35.0,
            "jovenes": 25.0
        }
    )
    
    # Mostrar resultados agregados
    print(f"Total proyectos analizados: {resultado['total_proyectos']}")
    print(f"Proyectos que cumplen estándares: {resultado['proyectos_cumplen']}")
    print(f"Tasa de cumplimiento: {resultado['tasa_cumplimiento']:.1f}%")
    print(f"Score promedio: {resultado['score_promedio']:.1f}/100")
    
    # Mostrar cumplimiento PDET
    if resultado['cumplimiento_pdet']:
        print("\nCumplimiento PDET:")
        pdet = resultado['cumplimiento_pdet']
        print(f"  Cumple requisitos: {pdet['cumple_requisitos']}")
        if pdet['alertas']:
            print(f"  Alertas: {len(pdet['alertas'])}")
    
    # Detallar proyectos
    print("\nDetalle por proyecto:")
    for r in resultado['resultados_proyectos']:
        proyecto = r['proyecto']
        res = r['resultado']
        print(f"\n  {proyecto}")
        print(f"    Score: {res.score_total:.1f}/100")
        print(f"    Nivel: {res.nivel_cumplimiento.value.upper()}")
        print(f"    Competencias: {'✓' if res.cumple_competencias else '✗'}")
        print(f"    MGA: {'✓' if res.cumple_mga else '✗'}")
        if res.es_municipio_pdet:
            print(f"    PDET: {'✓' if res.cumple_pdet else '✗'}")
    
    print()


def main():
    """Ejecutar todos los ejemplos"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "EJEMPLOS DE VALIDACIÓN DNP - FARFAN 2.0" + " " * 24 + "║")
    print("╚" + "═" * 78 + "╝")
    print("\n")
    
    # Ejecutar ejemplos
    ejemplo_1_competencias()
    input("Presione Enter para continuar...")
    
    ejemplo_2_indicadores_mga()
    input("Presione Enter para continuar...")
    
    ejemplo_3_lineamientos_pdet()
    input("Presione Enter para continuar...")
    
    ejemplo_4_validacion_integral()
    input("Presione Enter para continuar...")
    
    ejemplo_5_plan_completo()
    
    print("=" * 80)
    print("EJEMPLOS COMPLETADOS")
    print("=" * 80)
    print("\nPara más información, consulte DNP_INTEGRATION_DOCS.md")
    print()


if __name__ == "__main__":
    main()
