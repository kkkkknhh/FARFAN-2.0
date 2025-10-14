#!/usr/bin/env python3
"""
Demo Script para el Sistema de Orquestación FARFAN 2.0

Este script demuestra cómo usar el orquestador para:
1. Procesar un plan de desarrollo (simulado)
2. Generar las 300 respuestas
3. Crear reportes a 3 niveles (micro, meso, macro)

Nota: Requiere que todos los módulos estén instalados correctamente.
"""

import sys
import logging
from pathlib import Path
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger("demo_orchestrator")


def create_sample_pdf():
    """
    Crea un PDF de muestra para demostración
    
    En un entorno real, usaría un PDF real de un Plan de Desarrollo.
    Para esta demo, creamos un PDF simple con texto de ejemplo.
    """
    try:
        import fitz  # PyMuPDF
        
        # Create a simple PDF with sample text
        doc = fitz.open()
        page = doc.new_page()
        
        sample_text = """
PLAN DE DESARROLLO MUNICIPAL 2024-2027
MUNICIPIO DE EJEMPLO, COLOMBIA

1. DIAGNÓSTICO TERRITORIAL
El municipio cuenta con 45,000 habitantes, de los cuales 60% reside en zona rural.
La tasa de pobreza multidimensional es 42.3%, superior al promedio departamental.

Derechos de las mujeres e igualdad de género:
- Tasa de violencia de género: 28 por 100,000 habitantes (Fuente: Policía Nacional 2023)
- Participación de mujeres en concejos municipales: 45%
- Cobertura de programas de empoderamiento: 30% de las mujeres

2. VISIÓN ESTRATÉGICA
Para 2027, el municipio será reconocido por su desarrollo sostenible e inclusivo.

3. PLAN PLURIANUAL DE INVERSIONES
Se destinarán $12,500 millones al sector educación, con meta de construir 
3 instituciones educativas y capacitar 250 docentes en pedagogías innovadoras.

Programa de Seguridad:
- Responsable: Secretaría de Gobierno
- Recursos: $8,000 millones
- Meta: Reducir tasa de homicidios a 20 por 100,000
- Actividades: 
  * Implementación de sistema de videovigilancia
  * Fortalecimiento de la Policía Comunitaria
  * Programa de prevención con jóvenes

4. SEGUIMIENTO Y EVALUACIÓN
Se implementará el sistema de seguimiento al Plan mediante indicadores MGA.
"""
        
        page.insert_text((50, 50), sample_text, fontsize=11)
        
        # Save to temporary file
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        doc.save(temp_pdf.name)
        doc.close()
        
        logger.info(f"PDF de muestra creado: {temp_pdf.name}")
        return Path(temp_pdf.name)
        
    except ImportError:
        logger.error("PyMuPDF no está instalado. Instale con: pip install pymupdf")
        return None
    except Exception as e:
        logger.error(f"Error creando PDF de muestra: {e}")
        return None


def demo_orchestrator():
    """Demuestra el uso del orquestador"""
    
    print("="*80)
    print("FARFAN 2.0 - Demostración del Sistema de Orquestación")
    print("="*80)
    print()
    
    # Check if orchestrator can be imported
    try:
        from orchestrator import FARFANOrchestrator
        logger.info("✓ Orquestador importado correctamente")
    except ImportError as e:
        logger.error(f"✗ No se pudo importar el orquestador: {e}")
        logger.error("Asegúrese de que todos los módulos estén en el PYTHONPATH")
        return 1
    
    # Create sample PDF
    pdf_path = create_sample_pdf()
    if pdf_path is None:
        logger.error("No se pudo crear PDF de muestra")
        return 1
    
    # Create output directory
    output_dir = Path("./demo_resultados")
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Directorio de salida: {output_dir}")
    
    print()
    print("Inicializando orquestador...")
    print("-"*80)
    
    try:
        # Create orchestrator
        orchestrator = FARFANOrchestrator(
            output_dir=output_dir,
            log_level="INFO"
        )
        
        print()
        print("Procesando plan de desarrollo de muestra...")
        print("-"*80)
        
        # Process plan
        context = orchestrator.process_plan(
            pdf_path=pdf_path,
            policy_code="DEMO-2024",
            es_municipio_pdet=True
        )
        
        print()
        print("="*80)
        print("RESULTADOS DEL PROCESAMIENTO")
        print("="*80)
        
        # Display results
        print(f"\n📊 Estadísticas Generales:")
        print(f"  - Código de Política: {context.policy_code}")
        print(f"  - Texto extraído: {len(context.raw_text)} caracteres")
        print(f"  - Tablas identificadas: {len(context.tables)}")
        print(f"  - Secciones extraídas: {len(context.sections)}")
        print(f"  - Nodos causales: {len(context.nodes)}")
        print(f"  - Cadenas causales: {len(context.causal_chains)}")
        
        if context.dnp_validation_results:
            print(f"\n✅ Validación DNP:")
            print(f"  - Validaciones realizadas: {len(context.dnp_validation_results)}")
            print(f"  - Score de cumplimiento: {context.compliance_score:.1f}/100")
        
        if context.question_responses:
            print(f"\n❓ Sistema de Preguntas:")
            print(f"  - Preguntas respondidas: {len(context.question_responses)}")
            
            # Calculate average score
            if context.question_responses:
                notas = [r.nota_cuantitativa for r in context.question_responses.values()]
                promedio = sum(notas) / len(notas)
                print(f"  - Promedio de notas: {promedio:.3f}")
                print(f"  - Preguntas excelentes (≥0.85): {sum(1 for n in notas if n >= 0.85)}")
                print(f"  - Preguntas insuficientes (<0.55): {sum(1 for n in notas if n < 0.55)}")
        
        print(f"\n📝 Reportes Generados:")
        print(f"  - Micro: {len(context.micro_report)} elementos")
        print(f"  - Meso: {len(context.meso_report)} clústeres")
        print(f"  - Macro: Evaluación global completa")
        
        print(f"\n📁 Archivos de Salida:")
        for file in output_dir.glob("*DEMO-2024*"):
            print(f"  - {file.name} ({file.stat().st_size / 1024:.1f} KB)")
        
        print()
        print("="*80)
        print("✅ DEMOSTRACIÓN COMPLETADA EXITOSAMENTE")
        print("="*80)
        print()
        print(f"Revise los archivos generados en: {output_dir.absolute()}")
        print()
        
        # Cleanup temporary PDF
        pdf_path.unlink()
        
        return 0
        
    except Exception as e:
        logger.error(f"Error durante la demostración: {e}", exc_info=True)
        print()
        print("="*80)
        print("❌ LA DEMOSTRACIÓN ENCONTRÓ ERRORES")
        print("="*80)
        print()
        print("Detalles del error:")
        print(f"  {type(e).__name__}: {e}")
        print()
        print("Posibles soluciones:")
        print("  1. Verifique que todos los módulos estén instalados")
        print("  2. Asegúrese de tener el modelo spaCy: python -m spacy download es_core_news_lg")
        print("  3. Revise los logs arriba para más detalles")
        print()
        
        return 1


def demo_simple():
    """
    Demostración simple sin procesamiento completo
    Muestra la estructura del sistema sin requerir todos los módulos
    """
    print("="*80)
    print("FARFAN 2.0 - Demostración Simple (Sin Procesamiento Completo)")
    print("="*80)
    print()
    
    print("📋 Sistema de 300 Preguntas:")
    print()
    
    # Show question structure
    dimensiones = [
        ("D1", "Insumos (Diagnóstico y Líneas Base)", 5),
        ("D2", "Actividades (Formalizadas)", 5),
        ("D3", "Productos (Verificables)", 5),
        ("D4", "Resultados (Medibles)", 5),
        ("D5", "Impactos (Largo Plazo)", 5),
        ("D6", "Causalidad (Teoría de Cambio)", 5)
    ]
    
    puntos = [
        ("P1", "Derechos de las mujeres e igualdad de género"),
        ("P2", "Prevención de la violencia y protección frente al conflicto"),
        ("P3", "Ambiente sano, cambio climático, prevención y atención a desastres"),
        ("P4", "Derechos económicos, sociales y culturales"),
        ("P5", "Derechos de las víctimas y construcción de paz"),
        ("P6", "Derecho al buen futuro de la niñez, adolescencia, juventud"),
        ("P7", "Tierras y territorios"),
        ("P8", "Líderes y defensores de derechos humanos"),
        ("P9", "Crisis de derechos de personas privadas de la libertad"),
        ("P10", "Migración transfronteriza")
    ]
    
    print("Dimensiones Analíticas:")
    total_preguntas_base = 0
    for dim_id, dim_nombre, num_preguntas in dimensiones:
        print(f"  {dim_id}: {dim_nombre} - {num_preguntas} preguntas")
        total_preguntas_base += num_preguntas
    
    print(f"\n  Total preguntas base: {total_preguntas_base}")
    
    print("\nÁreas de Política (Decálogo):")
    for punto_id, punto_nombre in puntos:
        print(f"  {punto_id}: {punto_nombre}")
    
    print(f"\n  Total áreas: {len(puntos)}")
    
    total_preguntas = total_preguntas_base * len(puntos)
    print(f"\n✨ Total de preguntas: {total_preguntas_base} × {len(puntos)} = {total_preguntas}")
    
    print("\n" + "="*80)
    print("📊 Estructura de Reportes:")
    print("="*80)
    
    print("\n1. NIVEL MICRO:")
    print("   - 300 respuestas individuales")
    print("   - Cada respuesta incluye:")
    print("     * Texto de respuesta")
    print("     * Argumento doctoral (2+ párrafos)")
    print("     * Nota cuantitativa (0.0-1.0)")
    print("     * Evidencia del documento")
    print("     * Módulos utilizados")
    
    print("\n2. NIVEL MESO:")
    print("   - 4 clústeres temáticos:")
    print("     * C1: Seguridad, Paz y Protección (P1, P2, P8)")
    print("     * C2: Derechos Sociales (P4, P5, P6)")
    print("     * C3: Territorio y Ambiente (P3, P7)")
    print("     * C4: Poblaciones Especiales (P9, P10)")
    print("   - Análisis por 6 dimensiones en cada clúster")
    print("   - Evaluación cualitativa integral")
    
    print("\n3. NIVEL MACRO:")
    print("   - Score global de alineación")
    print("   - Análisis retrospectivo (¿qué tan lejos está?)")
    print("   - Análisis prospectivo (¿qué debe mejorar?)")
    print("   - Recomendaciones prioritarias")
    print("   - Fortalezas y debilidades críticas")
    
    print("\n" + "="*80)
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Demo del Sistema de Orquestación FARFAN 2.0")
    parser.add_argument("--simple", action="store_true", 
                       help="Ejecutar demo simple sin procesamiento completo")
    
    args = parser.parse_args()
    
    if args.simple:
        demo_simple()
        sys.exit(0)
    else:
        sys.exit(demo_orchestrator())
