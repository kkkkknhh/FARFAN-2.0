#!/usr/bin/env python3
"""
Example: Using PipelineCheckpoint with FARFAN Orchestrator
Demonstrates checkpointing in a pipeline workflow
"""

import logging
from pathlib import Path

from pipeline_checkpoint import (
    CloudBackupConfig,
    CloudBackupProvider,
    PipelineCheckpoint,
    RetentionPolicy,
    generate_encryption_key,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def ejemplo_basico():
    """Example 1: Basic checkpoint save and load"""
    print("\n" + "=" * 80)
    print("EJEMPLO 1: Checkpointing Básico")
    print("=" * 80)

    # Initialize checkpoint manager
    checkpoint = PipelineCheckpoint(
        checkpoint_dir="./checkpoints_ejemplo", enable_incremental=False
    )

    # Simulate pipeline state after Stage 1 (Document extraction)
    stage1_state = {
        "stage": "document_extraction",
        "policy_code": "PDM2024-ANT-MED",
        "raw_text": "Lorem ipsum dolor sit amet..." * 100,
        "sections": {
            "diagnostico": "Sección de diagnóstico...",
            "objetivos": "Sección de objetivos...",
        },
        "tables_count": 15,
        "timestamp": "2024-01-15T10:30:00",
    }

    # Save checkpoint
    ckpt_id = checkpoint.save(stage1_state, checkpoint_id="stage_1_extraction")
    print(f"✓ Checkpoint guardado: {ckpt_id}")

    # Load checkpoint
    restored = checkpoint.load(ckpt_id)
    print(f"✓ Checkpoint cargado: {restored['stage']}")
    print(f"  Secciones: {list(restored['sections'].keys())}")
    print(f"  Tablas: {restored['tables_count']}")

    # Get info
    info = checkpoint.get_checkpoint_info(ckpt_id)
    print(f"✓ Metadata:")
    print(f"  Tamaño: {info.size_bytes} bytes ({info.size_bytes / 1024:.1f} KB)")
    print(f"  Hash: {info.hash_sha256[:16]}...")
    print(f"  Versión: {info.version}")


def ejemplo_incremental():
    """Example 2: Incremental checkpointing across pipeline stages"""
    print("\n" + "=" * 80)
    print("EJEMPLO 2: Checkpointing Incremental")
    print("=" * 80)

    checkpoint = PipelineCheckpoint(
        checkpoint_dir="./checkpoints_incremental",
        enable_incremental=True,
        retention_policy=RetentionPolicy.KEEP_N_RECENT,
        retention_count=5,
    )

    # Stage 1: Document extraction
    state = {
        "policy_code": "PDM2024-ANT-MED",
        "raw_text": "Document text..." * 1000,
        "sections": {"diagnostico": "content", "objetivos": "content"},
    }
    checkpoint.save(state, checkpoint_id="full_stage1")
    print("✓ Stage 1 (full): Document extraction")

    # Stage 2: Add semantic analysis (incremental)
    state["semantic_analysis"] = {
        "chunks": ["chunk1", "chunk2", "chunk3"],
        "dimension_scores": {"D1": 0.85, "D2": 0.90},
    }
    checkpoint.save(state, checkpoint_id="incr_stage2")
    print("✓ Stage 2 (incremental): Semantic analysis added")

    # Stage 3: Add causal extraction (incremental)
    state["causal_graph"] = {
        "nodes": ["N1", "N2", "N3"],
        "edges": [("N1", "N2"), ("N2", "N3")],
    }
    checkpoint.save(state, checkpoint_id="incr_stage3")
    print("✓ Stage 3 (incremental): Causal graph added")

    # Stage 4: Add DNP validation (incremental)
    state["dnp_validation"] = {
        "compliance_score": 0.87,
        "validated_competencias": ["C1", "C2", "C3"],
    }
    checkpoint.save(state, checkpoint_id="incr_stage4")
    print("✓ Stage 4 (incremental): DNP validation added")

    # Show statistics
    stats = checkpoint.get_statistics()
    print(f"\n✓ Estadísticas:")
    print(f"  Total checkpoints: {stats['total_checkpoints']}")
    print(f"  Full checkpoints: {stats['full_checkpoints']}")
    print(f"  Incremental checkpoints: {stats['incremental_checkpoints']}")
    print(f"  Tamaño total: {stats['total_size_mb']} MB")

    # List checkpoints
    print(f"\n✓ Lista de checkpoints:")
    for ckpt in checkpoint.list_checkpoints():
        tipo = "FULL" if not ckpt.is_incremental else "INCR"
        print(f"  [{tipo}] {ckpt.checkpoint_id} - {ckpt.size_bytes / 1024:.1f} KB")


def ejemplo_encriptacion():
    """Example 3: Encryption for sensitive plan data"""
    print("\n" + "=" * 80)
    print("EJEMPLO 3: Encriptación para Datos Sensibles")
    print("=" * 80)

    try:
        # Generate encryption key
        key = generate_encryption_key()
        print("✓ Clave de encriptación generada")

        checkpoint = PipelineCheckpoint(
            checkpoint_dir="./checkpoints_encrypted",
            enable_encryption=True,
            encryption_key=key,
            enable_incremental=False,
        )

        # Sensitive plan data
        sensitive_state = {
            "policy_code": "PDM2024-ANT-MED",
            "financial_data": {
                "presupuesto_total": 150_000_000_000,  # 150 mil millones
                "allocations": {
                    "Educación": 45_000_000_000,
                    "Salud": 38_000_000_000,
                    "Infraestructura": 30_000_000_000,
                },
            },
            "sensitive_indicators": {
                "poblacion_victimas": 12500,
                "poblacion_desplazada": 3400,
                "indicadores_seguridad": ["indicador_1", "indicador_2"],
            },
        }

        # Save encrypted
        ckpt_id = checkpoint.save(sensitive_state, checkpoint_id="sensitive_plan")
        print(f"✓ Checkpoint encriptado guardado: {ckpt_id}")

        # Verify encryption
        info = checkpoint.get_checkpoint_info(ckpt_id)
        print(f"  Encriptado: {info.is_encrypted}")
        print(f"  Hash: {info.hash_sha256[:16]}...")

        # Load and decrypt
        restored = checkpoint.load(ckpt_id)
        print(f"✓ Checkpoint desencriptado y cargado")
        print(
            f"  Presupuesto total: ${restored['financial_data']['presupuesto_total']:,}"
        )
        print(
            f"  Población víctimas: {restored['sensitive_indicators']['poblacion_victimas']:,}"
        )

    except ImportError as e:
        print(f"⚠ Encriptación no disponible: {e}")
        print("  Instalar con: pip install cryptography")


def ejemplo_retencion():
    """Example 4: Retention policies for checkpoint cleanup"""
    print("\n" + "=" * 80)
    print("EJEMPLO 4: Políticas de Retención")
    print("=" * 80)

    # Policy: Keep only 3 most recent checkpoints
    checkpoint = PipelineCheckpoint(
        checkpoint_dir="./checkpoints_retention",
        retention_policy=RetentionPolicy.KEEP_N_RECENT,
        retention_count=3,
        enable_incremental=False,
    )

    # Create 6 checkpoints
    print("Creando 6 checkpoints...")
    for i in range(6):
        state = {"stage": i + 1, "data": f"Stage {i + 1} data..." * 100}
        checkpoint.save(state, checkpoint_id=f"stage_{i + 1}")
        print(f"  ✓ Checkpoint {i + 1} guardado")

    # List remaining checkpoints
    checkpoints = checkpoint.list_checkpoints()
    print(f"\n✓ Checkpoints retenidos (política: mantener 3 más recientes):")
    print(f"  Total: {len(checkpoints)}")
    for ckpt in checkpoints:
        print(f"  - {ckpt.checkpoint_id}")

    print("\n✓ Checkpoints antiguos eliminados automáticamente")


def ejemplo_consultas():
    """Example 5: Querying checkpoint metadata"""
    print("\n" + "=" * 80)
    print("EJEMPLO 5: Consultas de Metadata")
    print("=" * 80)

    checkpoint = PipelineCheckpoint(
        checkpoint_dir="./checkpoints_queries", enable_incremental=True
    )

    # Create checkpoints with custom metadata
    experiments = [
        {"stage": 1, "accuracy": 0.85, "experiment": "exp_001"},
        {"stage": 2, "accuracy": 0.88, "experiment": "exp_002"},
        {"stage": 3, "accuracy": 0.92, "experiment": "exp_003"},
        {"stage": 4, "accuracy": 0.87, "experiment": "exp_004"},
    ]

    for exp in experiments:
        state = {"stage": exp["stage"], "metrics": {"accuracy": exp["accuracy"]}}
        custom = {"experiment": exp["experiment"], "user": "researcher"}
        checkpoint.save(state, checkpoint_id=exp["experiment"], custom_metadata=custom)

    print("✓ Checkpoints con metadata personalizada creados")

    # Query 1: Filter by accuracy > 0.87
    print("\n✓ Query 1: Checkpoints con accuracy > 0.87")
    filtered = checkpoint.list_checkpoints(
        filter_fn=lambda c: c.custom_metadata.get("experiment", "").startswith("exp")
    )
    for ckpt in filtered:
        print(f"  - {ckpt.checkpoint_id}: {ckpt.custom_metadata}")

    # Query 2: Sort by size
    print("\n✓ Query 2: Top 2 checkpoints más grandes")
    by_size = checkpoint.list_checkpoints(sort_by="size_bytes", reverse=True)
    for ckpt in by_size[:2]:
        print(f"  - {ckpt.checkpoint_id}: {ckpt.size_bytes / 1024:.1f} KB")

    # Statistics
    stats = checkpoint.get_statistics()
    print(f"\n✓ Estadísticas generales:")
    print(f"  Total: {stats['total_checkpoints']} checkpoints")
    print(f"  Tamaño: {stats['total_size_mb']} MB")
    print(f"  Más antiguo: {stats['oldest_checkpoint']}")
    print(f"  Más reciente: {stats['newest_checkpoint']}")


def ejemplo_cloud_backup_stubs():
    """Example 6: Cloud backup configuration (stub)"""
    print("\n" + "=" * 80)
    print("EJEMPLO 6: Configuración de Backup en la Nube (Stub)")
    print("=" * 80)

    # Configure cloud backup
    cloud_config = CloudBackupConfig(
        enabled=True,
        provider=CloudBackupProvider.AWS_S3,
        bucket_name="farfan-checkpoints",
        region="us-east-1",
        auto_sync=False,
    )

    checkpoint = PipelineCheckpoint(
        checkpoint_dir="./checkpoints_cloud",
        cloud_backup_config=cloud_config,
        enable_incremental=False,
    )

    print(f"✓ Cloud backup configurado:")
    print(f"  Provider: {checkpoint.cloud_backup_config.provider.value}")
    print(f"  Bucket: {checkpoint.cloud_backup_config.bucket_name}")
    print(f"  Region: {checkpoint.cloud_backup_config.region}")
    print(f"  Auto-sync: {checkpoint.cloud_backup_config.auto_sync}")

    # Save checkpoint
    state = {"policy_code": "PDM2024", "data": "test"}
    ckpt_id = checkpoint.save(state, checkpoint_id="cloud_test")
    print(f"\n✓ Checkpoint guardado localmente: {ckpt_id}")

    # Manual sync (stub - no actual upload)
    print("\n✓ Iniciando sincronización manual (stub)...")
    checkpoint.sync_to_cloud([ckpt_id])
    print("  ℹ Nota: Sincronización es un stub - no se realizó upload real")

    # Reconfigure for different provider
    azure_config = CloudBackupConfig(
        enabled=True,
        provider=CloudBackupProvider.AZURE_BLOB,
        bucket_name="farfan-container",
        auto_sync=True,
    )
    checkpoint.configure_cloud_backup(azure_config)
    print(f"\n✓ Reconfigurado para Azure Blob Storage")


def ejemplo_workflow_completo():
    """Example 7: Complete orchestrator workflow with checkpointing"""
    print("\n" + "=" * 80)
    print("EJEMPLO 7: Workflow Completo con Checkpointing")
    print("=" * 80)

    checkpoint = PipelineCheckpoint(
        checkpoint_dir="./checkpoints_workflow",
        enable_incremental=True,
        retention_policy=RetentionPolicy.KEEP_N_RECENT,
        retention_count=10,
    )

    # Simulate complete FARFAN pipeline
    state = {"policy_code": "PDM2024-ANT-MED"}

    # Stage 1: Document extraction
    print("\n[STAGE 1] Document Extraction")
    state.update(
        {
            "raw_text": "Document content..." * 500,
            "sections": {"diagnostico": "content", "objetivos": "content"},
            "tables_count": 15,
        }
    )
    checkpoint.save(state, checkpoint_id="workflow_stage1")
    print("  ✓ Checkpoint guardado")

    # Stage 2: Semantic analysis
    print("\n[STAGE 2] Semantic Analysis")
    state.update(
        {
            "semantic_chunks": ["chunk1", "chunk2"],
            "dimension_scores": {"D1": 0.85, "D2": 0.90, "D3": 0.88},
        }
    )
    checkpoint.save(state, checkpoint_id="workflow_stage2")
    print("  ✓ Checkpoint guardado (incremental)")

    # Stage 3: Causal extraction
    print("\n[STAGE 3] Causal Extraction")
    state.update(
        {
            "causal_nodes": ["N1", "N2", "N3"],
            "causal_chains": [("N1", "N2"), ("N2", "N3")],
        }
    )
    checkpoint.save(state, checkpoint_id="workflow_stage3")
    print("  ✓ Checkpoint guardado (incremental)")

    # Stage 4: DNP validation
    print("\n[STAGE 4] DNP Validation")
    state.update(
        {"dnp_compliance_score": 0.87, "competencias_validadas": ["EDU", "SAL", "INF"]}
    )
    checkpoint.save(state, checkpoint_id="workflow_stage4")
    print("  ✓ Checkpoint guardado (incremental)")

    # Stage 5: Question answering
    print("\n[STAGE 5] Question Answering (300 preguntas)")
    state.update(
        {
            "question_responses": {
                "P1-D1-Q1": {"score": 0.85},
                "P1-D1-Q2": {"score": 0.90},
                # ... (300 total)
            }
        }
    )
    checkpoint.save(state, checkpoint_id="workflow_stage5")
    print("  ✓ Checkpoint guardado (incremental)")

    # Final stage: Reports
    print("\n[STAGE 6] Report Generation")
    state.update(
        {
            "micro_report": {"generated": True},
            "meso_report": {"generated": True},
            "macro_report": {"generated": True},
        }
    )
    checkpoint.save(state, checkpoint_id="workflow_complete", force_full=True)
    print("  ✓ Checkpoint final guardado (full)")

    # Show summary
    print("\n✓ Workflow completado")
    stats = checkpoint.get_statistics()
    print(f"\n📊 Estadísticas:")
    print(f"  Total checkpoints: {stats['total_checkpoints']}")
    print(
        f"  Full: {stats['full_checkpoints']}, Incremental: {stats['incremental_checkpoints']}"
    )
    print(f"  Tamaño total: {stats['total_size_mb']} MB")

    # Demonstrate recovery
    print(f"\n🔄 Demostración de recuperación:")
    print("  Simulando fallo en Stage 4...")
    recovered = checkpoint.load("workflow_stage3")
    print(f"  ✓ Estado recuperado desde Stage 3")
    print(f"  ✓ Puede continuar desde: {list(recovered.keys())[-3:]}")


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("FARFAN 2.0 - Pipeline Checkpoint System")
    print("Ejemplos de Uso")
    print("=" * 80)

    try:
        ejemplo_basico()
        ejemplo_incremental()
        ejemplo_encriptacion()
        ejemplo_retencion()
        ejemplo_consultas()
        ejemplo_cloud_backup_stubs()
        ejemplo_workflow_completo()

        print("\n" + "=" * 80)
        print("✅ Todos los ejemplos completados exitosamente")
        print("=" * 80)

    except Exception as e:
        logger.error(f"Error en ejemplos: {e}", exc_info=True)


if __name__ == "__main__":
    main()
