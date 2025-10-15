#!/usr/bin/env python3
"""
FARFAN 2.0 Orchestrator with Checkpoint Integration
Demonstrates how to add checkpointing to the orchestrator pipeline
"""

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from pipeline_checkpoint import (
    CloudBackupConfig,
    CloudBackupProvider,
    PipelineCheckpoint,
    RetentionPolicy,
)

logger = logging.getLogger(__name__)


class CheckpointedOrchestrator:
    """
    Wrapper around FARFANOrchestrator that adds checkpoint capabilities.

    This allows the orchestrator to:
    - Save state after each pipeline stage
    - Recover from failures at any stage
    - Track historical pipeline runs
    - Enable incremental processing
    """

    def __init__(
        self,
        output_dir: Path,
        checkpoint_dir: Optional[str] = None,
        enable_encryption: bool = False,
        encryption_key: Optional[bytes] = None,
        retention_count: int = 10,
        cloud_backup: bool = False,
        cloud_provider: str = "aws_s3",
        cloud_bucket: Optional[str] = None,
        log_level: str = "INFO",
    ):
        """
        Initialize checkpointed orchestrator.

        Args:
            output_dir: Directory for orchestrator outputs
            checkpoint_dir: Directory for checkpoints (default: output_dir/checkpoints)
            enable_encryption: Enable encryption for sensitive plan data
            encryption_key: Encryption key (generates new if None and encryption enabled)
            retention_count: Number of recent checkpoints to keep
            cloud_backup: Enable cloud backup stubs
            cloud_provider: Cloud provider (aws_s3, azure_blob, gcp_storage)
            cloud_bucket: Cloud bucket/container name
            log_level: Logging level
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize base orchestrator
        from orchestrator import FARFANOrchestrator

        self.orchestrator = FARFANOrchestrator(
            output_dir=self.output_dir, log_level=log_level
        )

        # Initialize checkpoint manager
        if checkpoint_dir is None:
            checkpoint_dir = str(self.output_dir / "checkpoints")

        cloud_config = None
        if cloud_backup:
            provider_map = {
                "aws_s3": CloudBackupProvider.AWS_S3,
                "azure_blob": CloudBackupProvider.AZURE_BLOB,
                "gcp_storage": CloudBackupProvider.GCP_STORAGE,
            }
            cloud_config = CloudBackupConfig(
                enabled=True,
                provider=provider_map.get(
                    cloud_provider, CloudBackupProvider.LOCAL_ONLY
                ),
                bucket_name=cloud_bucket,
                auto_sync=False,  # Manual sync only
            )

        self.checkpoint = PipelineCheckpoint(
            checkpoint_dir=checkpoint_dir,
            enable_encryption=enable_encryption,
            encryption_key=encryption_key,
            enable_incremental=True,
            retention_policy=RetentionPolicy.KEEP_N_RECENT,
            retention_count=retention_count,
            cloud_backup_config=cloud_config,
        )

        logger.info("CheckpointedOrchestrator initialized")
        logger.info(f"Output dir: {self.output_dir}")
        logger.info(f"Checkpoint dir: {checkpoint_dir}")
        logger.info(f"Encryption: {enable_encryption}")
        logger.info(f"Cloud backup: {cloud_backup}")

    def process_plan_with_checkpoints(
        self,
        pdf_path: Path,
        policy_code: str,
        es_municipio_pdet: bool = False,
        resume_from: Optional[str] = None,
    ):
        """
        Process plan with checkpoint after each stage.

        Args:
            pdf_path: Path to PDF plan
            policy_code: Policy code identifier
            es_municipio_pdet: Is PDET municipality
            resume_from: Checkpoint ID to resume from (None = start fresh)

        Returns:
            PipelineContext with final results
        """
        from orchestrator import PipelineContext

        logger.info("=" * 80)
        logger.info(f"Processing plan with checkpoints: {policy_code}")
        logger.info("=" * 80)

        # Resume from checkpoint or start fresh
        if resume_from:
            logger.info(f"Resuming from checkpoint: {resume_from}")
            ctx_dict = self.checkpoint.load(resume_from)
            ctx = PipelineContext(**ctx_dict)
            # Determine which stage to resume from
            resume_stage = ctx_dict.get("_last_completed_stage", 0)
            logger.info(f"Resuming from stage {resume_stage + 1}")
        else:
            ctx = PipelineContext(
                pdf_path=pdf_path, policy_code=policy_code, output_dir=self.output_dir
            )
            resume_stage = 0

        try:
            # Stage 1-2: Document extraction
            if resume_stage < 1:
                logger.info("\n[STAGE 1-2] Document Extraction")
                ctx = self.orchestrator._stage_extract_document(ctx)
                self._save_checkpoint(ctx, stage=1, policy_code=policy_code)

            # Stage 3: Semantic analysis
            if resume_stage < 2:
                logger.info("\n[STAGE 3] Semantic Analysis")
                ctx = self.orchestrator._stage_semantic_analysis(ctx)
                self._save_checkpoint(ctx, stage=2, policy_code=policy_code)

            # Stage 4: Causal extraction
            if resume_stage < 3:
                logger.info("\n[STAGE 4] Causal Extraction")
                ctx = self.orchestrator._stage_causal_extraction(ctx)
                self._save_checkpoint(ctx, stage=3, policy_code=policy_code)

            # Stage 5: Mechanism inference
            if resume_stage < 4:
                logger.info("\n[STAGE 5] Mechanism Inference")
                ctx = self.orchestrator._stage_mechanism_inference(ctx)
                self._save_checkpoint(ctx, stage=4, policy_code=policy_code)

            # Stage 6: Financial audit
            if resume_stage < 5:
                logger.info("\n[STAGE 6] Financial Audit")
                ctx = self.orchestrator._stage_financial_audit(ctx)
                self._save_checkpoint(ctx, stage=5, policy_code=policy_code)

            # Stage 7: DNP validation
            if resume_stage < 6:
                logger.info("\n[STAGE 7] DNP Validation")
                ctx = self.orchestrator._stage_dnp_validation(ctx, es_municipio_pdet)
                self._save_checkpoint(ctx, stage=6, policy_code=policy_code)

            # Stage 8: Question answering
            if resume_stage < 7:
                logger.info("\n[STAGE 8] Question Answering (300 preguntas)")
                ctx = self.orchestrator._stage_question_answering(ctx)
                self._save_checkpoint(ctx, stage=7, policy_code=policy_code)

            # Stage 9: Report generation
            if resume_stage < 8:
                logger.info("\n[STAGE 9] Report Generation")
                ctx = self.orchestrator._stage_report_generation(ctx)
                # Final checkpoint with force_full=True
                self._save_checkpoint(
                    ctx, stage=8, policy_code=policy_code, force_full=True
                )

            logger.info("=" * 80)
            logger.info(f"✅ Pipeline completed successfully: {policy_code}")
            logger.info("=" * 80)

            return ctx

        except Exception as e:
            logger.error(f"❌ Pipeline failed at stage: {e}", exc_info=True)
            logger.info(
                "💾 State has been checkpointed - can resume with resume_from parameter"
            )
            raise

    def _save_checkpoint(
        self, ctx, stage: int, policy_code: str, force_full: bool = False
    ):
        """Save checkpoint with stage metadata"""
        # Convert context to dict (handle non-serializable objects)
        ctx_dict = self._serialize_context(ctx)
        ctx_dict["_last_completed_stage"] = stage

        checkpoint_id = f"{policy_code}_stage_{stage}"

        custom_metadata = {
            "policy_code": policy_code,
            "stage": stage,
            "stage_name": self._get_stage_name(stage),
        }

        self.checkpoint.save(
            ctx_dict,
            checkpoint_id=checkpoint_id,
            force_full=force_full,
            custom_metadata=custom_metadata,
        )

        logger.info(f"✓ Checkpoint saved: {checkpoint_id}")

    def _serialize_context(self, ctx):
        """Convert PipelineContext to serializable dict"""
        ctx_dict = asdict(ctx)

        # Handle non-serializable objects
        # Convert Path objects to strings
        for key, value in ctx_dict.items():
            if isinstance(value, Path):
                ctx_dict[key] = str(value)

        return ctx_dict

    def _get_stage_name(self, stage: int) -> str:
        """Get human-readable stage name"""
        stage_names = {
            1: "Document Extraction",
            2: "Semantic Analysis",
            3: "Causal Extraction",
            4: "Mechanism Inference",
            5: "Financial Audit",
            6: "DNP Validation",
            7: "Question Answering",
            8: "Report Generation",
        }
        return stage_names.get(stage, f"Stage {stage}")

    def list_checkpoints(self, policy_code: Optional[str] = None):
        """
        List checkpoints, optionally filtered by policy code.

        Args:
            policy_code: Filter by policy code (None = all)

        Returns:
            List of checkpoint metadata
        """
        if policy_code:
            return self.checkpoint.list_checkpoints(
                filter_fn=lambda c: c.custom_metadata.get("policy_code") == policy_code
            )
        return self.checkpoint.list_checkpoints()

    def get_checkpoint_info(self, checkpoint_id: str):
        """Get metadata for a specific checkpoint"""
        return self.checkpoint.get_checkpoint_info(checkpoint_id)

    def delete_checkpoint(self, checkpoint_id: str):
        """Delete a specific checkpoint"""
        self.checkpoint.delete_checkpoint(checkpoint_id)

    def get_statistics(self):
        """Get checkpoint statistics"""
        return self.checkpoint.get_statistics()

    def sync_to_cloud(self, checkpoint_ids: Optional[list] = None):
        """
        Sync checkpoints to cloud storage (stub).

        Args:
            checkpoint_ids: List of checkpoint IDs (None = all)
        """
        self.checkpoint.sync_to_cloud(checkpoint_ids)


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description="FARFAN Orchestrator with Checkpointing"
    )
    parser.add_argument("pdf_path", help="Path to PDF plan")
    parser.add_argument("--policy-code", required=True, help="Policy code")
    parser.add_argument("--output-dir", default="./results", help="Output directory")
    parser.add_argument("--checkpoint-dir", help="Checkpoint directory")
    parser.add_argument("--pdet", action="store_true", help="Is PDET municipality")
    parser.add_argument("--resume-from", help="Resume from checkpoint ID")
    parser.add_argument("--encrypt", action="store_true", help="Enable encryption")
    parser.add_argument(
        "--cloud-backup", action="store_true", help="Enable cloud backup"
    )
    parser.add_argument(
        "--cloud-provider",
        default="aws_s3",
        choices=["aws_s3", "azure_blob", "gcp_storage"],
    )
    parser.add_argument("--cloud-bucket", help="Cloud bucket name")
    parser.add_argument(
        "--list-checkpoints", action="store_true", help="List checkpoints and exit"
    )
    parser.add_argument(
        "--checkpoint-stats",
        action="store_true",
        help="Show checkpoint statistics and exit",
    )

    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = CheckpointedOrchestrator(
        output_dir=Path(args.output_dir),
        checkpoint_dir=args.checkpoint_dir,
        enable_encryption=args.encrypt,
        cloud_backup=args.cloud_backup,
        cloud_provider=args.cloud_provider,
        cloud_bucket=args.cloud_bucket,
        log_level="INFO",
    )

    # Handle query commands
    if args.list_checkpoints:
        checkpoints = orchestrator.list_checkpoints(policy_code=args.policy_code)
        print(f"\nCheckpoints for {args.policy_code or 'all policies'}:")
        for ckpt in checkpoints:
            print(f"  {ckpt.checkpoint_id}")
            print(f"    Stage: {ckpt.custom_metadata.get('stage_name', 'Unknown')}")
            print(f"    Size: {ckpt.size_bytes / 1024:.1f} KB")
            print(f"    Time: {ckpt.timestamp}")
        return

    if args.checkpoint_stats:
        stats = orchestrator.get_statistics()
        print("\nCheckpoint Statistics:")
        print(f"  Total checkpoints: {stats['total_checkpoints']}")
        print(
            f"  Full: {stats['full_checkpoints']}, Incremental: {stats['incremental_checkpoints']}"
        )
        print(f"  Total size: {stats['total_size_mb']} MB")
        print(f"  Oldest: {stats.get('oldest_checkpoint', 'N/A')}")
        print(f"  Newest: {stats.get('newest_checkpoint', 'N/A')}")
        return

    # Process plan
    try:
        ctx = orchestrator.process_plan_with_checkpoints(
            pdf_path=Path(args.pdf_path),
            policy_code=args.policy_code,
            es_municipio_pdet=args.pdet,
            resume_from=args.resume_from,
        )

        print("\n" + "=" * 80)
        print("✅ Processing completed successfully")
        print("=" * 80)
        print(f"\nResults saved to: {args.output_dir}")
        print(
            f"Checkpoints saved to: {args.checkpoint_dir or args.output_dir + '/checkpoints'}"
        )

        # Show final statistics
        stats = orchestrator.get_statistics()
        print(f"\nCheckpoint Statistics:")
        print(f"  Total: {stats['total_checkpoints']} checkpoints")
        print(f"  Size: {stats['total_size_mb']} MB")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nYou can resume from the last checkpoint with:")
        print(f"  --resume-from {args.policy_code}_stage_N")
        return 1

    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    exit(main())
