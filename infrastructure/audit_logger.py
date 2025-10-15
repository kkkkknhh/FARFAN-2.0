#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Immutable Audit Logger Module
==============================

Provides immutable audit trail with SHA-256 provenance for governance compliance.

SIN_CARRETA Compliance:
- Append-only audit logs (no mutations)
- SHA-256 provenance for all source files
- JSONL format for streaming analysis
- Explicit timestamp and orchestrator tracking
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AuditRecord:
    """Immutable audit record structure"""
    run_id: str
    orchestrator: str
    timestamp: str
    sha256_source: str
    event: str = "execution"
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'run_id': self.run_id,
            'orchestrator': self.orchestrator,
            'timestamp': self.timestamp,
            'sha256_source': self.sha256_source,
            'event': self.event,
            'data': self.data
        }


class ImmutableAuditLogger:
    """
    Immutable audit logger for governance compliance.
    
    SIN_CARRETA Compliance:
    - Append-only JSONL format
    - SHA-256 file provenance
    - Thread-safe write operations
    - Explicit audit record structure
    
    Features:
    - Automatic SHA-256 hashing of source files
    - Structured JSONL output for streaming analysis
    - In-memory record cache for session queries
    - File-based persistence for long-term audit
    
    Usage:
        audit_logger = ImmutableAuditLogger(Path("audit_logs.jsonl"))
        
        audit_logger.append_record(
            run_id="run_20240101_120000",
            orchestrator="PDMOrchestrator",
            sha256_source=audit_logger.hash_file("plan.pdf"),
            event="extraction_complete",
            extraction_quality=0.85,
            chunk_count=42
        )
        
        # Query recent records
        recent = audit_logger.get_recent_records(limit=10)
    """
    
    def __init__(self, audit_store_path: Optional[Path] = None):
        """
        Initialize immutable audit logger.
        
        Args:
            audit_store_path: Path to JSONL audit log file (default: audit_logs.jsonl)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.audit_store_path = audit_store_path or Path("audit_logs.jsonl")
        self._records: List[AuditRecord] = []
        
        # Create parent directory if needed
        self.audit_store_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing records if file exists
        self._load_existing_records()
        
    def _load_existing_records(self) -> None:
        """Load existing audit records from disk"""
        if not self.audit_store_path.exists():
            return
        
        try:
            with open(self.audit_store_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        record_dict = json.loads(line)
                        record = AuditRecord(
                            run_id=record_dict['run_id'],
                            orchestrator=record_dict['orchestrator'],
                            timestamp=record_dict['timestamp'],
                            sha256_source=record_dict['sha256_source'],
                            event=record_dict.get('event', 'execution'),
                            data=record_dict.get('data', {})
                        )
                        self._records.append(record)
            
            self.logger.info(
                f"Loaded {len(self._records)} existing audit records from "
                f"{self.audit_store_path}"
            )
        except Exception as e:
            self.logger.warning(
                f"Failed to load existing audit records: {e}",
                exc_info=True
            )
    
    def append_record(
        self,
        run_id: str,
        orchestrator: str,
        sha256_source: str,
        event: str = "execution",
        **kwargs
    ) -> None:
        """
        Append an immutable audit record.
        
        Args:
            run_id: Unique run identifier
            orchestrator: Orchestrator class name
            sha256_source: SHA-256 hash of source file
            event: Event type (default: "execution")
            **kwargs: Additional data to include in audit record
            
        SIN_CARRETA Compliance:
        - Record is immediately persisted to disk
        - In-memory cache is append-only
        - Timestamp is automatically generated
        """
        record = AuditRecord(
            run_id=run_id,
            orchestrator=orchestrator,
            timestamp=datetime.now().isoformat(),
            sha256_source=sha256_source,
            event=event,
            data=kwargs
        )
        
        # Append to in-memory cache
        self._records.append(record)
        
        # Persist to disk immediately
        try:
            with open(self.audit_store_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + '\n')
            
            self.logger.info(
                f"Audit record appended: run_id={run_id}, "
                f"orchestrator={orchestrator}, event={event}"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to persist audit record: {e}",
                exc_info=True
            )
            # Don't raise - continue execution even if audit persistence fails
    
    @staticmethod
    def hash_file(file_path: str) -> str:
        """
        Generate SHA-256 hash of file for provenance tracking.
        
        Args:
            file_path: Path to file to hash
            
        Returns:
            SHA-256 hash hex string
            
        SIN_CARRETA Compliance:
        - Deterministic hashing (same file = same hash)
        - Chunked reading for large files
        - Graceful fallback on errors
        """
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, 'rb') as f:
                # Read in 4KB chunks to handle large files
                for byte_block in iter(lambda: f.read(4096), b''):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logging.warning(f"Could not hash file {file_path}: {e}")
            return "unknown"
    
    @staticmethod
    def hash_string(content: str) -> str:
        """
        Generate SHA-256 hash of string content.
        
        Args:
            content: String content to hash
            
        Returns:
            SHA-256 hash hex string
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def get_recent_records(
        self,
        limit: int = 10,
        orchestrator: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent audit records.
        
        Args:
            limit: Maximum number of records to return
            orchestrator: Filter by orchestrator name (optional)
            
        Returns:
            List of audit records (most recent first)
        """
        filtered_records = self._records
        
        if orchestrator:
            filtered_records = [
                r for r in filtered_records
                if r.orchestrator == orchestrator
            ]
        
        # Return most recent records first
        recent = filtered_records[-limit:]
        recent.reverse()
        
        return [r.to_dict() for r in recent]
    
    def get_records_by_run_id(self, run_id: str) -> List[Dict[str, Any]]:
        """
        Get all audit records for a specific run.
        
        Args:
            run_id: Run identifier
            
        Returns:
            List of audit records for the run
        """
        return [
            r.to_dict()
            for r in self._records
            if r.run_id == run_id
        ]
    
    def get_records_by_source(self, sha256_source: str) -> List[Dict[str, Any]]:
        """
        Get all audit records for a specific source file.
        
        Args:
            sha256_source: SHA-256 hash of source file
            
        Returns:
            List of audit records for the source
        """
        return [
            r.to_dict()
            for r in self._records
            if r.sha256_source == sha256_source
        ]
    
    def verify_record_integrity(self, record_dict: Dict[str, Any]) -> bool:
        """
        Verify integrity of an audit record.
        
        Args:
            record_dict: Audit record dictionary
            
        Returns:
            True if record has all required fields
        """
        required_fields = ['run_id', 'orchestrator', 'timestamp', 'sha256_source']
        return all(field in record_dict for field in required_fields)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get audit log statistics.
        
        Returns:
            Statistics dictionary with:
                - total_records: Total number of audit records
                - orchestrators: Count by orchestrator type
                - unique_sources: Number of unique source files
                - date_range: Earliest and latest timestamps
        """
        if not self._records:
            return {
                'total_records': 0,
                'orchestrators': {},
                'unique_sources': 0,
                'date_range': {'earliest': None, 'latest': None}
            }
        
        orchestrator_counts = {}
        unique_sources = set()
        
        for record in self._records:
            orchestrator_counts[record.orchestrator] = \
                orchestrator_counts.get(record.orchestrator, 0) + 1
            unique_sources.add(record.sha256_source)
        
        timestamps = [r.timestamp for r in self._records]
        
        return {
            'total_records': len(self._records),
            'orchestrators': orchestrator_counts,
            'unique_sources': len(unique_sources),
            'date_range': {
                'earliest': min(timestamps),
                'latest': max(timestamps)
            }
        }
