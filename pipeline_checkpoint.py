#!/usr/bin/env python3
"""
Pipeline Checkpoint - Sistema de guardado incremental del estado del pipeline

Proporciona:
- Guardado automático de estado después de cada etapa
- Recuperación de ejecuciones interrumpidas
- Checkpoints inmutables con versionado
- Trazabilidad completa de progreso
"""

import json
import pickle
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Dict, List
import logging
import hashlib

logger = logging.getLogger("pipeline_checkpoint")


@dataclass
class CheckpointMetadata:
    """Metadatos de un checkpoint"""
    checkpoint_id: str
    policy_code: str
    stage_name: str
    timestamp: datetime
    success: bool
    execution_time_ms: float
    data_hash: str
    previous_checkpoint: Optional[str] = None


class PipelineCheckpoint:
    """
    Sistema de checkpoints para guardar estado incremental del pipeline
    
    Cada checkpoint es inmutable y contiene:
    - Metadatos de ejecución
    - Estado completo del contexto hasta esa etapa
    - Hash para verificar integridad
    - Referencia al checkpoint anterior (cadena)
    """
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoints: List[CheckpointMetadata] = []
        self.latest_checkpoint: Optional[str] = None
        
        logger.info(f"PipelineCheckpoint inicializado: {self.checkpoint_dir}")
    
    def save(self, policy_code: str, stage_name: str, context: Any, 
             execution_time_ms: float, success: bool = True) -> str:
        """
        Guarda checkpoint después de una etapa
        
        Args:
            policy_code: Código de política
            stage_name: Nombre de la etapa
            context: Contexto completo del pipeline
            execution_time_ms: Tiempo de ejecución en ms
            success: Si la etapa fue exitosa
        
        Returns:
            ID del checkpoint creado
        """
        timestamp = datetime.now()
        checkpoint_id = self._generate_checkpoint_id(policy_code, stage_name, timestamp)
        
        # Serialize context
        try:
            context_data = self._serialize_context(context)
            data_hash = self._compute_hash(context_data)
        except Exception as e:
            logger.error(f"Error serializando contexto: {e}")
            raise
        
        # Create metadata
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            policy_code=policy_code,
            stage_name=stage_name,
            timestamp=timestamp,
            success=success,
            execution_time_ms=execution_time_ms,
            data_hash=data_hash,
            previous_checkpoint=self.latest_checkpoint
        )
        
        # Save to disk
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pkl"
        metadata_path = self.checkpoint_dir / f"{checkpoint_id}.meta.json"
        
        try:
            # Save pickled context
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(context_data, f)
            
            # Save metadata JSON
            with open(metadata_path, 'w') as f:
                json.dump(self._metadata_to_dict(metadata), f, indent=2)
            
            # Update registry
            self.checkpoints.append(metadata)
            self.latest_checkpoint = checkpoint_id
            
            logger.info(f"✓ Checkpoint guardado: {checkpoint_id} (stage: {stage_name}, {execution_time_ms:.1f}ms)")
            
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Error guardando checkpoint: {e}")
            raise
    
    def load(self, checkpoint_id: str) -> tuple[CheckpointMetadata, Any]:
        """
        Carga checkpoint por ID
        
        Args:
            checkpoint_id: ID del checkpoint
        
        Returns:
            Tupla (metadata, context)
        """
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pkl"
        metadata_path = self.checkpoint_dir / f"{checkpoint_id}.meta.json"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint_id}")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
            metadata = self._dict_to_metadata(metadata_dict)
        
        # Load context
        with open(checkpoint_path, 'rb') as f:
            context_data = pickle.load(f)
        
        # Verify integrity
        data_hash = self._compute_hash(context_data)
        if data_hash != metadata.data_hash:
            logger.warning(f"Hash mismatch en checkpoint {checkpoint_id}")
        
        logger.info(f"✓ Checkpoint cargado: {checkpoint_id}")
        
        return metadata, context_data
    
    def load_latest(self, policy_code: str) -> Optional[tuple[CheckpointMetadata, Any]]:
        """
        Carga el checkpoint más reciente para una política
        
        Args:
            policy_code: Código de política
        
        Returns:
            Tupla (metadata, context) o None si no existe
        """
        # Find all checkpoints for this policy
        policy_checkpoints = [
            cp for cp in self.checkpoints
            if cp.policy_code == policy_code and cp.success
        ]
        
        if not policy_checkpoints:
            logger.info(f"No hay checkpoints para {policy_code}")
            return None
        
        # Get most recent
        latest = max(policy_checkpoints, key=lambda cp: cp.timestamp)
        return self.load(latest.checkpoint_id)
    
    def list_checkpoints(self, policy_code: Optional[str] = None) -> List[CheckpointMetadata]:
        """
        Lista checkpoints disponibles
        
        Args:
            policy_code: Filtrar por código de política (opcional)
        
        Returns:
            Lista de metadatos de checkpoints
        """
        if policy_code:
            return [cp for cp in self.checkpoints if cp.policy_code == policy_code]
        return self.checkpoints.copy()
    
    def get_checkpoint_chain(self, checkpoint_id: str) -> List[CheckpointMetadata]:
        """
        Obtiene cadena de checkpoints hasta el especificado
        
        Args:
            checkpoint_id: ID del checkpoint final
        
        Returns:
            Lista de checkpoints en orden cronológico
        """
        chain = []
        current_id = checkpoint_id
        
        while current_id:
            # Find checkpoint
            checkpoint = next((cp for cp in self.checkpoints if cp.checkpoint_id == current_id), None)
            if not checkpoint:
                break
            
            chain.insert(0, checkpoint)
            current_id = checkpoint.previous_checkpoint
        
        return chain
    
    def clean_old_checkpoints(self, policy_code: str, keep_last_n: int = 5):
        """
        Limpia checkpoints antiguos, manteniendo los N más recientes
        
        Args:
            policy_code: Código de política
            keep_last_n: Número de checkpoints a mantener
        """
        policy_checkpoints = [
            cp for cp in self.checkpoints
            if cp.policy_code == policy_code
        ]
        
        if len(policy_checkpoints) <= keep_last_n:
            return
        
        # Sort by timestamp
        sorted_checkpoints = sorted(policy_checkpoints, key=lambda cp: cp.timestamp, reverse=True)
        to_delete = sorted_checkpoints[keep_last_n:]
        
        for checkpoint in to_delete:
            try:
                checkpoint_path = self.checkpoint_dir / f"{checkpoint.checkpoint_id}.pkl"
                metadata_path = self.checkpoint_dir / f"{checkpoint.checkpoint_id}.meta.json"
                
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                if metadata_path.exists():
                    metadata_path.unlink()
                
                self.checkpoints.remove(checkpoint)
                logger.info(f"Checkpoint eliminado: {checkpoint.checkpoint_id}")
                
            except Exception as e:
                logger.error(f"Error eliminando checkpoint {checkpoint.checkpoint_id}: {e}")
    
    def _generate_checkpoint_id(self, policy_code: str, stage_name: str, timestamp: datetime) -> str:
        """Genera ID único para checkpoint"""
        time_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")
        return f"{policy_code}_{stage_name}_{time_str}"
    
    def _serialize_context(self, context: Any) -> dict:
        """Serializa contexto a diccionario"""
        if hasattr(context, '__dict__'):
            return {k: v for k, v in context.__dict__.items() if not k.startswith('_')}
        return {'data': context}
    
    def _compute_hash(self, data: dict) -> str:
        """Calcula hash de los datos"""
        # Serialize to JSON for consistent hashing
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
    
    def _metadata_to_dict(self, metadata: CheckpointMetadata) -> dict:
        """Convierte metadata a diccionario JSON-serializable"""
        return {
            'checkpoint_id': metadata.checkpoint_id,
            'policy_code': metadata.policy_code,
            'stage_name': metadata.stage_name,
            'timestamp': metadata.timestamp.isoformat(),
            'success': metadata.success,
            'execution_time_ms': metadata.execution_time_ms,
            'data_hash': metadata.data_hash,
            'previous_checkpoint': metadata.previous_checkpoint
        }
    
    def _dict_to_metadata(self, data: dict) -> CheckpointMetadata:
        """Convierte diccionario a CheckpointMetadata"""
        return CheckpointMetadata(
            checkpoint_id=data['checkpoint_id'],
            policy_code=data['policy_code'],
            stage_name=data['stage_name'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            success=data['success'],
            execution_time_ms=data['execution_time_ms'],
            data_hash=data['data_hash'],
            previous_checkpoint=data.get('previous_checkpoint')
        )
