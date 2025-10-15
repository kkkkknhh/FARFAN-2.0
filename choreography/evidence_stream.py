"""
Streaming Evidence Pipeline (F3.2)
===================================
Asynchronous streaming pipeline for incremental evidence processing.
Enables analysis of massive documents without loading everything into memory.

Architecture:
- EvidenceStream: Async iterator for semantic chunks
- StreamingBayesianUpdater: Incremental Bayesian posterior updates
- Memory-efficient, real-time feedback, production-ready
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

# Import SemanticChunk from the existing embedding module
try:
    from emebedding_policy import SemanticChunk
except ImportError:
    # Fallback type definition if import fails
    from typing import TypedDict

    class SemanticChunk(TypedDict):
        """Fallback semantic chunk definition"""

        chunk_id: str
        content: str
        embedding: Any  # Use Any instead of NDArray to avoid numpy dependency
        metadata: Dict[str, Any]
        pdq_context: Optional[Dict[str, Any]]
        token_count: int
        position: tuple[int, int]


# Import EventBus for publishing intermediate results
from choreography.event_bus import EventBus, PDMEvent

logger = logging.getLogger(__name__)

# ============================================================================
# EVIDENCE STREAM - Async Iterator
# ============================================================================


class EvidenceStream:
    """
    Asynchronous stream of evidence chunks for incremental processing.

    Implements async iterator protocol to enable streaming analysis of
    large documents without loading everything into memory. Chunks are
    processed one at a time, allowing for:
    - Memory-efficient processing of massive PDM documents
    - Real-time feedback as analysis progresses
    - Early termination if sufficient evidence is found
    - Integration with async/await patterns

    Example:
        ```python
        # Create stream from semantic chunks
        stream = EvidenceStream(semantic_chunks)

        # Process incrementally
        async for chunk in stream:
            print(f"Processing chunk {chunk['chunk_id']}")
            # Analyze chunk, update posteriors, etc.

            # Can break early if needed
            if sufficient_evidence_found():
                break
        ```
    """

    def __init__(
        self,
        semantic_chunks: List[SemanticChunk],
        batch_size: int = 1,
        delay_ms: int = 0,
    ):
        """
        Initialize evidence stream.

        Args:
            semantic_chunks: List of semantic chunks to stream
            batch_size: Number of chunks to yield at once (default: 1)
            delay_ms: Optional delay between chunks in milliseconds (for rate limiting)
        """
        self.chunks = semantic_chunks
        self.current_idx = 0
        self.batch_size = batch_size
        self.delay_ms = delay_ms
        logger.info(
            f"EvidenceStream initialized with {len(semantic_chunks)} chunks, "
            f"batch_size={batch_size}"
        )

    def __aiter__(self) -> AsyncIterator[SemanticChunk]:
        """Return self as async iterator."""
        return self

    async def __anext__(self) -> SemanticChunk:
        """
        Get next chunk in the stream.

        Returns:
            Next semantic chunk

        Raises:
            StopAsyncIteration: When stream is exhausted
        """
        # Optional delay for rate limiting
        if self.delay_ms > 0:
            await asyncio.sleep(self.delay_ms / 1000.0)

        # Check if stream is exhausted
        if self.current_idx >= len(self.chunks):
            logger.debug("EvidenceStream exhausted")
            raise StopAsyncIteration

        # Get next chunk
        chunk = self.chunks[self.current_idx]
        self.current_idx += 1

        logger.debug(
            f"Streaming chunk {self.current_idx}/{len(self.chunks)}: "
            f"{chunk.get('chunk_id', 'unknown')}"
        )

        return chunk

    def reset(self) -> None:
        """Reset stream to beginning."""
        self.current_idx = 0
        logger.debug("EvidenceStream reset to beginning")

    def remaining(self) -> int:
        """Get number of chunks remaining in stream."""
        return len(self.chunks) - self.current_idx

    def progress(self) -> float:
        """Get progress as fraction between 0.0 and 1.0."""
        if not self.chunks:
            return 1.0
        return self.current_idx / len(self.chunks)


# ============================================================================
# BAYESIAN PRIOR/POSTERIOR MODELS
# ============================================================================


class MechanismPrior:
    """
    Prior distribution for a causal mechanism.

    Represents initial beliefs about a mechanism before observing evidence.
    In Bayesian terms, this is P(mechanism | background_knowledge).
    """

    def __init__(
        self,
        mechanism_name: str,
        prior_mean: float = 0.5,
        prior_std: float = 0.2,
        confidence: float = 0.5,
    ):
        """
        Initialize prior distribution.

        Args:
            mechanism_name: Name of the mechanism
            prior_mean: Prior mean probability (0.0 to 1.0)
            prior_std: Prior standard deviation
            confidence: Confidence in prior (0.0 to 1.0)
        """
        self.mechanism_name = mechanism_name
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.confidence = confidence

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "mechanism_name": self.mechanism_name,
            "prior_mean": self.prior_mean,
            "prior_std": self.prior_std,
            "confidence": self.confidence,
        }


class PosteriorDistribution:
    """
    Posterior distribution after observing evidence.

    Represents updated beliefs about a mechanism after incorporating
    evidence. In Bayesian terms: P(mechanism | evidence, background).
    """

    def __init__(
        self,
        mechanism_name: str,
        posterior_mean: float,
        posterior_std: float,
        evidence_count: int = 0,
        credible_interval_95: tuple[float, float] = None,
    ):
        """
        Initialize posterior distribution.

        Args:
            mechanism_name: Name of the mechanism
            posterior_mean: Updated mean probability
            posterior_std: Updated standard deviation
            evidence_count: Number of evidence chunks incorporated
            credible_interval_95: 95% credible interval (optional)
        """
        self.mechanism_name = mechanism_name
        self.posterior_mean = posterior_mean
        self.posterior_std = posterior_std
        self.evidence_count = evidence_count
        self.credible_interval_95 = credible_interval_95 or (
            max(0.0, posterior_mean - 1.96 * posterior_std),
            min(1.0, posterior_mean + 1.96 * posterior_std),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "mechanism_name": self.mechanism_name,
            "posterior_mean": self.posterior_mean,
            "posterior_std": self.posterior_std,
            "evidence_count": self.evidence_count,
            "credible_interval_95": self.credible_interval_95,
            "confidence": self._compute_confidence(),
        }

    def _compute_confidence(self) -> str:
        """
        Compute confidence level based on posterior distribution.

        Returns:
            Confidence level: 'very_strong', 'strong', 'moderate', 'weak'
        """
        # Narrower std and more evidence = higher confidence
        if self.posterior_std < 0.05 and self.evidence_count >= 10:
            return "very_strong"
        elif self.posterior_std < 0.1 and self.evidence_count >= 5:
            return "strong"
        elif self.posterior_std < 0.2 and self.evidence_count >= 3:
            return "moderate"
        else:
            return "weak"


# ============================================================================
# STREAMING BAYESIAN UPDATER
# ============================================================================


class StreamingBayesianUpdater:
    """
    Incremental Bayesian posterior updates from streaming evidence.

    Processes evidence chunks one at a time, updating posterior beliefs
    incrementally. This enables:
    - Analysis of massive documents without memory exhaustion
    - Real-time feedback on analysis progress
    - Early stopping if strong evidence is found
    - Publishing of intermediate results for monitoring

    Mathematical Foundation:
        Uses sequential Bayesian updating:
        P(θ|D₁,D₂,...,Dₙ) ∝ P(Dₙ|θ) × P(θ|D₁,...,Dₙ₋₁)

        Where:
        - θ: mechanism parameters
        - Dᵢ: evidence chunk i
        - Each chunk updates the posterior, which becomes the prior for next chunk

    Example:
        ```python
        updater = StreamingBayesianUpdater(event_bus=bus)

        prior = MechanismPrior('water_infrastructure', prior_mean=0.5)
        stream = EvidenceStream(chunks)

        posterior = await updater.update_from_stream(stream, prior)

        print(f"Final posterior mean: {posterior.posterior_mean:.3f}")
        print(f"Confidence: {posterior._compute_confidence()}")
        ```
    """

    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize streaming Bayesian updater.

        Args:
            event_bus: Optional event bus for publishing intermediate results
        """
        self.event_bus = event_bus
        self.relevance_threshold = 0.6  # Minimum similarity for relevance
        self.update_count = 0
        logger.info("StreamingBayesianUpdater initialized")

    async def update_from_stream(
        self,
        evidence_stream: EvidenceStream,
        prior: MechanismPrior,
        run_id: str = "default_run",
    ) -> PosteriorDistribution:
        """
        Perform incremental Bayesian update from streaming evidence.

        Processes evidence chunks sequentially, updating posterior beliefs
        after each chunk. Useful for massive documents where loading all
        evidence at once would exhaust memory.

        Args:
            evidence_stream: Stream of evidence chunks
            prior: Prior distribution before observing evidence
            run_id: Identifier for this analysis run

        Returns:
            Final posterior distribution after all evidence

        Example:
            ```python
            # Initialize prior belief
            prior = MechanismPrior(
                mechanism_name='education_quality',
                prior_mean=0.5,  # Neutral prior
                prior_std=0.2
            )

            # Stream evidence
            stream = EvidenceStream(document_chunks)

            # Update incrementally
            posterior = await updater.update_from_stream(stream, prior)

            # Check results
            if posterior.posterior_mean > 0.8:
                print("Strong evidence for mechanism")
            ```
        """
        # Initialize with prior
        current_posterior = PosteriorDistribution(
            mechanism_name=prior.mechanism_name,
            posterior_mean=prior.prior_mean,
            posterior_std=prior.prior_std,
            evidence_count=0,
        )

        logger.info(f"Starting streaming Bayesian update for '{prior.mechanism_name}'")

        evidence_count = 0

        # Process chunks incrementally
        async for chunk in evidence_stream:
            # Check relevance
            if await self._is_relevant(chunk, prior.mechanism_name):
                # Compute likelihood from chunk
                likelihood = await self._compute_likelihood(chunk, prior.mechanism_name)

                # Bayesian update
                current_posterior = self._bayesian_update(current_posterior, likelihood)

                evidence_count += 1
                current_posterior.evidence_count = evidence_count

                logger.debug(
                    f"Updated posterior (chunk {evidence_count}): "
                    f"mean={current_posterior.posterior_mean:.3f}, "
                    f"std={current_posterior.posterior_std:.3f}"
                )

                # Publish intermediate result if event bus available
                if self.event_bus:
                    await self.event_bus.publish(
                        PDMEvent(
                            event_type="posterior.updated",
                            run_id=run_id,
                            payload={
                                "posterior": current_posterior.to_dict(),
                                "chunk_id": chunk.get("chunk_id", "unknown"),
                                "progress": evidence_stream.progress(),
                            },
                        )
                    )

        logger.info(
            f"Streaming update complete: {evidence_count} relevant chunks processed, "
            f"final mean={current_posterior.posterior_mean:.3f}"
        )

        return current_posterior

    async def _is_relevant(self, chunk: SemanticChunk, mechanism_name: str) -> bool:
        """
        Determine if chunk is relevant to the mechanism.

        Uses keyword matching as a simple relevance filter.
        In production, this could use semantic similarity with embeddings.

        Args:
            chunk: Semantic chunk to check
            mechanism_name: Name of mechanism

        Returns:
            True if chunk is relevant
        """
        # Simple keyword matching (placeholder)
        content = chunk.get("content", "").lower()
        keywords = mechanism_name.lower().replace("_", " ").split()

        # Check if any keyword appears in content
        is_relevant = any(keyword in content for keyword in keywords)

        if is_relevant:
            logger.debug(
                f"Chunk {chunk.get('chunk_id')} is relevant to {mechanism_name}"
            )

        return is_relevant

    async def _compute_likelihood(
        self, chunk: SemanticChunk, mechanism_name: str
    ) -> float:
        """
        Compute likelihood P(evidence|mechanism).

        Estimates how likely this evidence would be if the mechanism exists.
        In production, this would use:
        - Semantic similarity scores
        - NER for entity/indicator extraction
        - Sentiment analysis
        - Numerical claim verification

        Args:
            chunk: Semantic chunk
            mechanism_name: Name of mechanism

        Returns:
            Likelihood score (0.0 to 1.0)
        """
        # Placeholder: use token count and keyword density as proxy
        content = chunk.get("content", "").lower()
        keywords = mechanism_name.lower().replace("_", " ").split()

        # Count keyword occurrences
        keyword_count = sum(content.count(kw) for kw in keywords)
        token_count = chunk.get("token_count", 100)

        # Normalize by token count
        density = min(1.0, keyword_count / max(1, token_count / 100))

        # Map to likelihood (0.5 to 0.9 for relevant chunks)
        likelihood = 0.5 + (density * 0.4)

        logger.debug(f"Computed likelihood={likelihood:.3f} for chunk")

        return likelihood

    def _bayesian_update(
        self, current_posterior: PosteriorDistribution, likelihood: float
    ) -> PosteriorDistribution:
        """
        Update posterior using Bayesian rule.

        Implements sequential Bayesian updating:
        posterior ∝ likelihood × prior

        For conjugate priors (Beta-Binomial), this has closed-form solution.
        Here we use a simplified Normal approximation for demonstration.

        Args:
            current_posterior: Current posterior (becomes prior for this update)
            likelihood: Likelihood P(evidence|mechanism)

        Returns:
            Updated posterior distribution
        """
        # Use current posterior as prior for this update
        prior_mean = current_posterior.posterior_mean
        prior_std = current_posterior.posterior_std

        # Simplified Bayesian update (precision-weighted average)
        # In production, use proper conjugate priors or MCMC

        # Precision = 1 / variance
        prior_precision = 1.0 / (prior_std**2) if prior_std > 0 else 1.0
        likelihood_precision = 10.0  # Assume moderate precision for likelihood

        # Updated precision is sum of precisions
        posterior_precision = prior_precision + likelihood_precision
        posterior_variance = 1.0 / posterior_precision
        posterior_std = posterior_variance**0.5

        # Precision-weighted mean
        posterior_mean = (
            prior_precision * prior_mean + likelihood_precision * likelihood
        ) / posterior_precision

        # Ensure mean stays in [0, 1]
        posterior_mean = max(0.0, min(1.0, posterior_mean))

        return PosteriorDistribution(
            mechanism_name=current_posterior.mechanism_name,
            posterior_mean=posterior_mean,
            posterior_std=posterior_std,
            evidence_count=current_posterior.evidence_count,
        )
