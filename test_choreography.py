"""
Tests for FRENTE 3: COREOGRAFÍA (Comunicación Descentralizada)
===============================================================
Comprehensive test suite for event bus and streaming evidence pipeline.
"""

import asyncio
from datetime import datetime
from typing import List

import pytest

# Import modules to test
from choreography.event_bus import ContradictionDetectorV2, EventBus, PDMEvent
from choreography.evidence_stream import (
    EvidenceStream,
    MechanismPrior,
    PosteriorDistribution,
    StreamingBayesianUpdater,
)

# ============================================================================
# TEST FIXTURES
# ============================================================================


@pytest.fixture
def event_bus():
    """Create a fresh event bus for testing."""
    return EventBus()


@pytest.fixture
def sample_chunks():
    """Create sample semantic chunks for testing."""
    return [
        {
            "chunk_id": f"chunk_{i}",
            "content": f"Este es un chunk sobre educación y calidad. Chunk {i}.",
            "embedding": None,
            "metadata": {"section": "education"},
            "pdq_context": None,
            "token_count": 20,
            "position": (i * 100, (i + 1) * 100),
        }
        for i in range(5)
    ]


@pytest.fixture
def mechanism_prior():
    """Create a sample mechanism prior."""
    return MechanismPrior(
        mechanism_name="education_quality",
        prior_mean=0.5,
        prior_std=0.2,
        confidence=0.5,
    )


# ============================================================================
# EVENT BUS TESTS
# ============================================================================


class TestPDMEvent:
    """Test PDMEvent model."""

    def test_event_creation(self):
        """Test creating a basic event."""
        event = PDMEvent(
            event_type="test.event", run_id="test_run", payload={"key": "value"}
        )

        assert event.event_type == "test.event"
        assert event.run_id == "test_run"
        assert event.payload == {"key": "value"}
        assert event.event_id is not None
        assert isinstance(event.timestamp, datetime)

    def test_event_with_custom_id(self):
        """Test creating event with custom ID."""
        event = PDMEvent(
            event_id="custom_id", event_type="test.event", run_id="test_run", payload={}
        )

        assert event.event_id == "custom_id"


class TestEventBus:
    """Test EventBus functionality."""

    def test_initialization(self, event_bus):
        """Test event bus initialization."""
        assert len(event_bus.subscribers) == 0
        assert len(event_bus.event_log) == 0

    def test_subscribe(self, event_bus):
        """Test subscribing to events."""

        async def handler(event):
            pass

        event_bus.subscribe("test.event", handler)

        assert "test.event" in event_bus.subscribers
        assert handler in event_bus.subscribers["test.event"]

    def test_unsubscribe(self, event_bus):
        """Test unsubscribing from events."""

        async def handler(event):
            pass

        event_bus.subscribe("test.event", handler)
        event_bus.unsubscribe("test.event", handler)

        assert handler not in event_bus.subscribers.get("test.event", [])

    @pytest.mark.asyncio
    async def test_publish_no_subscribers(self, event_bus):
        """Test publishing event with no subscribers."""
        event = PDMEvent(event_type="test.event", run_id="test", payload={})

        await event_bus.publish(event)

        # Event should be logged even with no subscribers
        assert len(event_bus.event_log) == 1
        assert event_bus.event_log[0] == event

    @pytest.mark.asyncio
    async def test_publish_with_handler(self, event_bus):
        """Test publishing event with handler."""
        received_events = []

        async def handler(event):
            received_events.append(event)

        event_bus.subscribe("test.event", handler)

        event = PDMEvent(
            event_type="test.event", run_id="test", payload={"data": "value"}
        )

        await event_bus.publish(event)

        # Wait for async handler
        await asyncio.sleep(0.1)

        assert len(received_events) == 1
        assert received_events[0].payload == {"data": "value"}

    @pytest.mark.asyncio
    async def test_publish_multiple_handlers(self, event_bus):
        """Test publishing to multiple handlers."""
        received_1 = []
        received_2 = []

        async def handler1(event):
            received_1.append(event)

        async def handler2(event):
            received_2.append(event)

        event_bus.subscribe("test.event", handler1)
        event_bus.subscribe("test.event", handler2)

        event = PDMEvent(event_type="test.event", run_id="test", payload={})

        await event_bus.publish(event)
        await asyncio.sleep(0.1)

        assert len(received_1) == 1
        assert len(received_2) == 1

    @pytest.mark.asyncio
    async def test_publish_sync_handler(self, event_bus):
        """Test publishing with synchronous handler."""
        received = []

        def sync_handler(event):
            received.append(event)

        event_bus.subscribe("test.event", sync_handler)

        event = PDMEvent(event_type="test.event", run_id="test", payload={})

        await event_bus.publish(event)
        await asyncio.sleep(0.1)

        assert len(received) == 1

    def test_get_event_log(self, event_bus):
        """Test retrieving event log."""
        event1 = PDMEvent(event_type="type1", run_id="run1", payload={})
        event2 = PDMEvent(event_type="type2", run_id="run1", payload={})
        event3 = PDMEvent(event_type="type1", run_id="run2", payload={})

        event_bus.event_log.extend([event1, event2, event3])

        # Filter by event type
        type1_events = event_bus.get_event_log(event_type="type1")
        assert len(type1_events) == 2

        # Filter by run_id
        run1_events = event_bus.get_event_log(run_id="run1")
        assert len(run1_events) == 2

        # Filter by both
        specific_events = event_bus.get_event_log(event_type="type1", run_id="run1")
        assert len(specific_events) == 1

        # Limit
        limited = event_bus.get_event_log(limit=2)
        assert len(limited) == 2

    def test_clear_log(self, event_bus):
        """Test clearing event log."""
        event_bus.event_log.append(
            PDMEvent(event_type="test", run_id="test", payload={})
        )

        event_bus.clear_log()

        assert len(event_bus.event_log) == 0


class TestContradictionDetectorV2:
    """Test ContradictionDetectorV2 example."""

    def test_initialization(self, event_bus):
        """Test detector initialization."""
        detector = ContradictionDetectorV2(event_bus)

        assert detector.event_bus == event_bus
        assert len(detector.detected_contradictions) == 0
        assert "graph.edge_added" in event_bus.subscribers

    @pytest.mark.asyncio
    async def test_edge_added_no_contradiction(self, event_bus):
        """Test edge addition without contradiction."""
        detector = ContradictionDetectorV2(event_bus)

        event = PDMEvent(
            event_type="graph.edge_added",
            run_id="test",
            payload={"source": "A", "target": "B", "relation": "contributes_to"},
        )

        await event_bus.publish(event)
        await asyncio.sleep(0.1)

        assert len(detector.detected_contradictions) == 0

    @pytest.mark.asyncio
    async def test_edge_added_self_loop(self, event_bus):
        """Test detecting self-loop contradiction."""
        detector = ContradictionDetectorV2(event_bus)
        contradictions = []

        async def contradiction_handler(event):
            contradictions.append(event)

        event_bus.subscribe("contradiction.detected", contradiction_handler)

        event = PDMEvent(
            event_type="graph.edge_added",
            run_id="test",
            payload={
                "source": "A",
                "target": "A",  # Self-loop
                "relation": "contributes_to",
            },
        )

        await event_bus.publish(event)
        await asyncio.sleep(0.1)

        assert len(detector.detected_contradictions) == 1
        assert len(contradictions) == 1
        assert contradictions[0].event_type == "contradiction.detected"


# ============================================================================
# EVIDENCE STREAM TESTS
# ============================================================================


class TestEvidenceStream:
    """Test EvidenceStream functionality."""

    def test_initialization(self, sample_chunks):
        """Test stream initialization."""
        stream = EvidenceStream(sample_chunks)

        assert len(stream.chunks) == 5
        assert stream.current_idx == 0

    @pytest.mark.asyncio
    async def test_iteration(self, sample_chunks):
        """Test async iteration."""
        stream = EvidenceStream(sample_chunks)

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert len(chunks) == 5
        assert stream.current_idx == 5

    @pytest.mark.asyncio
    async def test_early_termination(self, sample_chunks):
        """Test breaking out of iteration early."""
        stream = EvidenceStream(sample_chunks)

        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
            if len(chunks) >= 3:
                break

        assert len(chunks) == 3
        assert stream.remaining() == 2

    def test_reset(self, sample_chunks):
        """Test resetting stream."""
        stream = EvidenceStream(sample_chunks)
        stream.current_idx = 3

        stream.reset()

        assert stream.current_idx == 0

    def test_progress(self, sample_chunks):
        """Test progress calculation."""
        stream = EvidenceStream(sample_chunks)

        assert stream.progress() == pytest.approx(
            0.0, rel=1e-9, abs=1e-12
        )  # replaced float equality with pytest.approx

        stream.current_idx = 2
        assert stream.progress() == pytest.approx(
            0.4, rel=1e-9, abs=1e-12
        )  # replaced float equality with pytest.approx

        stream.current_idx = 5
        assert stream.progress() == pytest.approx(
            1.0, rel=1e-9, abs=1e-12
        )  # replaced float equality with pytest.approx

    def test_remaining(self, sample_chunks):
        """Test remaining count."""
        stream = EvidenceStream(sample_chunks)

        assert stream.remaining() == 5

        stream.current_idx = 3
        assert stream.remaining() == 2


class TestMechanismPrior:
    """Test MechanismPrior model."""

    def test_initialization(self):
        """Test prior initialization."""
        prior = MechanismPrior(
            mechanism_name="test_mechanism",
            prior_mean=0.6,
            prior_std=0.15,
            confidence=0.7,
        )

        assert prior.mechanism_name == "test_mechanism"
        assert prior.prior_mean == pytest.approx(
            0.6, rel=1e-9, abs=1e-12
        )  # replaced float equality with pytest.approx
        assert prior.prior_std == pytest.approx(
            0.15, rel=1e-9, abs=1e-12
        )  # replaced float equality with pytest.approx
        assert prior.confidence == pytest.approx(
            0.7, rel=1e-9, abs=1e-12
        )  # replaced float equality with pytest.approx

    def test_to_dict(self):
        """Test converting to dictionary."""
        prior = MechanismPrior("test", 0.5, 0.2, 0.5)
        d = prior.to_dict()

        assert d["mechanism_name"] == "test"
        assert d["prior_mean"] == pytest.approx(
            0.5, rel=1e-9, abs=1e-12
        )  # replaced float equality with pytest.approx
        assert d["prior_std"] == pytest.approx(
            0.2, rel=1e-9, abs=1e-12
        )  # replaced float equality with pytest.approx


class TestPosteriorDistribution:
    """Test PosteriorDistribution model."""

    def test_initialization(self):
        """Test posterior initialization."""
        posterior = PosteriorDistribution(
            mechanism_name="test",
            posterior_mean=0.7,
            posterior_std=0.1,
            evidence_count=5,
        )

        assert posterior.mechanism_name == "test"
        assert posterior.posterior_mean == pytest.approx(
            0.7, rel=1e-9, abs=1e-12
        )  # replaced float equality with pytest.approx
        assert posterior.evidence_count == 5

    def test_confidence_computation(self):
        """Test confidence level computation."""
        # Very strong
        p1 = PosteriorDistribution("test", 0.8, 0.03, 15)
        assert p1._compute_confidence() == "very_strong"

        # Strong
        p2 = PosteriorDistribution("test", 0.8, 0.08, 7)
        assert p2._compute_confidence() == "strong"

        # Moderate
        p3 = PosteriorDistribution("test", 0.8, 0.15, 4)
        assert p3._compute_confidence() == "moderate"

        # Weak
        p4 = PosteriorDistribution("test", 0.8, 0.3, 2)
        assert p4._compute_confidence() == "weak"


class TestStreamingBayesianUpdater:
    """Test StreamingBayesianUpdater functionality."""

    def test_initialization(self):
        """Test updater initialization."""
        updater = StreamingBayesianUpdater()

        assert updater.event_bus is None
        assert updater.relevance_threshold == pytest.approx(
            0.6, rel=1e-9, abs=1e-12
        )  # replaced float equality with pytest.approx

    def test_initialization_with_bus(self, event_bus):
        """Test updater with event bus."""
        updater = StreamingBayesianUpdater(event_bus)

        assert updater.event_bus == event_bus

    @pytest.mark.asyncio
    async def test_update_from_stream(self, sample_chunks, mechanism_prior):
        """Test streaming Bayesian update."""
        updater = StreamingBayesianUpdater()
        stream = EvidenceStream(sample_chunks)

        posterior = await updater.update_from_stream(stream, mechanism_prior)

        assert isinstance(posterior, PosteriorDistribution)
        assert posterior.mechanism_name == "education_quality"
        assert posterior.evidence_count > 0
        # Posterior should be different from prior
        assert posterior.posterior_mean != mechanism_prior.prior_mean

    @pytest.mark.asyncio
    async def test_update_with_event_publishing(
        self, event_bus, sample_chunks, mechanism_prior
    ):
        """Test update with intermediate event publishing."""
        updater = StreamingBayesianUpdater(event_bus)
        stream = EvidenceStream(sample_chunks)

        published_events = []

        async def event_handler(event):
            published_events.append(event)

        event_bus.subscribe("posterior.updated", event_handler)

        posterior = await updater.update_from_stream(
            stream, mechanism_prior, run_id="test_run"
        )

        await asyncio.sleep(0.1)

        # Should have published events for relevant chunks
        assert len(published_events) > 0
        assert all(e.event_type == "posterior.updated" for e in published_events)

    @pytest.mark.asyncio
    async def test_is_relevant(self, sample_chunks):
        """Test relevance checking."""
        updater = StreamingBayesianUpdater()

        # Chunk with 'educación' should be relevant
        is_rel = await updater._is_relevant(sample_chunks[0], "education_quality")
        assert is_rel is True

        # Chunk without keywords should not be relevant
        chunk_irrelevant = {
            "chunk_id": "test",
            "content": "Este es un texto sobre algo completamente diferente.",
            "token_count": 10,
        }
        is_rel = await updater._is_relevant(chunk_irrelevant, "education_quality")
        assert is_rel is False

    @pytest.mark.asyncio
    async def test_compute_likelihood(self, sample_chunks):
        """Test likelihood computation."""
        updater = StreamingBayesianUpdater()

        likelihood = await updater._compute_likelihood(sample_chunks[0], "education")

        assert 0.0 <= likelihood <= 1.0

    def test_bayesian_update(self):
        """Test Bayesian update logic."""
        updater = StreamingBayesianUpdater()

        current = PosteriorDistribution(
            mechanism_name="test",
            posterior_mean=0.5,
            posterior_std=0.2,
            evidence_count=1,
        )

        # High likelihood should increase posterior mean
        updated = updater._bayesian_update(current, likelihood=0.9)

        assert updated.posterior_mean > current.posterior_mean
        assert updated.posterior_std < current.posterior_std


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Test integration between components."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self, event_bus, sample_chunks):
        """Test complete pipeline with event bus and streaming."""
        # Setup detector
        detector = ContradictionDetectorV2(event_bus)

        # Setup streaming updater
        updater = StreamingBayesianUpdater(event_bus)

        # Track all events
        all_events = []

        async def track_events(event):
            all_events.append(event)

        event_bus.subscribe("posterior.updated", track_events)
        event_bus.subscribe("contradiction.detected", track_events)

        # Run streaming update
        prior = MechanismPrior("education_quality", 0.5, 0.2, 0.5)
        stream = EvidenceStream(sample_chunks)

        posterior = await updater.update_from_stream(
            stream, prior, run_id="integration_test"
        )

        # Publish some graph events
        await event_bus.publish(
            PDMEvent(
                event_type="graph.edge_added",
                run_id="integration_test",
                payload={"source": "A", "target": "A"},  # Self-loop
            )
        )

        await asyncio.sleep(0.2)

        # Verify results
        assert len(all_events) > 0
        assert any(e.event_type == "posterior.updated" for e in all_events)
        assert any(e.event_type == "contradiction.detected" for e in all_events)
        assert len(detector.detected_contradictions) > 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
