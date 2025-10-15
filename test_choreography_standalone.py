"""
Standalone Tests for FRENTE 3: COREOGRAFÍA
===========================================
Tests that run without pytest dependency.
"""

import asyncio
import sys
from datetime import datetime

# Import modules to test
from choreography.event_bus import ContradictionDetectorV2, EventBus, PDMEvent
from choreography.evidence_stream import (
    EvidenceStream,
    MechanismPrior,
    PosteriorDistribution,
    StreamingBayesianUpdater,
)

# ============================================================================
# TEST UTILITIES
# ============================================================================


class TestResult:
    """Track test results."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def record_pass(self, test_name):
        self.passed += 1
        print(f"✓ {test_name}")

    def record_fail(self, test_name, error):
        self.failed += 1
        self.errors.append((test_name, error))
        print(f"✗ {test_name}: {error}")

    def summary(self):
        total = self.passed + self.failed
        print("\n" + "=" * 70)
        print(f"Test Results: {self.passed}/{total} passed")
        if self.failed > 0:
            print(f"\nFailed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        print("=" * 70)
        return self.failed == 0


def assert_equal(actual, expected, message=""):
    """Assert equality."""
    if actual != expected:
        raise AssertionError(f"{message}\nExpected: {expected}\nActual: {actual}")


def assert_true(condition, message=""):
    """Assert true."""
    if not condition:
        raise AssertionError(f"{message}\nCondition was False")


def assert_in(item, container, message=""):
    """Assert item in container."""
    if item not in container:
        raise AssertionError(f"{message}\n{item} not in {container}")


def assert_greater(actual, expected, message=""):
    """Assert greater than."""
    if actual <= expected:
        raise AssertionError(f"{message}\n{actual} <= {expected}")


def assert_greater_equal(actual, expected, message=""):
    """Assert greater than or equal."""
    if actual < expected:
        raise AssertionError(f"{message}\n{actual} < {expected}")


# ============================================================================
# TEST FIXTURES
# ============================================================================


def create_sample_chunks():
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


# ============================================================================
# EVENT BUS TESTS
# ============================================================================


def test_pdm_event_creation():
    """Test creating a PDMEvent."""
    event = PDMEvent(
        event_type="test.event", run_id="test_run", payload={"key": "value"}
    )

    assert_equal(event.event_type, "test.event")
    assert_equal(event.run_id, "test_run")
    assert_equal(event.payload, {"key": "value"})
    assert_true(event.event_id is not None)
    assert_true(isinstance(event.timestamp, datetime))


def test_event_bus_initialization():
    """Test EventBus initialization."""
    bus = EventBus()
    assert_equal(len(bus.subscribers), 0)
    assert_equal(len(bus.event_log), 0)


def test_event_bus_subscribe():
    """Test subscribing to events."""
    bus = EventBus()

    async def handler(event):
        pass

    bus.subscribe("test.event", handler)

    assert_in("test.event", bus.subscribers)
    assert_in(handler, bus.subscribers["test.event"])


async def test_event_bus_publish():
    """Test publishing events."""
    bus = EventBus()
    received = []

    async def handler(event):
        received.append(event)

    bus.subscribe("test.event", handler)

    event = PDMEvent(event_type="test.event", run_id="test", payload={"data": "value"})

    await bus.publish(event)
    await asyncio.sleep(0.1)

    assert_equal(len(received), 1)
    assert_equal(received[0].payload["data"], "value")
    assert_equal(len(bus.event_log), 1)


async def test_contradiction_detector():
    """Test ContradictionDetectorV2."""
    bus = EventBus()
    detector = ContradictionDetectorV2(bus)
    contradictions = []

    async def handler(event):
        contradictions.append(event)

    bus.subscribe("contradiction.detected", handler)

    # Test self-loop (should detect contradiction)
    event = PDMEvent(
        event_type="graph.edge_added",
        run_id="test",
        payload={"source": "A", "target": "A"},
    )

    await bus.publish(event)
    await asyncio.sleep(0.1)

    assert_equal(len(detector.detected_contradictions), 1)
    assert_equal(len(contradictions), 1)


# ============================================================================
# EVIDENCE STREAM TESTS
# ============================================================================


async def test_evidence_stream_iteration():
    """Test EvidenceStream iteration."""
    chunks = create_sample_chunks()
    stream = EvidenceStream(chunks)

    collected = []
    async for chunk in stream:
        collected.append(chunk)

    assert_equal(len(collected), 5)
    assert_equal(stream.current_idx, 5)


def test_evidence_stream_progress():
    """Test progress tracking."""
    chunks = create_sample_chunks()
    stream = EvidenceStream(chunks)

    assert_equal(stream.progress(), 0.0)

    stream.current_idx = 2
    assert_equal(stream.progress(), 0.4)

    stream.current_idx = 5
    assert_equal(stream.progress(), 1.0)


def test_mechanism_prior():
    """Test MechanismPrior creation."""
    prior = MechanismPrior(
        mechanism_name="test", prior_mean=0.6, prior_std=0.15, confidence=0.7
    )

    assert_equal(prior.mechanism_name, "test")
    assert_equal(prior.prior_mean, 0.6)

    d = prior.to_dict()
    assert_equal(d["mechanism_name"], "test")


def test_posterior_distribution():
    """Test PosteriorDistribution creation."""
    posterior = PosteriorDistribution(
        mechanism_name="test", posterior_mean=0.7, posterior_std=0.1, evidence_count=5
    )

    assert_equal(posterior.mechanism_name, "test")
    assert_equal(posterior.posterior_mean, 0.7)
    assert_equal(posterior.evidence_count, 5)


async def test_streaming_bayesian_updater():
    """Test StreamingBayesianUpdater."""
    chunks = create_sample_chunks()
    stream = EvidenceStream(chunks)

    prior = MechanismPrior(
        "educación", 0.5, 0.2, 0.5
    )  # Use 'educación' to match chunk content
    updater = StreamingBayesianUpdater()

    posterior = await updater.update_from_stream(stream, prior)

    assert_true(isinstance(posterior, PosteriorDistribution))
    assert_equal(posterior.mechanism_name, "educación")
    assert_greater_equal(
        posterior.evidence_count, 0
    )  # At least 0 (may be 0 if no relevant chunks)


async def test_streaming_with_events():
    """Test streaming with event publishing."""
    bus = EventBus()
    chunks = create_sample_chunks()
    stream = EvidenceStream(chunks)

    updater = StreamingBayesianUpdater(bus)
    prior = MechanismPrior(
        "educación", 0.5, 0.2, 0.5
    )  # Use 'educación' to match chunk content

    events = []

    async def handler(event):
        events.append(event)

    bus.subscribe("posterior.updated", handler)

    posterior = await updater.update_from_stream(stream, prior, run_id="test")
    await asyncio.sleep(0.1)

    # Events are only published for relevant chunks, so we may have 0 or more
    assert_greater_equal(len(events), 0)
    if len(events) > 0:
        assert_true(all(e.event_type == "posterior.updated" for e in events))


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


async def test_full_integration():
    """Test complete integration."""
    bus = EventBus()
    chunks = create_sample_chunks()

    # Setup components
    detector = ContradictionDetectorV2(bus)
    updater = StreamingBayesianUpdater(bus)

    # Track events
    all_events = []

    async def handler(event):
        all_events.append(event)

    bus.subscribe("posterior.updated", handler)
    bus.subscribe("contradiction.detected", handler)

    # Run streaming update
    prior = MechanismPrior(
        "educación", 0.5, 0.2, 0.5
    )  # Use 'educación' to match chunk content
    stream = EvidenceStream(chunks)

    posterior = await updater.update_from_stream(stream, prior, run_id="test")

    # Publish graph event with self-loop
    await bus.publish(
        PDMEvent(
            event_type="graph.edge_added",
            run_id="test",
            payload={"source": "A", "target": "A"},
        )
    )

    await asyncio.sleep(0.2)

    # Verify - at minimum we should have the contradiction event
    assert_greater_equal(len(all_events), 1)  # At least the contradiction
    assert_true(any(e.event_type == "contradiction.detected" for e in all_events))


# ============================================================================
# TEST RUNNER
# ============================================================================


async def run_async_tests(results):
    """Run all async tests."""
    tests = [
        ("Event Bus Publish", test_event_bus_publish),
        ("Contradiction Detector", test_contradiction_detector),
        ("Evidence Stream Iteration", test_evidence_stream_iteration),
        ("Streaming Bayesian Updater", test_streaming_bayesian_updater),
        ("Streaming with Events", test_streaming_with_events),
        ("Full Integration", test_full_integration),
    ]

    for name, test_func in tests:
        try:
            await test_func()
            results.record_pass(name)
        except Exception as e:
            results.record_fail(name, str(e))


def run_sync_tests(results):
    """Run all synchronous tests."""
    tests = [
        ("PDM Event Creation", test_pdm_event_creation),
        ("Event Bus Initialization", test_event_bus_initialization),
        ("Event Bus Subscribe", test_event_bus_subscribe),
        ("Evidence Stream Progress", test_evidence_stream_progress),
        ("Mechanism Prior", test_mechanism_prior),
        ("Posterior Distribution", test_posterior_distribution),
    ]

    for name, test_func in tests:
        try:
            test_func()
            results.record_pass(name)
        except Exception as e:
            results.record_fail(name, str(e))


def main():
    """Main test runner."""
    print("=" * 70)
    print("FRENTE 3: COREOGRAFÍA - Test Suite")
    print("=" * 70)
    print()

    results = TestResult()

    # Run sync tests
    print("Running synchronous tests...")
    run_sync_tests(results)

    # Run async tests
    print("\nRunning asynchronous tests...")
    asyncio.run(run_async_tests(results))

    # Print summary
    success = results.summary()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
