#!/usr/bin/env python3
"""
Quick validation test for refactored event_bus.py
"""

import asyncio
import sys

from choreography.event_bus import EventBus, PDMEvent


async def test_basic_publish():
    """Test basic publish-subscribe with refactored code."""
    bus = EventBus()
    received = []

    async def handler(event):
        received.append(event)

    bus.subscribe("test.event", handler)

    event = PDMEvent(
        event_type="test.event", run_id="test_run", payload={"key": "value"}
    )

    await bus.publish(event)
    await asyncio.sleep(0.1)

    assert len(received) == 1, f"Expected 1 event, got {len(received)}"
    assert received[0].payload["key"] == "value"
    print("✓ Basic publish-subscribe works")


async def test_multiple_handlers():
    """Test multiple handlers with refactored code."""
    bus = EventBus()
    received1 = []
    received2 = []

    async def handler1(event):
        received1.append(event)

    async def handler2(event):
        received2.append(event)

    bus.subscribe("test.event", handler1)
    bus.subscribe("test.event", handler2)

    event = PDMEvent(
        event_type="test.event", run_id="test_run", payload={"data": "test"}
    )

    await bus.publish(event)
    await asyncio.sleep(0.1)

    assert len(received1) == 1, f"Handler1: Expected 1 event, got {len(received1)}"
    assert len(received2) == 1, f"Handler2: Expected 1 event, got {len(received2)}"
    print("✓ Multiple handlers work correctly")


async def test_error_handling():
    """Test error handling with refactored code."""
    bus = EventBus()
    received_good = []

    async def failing_handler(event):
        raise ValueError("Intentional error")

    async def good_handler(event):
        received_good.append(event)

    bus.subscribe("test.event", failing_handler)
    bus.subscribe("test.event", good_handler)

    event = PDMEvent(
        event_type="test.event", run_id="test_run", payload={"test": "error_handling"}
    )

    await bus.publish(event)
    await asyncio.sleep(0.1)

    # Good handler should still execute even if failing handler raises
    assert len(received_good) == 1, (
        f"Expected good handler to run, got {len(received_good)}"
    )
    print("✓ Error handling preserves other handlers")


async def test_sync_handler():
    """Test synchronous handler support."""
    bus = EventBus()
    received = []

    def sync_handler(event):
        received.append(event)

    bus.subscribe("test.event", sync_handler)

    event = PDMEvent(
        event_type="test.event", run_id="test_run", payload={"sync": "test"}
    )

    await bus.publish(event)
    await asyncio.sleep(0.1)

    assert len(received) == 1, f"Expected 1 event, got {len(received)}"
    print("✓ Synchronous handler works")


async def test_event_types():
    """Test specific event types mentioned in requirements."""
    bus = EventBus()
    events = []

    async def handler(event):
        events.append(event.event_type)

    # Subscribe to the specific event types
    bus.subscribe("graph.edge_added", handler)
    bus.subscribe("contradiction.detected", handler)
    bus.subscribe("posterior.updated", handler)

    # Publish each event type
    await bus.publish(
        PDMEvent(
            event_type="graph.edge_added",
            run_id="test",
            payload={"source": "A", "target": "B"},
        )
    )

    await bus.publish(
        PDMEvent(
            event_type="contradiction.detected",
            run_id="test",
            payload={"severity": "high"},
        )
    )

    await bus.publish(
        PDMEvent(event_type="posterior.updated", run_id="test", payload={"mean": 0.75})
    )

    await asyncio.sleep(0.1)

    assert len(events) == 3, f"Expected 3 events, got {len(events)}"
    assert "graph.edge_added" in events
    assert "contradiction.detected" in events
    assert "posterior.updated" in events
    print("✓ All required event types work")


async def main():
    """Run all tests."""
    try:
        await test_basic_publish()
        await test_multiple_handlers()
        await test_error_handling()
        await test_sync_handler()
        await test_event_types()
        print("\n✅ All tests passed!")
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
