"""Two ways to move messages: a real broker, and an in-process loopback.

The loopback exists so the pipeline can be tested and demonstrated without installing Mosquitto.
It implements the same three methods, so `line.py` and `gate.py` never learn which one they have
-- and the tests exercise the production code path rather than a parallel fake of it.

What the real transport adds is the part SSE cannot do: a last will. The broker holds a message
on the cell's status topic and publishes it *for* us if the connection drops without a clean
disconnect. A crashed line therefore announces itself. An HTTP stream that stops just stops, and
every consumer has to invent its own timeout to notice.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from . import topics
from .codec import encode

Handler = Callable[[str, bytes], None]

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 1883
KEEPALIVE_S = 30
CONNECT_TIMEOUT_S = 5.0


class Transport(Protocol):
    """The whole surface the producer and consumer need."""

    def publish(self, topic: str, payload: dict, qos: int = 0, retain: bool = False) -> None: ...

    def subscribe(self, topic: str, handler: Handler, qos: int = 0) -> None: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class LoopbackTransport:
    """An in-memory broker: no network, no daemon, same interface.

    Wildcards are supported only at the level MQTT actually uses here (`+` for one segment),
    which is enough for `defectlab/+/telemetry` and not a general implementation.
    """

    handlers: dict[str, list[Handler]] = field(default_factory=lambda: defaultdict(list))
    published: list[tuple[str, dict]] = field(default_factory=list)
    retained: dict[str, dict] = field(default_factory=dict)

    def publish(self, topic: str, payload: dict, qos: int = 0, retain: bool = False) -> None:
        self.published.append((topic, payload))
        if retain:
            self.retained[topic] = payload
        for pattern, handlers in list(self.handlers.items()):
            if _matches(pattern, topic):
                for handler in handlers:
                    handler(topic, encode(payload))

    def subscribe(self, topic: str, handler: Handler, qos: int = 0) -> None:
        self.handlers[topic].append(handler)
        for retained, payload in self.retained.items():
            if _matches(topic, retained):
                handler(retained, encode(payload))

    def close(self) -> None:
        self.handlers.clear()


def _matches(pattern: str, topic: str) -> bool:
    if pattern == topic:
        return True
    expected, actual = pattern.split("/"), topic.split("/")
    if len(expected) != len(actual):
        return False
    return all(want in ("+", got) for want, got in zip(expected, actual, strict=True))


class MqttTransport:
    """paho-mqtt, connected and looping in a background thread."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        cell: str = topics.DEFAULT_CELL,
        client_id: str = "",
    ) -> None:
        from paho.mqtt import client as mqtt

        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2, client_id=client_id, protocol=mqtt.MQTTv5
        )
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._client.on_message = self._dispatch
        self._arm_will(cell)
        self._client.connect(host, port, keepalive=KEEPALIVE_S)
        self._client.loop_start()

    def _arm_will(self, cell: str) -> None:
        """Registered before connect: the broker cannot honour a will it was never told about."""
        self._client.will_set(
            topics.status(cell),
            encode({"state": "offline", "cell": cell, "detail": "connection lost"}),
            qos=topics.AT_LEAST_ONCE,
            retain=True,
        )

    def _dispatch(self, _client, _userdata, message) -> None:
        for pattern, handlers in self._handlers.items():
            if _matches(pattern, message.topic):
                for handler in handlers:
                    handler(message.topic, message.payload)

    def publish(self, topic: str, payload: dict, qos: int = 0, retain: bool = False) -> None:
        self._client.publish(topic, encode(payload), qos=qos, retain=retain)

    def subscribe(self, topic: str, handler: Handler, qos: int = 0) -> None:
        self._handlers[topic].append(handler)
        self._client.subscribe(topic, qos=qos)

    def close(self) -> None:
        """A clean disconnect suppresses the will, which is the point of distinguishing them."""
        self._client.loop_stop()
        self._client.disconnect()


def connect(host: str, port: int, cell: str, client_id: str = "") -> Transport:
    """A real broker if one answers, otherwise fail loudly rather than silently going nowhere."""
    return MqttTransport(host=host, port=port, cell=cell, client_id=client_id)
