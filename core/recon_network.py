from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set
from .recon_core import EXTERNAL_REQUEST_SENDER, Inbox, Msg, Node, NodeKind, ScriptState


@dataclass
class Network:
    nodes: Dict[str, Node] = field(default_factory=dict)
    tick: int = 0
    external_requests: Set[str] = field(default_factory=set)
    recon_mode: str = "compat"

    def add_node(self, node: Node) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"Duplicate node_id: {node.node_id}")
        self.nodes[node.node_id] = node

    def get(self, node_id: str) -> Node:
        return self.nodes[node_id]

    def connect_parent_child(self, parent_id: str, child_id: str) -> None:
        p = self.get(parent_id)
        c = self.get(child_id)
        c.parent = parent_id
        if child_id not in p.children:
            p.children.append(child_id)

    def connect_sequence(self, predecessor_id: str, successor_id: str) -> None:
        pre = self.get(predecessor_id)
        suc = self.get(successor_id)
        pre.successor = successor_id
        suc.predecessor = predecessor_id

    def set_sensor_value(self, sensor_id: str, value: float) -> None:
        n = self.get(sensor_id)
        if n.kind != NodeKind.SENSOR:
            raise ValueError("set_sensor_value called on non-sensor")
        n.value = float(value)

    def set_external_request(self, node_id: str, on: bool = True) -> None:
        if on:
            self.external_requests.add(node_id)
        else:
            self.external_requests.discard(node_id)

    def set_root_requested(self, on: bool = True) -> None:
        for nid, node in self.nodes.items():
            if node.parent is None:
                self.set_external_request(nid, on)

    def _strict_fsm_enabled(self) -> bool:
        for node in self.nodes.values():
            if node.config.strict_fsm:
                return True
        return False

    def start_root(self, on: bool = True) -> None:
        if self._strict_fsm_enabled():
            self.set_root_requested(on)
            return
        for node in self.nodes.values():
            if node.parent is None and node.kind == NodeKind.SCRIPT:
                node.script_state = ScriptState.ACTIVE if on else ScriptState.INACTIVE

    def _inboxes_from_emit(self) -> Dict[str, Inbox]:
        inboxes = {nid: Inbox() for nid in self.nodes}
        for nid, node in self.nodes.items():
            for dst, msg in node.emit_messages():
                inboxes[dst].add(nid, msg)
        return inboxes

    def step(self) -> Dict[str, Inbox]:
        inboxes = self._inboxes_from_emit()
        for nid in self.external_requests:
            if nid in inboxes:
                inboxes[nid].add(EXTERNAL_REQUEST_SENDER, Msg.REQUEST)
        for nid, node in self.nodes.items():
            node.update(inboxes[nid], tick=self.tick)
        self.tick += 1
        return inboxes
