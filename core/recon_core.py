from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Set, Tuple
import math
import os


class Msg(Enum):
    REQUEST = auto()
    CONFIRM = auto()
    WAIT = auto()
    INHIBIT_REQUEST = auto()
    INHIBIT_CONFIRM = auto()
    FAIL = auto()


EXTERNAL_REQUEST_SENDER = "__external_request__"


@dataclass
class ReCoNConfig:
    strict_table1: bool = False
    strict_fsm: bool = False
    strict_terminal: bool = False


def _env_flag(name: str) -> bool:
    val = os.getenv(name, "")
    return val.strip().lower() in ("1", "true", "yes", "on")


def config_from_env() -> ReCoNConfig:
    if os.getenv("RECON_STRICT") is None:
        return ReCoNConfig(strict_table1=True, strict_fsm=True, strict_terminal=True)
    if _env_flag("RECON_STRICT"):
        return ReCoNConfig(strict_table1=True, strict_fsm=True, strict_terminal=True)
    return ReCoNConfig(
        strict_table1=_env_flag("RECON_STRICT_TABLE1"),
        strict_fsm=_env_flag("RECON_STRICT_FSM"),
        strict_terminal=_env_flag("RECON_STRICT_TERMINAL"),
    )


class ScriptState(Enum):
    INACTIVE = auto()
    REQUESTED = auto()   # retained for readability; not required by the accelerated semantics
    ACTIVE = auto()
    SUPPRESSED = auto()
    WAITING = auto()
    TRUE = auto()
    CONFIRMED = auto()
    FAILED = auto()


class TerminalState(Enum):
    INACTIVE = auto()
    ACTIVE = auto()
    CONFIRMED = auto()
    FAILED = auto()


class NodeKind(Enum):
    SCRIPT = auto()
    SENSOR = auto()
    ACTUATOR = auto()


@dataclass
class Inbox:
    by_sender: Dict[str, Set[Msg]] = field(default_factory=dict)

    def add(self, sender_id: str, msg: Msg) -> None:
        self.by_sender.setdefault(sender_id, set()).add(msg)

    def from_sender(self, sender_id: Optional[str]) -> Set[Msg]:
        if sender_id is None:
            return set()
        return self.by_sender.get(sender_id, set())


@dataclass
class Node:
    node_id: str
    kind: NodeKind

    # Relations
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    predecessor: Optional[str] = None
    successor: Optional[str] = None

    # State
    script_state: ScriptState = ScriptState.INACTIVE
    terminal_state: TerminalState = TerminalState.INACTIVE
    fail_grace_ticks_remaining: int = 0

    # Continuous scalars
    activation: float = 0.0
    value: float = 0.0
    threshold: float = 0.5

    # Actuator effect
    actuator_effect: Optional[Callable[[float], None]] = None

    config: ReCoNConfig = field(default_factory=ReCoNConfig)

    def is_script(self) -> bool:
        return self.kind == NodeKind.SCRIPT

    def state_repr(self) -> str:
        return self.script_state.name if self.is_script() else self.terminal_state.name

    # -------------------------
    # Message emission
    # -------------------------
    def emit_messages(self) -> List[Tuple[str, Msg]]:
        out: List[Tuple[str, Msg]] = []

        if self.kind == NodeKind.SCRIPT:
            s = self.script_state

            if self.config.strict_table1:
                if s == ScriptState.REQUESTED:
                    if self.successor is not None:
                        out.append((self.successor, Msg.INHIBIT_REQUEST))
                    if self.predecessor is not None:
                        out.append((self.predecessor, Msg.INHIBIT_CONFIRM))
                    if self.parent is not None:
                        out.append((self.parent, Msg.WAIT))

                elif s in (ScriptState.ACTIVE, ScriptState.WAITING):
                    if self.successor is not None:
                        out.append((self.successor, Msg.INHIBIT_REQUEST))
                    if self.predecessor is not None:
                        out.append((self.predecessor, Msg.INHIBIT_CONFIRM))
                    for c in self.children:
                        out.append((c, Msg.REQUEST))
                    if self.parent is not None:
                        out.append((self.parent, Msg.WAIT))

                elif s in (ScriptState.SUPPRESSED, ScriptState.FAILED):
                    if self.successor is not None:
                        out.append((self.successor, Msg.INHIBIT_REQUEST))
                    if self.predecessor is not None:
                        out.append((self.predecessor, Msg.INHIBIT_CONFIRM))

                elif s == ScriptState.TRUE:
                    if self.predecessor is not None:
                        out.append((self.predecessor, Msg.INHIBIT_CONFIRM))

                elif s == ScriptState.CONFIRMED:
                    if self.predecessor is not None:
                        out.append((self.predecessor, Msg.INHIBIT_CONFIRM))
                    if self.parent is not None:
                        out.append((self.parent, Msg.CONFIRM))

            else:
                if s in (ScriptState.ACTIVE, ScriptState.WAITING):
                    # hold parent
                    if self.parent is not None:
                        out.append((self.parent, Msg.WAIT))
                    # drive children
                    for c in self.children:
                        out.append((c, Msg.REQUEST))
                    # sequence inhibition (optional)
                    if self.successor is not None:
                        out.append((self.successor, Msg.INHIBIT_REQUEST))
                    if self.predecessor is not None:
                        out.append((self.predecessor, Msg.INHIBIT_CONFIRM))

                elif s == ScriptState.TRUE:
                    # TRUE releases successor request inhibition, but still blocks predecessor confirmation
                    if self.predecessor is not None:
                        out.append((self.predecessor, Msg.INHIBIT_CONFIRM))

                elif s == ScriptState.CONFIRMED:
                    if self.parent is not None:
                        out.append((self.parent, Msg.CONFIRM))
                    if self.predecessor is not None:
                        out.append((self.predecessor, Msg.INHIBIT_CONFIRM))

                elif s == ScriptState.FAILED:
                    if self.parent is not None:
                        out.append((self.parent, Msg.FAIL))

        else:
            ts = self.terminal_state
            strict_terminal = self.config.strict_terminal or self.config.strict_fsm
            if strict_terminal:
                if ts == TerminalState.ACTIVE and self.parent is not None:
                    out.append((self.parent, Msg.WAIT))
                elif ts == TerminalState.CONFIRMED and self.parent is not None:
                    out.append((self.parent, Msg.CONFIRM))
            else:
                if ts == TerminalState.CONFIRMED and self.parent is not None:
                    out.append((self.parent, Msg.CONFIRM))
                elif ts == TerminalState.FAILED and self.parent is not None:
                    out.append((self.parent, Msg.FAIL))

        return out

    # -------------------------
    # δ transitions
    # -------------------------
    def update(self, inbox: Inbox, tick: int) -> None:
        if self.kind == NodeKind.SCRIPT:
            self._update_script(inbox, tick)
        elif self.kind == NodeKind.SENSOR:
            self._update_sensor(inbox)
        elif self.kind == NodeKind.ACTUATOR:
            self._update_actuator(inbox)
        else:
            raise ValueError("Unknown node kind")

    def _update_script(self, inbox: Inbox, tick: int) -> None:
        s = self.script_state
        parent_msgs = inbox.from_sender(self.parent)
        pred_msgs = inbox.from_sender(self.predecessor)
        succ_msgs = inbox.from_sender(self.successor)
        external_msgs = inbox.from_sender(EXTERNAL_REQUEST_SENDER)

        requested = (Msg.REQUEST in parent_msgs) or (Msg.REQUEST in external_msgs)
        inhibited = (Msg.INHIBIT_REQUEST in pred_msgs)

        if self.config.strict_fsm:
            r = requested
            ir = (Msg.INHIBIT_REQUEST in pred_msgs)
            ic = (Msg.INHIBIT_CONFIRM in succ_msgs)
            grace = int(self.fail_grace_ticks_remaining)
            has_children = bool(self.children)

            def grace_active() -> bool:
                return grace > 0 and has_children and r

            def any_child_msg(msg: Msg) -> bool:
                for c in self.children:
                    if msg in inbox.from_sender(c):
                        return True
                return False

            w = any_child_msg(Msg.WAIT)
            c = any_child_msg(Msg.CONFIRM)
            prev_state = s
            new_state = s

            if s == ScriptState.INACTIVE:
                new_state = ScriptState.REQUESTED if r else ScriptState.INACTIVE

            elif s == ScriptState.REQUESTED:
                if not r:
                    new_state = ScriptState.INACTIVE
                elif ir:
                    new_state = ScriptState.SUPPRESSED
                else:
                    new_state = ScriptState.ACTIVE

            elif s == ScriptState.SUPPRESSED:
                if not r:
                    new_state = ScriptState.INACTIVE
                elif ir:
                    new_state = ScriptState.SUPPRESSED
                else:
                    new_state = ScriptState.ACTIVE

            elif s == ScriptState.ACTIVE:
                if not r:
                    new_state = ScriptState.INACTIVE
                elif w:
                    new_state = ScriptState.WAITING
                elif grace_active():
                    new_state = ScriptState.ACTIVE
                else:
                    new_state = ScriptState.FAILED

            elif s == ScriptState.WAITING:
                if not r:
                    new_state = ScriptState.INACTIVE
                elif c and ic:
                    new_state = ScriptState.TRUE
                elif c and (not ic):
                    new_state = ScriptState.CONFIRMED
                elif w:
                    new_state = ScriptState.WAITING
                elif grace_active():
                    new_state = ScriptState.WAITING
                else:
                    new_state = ScriptState.FAILED

            elif s == ScriptState.FAILED:
                if not r:
                    new_state = ScriptState.INACTIVE
                else:
                    new_state = ScriptState.FAILED

            elif s == ScriptState.TRUE:
                if not r:
                    new_state = ScriptState.INACTIVE
                else:
                    new_state = ScriptState.TRUE

            elif s == ScriptState.CONFIRMED:
                if not r:
                    new_state = ScriptState.INACTIVE
                else:
                    new_state = ScriptState.CONFIRMED

            self.script_state = new_state

            if new_state == ScriptState.ACTIVE and prev_state not in (ScriptState.ACTIVE, ScriptState.WAITING) and has_children and r:
                grace = max(grace, 1)

            if grace > 0 and has_children and r and prev_state in (ScriptState.ACTIVE, ScriptState.WAITING):
                grace -= 1

            if new_state in (ScriptState.INACTIVE, ScriptState.TRUE, ScriptState.CONFIRMED, ScriptState.FAILED):
                grace = 0

            self.fail_grace_ticks_remaining = grace

            if self.script_state == ScriptState.INACTIVE:
                self.activation = 0.0
            return

        def all_children_confirmed() -> bool:
            for c in self.children:
                if Msg.CONFIRM not in inbox.from_sender(c):
                    return False
            return True

        def any_child_failed() -> bool:
            for c in self.children:
                if Msg.FAIL in inbox.from_sender(c):
                    return True
            return False

        # IMPORTANT: request-drop reset for non-root scripts
        # (Root has parent=None and is manually started.)
        if self.parent is not None and (not requested):
            if s != ScriptState.INACTIVE:
                self.script_state = ScriptState.INACTIVE
                self.activation = 0.0
            return

        # Accelerated handshake: INACTIVE -> ACTIVE directly when requested
        if s == ScriptState.INACTIVE:
            if requested:
                self.script_state = ScriptState.SUPPRESSED if inhibited else ScriptState.ACTIVE

        elif s == ScriptState.SUPPRESSED:
            if requested and (not inhibited):
                self.script_state = ScriptState.ACTIVE

        elif s == ScriptState.ACTIVE:
            # one tick to issue requests; then wait
            self.script_state = ScriptState.WAITING

        elif s == ScriptState.WAITING:
            if any_child_failed():
                self.script_state = ScriptState.FAILED
            elif all_children_confirmed():
                self.script_state = ScriptState.TRUE

        elif s == ScriptState.TRUE:
            if self.successor is None:
                self.script_state = ScriptState.CONFIRMED
            else:
                if Msg.CONFIRM in succ_msgs:
                    self.script_state = ScriptState.CONFIRMED

        elif s == ScriptState.CONFIRMED:
            # Re-entrant semantics: if request persists, restart after emitting CONFIRM for one tick.
            # (CONFIRM is emitted pre-update.)
            if requested:
                self.script_state = ScriptState.SUPPRESSED if inhibited else ScriptState.ACTIVE
            else:
                self.script_state = ScriptState.INACTIVE
                self.activation = 0.0

        elif s == ScriptState.FAILED:
            if requested:
                self.script_state = ScriptState.SUPPRESSED if inhibited else ScriptState.ACTIVE
            else:
                self.script_state = ScriptState.INACTIVE
                self.activation = 0.0

        elif s == ScriptState.REQUESTED:
            # kept for compatibility; treat as ACTIVE
            self.script_state = ScriptState.SUPPRESSED if inhibited else ScriptState.ACTIVE

    def _update_sensor(self, inbox: Inbox) -> None:
        """
        Sensors WAIT while requested; they do NOT immediately FAIL.
        They CONFIRM once value crosses threshold.
        """
        parent_msgs = inbox.from_sender(self.parent)
        external_msgs = inbox.from_sender(EXTERNAL_REQUEST_SENDER)
        requested = (Msg.REQUEST in parent_msgs) or (Msg.REQUEST in external_msgs)

        self.activation = float(self.value)

        if not requested:
            self.terminal_state = TerminalState.INACTIVE
            return

        if self.terminal_state == TerminalState.INACTIVE:
            self.terminal_state = TerminalState.ACTIVE

        if self.terminal_state == TerminalState.ACTIVE:
            if self.value >= self.threshold:
                self.terminal_state = TerminalState.CONFIRMED

        # CONFIRMED persists until request ends (handled above)

    def _update_actuator(self, inbox: Inbox) -> None:
        """
        Actuators are continuous while requested:
          - INACTIVE -> ACTIVE on request
          - ACTIVE executes effect, enters CONFIRMED
          - CONFIRMED continues executing effect while request persists
        """
        parent_msgs = inbox.from_sender(self.parent)
        external_msgs = inbox.from_sender(EXTERNAL_REQUEST_SENDER)
        requested = (Msg.REQUEST in parent_msgs) or (Msg.REQUEST in external_msgs)

        if not requested:
            self.terminal_state = TerminalState.INACTIVE
            self.activation = 0.0
            return

        if self.terminal_state == TerminalState.INACTIVE:
            self.terminal_state = TerminalState.ACTIVE

        if self.terminal_state == TerminalState.ACTIVE:
            if self.actuator_effect is not None:
                self.actuator_effect(self.activation)
            self.terminal_state = TerminalState.CONFIRMED
            return

        if self.terminal_state == TerminalState.CONFIRMED:
            if self.actuator_effect is not None:
                self.actuator_effect(self.activation)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))
