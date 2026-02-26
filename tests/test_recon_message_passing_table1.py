import pytest

from core.recon_core import Msg, Node, NodeKind, ReCoNConfig, ScriptState, TerminalState
from core.recon_network import Network


def _strict_script() -> ReCoNConfig:
    return ReCoNConfig(strict_table1=True, strict_fsm=True)


def _strict_terminal() -> ReCoNConfig:
    return ReCoNConfig(strict_terminal=True)


def _table1_script_node() -> Node:
    n = Node(node_id="U", kind=NodeKind.SCRIPT, config=ReCoNConfig(strict_table1=True))
    n.parent = "P"
    n.predecessor = "PRE"
    n.successor = "SUC"
    n.children = ["C1", "C2"]
    return n


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (ScriptState.INACTIVE, set()),
        (
            ScriptState.REQUESTED,
            {
                ("SUC", Msg.INHIBIT_REQUEST),
                ("PRE", Msg.INHIBIT_CONFIRM),
                ("P", Msg.WAIT),
            },
        ),
        (
            ScriptState.ACTIVE,
            {
                ("SUC", Msg.INHIBIT_REQUEST),
                ("PRE", Msg.INHIBIT_CONFIRM),
                ("C1", Msg.REQUEST),
                ("C2", Msg.REQUEST),
                ("P", Msg.WAIT),
            },
        ),
        (
            ScriptState.SUPPRESSED,
            {
                ("SUC", Msg.INHIBIT_REQUEST),
                ("PRE", Msg.INHIBIT_CONFIRM),
            },
        ),
        (
            ScriptState.WAITING,
            {
                ("SUC", Msg.INHIBIT_REQUEST),
                ("PRE", Msg.INHIBIT_CONFIRM),
                ("C1", Msg.REQUEST),
                ("C2", Msg.REQUEST),
                ("P", Msg.WAIT),
            },
        ),
        (ScriptState.TRUE, {("PRE", Msg.INHIBIT_CONFIRM)}),
        (
            ScriptState.CONFIRMED,
            {
                ("PRE", Msg.INHIBIT_CONFIRM),
                ("P", Msg.CONFIRM),
            },
        ),
        (
            ScriptState.FAILED,
            {
                ("SUC", Msg.INHIBIT_REQUEST),
                ("PRE", Msg.INHIBIT_CONFIRM),
            },
        ),
    ],
)
def test_table1_emission_script_states(state: ScriptState, expected) -> None:
    n = _table1_script_node()
    n.script_state = state
    assert set(n.emit_messages()) == expected


def test_table1_no_inhibit_request_without_successor() -> None:
    n = _table1_script_node()
    n.successor = None
    n.script_state = ScriptState.ACTIVE
    msgs = set(n.emit_messages())
    assert ("SUC", Msg.INHIBIT_REQUEST) not in msgs


def test_table1_no_inhibit_confirm_without_predecessor() -> None:
    n = _table1_script_node()
    n.predecessor = None
    n.script_state = ScriptState.WAITING
    msgs = set(n.emit_messages())
    assert ("PRE", Msg.INHIBIT_CONFIRM) not in msgs


def test_table1_no_parent_messages_for_root() -> None:
    n = _table1_script_node()
    n.parent = None
    n.script_state = ScriptState.CONFIRMED
    msgs = set(n.emit_messages())
    assert ("P", Msg.CONFIRM) not in msgs

    n.script_state = ScriptState.ACTIVE
    msgs = set(n.emit_messages())
    assert ("P", Msg.WAIT) not in msgs


def _build_fsm_network(with_successor: bool, terminal_wait: bool = True) -> Network:
    net = Network()

    p = Node(node_id="P", kind=NodeKind.SCRIPT, config=ReCoNConfig(strict_table1=True))
    u = Node(node_id="U", kind=NodeKind.SCRIPT, config=_strict_script())
    x_cfg = _strict_terminal() if terminal_wait else ReCoNConfig()
    x = Node(node_id="X", kind=NodeKind.SENSOR, config=x_cfg)

    net.add_node(p)
    net.add_node(u)
    net.add_node(x)
    net.connect_parent_child("P", "U")
    net.connect_parent_child("U", "X")

    if with_successor:
        s = Node(node_id="S", kind=NodeKind.SCRIPT, config=_strict_script())
        net.add_node(s)
        net.connect_sequence("U", "S")
        net.set_external_request("S", True)

    net.set_external_request("P", True)
    return net


def test_strict_fsm_wait_to_true_with_inhibit_confirm() -> None:
    net = _build_fsm_network(with_successor=True)
    x = net.get("X")
    x.threshold = 1.0
    x.value = 0.0

    states = []
    for _ in range(5):
        net.step()
        states.append(net.get("U").script_state)
    assert states == [
        ScriptState.INACTIVE,
        ScriptState.REQUESTED,
        ScriptState.ACTIVE,
        ScriptState.ACTIVE,
        ScriptState.WAITING,
    ]

    x.value = 1.0
    net.step()  # X becomes CONFIRMED
    net.step()  # U sees CONFIRM with ic=True
    assert net.get("U").script_state == ScriptState.TRUE


def test_strict_fsm_wait_to_confirm_without_successor() -> None:
    net = _build_fsm_network(with_successor=False)
    x = net.get("X")
    x.threshold = 1.0
    x.value = 0.0

    for _ in range(4):
        net.step()
    assert net.get("U").script_state == ScriptState.ACTIVE

    net.step()
    assert net.get("U").script_state == ScriptState.WAITING

    x.value = 1.0
    net.step()
    net.step()
    assert net.get("U").script_state == ScriptState.CONFIRMED


def test_strict_fsm_inhibit_request_suppresses_then_releases() -> None:
    net = _build_fsm_network(with_successor=False)
    pre = Node(node_id="PRE", kind=NodeKind.SCRIPT, config=_strict_script())
    net.add_node(pre)
    net.connect_sequence("PRE", "U")
    net.set_external_request("PRE", True)

    for _ in range(2):
        net.step()
    assert net.get("U").script_state == ScriptState.REQUESTED

    net.step()
    assert net.get("U").script_state == ScriptState.SUPPRESSED

    net.get("U").predecessor = None
    pre.successor = None
    net.step()
    assert net.get("U").script_state == ScriptState.ACTIVE


def test_strict_fsm_fails_without_wait_or_confirm() -> None:
    net = _build_fsm_network(with_successor=False, terminal_wait=False)
    x = net.get("X")
    x.threshold = 1.0
    x.value = 0.0

    states = []
    for _ in range(5):
        net.step()
        states.append(net.get("U").script_state)
    assert states[-2:] == [ScriptState.ACTIVE, ScriptState.FAILED]


def test_strict_terminal_emits_wait() -> None:
    net = Network()
    p = Node(node_id="P", kind=NodeKind.SCRIPT, config=ReCoNConfig(strict_table1=True))
    y = Node(node_id="Y", kind=NodeKind.SENSOR, config=_strict_terminal())
    y.threshold = 1.0
    y.value = 0.0

    net.add_node(p)
    net.add_node(y)
    net.connect_parent_child("P", "Y")
    net.set_external_request("P", True)

    wait_ticks = 0
    for _ in range(6):
        prev_state = y.terminal_state
        inboxes = net.step()
        if prev_state == TerminalState.ACTIVE:
            wait_ticks += 1
            assert Msg.WAIT in inboxes["P"].from_sender("Y")
    assert wait_ticks > 0


def test_cease_activity_when_request_removed() -> None:
    net = Network()
    cfg = ReCoNConfig(strict_table1=True, strict_fsm=True, strict_terminal=True)
    p = Node(node_id="P", kind=NodeKind.SCRIPT, config=cfg)
    u = Node(node_id="U", kind=NodeKind.SCRIPT, config=cfg)
    x = Node(node_id="X", kind=NodeKind.SENSOR, config=cfg)

    net.add_node(p)
    net.add_node(u)
    net.add_node(x)
    net.connect_parent_child("P", "U")
    net.connect_parent_child("U", "X")

    net.set_external_request("P", True)

    # Allow depth+1 ticks for request removal to propagate to X via P -> U -> X.
    for _ in range(5):
        net.step()

    net.set_external_request("P", False)

    for _ in range(5):
        net.step()

    assert net.get("P").script_state == ScriptState.INACTIVE
    assert net.get("U").script_state == ScriptState.INACTIVE
    assert net.get("X").terminal_state == TerminalState.INACTIVE


def test_strict_root_only_request_grace_allows_wait() -> None:
    net = Network()
    cfg = ReCoNConfig(strict_table1=True, strict_fsm=True, strict_terminal=True)
    root = Node(node_id="Root", kind=NodeKind.SCRIPT, config=cfg)
    child = Node(node_id="Child", kind=NodeKind.SCRIPT, config=cfg)
    leaf = Node(node_id="Leaf", kind=NodeKind.SENSOR, config=cfg)

    net.add_node(root)
    net.add_node(child)
    net.add_node(leaf)
    net.connect_parent_child("Root", "Child")
    net.connect_parent_child("Child", "Leaf")

    net.set_external_request("Root", True)

    net.step()
    assert net.get("Root").script_state == ScriptState.REQUESTED
    assert net.get("Child").script_state == ScriptState.INACTIVE

    net.step()
    assert net.get("Root").script_state == ScriptState.ACTIVE
    assert net.get("Child").script_state == ScriptState.INACTIVE

    net.step()
    assert net.get("Root").script_state == ScriptState.ACTIVE
    assert net.get("Child").script_state == ScriptState.REQUESTED

    net.step()
    assert net.get("Root").script_state == ScriptState.WAITING
    assert net.get("Child").script_state == ScriptState.ACTIVE
