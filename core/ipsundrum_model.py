from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import numpy as np

from .recon_core import ReCoNConfig, config_from_env, Node, NodeKind, ScriptState, clamp01, sigmoid
from .recon_network import Network
from core.driver.ipsundrum_dynamics import ipsundrum_step


# -------------------------
# Parameters
# -------------------------

@dataclass
class LoopParams:
    """
    Humphrey loop parameters (sensorimotor → privatized feedback → attractor).

    Core ipsundrum dynamics:
      Ns_t = F( drive_t )
      X_t  = d X_{t-1} + (1-d) Ns_t
      M_t  = h X_t
      Ne_t = filtered_copy(M_t)
      E_t  = g_eff M_t

    where:
      drive_t = I_t + precision_t * E_{t-1} + bias + noise
    """

    # Reafferent gain: E_t = g * M_t
    g: float = 0.9

    # Motor gain: M_t = h * X_t
    h: float = 1.0

    # terminal confirm threshold (NOT phenomenal threshold used in evaluation)
    sensor_threshold: float = 0.01

    # bias into sensory drive (for signed I_ext in [-1,1], bias=0.5 often helps)
    sensor_bias: float = 0.0

    # sensory nonlinearity F
    nonlinearity: str = "linear"   # "linear" or "sigmoid"
    saturation: bool = True

    # thick-moment integrator: X_t = d X_{t-1} + (1-d) Ns_t
    internal_decay: float = 0.5

    # sensory noise
    noise_std: float = 0.0

    # fatigue: activity-dependent decay of g (optional)
    fatigue: float = 0.0

    # efference copy filtering: Ne_t = d_e Ne_{t-1} + (1-d_e) M_t
    efference_decay: float = 0.7

    # NEW: optional divisive normalization to prevent hard saturation of Ns
    # drive <- drive / (1 + divisive_norm * |precision*E_{t-1}|)
    # Set to ~0.3–1.0 for graded sensations under strong feedback.
    divisive_norm: float = 0.0


@dataclass
class AffectParams:
    """
    Barrett-style interoceptive prediction loop + affect readout.

    IMPORTANT FIX: stimulus term is SIGNED (deposit vs cost).
    - positive I_ext contributes cost (hurts body budget)
    - negative I_ext contributes deposit (replenishes body budget)

    This enables genuine internally-generated preference gradients.
    """

    enabled: bool = True

    setpoint: float = 0.0
    k_homeo: float = 0.10
    k_pe: float = 0.50

    demand_motor: float = 0.15
    demand_stim: float = 0.25

    stim_cost_pos: float = 1.0
    stim_gain_neg: float = 0.5

    bb_noise_std: float = 0.0

    valence_scale: float = 1.0
    arousal_scale: float = 1.0

    # coupling from affect -> ipsundrum
    modulate_g: bool = False
    k_g_arousal: float = 0.0
    k_g_unpleasant: float = 0.0

    modulate_precision: bool = False
    precision_base: float = 1.0
    k_precision_arousal: float = 0.0


# -------------------------
# Ipsundrum percept node (Humphrey gating)
# -------------------------

class IpsundrumPercept(Node):
    """
    Script node P with recurrence gating:
      - cycles_required: minimum loops before confirming
      - loop_until: max loops (None = unlimited)
      - efference_threshold: continue looping while Ne >= threshold
    """
    def __init__(
        self,
        node_id: str,
        cycles_required: int = 1,
        loop_until: Optional[int] = None,
        efference_threshold: float = 0.0,
        config: Optional[ReCoNConfig] = None,
    ):
        super().__init__(node_id=node_id, kind=NodeKind.SCRIPT, config=config or ReCoNConfig())
        self.cycles_required = int(cycles_required)
        self.loop_until = loop_until
        self.efference_threshold = float(efference_threshold)
        self.cycles_done = 0
        self.network_ref: Optional[Network] = None

    def _efference_child_id(self) -> Optional[str]:
        for c in self.children:
            if c.lower().startswith("ne"):
                return c
        return None

    def update(self, inbox, tick: int) -> None:
        prev = self.script_state
        super()._update_script(inbox, tick)

        if self.script_state == ScriptState.TRUE:
            self.cycles_done += 1

            eff = 0.0
            ne = self._efference_child_id()
            if ne is not None and self.network_ref is not None:
                eff = float(self.network_ref.get(ne).activation)

            want_loop = eff >= self.efference_threshold
            under_limit = (self.loop_until is None) or (self.cycles_done < self.loop_until)

            if self.cycles_done < self.cycles_required:
                self.script_state = ScriptState.ACTIVE
            elif want_loop and under_limit:
                self.script_state = ScriptState.ACTIVE
            else:
                self.script_state = ScriptState.CONFIRMED

        if prev in (ScriptState.CONFIRMED, ScriptState.FAILED) and self.script_state == ScriptState.INACTIVE:
            self.cycles_done = 0


@dataclass
class Wiring:
    Ns: str
    R: str
    Nm_or_Nr: str
    Ne: Optional[str]
    P: Optional[str]

    # Barrett:
    Ni: Optional[str] = None
    Nv: Optional[str] = None
    Na: Optional[str] = None


# -------------------------
# Builder
# -------------------------

class Builder:
    def __init__(
        self,
        params: Optional[LoopParams] = None,
        affect: Optional[AffectParams] = None,
        recon_config: Optional[ReCoNConfig] = None,
    ):
        self.params = params or LoopParams()
        self.affect = affect or AffectParams()
        env_cfg = config_from_env()
        if env_cfg.strict_fsm and recon_config is not None:
            if not (recon_config.strict_table1 and recon_config.strict_fsm and recon_config.strict_terminal):
                raise ValueError("RECON_STRICT=1 requires strict ReCoNConfig overrides.")
        self.recon_config = recon_config or env_cfg

    def _mark_recon_mode(self, net: Network) -> None:
        strict = self.recon_config.strict_table1 or self.recon_config.strict_fsm or self.recon_config.strict_terminal
        net.recon_mode = "strict" if strict else "compat"

    def _sensor(self, node_id: str) -> Node:
        n = Node(node_id=node_id, kind=NodeKind.SENSOR, config=self.recon_config)
        n.threshold = self.params.sensor_threshold
        return n

    def _actuator(self, node_id: str, effect: Optional[Callable[[float], None]] = None) -> Node:
        n = Node(node_id=node_id, kind=NodeKind.ACTUATOR, config=self.recon_config)
        n.actuator_effect = effect
        return n

    def _script(self, node_id: str) -> Node:
        return Node(node_id=node_id, kind=NodeKind.SCRIPT, config=self.recon_config)

    def stage_A(self) -> Tuple[Network, Wiring]:
        net = Network()
        Ns = self._sensor("Ns")
        R = self._script("R")
        Nm = self._actuator("Nm")
        Root = self._script("Root")

        for n in (Ns, R, Nm, Root):
            net.add_node(n)

        net.connect_parent_child("Root", "R")
        net.connect_parent_child("R", "Ns")
        net.connect_parent_child("R", "Nm")

        self._mark_recon_mode(net)
        return net, Wiring(Ns="Ns", R="R", Nm_or_Nr="Nm", Ne=None, P=None)

    def stage_B(self) -> Tuple[Network, Wiring]:
        net, _ = self.stage_A()
        Ne = self._sensor("Ne")
        P = self._script("P")
        net.add_node(Ne)
        net.add_node(P)

        net.get("Root").children = ["P"]
        net.get("P").parent = "Root"

        net.get("R").parent = "P"
        net.get("P").children.append("R")
        net.connect_parent_child("P", "Ne")

        ne_memory = {"value": 0.0, "last_tick": None}
        orig_set_sensor_value = net.set_sensor_value

        def _update_efference(m_cmd: float) -> None:
            d_e = float(getattr(self.params, "efference_decay", 0.7))
            ne_memory["value"] = d_e * float(ne_memory["value"]) + (1.0 - d_e) * abs(float(m_cmd))
            ne_memory["last_tick"] = net.tick
            orig_set_sensor_value("Ne", clamp01(abs(float(ne_memory["value"]))))

        def efference_effect(_a: float) -> None:
            # Efference copy: low-pass filtered copy of outgoing motor command.
            # Stage B uses a reflexive motor command derived from current Ns.
            if ne_memory.get("last_tick") == net.tick:
                orig_set_sensor_value("Ne", clamp01(abs(float(ne_memory["value"]))))
                return
            m_cmd = float(net.get("Ns").activation)
            _update_efference(m_cmd)

        def set_sensor_value(sensor_id: str, value: float) -> None:
            orig_set_sensor_value(sensor_id, value)
            if sensor_id == "Ns":
                _update_efference(float(value))

        net.set_sensor_value = set_sensor_value  # type: ignore[assignment]

        def update_sensor(I_ext: float, rng: Optional[np.random.Generator] = None) -> None:
            _ = rng
            I_drive = float(I_ext)
            if not bool(self.affect.enabled):
                # Stage-B / non-affect: negative I does not confer positive value.
                I_drive = max(0.0, I_drive)
            ns_val = float(clamp01(0.5 + 0.5 * I_drive))
            net.set_sensor_value("Ns", ns_val)

        net._update_ipsundrum_sensor = update_sensor  # type: ignore[attr-defined]

        net.get("Nm").actuator_effect = efference_effect
        self._mark_recon_mode(net)
        return net, Wiring(Ns="Ns", R="R", Nm_or_Nr="Nm", Ne="Ne", P="P")

    def stage_C(self, cycles_required: int = 8) -> Tuple[Network, Wiring]:
        net = Network()
        Ns = self._sensor("Ns")
        R = self._script("R")
        Nr = self._actuator("Nr")
        Ne = self._sensor("Ne")
        P = IpsundrumPercept(
            "P",
            cycles_required=cycles_required,
            loop_until=cycles_required,
            efference_threshold=0.0,
            config=self.recon_config,
        )
        Root = self._script("Root")

        aff_enabled = bool(self.affect.enabled)
        Ni = Nv = Na = None
        nodes = [Ns, R, Nr, Ne, P, Root]
        if aff_enabled:
            Ni = self._sensor("Ni")
            Nv = self._sensor("Nv")
            Na = self._sensor("Na")
            nodes.extend([Ni, Nv, Na])
        for n in nodes:
            net.add_node(n)

        net.connect_parent_child("Root", "P")
        net.connect_parent_child("P", "R")
        net.connect_parent_child("P", "Ne")
        net.connect_parent_child("R", "Ns")
        net.connect_parent_child("R", "Nr")

        self._attach_loop(net)
        P.network_ref = net
        self._mark_recon_mode(net)
        return net, Wiring(
            Ns="Ns",
            R="R",
            Nm_or_Nr="Nr",
            Ne="Ne",
            P="P",
            Ni=("Ni" if aff_enabled else None),
            Nv=("Nv" if aff_enabled else None),
            Na=("Na" if aff_enabled else None),
        )

    def stage_D(self, efference_threshold: float = 0.05) -> Tuple[Network, Wiring]:
        net = Network()
        Ns = self._sensor("Ns")
        R = self._script("R")
        Nr = self._actuator("Nr")
        Ne = self._sensor("Ne")
        P = IpsundrumPercept(
            "P",
            cycles_required=2,
            loop_until=None,
            efference_threshold=efference_threshold,
            config=self.recon_config,
        )
        Root = self._script("Root")

        aff_enabled = bool(self.affect.enabled)
        Ni = Nv = Na = None
        nodes = [Ns, R, Nr, Ne, P, Root]
        if aff_enabled:
            Ni = self._sensor("Ni")
            Nv = self._sensor("Nv")
            Na = self._sensor("Na")
            nodes.extend([Ni, Nv, Na])
        for n in nodes:
            net.add_node(n)

        net.connect_parent_child("Root", "P")
        net.connect_parent_child("P", "R")
        net.connect_parent_child("P", "Ne")
        net.connect_parent_child("R", "Ns")
        net.connect_parent_child("R", "Nr")

        self._attach_loop(net)
        P.network_ref = net
        self._mark_recon_mode(net)
        return net, Wiring(
            Ns="Ns",
            R="R",
            Nm_or_Nr="Nr",
            Ne="Ne",
            P="P",
            Ni=("Ni" if aff_enabled else None),
            Nv=("Nv" if aff_enabled else None),
            Na=("Na" if aff_enabled else None),
        )

    def _attach_loop(self, net: Network) -> None:
        params = self.params
        aff = self.affect
        state = {
            "reafferent": 0.0,
            "internal": 0.0,
            "motor": 0.0,
            "efference": 0.0,
            "g": float(params.g),

            # DEBUG fields (optional to plot)
            "drive": 0.0,
            "drive_base": 0.0,
            "precision_eff": 1.0,
            "g_eff": float(params.g),
            "alpha_eff": 0.0,
        }
        if aff.enabled:
            state.update(
                {
                    "demand": 0.0,
                    "bb_true": 0.0,
                    "bb_model": 0.0,
                    "bb_pred": 0.0,
                    "pe": 0.0,
                    "valence": 1.0,
                    "arousal": 0.0,
                }
            )

        def nr_effect(_a: float) -> None:
            net.set_sensor_value("Ne", clamp01(abs(float(state["efference"]))))

        net.get("Nr").actuator_effect = nr_effect

        def F_sensory(x: float) -> float:
            if params.nonlinearity == "linear":
                y = x
            elif params.nonlinearity == "sigmoid":
                y = sigmoid(x)
            else:
                raise ValueError(f"Unknown nonlinearity: {params.nonlinearity}")
            if params.saturation:
                y = clamp01(y)
            return y

        def stim_cost(I_ext: float) -> float:
            # positive = cost, negative = deposit
            if I_ext >= 0.0:
                return aff.stim_cost_pos * abs(I_ext)
            else:
                return -aff.stim_gain_neg * abs(I_ext)

        def update_affect(I_ext: float, M: float, rng: np.random.Generator) -> None:
            if not aff.enabled:
                return

            # signed stimulus contribution (deposit vs cost)
            demand = aff.demand_motor * abs(M) + aff.demand_stim * stim_cost(I_ext)
            state["demand"] = float(demand)

            u = -aff.k_homeo * (float(state["bb_model"]) - float(aff.setpoint))
            bb_pred = float(state["bb_model"]) + u

            bb_true = float(state["bb_true"]) + u - float(demand)
            y = bb_true + (float(rng.normal(0.0, aff.bb_noise_std)) if aff.bb_noise_std > 0.0 else 0.0)

            pe = y - bb_pred
            bb_model = float(state["bb_model"]) + aff.k_pe * pe

            dist = abs(bb_model - float(aff.setpoint))
            val = 1.0 - (dist / max(1e-9, float(aff.valence_scale)))
            val = clamp01(val)

            # arousal should track magnitude, even if demand is negative (deposit)
            aro = float(aff.arousal_scale) * (abs(pe) + abs(demand))
            aro = clamp01(aro)

            state["bb_true"] = bb_true
            state["bb_pred"] = bb_pred
            state["pe"] = pe
            state["bb_model"] = bb_model
            state["valence"] = val
            state["arousal"] = aro

            if "Ni" in net.nodes:
                net.set_sensor_value("Ni", clamp01(0.5 + 0.5 * bb_model))
            if "Nv" in net.nodes:
                net.set_sensor_value("Nv", val)
            if "Na" in net.nodes:
                net.set_sensor_value("Na", aro)

        def update_sensor(I_ext: float, rng: Optional[np.random.Generator] = None) -> None:
            rg = rng or np.random.default_rng(0)

            # Single source of truth for ipsundrum dynamics
            next_state = ipsundrum_step(state, float(I_ext), params, aff, rng=rg)
            state.clear()
            state.update(next_state)

            # Update visible sensors from state
            if "Ns" in net.nodes:
                net.set_sensor_value("Ns", float(state.get("Ns", 0.0)))
            if "Ne" in net.nodes:
                net.set_sensor_value("Ne", clamp01(abs(float(state.get("efference", 0.0)))))

            lesion_affect = bool(state.get("lesion_affect", False))
            if aff.enabled and not lesion_affect:
                if "Ni" in net.nodes:
                    net.set_sensor_value("Ni", clamp01(0.5 + 0.5 * float(state.get("bb_model", 0.0))))
                if "Nv" in net.nodes:
                    net.set_sensor_value("Nv", clamp01(float(state.get("valence", 0.0))))
                if "Na" in net.nodes:
                    net.set_sensor_value("Na", clamp01(float(state.get("arousal", 0.0))))

        net._update_ipsundrum_sensor = update_sensor  # type: ignore[attr-defined]
        net._ipsundrum_state = state                  # type: ignore[attr-defined]
