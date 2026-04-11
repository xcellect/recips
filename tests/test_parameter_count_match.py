from core.perspective_model import make_perspective_params
from core.workspace_model import make_workspace_params


def test_parameter_counts_within_twenty_percent():
    perspective = make_perspective_params(arch_seed=0).parameter_count()
    workspace = make_workspace_params(arch_seed=0).parameter_count()
    bigger = max(perspective, workspace)
    smaller = min(perspective, workspace)
    assert smaller / bigger >= 0.8
