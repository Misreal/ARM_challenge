# The search space's version and the study name derived from it.
#
# Kept dependency-free and in one place: the name is built by `src.app`, the study
# driver, the final test and the dashboard, and when those disagree the study
# writes a file the pipeline then cannot find.

from __future__ import annotations

# Bumped when the objectives or the searched knobs change, so a resumed study can
# never mix trials drawn from two different spaces. v2 added top-1 as a fourth
# objective and unfroze the four runtime knobs; v3 dropped dynamic quantization
# from the sampled space; v4 enforces the reduced space, so a v4 study samples
# only the groups the reduction left searchable.
SPACE_VERSION = "v4"

# Optuna's own default. A study at this size records no suffix, so the names
# minted before the knob was reachable keep resolving.
DEFAULT_POPULATION_SIZE = 50


def study_name(
    model: str,
    budget_pt: float,
    mock: bool = False,
    population_size: int = DEFAULT_POPULATION_SIZE,
) -> str:
    """Everything that must not be resumed across, spelled into one filename.

    Budget, venue, space version and population all change what the trials mean,
    and `load_if_exists` would silently blend two of them into one front.
    """
    population = "" if population_size == DEFAULT_POPULATION_SIZE else f"_pop{population_size}"
    suffix = "_mock" if mock else ""
    return f"{model}_budget{budget_pt:g}_{SPACE_VERSION}{population}{suffix}"
