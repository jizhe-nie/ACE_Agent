import logging

from ACE_Agent.expert_sub_agents.centroid_expert import CentroidExpert
from ACE_Agent.expert_sub_agents.critic_expert import CriticExpert
from ACE_Agent.expert_sub_agents.dimension_expert import DimensionExpert
from ACE_Agent.expert_sub_agents.ensemble_expert import EnsembleConsensusExpert
from ACE_Agent.expert_sub_agents.topology_expert import TopologyExpert
from ACE_Agent.expert_sub_agents.zoo_expert import ZooExpert

_logger = logging.getLogger(__name__)


def build_expert_registry():
    """Build the full expert registry.

    Active (6): centroid, topology, zoo, critic, dimension, ensemble.

    Removed from registry (2026-05-04 cleanup):
    - multi_view: old skeleton using legacy run() pattern, lacks _generate_code().
    - deep_representation: thin LLM prompt wrapper, functionally overlapped by
      dimension_expert's deep clustering pipelines (SelfLabel, AE_KMeans).
    """
    candidates = [
        CentroidExpert,
        TopologyExpert,
        ZooExpert,
        CriticExpert,
        DimensionExpert,
        EnsembleConsensusExpert,
    ]
    registry = {}
    for cls in candidates:
        try:
            expert = cls()
            registry[expert.key] = expert
        except TypeError as exc:
            _logger.debug(
                "Expert '%s' skipped (abstract method not implemented): %s",
                cls.__name__,
                exc,
            )
    return registry
