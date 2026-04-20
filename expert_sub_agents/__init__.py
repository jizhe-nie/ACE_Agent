from ACE_Agent.expert_sub_agents.centroid_expert import CentroidExpert
from ACE_Agent.expert_sub_agents.critic_expert import CriticExpert
from ACE_Agent.expert_sub_agents.deep_representation import DeepRepresentationExpert
from ACE_Agent.expert_sub_agents.dimension_expert import DimensionExpert
from ACE_Agent.expert_sub_agents.multi_view_expert import MultiViewExpert
from ACE_Agent.expert_sub_agents.topology_expert import TopologyExpert
from ACE_Agent.expert_sub_agents.zoo_expert import ZooExpert
import logging

_logger = logging.getLogger(__name__)


def build_expert_registry():
    """Build the full expert registry.

    Ready experts: centroid, topology, zoo, critic, dimension, deep_representation.
    WIP (Phase 2+): multi_view.
    """
    candidates = [
        CentroidExpert,
        TopologyExpert,
        ZooExpert,
        CriticExpert,
        DimensionExpert,
        DeepRepresentationExpert,
        MultiViewExpert,
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

