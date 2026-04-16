from ACE_Agent.expert_sub_agents.centroid_expert import CentroidExpert
from ACE_Agent.expert_sub_agents.deep_representation import DeepRepresentationExpert
from ACE_Agent.expert_sub_agents.dimension_expert import DimensionExpert
from ACE_Agent.expert_sub_agents.multi_view_expert import MultiViewExpert
from ACE_Agent.expert_sub_agents.topology_expert import TopologyExpert
from ACE_Agent.expert_sub_agents.zoo_expert import ZooExpert


def build_expert_registry():
    experts = [
        CentroidExpert(),
        TopologyExpert(),
        DimensionExpert(),
        DeepRepresentationExpert(),
        MultiViewExpert(),
        ZooExpert(),
    ]
    return {expert.key: expert for expert in experts}

