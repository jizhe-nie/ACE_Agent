import pytest
import numpy as np
from ACE_Agent.agent_core.router import MasterRouter
from ACE_Agent.agent_core.supervisor import ACESupervisor
from ACE_Agent.tools.data_factory import generate_dataset

def test_router_blobs():
    router = MasterRouter()
    dataset = generate_dataset("blobs")
    decision = router.route(dataset)
    
    # Blobs should prefer centroid
    expert_keys = [e.expert_key for e in decision.selected_experts]
    assert "centroid" in expert_keys
    # For blobs, centroid role should be primary
    centroid_rec = next(e for e in decision.selected_experts if e.expert_key == "centroid")
    assert centroid_rec.role == "primary"

def test_router_moons():
    router = MasterRouter()
    dataset = generate_dataset("moons")
    decision = router.route(dataset)
    
    # Moons should prefer topology
    expert_keys = [e.expert_key for e in decision.selected_experts]
    assert "topology" in expert_keys
    topology_rec = next(e for e in decision.selected_experts if e.expert_key == "topology")
    assert topology_rec.role == "primary"

def test_supervisor_run_lite():
    supervisor = ACESupervisor()
    # Use a very small dataset for quick testing
    dataset = generate_dataset("blobs", n_samples=50)
    report = supervisor.run(dataset)
    
    assert report is not None
    assert len(report.results) > 0
    assert report.ranking[0].metrics["score"] >= 0
    assert report.output_dir.exists()
