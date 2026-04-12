from __future__ import annotations

import textwrap
from pathlib import Path

from ACE_Agent.agent_core.schemas import AlgorithmRunResult, DatasetBundle
from ACE_Agent.expert_sub_agents.base import BaseExpert


class DeepRepresentationExpert(BaseExpert):
    key = "deep"
    label = "深度表征专家"

    def run(self, dataset: DatasetBundle, output_dir: Path) -> list[AlgorithmRunResult]:
        expected_clusters = int(dataset.metadata.get("expected_clusters", 3))

        autoencoder_code = textwrap.dedent(
            f"""
            import torch
            import torch.nn as nn
            from sklearn.cluster import KMeans

            torch.manual_seed(42)
            scaled = StandardScaler().fit_transform(X).astype("float32")
            tensor_x = torch.tensor(scaled)

            class AutoEncoder(nn.Module):
                def __init__(self, input_dim, latent_dim=2):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(input_dim, 16),
                        nn.ReLU(),
                        nn.Linear(16, 8),
                        nn.ReLU(),
                        nn.Linear(8, latent_dim),
                    )
                    self.decoder = nn.Sequential(
                        nn.Linear(latent_dim, 8),
                        nn.ReLU(),
                        nn.Linear(8, 16),
                        nn.ReLU(),
                        nn.Linear(16, input_dim),
                    )

                def forward(self, value):
                    latent = self.encoder(value)
                    reconstructed = self.decoder(latent)
                    return latent, reconstructed

            model = AutoEncoder(input_dim=scaled.shape[1], latent_dim=2)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.MSELoss()

            model.train()
            for _ in range(80):
                optimizer.zero_grad()
                latent, reconstructed = model(tensor_x)
                loss = criterion(reconstructed, tensor_x)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                latent, _ = model(tensor_x)

            embedding = latent.numpy()
            labels = KMeans(n_clusters={expected_clusters}, n_init=20, random_state=42).fit_predict(embedding)
            metrics = evaluate_labels(embedding, y_true, labels)
            plot_path = save_cluster_plot(embedding, labels, output_path, "深度专家 - 自动编码器 + KMeans")
            result = {{
                "labels": labels.tolist(),
                "metrics": metrics,
                "plot_path": plot_path,
            }}
            """
        )
        return [
            self._execute_code(
                dataset=dataset,
                output_dir=output_dir,
                algorithm_name="AutoEncoderPlusKMeans",
                params={"latent_dim": 2, "epochs": 80, "n_clusters": expected_clusters},
                code=autoencoder_code,
                plot_filename=f"{dataset.name}_deep_autoencoder_kmeans.png",
                trace=[
                    "训练了一个小型自动编码器来学习非线性 2D 潜空间。",
                    "对潜码使用 KMeans 进行聚类，作为轻量级的深度聚类代理方案。",
                ],
            )
        ]

