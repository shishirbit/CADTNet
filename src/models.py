"""Released CA-DTNet and baseline model definitions.

The module names and tensor dimensions intentionally match the published
checkpoints in ``checkpoints/``. Inputs use ``[batch, time, node, feature]``.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _row_normalize(adjacency: Tensor) -> Tensor:
    adjacency = adjacency.float()
    return adjacency / adjacency.sum(-1, keepdim=True).clamp_min(1.0)


class MaskedGraphAttention(nn.Module):
    """Single-head scaled attention constrained by the road graph."""

    def __init__(self, hidden_dim: int, adjacency: Tensor, dropout: float = 0.15):
        super().__init__()
        mask = adjacency.bool() | torch.eye(adjacency.shape[0], dtype=torch.bool)
        self.register_buffer("adjacency_mask", mask)
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, states: Tensor) -> Tensor:
        scores = torch.matmul(self.query(states), self.key(states).transpose(-1, -2))
        scores = scores / math.sqrt(states.shape[-1])
        scores = scores.masked_fill(~self.adjacency_mask, torch.finfo(scores.dtype).min)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        update = self.output(torch.matmul(weights, self.value(states)))
        return self.norm(states + update)


class CADTNet(nn.Module):
    """Communication-aware digital-twin forecasting network.

    The released configuration uses 8 traffic features, 27 prepared
    communication/context features, 61 graph nodes, a 64-dimensional latent
    state, three horizons, three targets, and quantiles (0.1, 0.5, 0.9).
    """

    def __init__(
        self,
        adjacency: Tensor,
        traffic_features: int = 8,
        communication_features: int = 27,
        hidden_dim: int = 64,
        horizons: int = 3,
        targets: int = 3,
        quantiles: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()
        nodes = int(adjacency.shape[0])
        self.horizons = horizons
        self.targets = targets
        self.quantiles = quantiles
        self.node_embedding = nn.Parameter(torch.empty(nodes, hidden_dim))
        nn.init.normal_(self.node_embedding, std=0.02)

        self.traffic_projection = nn.Sequential(
            nn.Linear(traffic_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.communication_encoder = nn.Sequential(
            nn.Linear(communication_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.reliability_gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.temporal_encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.graph_attention = MaskedGraphAttention(hidden_dim, adjacency, dropout)
        self.digital_twin_refinement = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.reconstruction_head = nn.Linear(hidden_dim, traffic_features)
        self.forecast_head = nn.Linear(hidden_dim, horizons * targets * quantiles)

    def forward(self, traffic: Tensor, communication_context: Tensor) -> dict[str, Tensor]:
        if traffic.ndim != 4 or communication_context.ndim != 4:
            raise ValueError("traffic and communication_context must be [B,T,N,F]")
        if traffic.shape[:3] != communication_context.shape[:3]:
            raise ValueError("traffic and communication tensors must share B,T,N")

        batch, steps, nodes, _ = traffic.shape
        traffic_state = self.traffic_projection(traffic)
        communication_state = self.communication_encoder(communication_context)
        reliability = self.reliability_gate(communication_state)
        fused = traffic_state * reliability + self.node_embedding[None, None, :, :]

        temporal_in = fused.permute(0, 2, 1, 3).reshape(batch * nodes, steps, -1)
        _, final_state = self.temporal_encoder(temporal_in)
        temporal = final_state[-1].reshape(batch, nodes, -1)
        spatial = self.graph_attention(temporal)
        twin_state = self.digital_twin_refinement(torch.cat([temporal, spatial], dim=-1))

        reconstruction = self.reconstruction_head(twin_state)
        quantile_forecast = self.forecast_head(twin_state).view(
            batch, nodes, self.horizons, self.targets, self.quantiles
        )
        quantile_forecast = quantile_forecast.permute(0, 2, 1, 3, 4).contiguous()
        median_index = self.quantiles // 2
        return {
            "prediction": quantile_forecast[..., median_index],
            "quantiles": quantile_forecast,
            "reconstruction": reconstruction,
            "reliability": reliability,
        }


class GraphGRU(nn.Module):
    def __init__(self, adjacency: Tensor, features: int = 8, hidden_dim: int = 64,
                 horizons: int = 3, targets: int = 3, dropout: float = 0.15):
        super().__init__()
        self.horizons, self.targets = horizons, targets
        self.register_buffer("adjacency", _row_normalize(adjacency))
        self.input_projection = nn.Linear(features, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.graph_update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.output_head = nn.Linear(hidden_dim, horizons * targets)

    def forward(self, traffic: Tensor) -> Tensor:
        batch, steps, nodes, _ = traffic.shape
        x = self.input_projection(traffic).permute(0, 2, 1, 3).reshape(batch * nodes, steps, -1)
        _, state = self.gru(x)
        state = state[-1].reshape(batch, nodes, -1)
        neighbors = torch.einsum("nm,bmh->bnh", self.adjacency, state)
        state = self.graph_update(torch.cat([state, neighbors], dim=-1))
        return self.output_head(state).view(batch, nodes, self.horizons, self.targets).permute(0, 2, 1, 3)


class DiffusionConvolution(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, forward: Tensor, backward: Tensor, order: int = 2):
        super().__init__()
        self.order = order
        self.register_buffer("forward_transition", _row_normalize(forward))
        self.register_buffer("backward_transition", _row_normalize(backward))
        self.projection = nn.Linear(input_dim * (1 + 2 * order), output_dim)

    def forward(self, signal: Tensor) -> Tensor:
        terms = [signal]
        for support in (self.forward_transition, self.backward_transition):
            previous, current = signal, torch.einsum("nm,bmc->bnc", support, signal)
            terms.append(current)
            for _ in range(2, self.order + 1):
                following = 2 * torch.einsum("nm,bmc->bnc", support, current) - previous
                terms.append(following)
                previous, current = current, following
        return self.projection(torch.cat(terms, dim=-1))


class DCGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, forward: Tensor, backward: Tensor):
        super().__init__()
        joint = input_dim + hidden_dim
        self.hidden_dim = hidden_dim
        self.gate_convolution = DiffusionConvolution(joint, 2 * hidden_dim, forward, backward)
        self.candidate_convolution = DiffusionConvolution(joint, hidden_dim, forward, backward)

    def forward(self, inputs: Tensor, state: Tensor) -> Tensor:
        reset, update = torch.sigmoid(self.gate_convolution(torch.cat([inputs, state], -1))).chunk(2, -1)
        candidate = torch.tanh(self.candidate_convolution(torch.cat([inputs, reset * state], -1)))
        return update * state + (1.0 - update) * candidate


class DCRNN(nn.Module):
    def __init__(self, adjacency: Tensor, features: int = 8, hidden_dim: int = 64,
                 horizons: int = 3, targets: int = 3):
        super().__init__()
        self.horizons, self.targets, self.hidden_dim = horizons, targets, hidden_dim
        forward = _row_normalize(adjacency)
        backward = _row_normalize(adjacency.transpose(0, 1))
        self.input_projection = nn.Linear(features, hidden_dim)
        self.recurrent_cell = DCGRUCell(hidden_dim, hidden_dim, forward, backward)
        self.output_head = nn.Linear(hidden_dim, horizons * targets)

    def forward(self, traffic: Tensor) -> Tensor:
        batch, _, nodes, _ = traffic.shape
        state = traffic.new_zeros(batch, nodes, self.hidden_dim)
        for step in self.input_projection(traffic).unbind(dim=1):
            state = self.recurrent_cell(step, state)
        return self.output_head(state).view(batch, nodes, self.horizons, self.targets).permute(0, 2, 1, 3)


class NConv(nn.Module):
    def forward(self, signal: Tensor, adjacency: Tensor) -> Tensor:
        return torch.einsum("bcnt,nm->bcmt", signal, adjacency)


class GraphConvolution(nn.Module):
    def __init__(self, channels: int, supports: int = 3, order: int = 2):
        super().__init__()
        self.order = order
        self.nconv = NConv()
        self.projection = nn.Conv2d(channels * (1 + supports * order), channels, kernel_size=(1, 1))

    def forward(self, signal: Tensor, supports: Iterable[Tensor]) -> Tensor:
        terms = [signal]
        for support in supports:
            current = self.nconv(signal, support)
            terms.append(current)
            for _ in range(2, self.order + 1):
                current = self.nconv(current, support)
                terms.append(current)
        return self.projection(torch.cat(terms, dim=1))


class GraphWaveNet(nn.Module):
    def __init__(self, adjacency: Tensor, features: int = 8, horizons: int = 3, targets: int = 3,
                 residual_channels: int = 32, skip_channels: int = 64, dropout: float = 0.15):
        super().__init__()
        nodes = adjacency.shape[0]
        self.horizons, self.targets, self.dropout = horizons, targets, dropout
        self.node_vector_left = nn.Parameter(torch.randn(nodes, 10) * 0.1)
        self.node_vector_right = nn.Parameter(torch.randn(10, nodes) * 0.1)
        self.register_buffer("fixed_adjacency", _row_normalize(adjacency))
        self.register_buffer("reverse_adjacency", _row_normalize(adjacency.transpose(0, 1)))
        self.start_convolution = nn.Conv2d(features, residual_channels, (1, 1))
        dilations = (1, 2, 4, 1, 2, 4)
        self.filter_convolutions = nn.ModuleList([
            nn.Conv2d(residual_channels, residual_channels, (1, 2), dilation=(1, d)) for d in dilations
        ])
        self.gate_convolutions = nn.ModuleList([
            nn.Conv2d(residual_channels, residual_channels, (1, 2), dilation=(1, d)) for d in dilations
        ])
        self.residual_convolutions = nn.ModuleList([
            nn.Conv2d(residual_channels, residual_channels, (1, 1)) for _ in dilations
        ])
        self.skip_convolutions = nn.ModuleList([
            nn.Conv2d(residual_channels, skip_channels, (1, 1)) for _ in dilations
        ])
        self.graph_convolutions = nn.ModuleList([GraphConvolution(residual_channels) for _ in dilations])
        self.batch_norms = nn.ModuleList([nn.BatchNorm2d(residual_channels) for _ in dilations])
        self.end_convolution_1 = nn.Conv2d(skip_channels, 128, (1, 1))
        self.end_convolution_2 = nn.Conv2d(128, horizons * targets, (1, 1))

    def forward(self, traffic: Tensor) -> Tensor:
        x = traffic.permute(0, 3, 2, 1)
        receptive_field = 15
        if x.shape[-1] < receptive_field:
            x = F.pad(x, (receptive_field - x.shape[-1], 0, 0, 0))
        x = self.start_convolution(x)
        skip = None
        adaptive = torch.softmax(F.relu(self.node_vector_left @ self.node_vector_right), dim=1)
        supports = (self.fixed_adjacency, self.reverse_adjacency, adaptive)
        for filt, gate, residual, skip_conv, graph, norm in zip(
            self.filter_convolutions, self.gate_convolutions, self.residual_convolutions,
            self.skip_convolutions, self.graph_convolutions, self.batch_norms
        ):
            previous = x
            x = torch.tanh(filt(x)) * torch.sigmoid(gate(x))
            contribution = skip_conv(x)
            skip = contribution if skip is None else skip[..., -contribution.shape[-1]:] + contribution
            x = graph(x, supports)
            x = x + residual(previous)[..., -x.shape[-1]:]
            x = norm(x)
            x = F.dropout(x, self.dropout, training=self.training)
        output = self.end_convolution_2(F.relu(self.end_convolution_1(F.relu(skip))))[..., -1]
        return output.permute(0, 2, 1).reshape(output.shape[0], output.shape[2], self.horizons, self.targets).permute(0, 2, 1, 3)


def load_released_checkpoint(model: nn.Module, path: str, map_location: str | torch.device = "cpu") -> dict:
    """Load a released checkpoint and return its provenance metadata."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state"])
    return payload.get("metadata", {})
