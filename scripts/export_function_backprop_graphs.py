#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.func import grad_and_value


def make_grid(grid_size: int) -> torch.Tensor:
    coords = []
    for index in range(grid_size * grid_size):
        x = index % grid_size
        y = index // grid_size
        coords.append([(x / (grid_size - 1)) * 2 - 1, (y / (grid_size - 1)) * 2 - 1])
    return torch.tensor(coords, dtype=torch.float32)


def make_target(coords: torch.Tensor, name: str) -> torch.Tensor:
    x = coords[:, 0]
    y = coords[:, 1]
    match name:
        case "smoke":
            return (torch.sin(x * 2.1) * torch.cos(y * 1.7)).unsqueeze(1)
        case "smile":
            head = torch.where(torch.sqrt(x * x + y * y) < 0.92, 0.25, -0.85)
            left_eye = torch.exp(-85 * ((x + 0.32) ** 2 + (y + 0.28) ** 2))
            right_eye = torch.exp(-85 * ((x - 0.32) ** 2 + (y + 0.28) ** 2))
            mouth = torch.exp(-120 * (torch.sqrt(x * x + (y - 0.05) ** 2) - 0.48) ** 2) * (y > 0.05).float()
            return (head - left_eye * 1.2 - right_eye * 1.2 + mouth * 1.1).unsqueeze(1)
        case _:
            raise ValueError(f"unknown target {name}")


def init_params(dims: list[int], seed: int) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    generator = torch.Generator().manual_seed(seed)
    weights = []
    biases = []
    for fan_in, fan_out in zip(dims, dims[1:]):
        weights.append((torch.rand((fan_out, fan_in), generator=generator) * 2 - 1) * (2 / fan_in) ** 0.5)
        biases.append((torch.rand((fan_out,), generator=generator) * 2 - 1) * 0.1)
    return tuple(weights), tuple(biases)


def apply_activation(value: torch.Tensor, activation: str) -> torch.Tensor:
    match activation:
        case "relu":
            return torch.relu(value)
        case "sigmoid":
            return torch.sigmoid(value)
        case "tanh":
            return torch.tanh(value)
        case _:
            raise ValueError(f"unknown activation {activation}")


def mlp_forward(weights: tuple[torch.Tensor, ...], biases: tuple[torch.Tensor, ...], coords: torch.Tensor, activation: str) -> torch.Tensor:
    current = coords
    for layer, (weight, bias) in enumerate(zip(weights, biases)):
        current = current @ weight.t() + bias
        if layer < len(weights) - 1:
            current = apply_activation(current, activation)
    return torch.tanh(current)


def loss_fn(weights, biases, coords, target, activation):
    prediction = mlp_forward(weights, biases, coords, activation)
    return ((prediction - target) ** 2).mean() * 0.5


def sgd_step(weights, biases, coords, target, lr: float, activation: str):
    (weight_grads, bias_grads), loss = grad_and_value(loss_fn, argnums=(0, 1))(weights, biases, coords, target, activation)
    next_weights = tuple(weight - lr * grad for weight, grad in zip(weights, weight_grads))
    next_biases = tuple(bias - lr * grad for bias, grad in zip(biases, bias_grads))
    return next_weights, next_biases, loss


def tensor_to_json(value: torch.Tensor):
    return value.detach().cpu().tolist()


def target_name(target) -> str:
    if hasattr(target, "__name__"):
        return target.__name__
    return str(target)


def export_graph(weights, biases, coords, target, lr: float, activation: str):
    def traced_step(weights_arg, biases_arg, coords_arg, target_arg):
        return sgd_step(weights_arg, biases_arg, coords_arg, target_arg, lr, activation)

    graph_module = make_fx(traced_step)(weights, biases, coords, target)
    nodes = []
    for node in graph_module.graph.nodes:
        nodes.append(
            {
                "name": node.name,
                "op": node.op,
                "target": target_name(node.target),
                "args": str(node.args),
                "kwargs": str(node.kwargs),
            }
        )
    return graph_module, nodes


def unsupported_ops(nodes):
    allowed_call_targets = {
        "t.default",
        "mm.default",
        "add.Tensor",
        "sub.Tensor",
        "mul.Tensor",
        "mul.Scalar",
        "div.Tensor",
        "div.Scalar",
        "pow.Tensor_Scalar",
        "mean.default",
        "sum.dim_IntList",
        "tanh.default",
        "relu.default",
        "sigmoid.default",
        "tanh_backward.default",
        "threshold_backward.default",
        "ones_like.default",
        "detach.default",
        "expand.default",
        "view.default",
    }
    unsupported = set()
    for node in nodes:
        if node["op"] in {"placeholder", "output"}:
            continue
        if node["target"] not in allowed_call_targets:
            unsupported.add(node["target"])
    return sorted(unsupported)


def build_fixture(args):
    dims = [2, *([args.hidden] * args.layers), 1]
    coords = make_grid(args.grid)
    target = make_target(coords, args.target)
    weights, biases = init_params(dims, args.seed)
    next_weights, next_biases, loss = sgd_step(weights, biases, coords, target, args.lr, args.activation)
    graph_module, nodes = export_graph(weights, biases, coords, target, args.lr, args.activation)
    return {
        "meta": {
            "grid": args.grid,
            "layers": args.layers,
            "hidden": args.hidden,
            "activation": args.activation,
            "target": args.target,
            "lr": args.lr,
            "seed": args.seed,
            "torch": torch.__version__,
        },
        "params": {
            "weights": [tensor_to_json(weight) for weight in weights],
            "biases": [tensor_to_json(bias) for bias in biases],
        },
        "data": {
            "coords": tensor_to_json(coords),
            "target": tensor_to_json(target),
        },
        "expected": {
            "loss": float(loss.detach().cpu()),
            "weights": [tensor_to_json(weight) for weight in next_weights],
            "biases": [tensor_to_json(bias) for bias in next_biases],
        },
        "fx": {
            "code": graph_module.code,
            "nodes": nodes,
            "unsupported_ops": unsupported_ops(nodes),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=4)
    parser.add_argument("--activation", choices=["relu", "sigmoid", "tanh"], default="relu")
    parser.add_argument("--target", choices=["smoke", "smile"], default="smoke")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    fixture = build_fixture(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture, indent=2))
    print(f"wrote {args.out}")
    print(f"unsupported_ops={fixture['fx']['unsupported_ops']}")


if __name__ == "__main__":
    main()
