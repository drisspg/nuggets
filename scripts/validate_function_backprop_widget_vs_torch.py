#!/usr/bin/env python3
import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch


GRID_SIZE = 64
ROOT = Path(__file__).resolve().parents[1]
NODE_REF = ROOT / "scripts" / "function_backprop_widget_ref.mjs"
DTYPE = torch.float64


@dataclass
class JsRng:
    seed: int = 0xDECAFBAD

    def rand(self, scale: float) -> float:
        self.seed = (1664525 * self.seed + 1013904223) & 0xFFFFFFFF
        return ((self.seed / 4294967296) * 2 - 1) * scale


def coord_for_cell(value: int) -> float:
    return (value / (GRID_SIZE - 1)) * 2 - 1


def index_for_cell(x: int, y: int) -> int:
    return y * GRID_SIZE + x


def coords_for_index(index: int) -> tuple[float, float]:
    return coord_for_cell(index % GRID_SIZE), coord_for_cell(index // GRID_SIZE)


def smile_target() -> torch.Tensor:
    values = []
    for y_index in range(GRID_SIZE):
        for x_index in range(GRID_SIZE):
            x = coord_for_cell(x_index)
            y = coord_for_cell(y_index)
            head = 0.25 if math.hypot(x, y) < 0.92 else -0.85
            left_eye = math.exp(-85 * ((x + 0.32) ** 2 + (y + 0.28) ** 2))
            right_eye = math.exp(-85 * ((x - 0.32) ** 2 + (y + 0.28) ** 2))
            mouth = math.exp(-120 * (math.hypot(x, y - 0.05) - 0.48) ** 2) * (1 if y > 0.05 else 0)
            values.append(max(-1, min(1, head - left_eye * 1.2 - right_eye * 1.2 + mouth * 1.1)))
    return torch.tensor(values, dtype=DTYPE)


def random_matrix(rng: JsRng, rows: int, cols: int, scale: float) -> torch.Tensor:
    return torch.tensor([[rng.rand(scale) for _ in range(cols)] for _ in range(rows)], dtype=DTYPE)


def random_vector(rng: JsRng, length: int, scale: float) -> torch.Tensor:
    return torch.tensor([rng.rand(scale) for _ in range(length)], dtype=DTYPE)


def init_model(layers: int, hidden: int):
    rng = JsRng()
    dims = [2, *([hidden] * layers), 1]
    weights = []
    biases = []
    for fan_in, fan_out in zip(dims, dims[1:]):
        weights.append(random_matrix(rng, fan_out, fan_in, math.sqrt(2 / fan_in)).requires_grad_())
        biases.append(random_vector(rng, fan_out, 0.1).requires_grad_())
    return {
        "weights": weights,
        "biases": biases,
        "weight_m1": [torch.zeros_like(weight) for weight in weights],
        "weight_m2": [torch.zeros_like(weight) for weight in weights],
        "bias_m1": [torch.zeros_like(bias) for bias in biases],
        "bias_m2": [torch.zeros_like(bias) for bias in biases],
    }


def activate(value: torch.Tensor, activation_name: str) -> torch.Tensor:
    match activation_name:
        case "relu":
            return torch.relu(value)
        case "sigmoid":
            return torch.sigmoid(value)
        case _:
            return torch.tanh(value)


def forward(model, coords: torch.Tensor, activation_name: str) -> torch.Tensor:
    current = coords
    for layer, (weight, bias) in enumerate(zip(model["weights"], model["biases"])):
        current = current @ weight.t() + bias
        if layer < len(model["weights"]) - 1:
            current = activate(current, activation_name)
    return torch.tanh(current.squeeze(-1))


def loss_values(prediction: torch.Tensor, target: torch.Tensor, loss_name: str) -> torch.Tensor:
    error = prediction - target
    match loss_name:
        case "l1":
            return torch.abs(error)
        case "huber":
            return torch.where(torch.abs(error) <= 0.25, 0.5 * error * error, 0.25 * (torch.abs(error) - 0.125))
        case _:
            return 0.5 * error * error


def batch_indices(train_cursor: int, inspect_only: bool) -> tuple[list[int], int]:
    if inspect_only:
        return [index_for_cell(7, 11)], train_cursor
    indices = []
    for _ in range(32):
        train_cursor = (train_cursor + 37) % (GRID_SIZE * GRID_SIZE)
        indices.append(train_cursor)
    return indices, train_cursor


def train_batch(model, target: torch.Tensor, settings: dict, step: int, train_cursor: int) -> int:
    indices, train_cursor = batch_indices(train_cursor, settings["inspectOnly"])
    coords = torch.tensor([coords_for_index(index) for index in indices], dtype=DTYPE)
    y_target = target[indices]
    loss = loss_values(forward(model, coords, settings["activationName"]), y_target, settings["lossName"]).mean()
    for param in [*model["weights"], *model["biases"]]:
        param.grad = None
    loss.backward()
    beta1 = 0.9
    beta2 = 0.999
    correction1 = 1 - beta1 ** (step + 1)
    correction2 = 1 - beta2 ** (step + 1)
    with torch.no_grad():
        for layer in range(len(model["weights"])):
            if settings["optimizerName"] == "adam":
                model["weight_m1"][layer].mul_(beta1).add_(model["weights"][layer].grad, alpha=1 - beta1)
                model["weight_m2"][layer].mul_(beta2).addcmul_(
                    model["weights"][layer].grad,
                    model["weights"][layer].grad,
                    value=1 - beta2,
                )
                model["weights"][layer].sub_(
                    settings["lr"]
                    * model["weight_m1"][layer]
                    / correction1
                    / (torch.sqrt(model["weight_m2"][layer] / correction2) + 1e-8)
                )
                model["bias_m1"][layer].mul_(beta1).add_(model["biases"][layer].grad, alpha=1 - beta1)
                model["bias_m2"][layer].mul_(beta2).addcmul_(
                    model["biases"][layer].grad,
                    model["biases"][layer].grad,
                    value=1 - beta2,
                )
                model["biases"][layer].sub_(
                    settings["lr"]
                    * model["bias_m1"][layer]
                    / correction1
                    / (torch.sqrt(model["bias_m2"][layer] / correction2) + 1e-8)
                )
            else:
                model["weights"][layer].sub_(settings["lr"] * model["weights"][layer].grad)
                model["biases"][layer].sub_(settings["lr"] * model["biases"][layer].grad)
    for param in [*model["weights"], *model["biases"]]:
        param.requires_grad_()
    return train_cursor


def flatten_model(model) -> list[float]:
    tensors = [
        *model["weights"],
        *model["biases"],
        *model["weight_m1"],
        *model["weight_m2"],
        *model["bias_m1"],
        *model["bias_m2"],
    ]
    return [float(value) for tensor in tensors for value in tensor.reshape(-1)]


def average_loss(model, target: torch.Tensor, settings: dict) -> float:
    indices = list(range(0, target.numel(), 5))
    coords = torch.tensor([coords_for_index(index) for index in indices], dtype=DTYPE)
    with torch.no_grad():
        return float(loss_values(forward(model, coords, settings["activationName"]), target[indices], settings["lossName"]).mean())


def torch_case(settings: dict) -> dict:
    model = init_model(settings["layers"], settings["hidden"])
    target = smile_target()
    train_cursor = 0
    for step in range(settings["steps"]):
        train_cursor = train_batch(model, target, settings, step, train_cursor)
    probes = [index_for_cell(7, 11), index_for_cell(32, 32), index_for_cell(51, 19), index_for_cell(4, 58)]
    coords = torch.tensor([coords_for_index(index) for index in probes], dtype=DTYPE)
    with torch.no_grad():
        return {
            "step": settings["steps"],
            "trainCursor": train_cursor,
            "modelFlat": flatten_model(model),
            "probes": [float(value) for value in forward(model, coords, settings["activationName"])],
            "averageLoss": average_loss(model, target, settings),
        }


def max_abs_delta(actual, expected) -> float:
    if isinstance(actual, list):
        return max((max_abs_delta(a, e) for a, e in zip(actual, expected)), default=0.0)
    return abs(actual - expected)


def build_cases(args) -> list[dict]:
    cases = []
    for layers in args.layers:
        for hidden in args.hidden:
            for activation_name in ["tanh", "relu", "sigmoid"]:
                for loss_name in ["mse", "l1", "huber"]:
                    for optimizer_name in ["sgd", "adam"]:
                        for inspect_only in [True, False]:
                            cases.append(
                                {
                                    "layers": layers,
                                    "hidden": hidden,
                                    "activationName": activation_name,
                                    "lossName": loss_name,
                                    "optimizerName": optimizer_name,
                                    "inspectOnly": inspect_only,
                                    "lr": args.lr,
                                    "steps": args.steps,
                                }
                            )
    return cases


def node_widget_cases(cases: list[dict]) -> list[dict]:
    completed = subprocess.run(
        ["node", str(NODE_REF), json.dumps(cases)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=ROOT,
    )
    return json.loads(completed.stdout)


def case_name(settings: dict) -> str:
    return "/".join(
        [
            f"L{settings['layers']}",
            f"H{settings['hidden']}",
            settings["activationName"],
            settings["lossName"],
            settings["optimizerName"],
            "inspect" if settings["inspectOnly"] else "batch",
            f"steps{settings['steps']}",
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Compare function-backprop widget CPU math against PyTorch.")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 3, 6])
    parser.add_argument("--hidden", type=int, nargs="+", default=[4, 24])
    parser.add_argument("--atol", type=float, default=2e-10)
    args = parser.parse_args()

    cases = build_cases(args)
    widget_results = node_widget_cases(cases)
    worst = {"case": "", "model": 0.0, "probes": 0.0, "loss": 0.0}
    failures = []
    for widget_result in widget_results:
        settings = widget_result["settings"]
        torch_result = torch_case(settings)
        deltas = {
            "model": max_abs_delta(widget_result["modelFlat"], torch_result["modelFlat"]),
            "probes": max_abs_delta(widget_result["probes"], torch_result["probes"]),
            "loss": abs(widget_result["averageLoss"] - torch_result["averageLoss"]),
        }
        name = case_name(settings)
        for key, value in deltas.items():
            if value > worst[key]:
                worst[key] = value
                worst["case"] = name
        if widget_result["step"] != torch_result["step"] or widget_result["trainCursor"] != torch_result["trainCursor"]:
            failures.append(f"{name}: step/cursor mismatch widget={widget_result['step']}/{widget_result['trainCursor']} torch={torch_result['step']}/{torch_result['trainCursor']}")
        for key, value in deltas.items():
            if value > args.atol:
                failures.append(f"{name}: {key} delta {value:.3e} > {args.atol:.3e}")

    print(f"validated {len(cases)} widget-vs-torch cases")
    print(
        "worst deltas: "
        f"model={worst['model']:.3e}, probes={worst['probes']:.3e}, loss={worst['loss']:.3e} ({worst['case']})"
    )
    if failures:
        raise SystemExit("\n".join(failures[:20]))


if __name__ == "__main__":
    main()
