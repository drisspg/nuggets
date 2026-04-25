#!/usr/bin/env python3
import math

import torch


GRID_SIZE = 64


class JsRng:
    def __init__(self):
        self.seed = 0xDECAFBAD

    def random(self):
        self.seed = (1664525 * self.seed + 1013904223) & 0xFFFFFFFF
        return self.seed / 4294967296

    def rand(self, scale):
        return (self.random() * 2 - 1) * scale


def coord_for_cell(value):
    return (value / (GRID_SIZE - 1)) * 2 - 1


def smile_target(x, y):
    head = 0.25 if math.hypot(x, y) < 0.92 else -0.85
    left_eye = math.exp(-85 * ((x + 0.32) ** 2 + (y + 0.28) ** 2))
    right_eye = math.exp(-85 * ((x - 0.32) ** 2 + (y + 0.28) ** 2))
    mouth = math.exp(-120 * (math.hypot(x, y - 0.05) - 0.48) ** 2) * (1 if y > 0.05 else 0)
    return max(-1, min(1, head - left_eye * 1.2 - right_eye * 1.2 + mouth * 1.1))


def random_matrix(rng, rows, cols, scale):
    return torch.tensor([[rng.rand(scale) for _ in range(cols)] for _ in range(rows)], dtype=torch.float32)


def random_vector(rng, length, scale):
    return torch.tensor([rng.rand(scale) for _ in range(length)], dtype=torch.float32)


def activate(value, name):
    match name:
        case "relu":
            return torch.relu(value)
        case "sigmoid":
            return torch.sigmoid(value)
        case _:
            return torch.tanh(value)


def loss_value(prediction, target, name):
    error = prediction - target
    match name:
        case "l1":
            return torch.abs(error)
        case "huber":
            return torch.where(torch.abs(error) <= 0.25, 0.5 * error * error, 0.25 * (torch.abs(error) - 0.125))
        case _:
            return 0.5 * error * error


def init_mlp(layers, hidden):
    rng = JsRng()
    dims = [2, *([hidden] * layers), 1]
    weights = []
    biases = []
    for fan_in, fan_out in zip(dims, dims[1:]):
        weights.append(random_matrix(rng, fan_out, fan_in, math.sqrt(2 / fan_in)).requires_grad_())
        biases.append(random_vector(rng, fan_out, 0.1).requires_grad_())
    return weights, biases


def mlp_forward(weights, biases, coord, activation):
    current = coord
    for layer, (weight, bias) in enumerate(zip(weights, biases)):
        current = current @ weight.t() + bias
        if layer < len(weights) - 1:
            current = activate(current, activation)
    return torch.tanh(current[0])


def assert_finite_grads(name, params):
    for index, param in enumerate([*params[0], *params[1]]):
        if param.grad is None or not torch.isfinite(param.grad).all():
            raise AssertionError(f"{name} grad {index} is invalid")


def validate_mlp():
    coord = torch.tensor([coord_for_cell(7), coord_for_cell(11)], dtype=torch.float32)
    target = torch.tensor(smile_target(coord[0].item(), coord[1].item()), dtype=torch.float32)
    for activation in ["tanh", "relu", "sigmoid"]:
        for loss in ["mse", "l1", "huber"]:
            weights, biases = init_mlp(3, 24)
            value = loss_value(mlp_forward(weights, biases, coord, activation), target, loss)
            value.backward()
            assert_finite_grads(f"mlp {activation}/{loss}", (weights, biases))


def main():
    validate_mlp()
    print("torch reference gradients validated for mlp")


if __name__ == "__main__":
    main()
