#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch

from export_function_backprop_graphs import sgd_step


def tensors(values):
    return tuple(torch.tensor(value, dtype=torch.float32) for value in values)


def assert_close(name, actual, expected, rtol, atol):
    if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
        diff = (actual - expected).abs().max().item()
        raise AssertionError(f"{name} mismatch: max_abs_diff={diff}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()

    fixture = json.loads(args.fixture.read_text())
    meta = fixture["meta"]
    if fixture["fx"]["unsupported_ops"]:
        raise AssertionError(f"unsupported FX ops: {fixture['fx']['unsupported_ops']}")

    weights = tensors(fixture["params"]["weights"])
    biases = tensors(fixture["params"]["biases"])
    coords = torch.tensor(fixture["data"]["coords"], dtype=torch.float32)
    target = torch.tensor(fixture["data"]["target"], dtype=torch.float32)
    next_weights, next_biases, loss = sgd_step(weights, biases, coords, target, meta["lr"], meta["activation"])

    expected_weights = tensors(fixture["expected"]["weights"])
    expected_biases = tensors(fixture["expected"]["biases"])
    expected_loss = torch.tensor(fixture["expected"]["loss"], dtype=torch.float32)

    assert_close("loss", loss, expected_loss, args.rtol, args.atol)
    for index, (actual, expected) in enumerate(zip(next_weights, expected_weights)):
        assert_close(f"weight[{index}]", actual, expected, args.rtol, args.atol)
    for index, (actual, expected) in enumerate(zip(next_biases, expected_biases)):
        assert_close(f"bias[{index}]", actual, expected, args.rtol, args.atol)

    print(f"validated {args.fixture}")
    print(f"loss={loss.item():.8f}")


if __name__ == "__main__":
    main()
