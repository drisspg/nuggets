#!/usr/bin/env node
const gridSize = 64

function makeRng() {
  let seed = 0xdecafbad
  return {
    rand(scale) {
      seed = (1664525 * seed + 1013904223) >>> 0
      return ((seed / 4294967296) * 2 - 1) * scale
    },
  }
}

function coordForCell(value) {
  return (value / (gridSize - 1)) * 2 - 1
}

function indexForCell(x, y) {
  return y * gridSize + x
}

function coordsForIndex(index) {
  const x = index % gridSize
  const y = Math.floor(index / gridSize)
  return [coordForCell(x), coordForCell(y), x, y]
}

function setTargetCell(target, x, y, value) {
  target[indexForCell(x, y)] = Math.max(-1, Math.min(1, value))
}

function smileTarget() {
  const target = new Float64Array(gridSize * gridSize)
  for (let y = 0; y < gridSize; y++) {
    for (let xIndex = 0; xIndex < gridSize; xIndex++) {
      const x = coordForCell(xIndex)
      const yCoord = coordForCell(y)
      const head = Math.hypot(x, yCoord) < 0.92 ? 0.25 : -0.85
      const leftEye = Math.exp(-85 * ((x + 0.32) ** 2 + (yCoord + 0.28) ** 2))
      const rightEye = Math.exp(-85 * ((x - 0.32) ** 2 + (yCoord + 0.28) ** 2))
      const mouth =
        Math.exp(-120 * (Math.hypot(x, yCoord - 0.05) - 0.48) ** 2) *
        (yCoord > 0.05 ? 1 : 0)
      setTargetCell(target, xIndex, y, head - leftEye * 1.2 - rightEye * 1.2 + mouth * 1.1)
    }
  }
  return Array.from(target)
}

function zeroVector(length) {
  return Array.from({ length }, () => 0)
}

function zeroMatrix(rows, cols) {
  return Array.from({ length: rows }, () => zeroVector(cols))
}

function randomMatrix(rng, rows, cols, scale) {
  return Array.from({ length: rows }, () => Array.from({ length: cols }, () => rng.rand(scale)))
}

function randomVector(rng, length, scale) {
  return Array.from({ length }, () => rng.rand(scale))
}

function likeZeros(value) {
  return Array.isArray(value[0]) ? value.map((row) => row.map(() => 0)) : value.map(() => 0)
}

function resetMlpModel(layers, hidden) {
  const rng = makeRng()
  const dims = [2, ...Array.from({ length: layers }, () => hidden), 1]
  const model = {
    weights: [],
    biases: [],
    weightM1: [],
    weightM2: [],
    biasM1: [],
    biasM2: [],
  }
  for (let layer = 1; layer < dims.length; layer++) {
    const fanIn = dims[layer - 1]
    model.weights.push(randomMatrix(rng, dims[layer], fanIn, Math.sqrt(2 / fanIn)))
    model.biases.push(randomVector(rng, dims[layer], 0.1))
    model.weightM1.push(zeroMatrix(dims[layer], fanIn))
    model.weightM2.push(zeroMatrix(dims[layer], fanIn))
    model.biasM1.push(zeroVector(dims[layer]))
    model.biasM2.push(zeroVector(dims[layer]))
  }
  return model
}

function activate(value, activationName) {
  switch (activationName) {
    case "relu":
      return Math.max(0, value)
    case "sigmoid":
      return 1 / (1 + Math.exp(-value))
    default:
      return Math.tanh(value)
  }
}

function activationDerivative(value, activated, activationName) {
  switch (activationName) {
    case "relu":
      return value > 0 ? 1 : 0
    case "sigmoid":
      return activated * (1 - activated)
    default:
      return 1 - activated * activated
  }
}

function forward(model, x, y, activationName) {
  const activations = [[x, y]]
  const preactivations = []
  let current = [x, y]
  for (let layer = 0; layer < model.weights.length; layer++) {
    const weights = model.weights[layer]
    const biases = model.biases[layer]
    const z = new Array(weights.length)
    for (let i = 0; i < weights.length; i++) {
      let value = biases[i]
      for (let j = 0; j < weights[i].length; j++) value += weights[i][j] * current[j]
      z[i] = value
    }
    const isOutput = layer === model.weights.length - 1
    if (isOutput) {
      current = z
    } else {
      current = new Array(z.length)
      for (let i = 0; i < z.length; i++) current[i] = activate(z[i], activationName)
    }
    preactivations.push(z)
    activations.push(current)
  }
  return { preactivations, activations, value: Math.tanh(current[0]) }
}

function lossDerivative(prediction, yTarget, lossName) {
  const error = prediction - yTarget
  switch (lossName) {
    case "l1":
      return Math.sign(error)
    case "huber":
      return Math.abs(error) <= 0.25 ? error : 0.25 * Math.sign(error)
    default:
      return error
  }
}

function lossValue(prediction, yTarget, lossName) {
  const error = prediction - yTarget
  switch (lossName) {
    case "l1":
      return Math.abs(error)
    case "huber":
      return Math.abs(error) <= 0.25 ? 0.5 * error * error : 0.25 * (Math.abs(error) - 0.125)
    default:
      return 0.5 * error * error
  }
}

function zeroGradients(model) {
  return {
    weights: model.weights.map((layer) => layer.map((row) => row.map(() => 0))),
    biases: model.biases.map((layer) => layer.map(() => 0)),
  }
}

function addSampleGradients(model, target, index, settings, grads) {
  const [x, y] = coordsForIndex(index)
  const cache = forward(model, x, y, settings.activationName)
  let delta = [
    lossDerivative(cache.value, target[index], settings.lossName) * (1 - cache.value * cache.value),
  ]
  for (let layer = model.weights.length - 1; layer >= 0; layer--) {
    const weights = model.weights[layer]
    const nextDelta = layer > 0 ? new Array(weights[0].length) : null
    if (nextDelta) {
      for (let j = 0; j < nextDelta.length; j++) {
        let upstream = 0
        for (let i = 0; i < delta.length; i++) upstream += weights[i][j] * delta[i]
        nextDelta[j] =
          upstream *
          activationDerivative(
            cache.preactivations[layer - 1][j],
            cache.activations[layer][j],
            settings.activationName,
          )
      }
    }
    const prev = cache.activations[layer]
    for (let i = 0; i < delta.length; i++) {
      grads.biases[layer][i] += delta[i]
      for (let j = 0; j < prev.length; j++) grads.weights[layer][i][j] += delta[i] * prev[j]
    }
    delta = nextDelta
  }
}

function updateBatch(model, grads, batchSize, settings, step) {
  const beta1 = 0.9
  const beta2 = 0.999
  const t = step + 1
  const correction1 = 1 - beta1 ** t
  const correction2 = 1 - beta2 ** t
  for (let layer = 0; layer < model.weights.length; layer++) {
    for (let i = 0; i < model.weights[layer].length; i++) {
      const biasGrad = grads.biases[layer][i] / batchSize
      if (settings.optimizerName === "adam") {
        model.biasM1[layer][i] = beta1 * model.biasM1[layer][i] + (1 - beta1) * biasGrad
        model.biasM2[layer][i] = beta2 * model.biasM2[layer][i] + (1 - beta2) * biasGrad * biasGrad
        model.biases[layer][i] -=
          (settings.lr * model.biasM1[layer][i]) /
          correction1 /
          (Math.sqrt(model.biasM2[layer][i] / correction2) + 1e-8)
      } else {
        model.biases[layer][i] -= settings.lr * biasGrad
      }
      for (let j = 0; j < model.weights[layer][i].length; j++) {
        const weightGrad = grads.weights[layer][i][j] / batchSize
        if (settings.optimizerName === "adam") {
          model.weightM1[layer][i][j] =
            beta1 * model.weightM1[layer][i][j] + (1 - beta1) * weightGrad
          model.weightM2[layer][i][j] =
            beta2 * model.weightM2[layer][i][j] + (1 - beta2) * weightGrad * weightGrad
          model.weights[layer][i][j] -=
            (settings.lr * model.weightM1[layer][i][j]) /
            correction1 /
            (Math.sqrt(model.weightM2[layer][i][j] / correction2) + 1e-8)
        } else {
          model.weights[layer][i][j] -= settings.lr * weightGrad
        }
      }
    }
  }
}

function trainBatch(state, settings) {
  const grads = zeroGradients(state.model)
  const batchSize = settings.inspectOnly ? 1 : 32
  for (let i = 0; i < batchSize; i++) {
    if (settings.inspectOnly) {
      addSampleGradients(state.model, state.target, indexForCell(7, 11), settings, grads)
    } else {
      state.trainCursor = (state.trainCursor + 37) % state.target.length
      addSampleGradients(state.model, state.target, state.trainCursor, settings, grads)
    }
  }
  updateBatch(state.model, grads, batchSize, settings, state.step)
  state.step++
}

function flattenModel(model) {
  return [
    ...model.weights.flat(2),
    ...model.biases.flat(1),
    ...model.weightM1.flat(2),
    ...model.weightM2.flat(2),
    ...model.biasM1.flat(1),
    ...model.biasM2.flat(1),
  ]
}

function averageLoss(model, target, activationName, lossName) {
  let total = 0
  let count = 0
  for (let i = 0; i < target.length; i += 5) {
    const [x, y] = coordsForIndex(i)
    total += lossValue(forward(model, x, y, activationName).value, target[i], lossName)
    count++
  }
  return total / count
}

function runCase(settings) {
  const state = {
    model: resetMlpModel(settings.layers, settings.hidden),
    target: smileTarget(),
    step: 0,
    trainCursor: 0,
  }
  for (let i = 0; i < settings.steps; i++) trainBatch(state, settings)
  const probes = [indexForCell(7, 11), indexForCell(32, 32), indexForCell(51, 19), indexForCell(4, 58)]
  return {
    settings,
    step: state.step,
    trainCursor: state.trainCursor,
    modelFlat: flattenModel(state.model),
    probes: probes.map((index) => {
      const [x, y] = coordsForIndex(index)
      return forward(state.model, x, y, settings.activationName).value
    }),
    averageLoss: averageLoss(state.model, state.target, settings.activationName, settings.lossName),
  }
}

function defaultCases() {
  const cases = []
  for (const layers of [1, 3, 6]) {
    for (const hidden of [4, 24]) {
      for (const activationName of ["tanh", "relu", "sigmoid"]) {
        for (const lossName of ["mse", "l1", "huber"]) {
          for (const optimizerName of ["sgd", "adam"]) {
            for (const inspectOnly of [true, false]) {
              cases.push({
                layers,
                hidden,
                activationName,
                lossName,
                optimizerName,
                inspectOnly,
                lr: 1e-3,
                steps: 6,
              })
            }
          }
        }
      }
    }
  }
  return cases
}

const cases = process.argv[2] ? JSON.parse(process.argv[2]) : defaultCases()
process.stdout.write(JSON.stringify(cases.map(runCase)))
