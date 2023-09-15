// Compute Aritifical Neural Networks.
// Programmed by Dr. Jaywan Chung
// v0.1 updated on Sep 13, 2023

"use strict";

class Matrix {
    constructor(nRows, nCols, array=null) {
        this.nRows = nRows;
        this.nCols = nCols;

        this.setFromArray(nRows, nCols, array);
    }
    clone() {
        return new Matrix(this.nRows, this.nCols, this.array);
    }
    setFromArray(nRows, nCols, array) {
        if (array == null) {
            this.array = new Float64Array(nRows * nCols);
            return;
        }
        if (nRows * nCols != array.length) {
            throw new Error(`Matrix initalization failure: size does not match; nRows*nCols=${nRows * nCols} != length=${array.length}`);
        }
        this.array = new Float64Array(array);
    }
    // 'row' and 'col' starts with zero.
    getElement(row, col) {
        return this.array[this.nCols*row + col];
    }
    setElement(row, col, value) {
        this.array[this.nCols*row + col] = value;
    }
    isEqualAsArray(array) {
        for (let i=0; i<this.nRows*this.nCols; i++) {
            if (this.array[i] != array[i]) {
                return false;
            }
        }
        return true;
    }
    hasSameSize(otherMatrix) {
        if ((this.nRows == otherMatrix.nRows) && (this.nCols == otherMatrix.nCols)) {
            return true;
        }
        return false;
    }
    // compute 'this * otherMatrix' and write its value to 'outputMatrix'.
    multiply(otherMatrix, outputMatrix) {
        if ((this.nCols != otherMatrix.nRows) || (this.nRows != outputMatrix.nRows) || (otherMatrix.nCols != outputMatrix.nCols)) {
            throw new Error("Matrix multiplication failure: size does not match!");
        };
        let value;
        for (let i=0; i<outputMatrix.nRows; i++) {
            for (let j=0; j<outputMatrix.nCols; j++) {
                value = 0.0;
                for (let k=0; k<this.nCols; k++) {
                    value += this.getElement(i, k) * otherMatrix.getElement(k, j);
                }
                outputMatrix.setElement(i, j, value);
            }
        }
        return;
    }
    plus(otherMatrix, outputMatrix) {
        if (!this.hasSameSize(otherMatrix) || !(this.hasSameSize(outputMatrix))) {
            throw new Error("Matrix addition failure: size does not match!");
        };
        for (let i=0; i<outputMatrix.nRows; i++) {
            for (let j=0; j<outputMatrix.nCols; j++) {
                outputMatrix.setElement(i, j, this.getElement(i, j) + otherMatrix.getElement(i, j));
            }
        }
        return;
    }
    composition(compositeFunction, outputMatrix) {
        for (let i=0; i<this.nRows; i++) {
            for (let j=0; j<this.nCols; j++) {
                outputMatrix.setElement(i, j, compositeFunction(this.getElement(i, j)));
            }
        }
    }
    fill(value) {
        for (let i=0; i<this.nRows; i++) {
            for (let j=0; j<this.nCols; j++) {
                this.setElement(i, j, value);
            }
        }
    }
}

class Layer {
    constructor(nInputs, nOutputs, weights, biases, activation="linear") {
        Layer.supportedActivations = ["linear", "elu", "softplus"];
        if (!Layer.supportedActivations.includes(activation)) {
            throw new Error(`Layer initialization failure: Unknown activation function ("${activation}")!`);
        };

        this.nInputs = nInputs;
        this.nOutputs = nOutputs;
        this.weightMatrix = new Matrix(nOutputs, nInputs, weights);
        this.biasMatrix = new Matrix(nOutputs, 1, biases);
        this.activation = activation;
        this.outputMatrix = new Matrix(nOutputs, 1);
    }
    // evaluate the layer and save the result into 'outputMatrix'.
    evaluate(inputMatrix) {
        // compute 'Wx + b'
        this.weightMatrix.multiply(inputMatrix, this.outputMatrix);
        this.outputMatrix.plus(this.biasMatrix, this.outputMatrix);
        // composite activation functions
        switch (this.activation) {
            case "linear":
                // do nothing
                break;
            case "elu":
                this.outputMatrix.composition(eluActivation, this.outputMatrix);
                break;
            case "softplus":
                this.outputMatrix.composition(softplusActivation, this.outputMatrix);
                break;
            default:
                throw new Error(`Layer evaluation failure: Unknown activation function ("${this.activation}")!`);
        }
    }
}

function eluActivation(x) {
    if (x >= 0.0) {
        return x;
    }
    return Math.exp(x) - 1;
}

function softplusActivation(x) {
    return Math.log(Math.exp(x) + 1);
}

class FullyConnectedNeuralNetwork {
    constructor(nInputs, weightsArray, biasesArray, activationArray) {
        const nLayers = weightsArray.length
        if ((biasesArray.length != nLayers) || (activationArray.length != nLayers)) {
            throw new Error("Neural network initialization failure: number of layers mismatch!");
        }
        this.layers = new Array(nLayers);
        this.nLayers = nLayers;
        this.nInputs = nInputs;

        // generate hidden layers
        let nInputsOfLayer, nOutputsOfLayer, weights, biases, activation;
        nInputsOfLayer = nInputs;
        for (let i=0; i<nLayers; i++) {
            weights = weightsArray[i];
            biases = biasesArray[i];
            activation = activationArray[i];
            nOutputsOfLayer = Math.floor(weights.length / nInputsOfLayer);
            let layer = new Layer(nInputsOfLayer, nOutputsOfLayer, weights, biases, activation);
            this.layers[i] = layer;
            nInputsOfLayer = nOutputsOfLayer;
        }
        // infer the number of outputs
        this.nOutputs = nOutputsOfLayer;
        this.outputMatrix = null;
    }
    // evaluate the neural network and save the result into 'outputMatrix'.
    evaluate(inputMatrix) {
        if (inputMatrix.nCols != 1) {
            throw new Error("LaNN evaluation error: input matrix must have only one column!");
        }
        let prevLayerOutput = inputMatrix;
        for (const layer of this.layers) {
            layer.evaluate(prevLayerOutput);
            prevLayerOutput = layer.outputMatrix;
        }
        // copy the final result as the output
        this.outputMatrix = prevLayerOutput;
    }
}

class LatentSpaceNeuralNetwork {
    constructor(embeddingNet, dictionaryNet) {
        this.embeddingNet = embeddingNet;
        this.dictionaryNet = dictionaryNet;
        this.dimLatentSpace = embeddingNet.nOutputs;
        this.nSurfaceInputs = dictionaryNet.nInputs - this.dimLatentSpace;
        this.nNonSurfaceInputs = embeddingNet.nInputs;
        this.nInputs = this.nNonSurfaceInputs + this.nSurfaceInputs;
        this.nOutputs = dictionaryNet.nOutputs;
        this.outputMatrix = null;
        this.dictionaryInput = new Matrix(this.dictionaryNet.nInputs, 1);
    }
    // evaluate the neural network and save the result into 'outputMatrix'.
    evaluate(inputMatrix) {
        if (inputMatrix.nCols != 1) {
            throw new Error("LaNN evaluation error: input matrix must have only one column!");
        }
        const embeddingInput = new Matrix(this.nNonSurfaceInputs, 1, inputMatrix.array.slice(0, this.nNonSurfaceInputs));
        this.embeddingNet.evaluate(embeddingInput);
        for (let i=0; i<this.dimLatentSpace; i++) {
            this.dictionaryInput.setElement(i, 0, this.embeddingNet.outputMatrix.getElement(i, 0));
        }
        for (let i=0; i<this.nSurfaceInputs; i++) {
            this.dictionaryInput.setElement(i+this.dimLatentSpace, 0, inputMatrix.getElement(i+this.nNonSurfaceInputs, 0));
        }
        this.dictionaryNet.evaluate(this.dictionaryInput);
        this.outputMatrix = this.dictionaryNet.outputMatrix;
    }
}