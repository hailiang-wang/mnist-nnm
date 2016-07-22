'use strict';
/**
 * [mnist description]
 * @type {[type]}
 */
const debug = require('debug')('mnist-nnm');

// 
// Synaptic is a javascript neural network library for
// node.js and the browser, its generalized algorithm 
// is architecture-free, so you can build and train
// basically any type of first order or even second 
// order neural network architectures.
const synaptic = require('synaptic');
const Trainer = synaptic.Trainer;
const Layer = synaptic.Layer;
const Network = synaptic.Network;

function Model() {
    this.isTrained = false;
    this.inputLayer = new Layer(784);
    this.hiddenLayer = new Layer(100);
    this.outputLayer = new Layer(10);

    this.inputLayer.project(this.hiddenLayer);
    this.hiddenLayer.project(this.outputLayer);

    this.mnistNN = new Network({
        input: this.inputLayer,
        hidden: [this.hiddenLayer],
        output: this.outputLayer
    });

    this.trainer = new Trainer(this.mnistNN);
};

// train the network with trainingSet and options
// https://github.com/cazala/synaptic/wiki/Trainer
Model.prototype.train = function(trainingSet, options) {
    this.trainer.train(trainingSet, options);
    debug('model training done.');
    this.isTrained = true;
};

Model.prototype.isTrained = function() {
    return this.isTrained;
}

/**
 * [activate description]
 * @param  {[type]} testData node mnist data
 * @return {[type]}          [description]
 */
Model.prototype.activate = function(testData) {
    if (this.isTrained) {
        return ((raw) => {
            debug("softmax");
            debug("------------------------------------");

            let maximum = raw.reduce(function(p, c) {
                return p > c ? p : c;
            });
            let nominators = raw.map(function(e) {
                return Math.exp(e - maximum);
            });
            let denominator = nominators.reduce(function(p, c) {
                return p + c;
            });
            let softmax = nominators.map(function(e) {
                return e / denominator;
            });

            let maxIndex = 0;
            softmax.reduce((p, c, i) => {
                if (p < c) {
                    maxIndex = i;
                    return c;
                } else return p;
            });


            let result = [];
            for (let i = 0; i < raw.length; i++) {
                if (i == maxIndex)
                    result.push(1);
                else
                    result.push(0);
            }

            return result;
        })(this.mnistNN.activate(testData.input));
    } else {
        throw new Error('neural network is not trained yet.');
    }
}

Model.prototype.validate = function(testData) {
    var resolveDigit = t => {
        let val;
        for (let i = 0; i <= 9; i++) {
            if (t[i] == 1) {
                val = i;
                break;
            }
        }
        return val
    };

    let realVal = resolveDigit(testData.output);
    debug('real value %s', realVal);

    let recognizedVal = resolveDigit(this.activate(testData));
    debug('recognized value %s', recognizedVal);

    return realVal === recognizedVal;
}

exports.Model = Model;
