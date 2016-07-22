'use strict';
/**
 * testcase for nn
 * @type {[type]}
 */
const test = require('ava');
const nn = require('../');
const synaptic = require('synaptic');
const Trainer = synaptic.Trainer;
const log = require('log4js').getLogger('nn.test');
//
// The goal of this library is to provide an easy-to-use 
// way for training and testing MNIST digits for neural 
// networks (either in the browser or node.js). 
// It includes 10000 different samples of mnist digits. 
const mnist = require('mnist');

/**
 * A Simple NNM to verify the functions.
 * @param  {[type]} t [description]
 * @return {[type]}   [description]
 */
test.cb('Simple Neural Network', t => {

    // nn model
    let nnm = new nn.Model();
    // https://github.com/cazala/synaptic/wiki/Trainer
    let options = {
        rate: .2,
        iterations: 1,
        error: .1,
        shuffle: true,
        log: 1,
        cost: Trainer.cost.CROSS_ENTROPY
    };
    // mnist.set(trainingAmount, testAmount) 
    let set = mnist.set(700, 20);
    let trainingSet = set.training;
    let testSet = set.test;

    log.info('start to train', JSON.stringify(options));
    nnm.train(trainingSet, options);

    t.true(nnm.isTrained);
    log.info('training done.');
    nnm.validate(testSet[0]);
    t.pass();
    t.end();
});


/**
 * An Advanced NNM to check result.
 * @param  {[type]} t [description]
 * @return {[type]}   [description]
 */
test.only.cb('Advanced Neural Network', t => {

    // nn model
    let nnm = new nn.Model();
    // https://github.com/cazala/synaptic/wiki/Trainer
    let options = {
        rate: .1,
        iterations: 20,
        error: .01,
        shuffle: true,
        log: 1,
        cost: Trainer.cost.CROSS_ENTROPY
    };
    // mnist.set(trainingAmount, testAmount) 
    let set = mnist.set(700, 20);
    let trainingSet = set.training;
    let testSet = set.test;

    log.info('start to train', JSON.stringify(options));
    nnm.train(trainingSet, options);

    t.true(nnm.isTrained);
    log.info('training done.');
    for (let i = 0; i < 100; i++) {
        nnm.validate(testSet[i]);
    }
    t.pass();
    t.end();
});
