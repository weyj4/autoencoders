import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import MnistData from './data';

function component() {
  const element = document.createElement('div');

  // Lodash, currently included via a script, is required for this line to work
  element.innerHTML = "Hello, webpack";

  return element;
}

async function showExamples(data, input, output) {
  const surface = tfvis.visor().surface({
    name: 'Input Data Examples',
    tab: 'Input Data',
  });

  const examples = data.nextTestBatch(20);
  console.log(`examples: ${JSON.stringify(examples)}`);
  console.log(`input: ${JSON.stringify(input)}`);
  console.log(`output: ${JSON.stringify(output)}`);
  const numExamples = 20;

  for (let i = 0; i < numExamples; i++) {
    const inputTensor = tf.tidy(() => {
      const shape = 784;
      return input.slice([i, 0], [1, shape]).reshape([28, 28, 1]);
      // return examples.xs
      //   .slice([i, 0], [1, shape])
      //   .reshape([28, 28, 1]);
    });

    const outputTensor = tf.tidy(() => {
      const shape = 784;
      return output.slice([i, 0], [1, shape]).reshape([28, 28, 1]);
    })

    const canvasOne = document.createElement('canvas');
    canvasOne.width = 28;
    canvasOne.height = 28;
    canvasOne.style = 'margin: 4px;';
    await tf.browser.toPixels(inputTensor, canvasOne);
    const canvasTwo = document.createElement('canvas');
    canvasTwo.width = 28;
    canvasTwo.height = 28;
    canvasTwo.style = 'margin: 4px;';
    await tf.browser.toPixels(outputTensor, canvasTwo);
    surface.drawArea.appendChild(canvasOne);
    surface.drawArea.appendChild(canvasTwo);

    inputTensor.dispose();
    outputTensor.dispose();
  }
};

function constructDense() {
  const model = tf.sequential({
    layers: [
      tf.layers.dense({ inputShape: [784], units: 128, activation: 'relu', name: 'encoder' }),
      tf.layers.dense({ units: 784, activation: 'sigmoid', name: 'decoder' }),
    ],
  });

  model.compile({
    optimizer: tf.train.adam(),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  });

  model.summary();

  return model;
}

function constructConv2d() {
  const model = tf.sequential({
    layers: [
      tf.layers.conv2d({
        inputShape: [28, 28, 1],
        filters: 16,
        kernelSize: [3, 3],
        activation: 'relu',
        padding: 'same',
      }),
      tf.layers.maxPooling2d({
        poolSize: [2, 2],
        padding: 'same',
      }),
      tf.layers.conv2d({
        filters: 8,
        kernelSize: [3, 3],
        activation: 'relu',
        padding: 'same',
      }),
      tf.layers.maxPooling2d({
        poolSize: [2, 2],
        padding: 'same',
      }),
      tf.layers.conv2d({
        filters: 8,
        kernelSize: [3, 3],
        activation: 'relu',
        padding: 'same',
      }),
      tf.layers.maxPooling2d({
        poolSize: [2, 2],
        padding: 'same',
      }),
      tf.layers.conv2d({
        filters: 8,
        kernelSize: [3, 3],
        activation: 'relu',
        padding: 'same',
      }),
      tf.layers.upSampling2d({
        size: [2, 2],
      }),
      tf.layers.conv2d({
        filters: 8,
        kernelSize: [3, 3],
        activation: 'relu',
        padding: 'same',
      }),
      tf.layers.upSampling2d({
        size: [2, 2],
      }),
      tf.layers.conv2d({
        filters: 16,
        kernelSize: [3, 3],
        activation: 'relu',
        padding: 'same',
      }),
      tf.layers.upSampling2d({
        size: [2, 2],
      }),
      tf.layers.conv2d({
        filters: 1,
        kernelSize: [3, 3],
        activation: 'sigmoid',
        padding: 'same',
      }),
    ],
  });

  model.compile({
    optimizer: tf.train.adam(),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  })

  model.summary();
  
  return model;
}

async function getTrainTestData(data) {
  const TRAIN_DATA_SIZE = 5500;
  const TEST_DATA_SIZE = 1000;

  const [ trainXs, trainYs ] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
    return [
      d.xs,
      // d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
      d.labels
    ];
  });

  const [ testXs, testYs ] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE);
    console.log(`d: ${JSON.stringify(d)}`);
    return [
      d.xs,
      // d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
      d.labels
    ];
  });

  return {
    trainXs,
    trainYs,
    testXs,
    testYs
  };
}

async function train(model, trainXs, trainYs, testXs, testYs) {
  const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
  const container = {
    name: 'Model Training', styles: { height: '1000px' }
  };
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
  const BATCH_SIZE = 256;

  console.log(`data.trainXs.shape: ${trainXs.shape}`);
  console.log(`data.trainYs.shape: ${trainYs.shape}`);
  console.log(`data.testXs.shape: ${testXs.shape}`);
  console.log(`data.testYs.shape: ${testYs.shape}`);

  const fit = await model.fit(trainXs, trainXs, {
    batchSize: BATCH_SIZE,
    validationData: [ testXs, testXs ],
    epochs: 10,
    shuffle: true,
    callbacks: fitCallbacks,
  });
  const predictions = await model.predict(testXs);
  return predictions
}

async function run() {
  const data = new MnistData();
  await data.load();
  // const vae = constructDense();
  const vae = constructConv2d();
  const {
    trainXs,
    trainYs,
    testXs,
    testYs
  } = await getTrainTestData(data);
  const output = await train(vae, trainXs, trainYs, testXs, testYs);
  // const output = await train(vae, trainXs.reshape(5500, 28, 28, 1), trainYs, testXs.reshape(1000, 28, 28, 1), testYs);
  // const output = fit.predict(testXs);
  await showExamples(data, testXs, output);
  const element = document.createElement('div');

  element.innerHTML = "Hello, webpack";

  return element;
}

import HelloWorld from 'components/HelloWorld';
import 'main.css';
import { type } from 'os';
import { modelSummary } from '@tensorflow/tfjs-vis/dist/show/model';

const main = async () => {
    HelloWorld();
    await run();
}

main().then(() => console.log('Started'));