import { MnistData } from './data.js';

async function predict() {
    const data = new MnistData();
    await data.load();

    // Get a random image from the dataset for prediction
    const testDataSize = 1; // Number of images to predict
    const [testXs, testYs] = tf.tidy(() => {
        const testData = data.nextTestBatch(testDataSize);
        return [
            testData.xs.reshape([testDataSize, 28, 28, 1]),
            testData.labels.argMax(-1)
        ];
    });

    const model = await tf.loadLayersModel('path_to_your_model/model.json');

    // Predict the label for the image
    const prediction = model.predict(testXs).argMax(-1).dataSync()[0];
    const actualLabel = testYs.dataSync()[0];

    // Show the image, actual label, and predicted label
    showImage(testDataSize, testXs, actualLabel, prediction);

    testXs.dispose();
    testYs.dispose();
}

function showImage(testDataSize, testXs, actualLabel, predictedLabel) {
    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    const ctx = canvas.getContext('2d');

    // Draw the image on the canvas
    for (let i = 0; i < testDataSize; i++) {
        const image = tf.tidy(() => testXs.slice([i, 0], [1, testXs.shape[1]]).reshape([28, 28, 1]));
        tf.browser.toPixels(image, canvas).then(() => {
            // Display the canvas with the image
            const imageDiv = document.getElementById('image-div');
            imageDiv.innerHTML = ''; // Clear previous image
            imageDiv.appendChild(canvas);

            // Display actual and predicted labels
            const infoDiv = document.getElementById('info-div');
            infoDiv.innerHTML = `Actual Label: ${actualLabel}<br>Predicted Label: ${predictedLabel}`;
        });
    }
}

document.getElementById('predict-btn').addEventListener('click', predict);
