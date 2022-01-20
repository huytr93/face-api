const input = document.querySelector("#input-file");
const showImage = document.querySelector("#show-image");
const MODEL_URL = './models'
let faceMatcher = null;
async function loadData () {
    const labels = ["Hamabe Minami", "Imada Mio", "Shiraishi Mai", "Clark Kent"];
    const faceDescriptors = [];
    for(const label of labels) {
        const descriptors = [];
        for(let i = 1; i <= 4; i++) {
            const image = await faceapi.fetchImage(`/data/${label}/${i}.png`);
            const detection = await faceapi.detectSingleFace(image).withFaceLandmarks().withFaceDescriptor();
            descriptors.push(detection.descriptor);
        }
        faceDescriptors.push(new faceapi.LabeledFaceDescriptors(label, descriptors));
    }
    alert(`Load data done`);
    return faceDescriptors;
}

window.addEventListener("load", async () => {
    await Promise.all([
        faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
        faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
        faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
    ])
    const humanData = await loadData();
    faceMatcher = new faceapi.FaceMatcher(humanData, 0.7);
});

input.addEventListener("change", async() => {
    const file = input.files[0];
    const image = await faceapi.bufferToImage(file);
    const canvas = await faceapi.createCanvasFromMedia(image);

    showImage.innerHTML = ""
    showImage.append(image);
    showImage.append(canvas);

    const displaySize = { width: image.width, height: image.height };
    faceapi.matchDimensions(canvas, displaySize);

    const detections = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors();
    const resizedDetections = faceapi.resizeResults(detections, displaySize)
    // draw detections into the canvas
    faceapi.draw.drawDetections(canvas, resizedDetections);

    for(const detection of resizedDetections) {
        const box = detection.detection.box;
        const drawOptions = {
            label: faceMatcher.findBestMatch(detection.descriptor),
        }
        const drawBox = new faceapi.draw.DrawBox(box, drawOptions);
        drawBox.draw(canvas)
    }
})




