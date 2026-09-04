import "./style.css";
import { conditionsAt, debruijn, decodePrediction, randomOffsets } from "./geometry.js";
import { BrowserPredictor, loadManifest } from "./model.js";
import { PhasePlot } from "./phase-plot.js";
import { RhombusRenderer } from "./renderer.js";

const elements = {
  canvas: document.querySelector("#tiling"),
  loading: document.querySelector("#loading"),
  backend: document.querySelector("#backend"),
  latency: document.querySelector("#latency"),
  play: document.querySelector("#play"),
  time: document.querySelector("#time"),
  timeValue: document.querySelector("#time-value"),
  speed: document.querySelector("#speed"),
  model: document.querySelector("#model"),
  thetaMode: document.querySelector("#theta-mode"),
  randomize: document.querySelector("#randomize"),
  showExact: document.querySelector("#show-exact"),
  showPrediction: document.querySelector("#show-prediction"),
  showAArcs: document.querySelector("#show-a-arcs"),
  showCArcs: document.querySelector("#show-c-arcs"),
  gamma01: document.querySelector("#gamma01"),
  gamma23: document.querySelector("#gamma23"),
  gamma01Value: document.querySelector("#gamma01-value"),
  gamma23Value: document.querySelector("#gamma23-value"),
};

const renderer = new RhombusRenderer(elements.canvas);
const gamma01Plot = new PhasePlot(elements.gamma01);
const gamma23Plot = new PhasePlot(elements.gamma23);
let offsets = randomOffsets();
let time = 0;
let playing = true;
let previousFrame = performance.now();
let models = [];
let exact = null;
let prediction = null;

function updateReadout(condition, appendTrail) {
  gamma01Plot.update(condition.gamma[0], condition.gamma[1], appendTrail);
  gamma23Plot.update(condition.gamma[2], condition.gamma[3], appendTrail);
  elements.gamma01Value.textContent =
    `${condition.gamma[0].toFixed(3)}, ${condition.gamma[1].toFixed(3)}`;
  elements.gamma23Value.textContent =
    `${condition.gamma[2].toFixed(3)}, ${condition.gamma[3].toFixed(3)}`;
  elements.timeValue.textContent = time.toFixed(2);
  elements.time.value = String(time);
}

const predictor = new BrowserPredictor((status) => {
  if (status.state === "loading") {
    elements.loading.classList.remove("hidden", "error");
    elements.loading.textContent = `Loading ${status.model}…`;
    elements.backend.textContent = "backend · loading";
  } else if (status.state === "ready") {
    elements.loading.classList.add("hidden");
    elements.backend.textContent = `backend · ${status.backend}`;
    requestCurrentPrediction();
  } else {
    elements.loading.classList.remove("hidden");
    elements.loading.classList.add("error");
    elements.loading.textContent = `Model error · ${status.error?.message ?? status.error}`;
    elements.backend.textContent = "backend · error";
  }
});

function updateGeometry(appendTrail = false) {
  const condition = conditionsAt(time, offsets, elements.thetaMode.value === "moving");
  exact = debruijn(condition.gamma, condition.thetaOffset);
  renderer.setGeometry(exact, prediction);
  updateReadout(condition, appendTrail);
  if (predictor.session) {
    predictor.request([...condition.gamma, condition.thetaOffset], (data, latency) => {
      prediction = decodePrediction(data);
      elements.latency.textContent = `${latency.toFixed(1)} ms`;
      renderer.setGeometry(exact, prediction);
    });
  }
}

function requestCurrentPrediction() {
  updateGeometry();
}

async function selectModel() {
  prediction = null;
  renderer.setGeometry(exact, null);
  const model = models.find((candidate) => candidate.id === elements.model.value);
  if (model) await predictor.load(model);
}

elements.play.addEventListener("click", () => {
  playing = !playing;
  elements.play.textContent = playing ? "Pause" : "Play";
  previousFrame = performance.now();
});

elements.time.addEventListener("input", () => {
  time = Number(elements.time.value);
  gamma01Plot.reset();
  gamma23Plot.reset();
  updateGeometry();
});

elements.randomize.addEventListener("click", () => {
  offsets = randomOffsets();
  prediction = null;
  gamma01Plot.reset();
  gamma23Plot.reset();
  updateGeometry();
});

elements.model.addEventListener("change", () => void selectModel());
elements.thetaMode.addEventListener("change", updateGeometry);
elements.speed.addEventListener("change", () => {
  previousFrame = performance.now();
});

function updateVisibility() {
  renderer.setVisibility(
    elements.showExact.checked,
    elements.showPrediction.checked,
    elements.showAArcs.checked,
    elements.showCArcs.checked,
  );
}
elements.showExact.addEventListener("change", updateVisibility);
elements.showPrediction.addEventListener("change", updateVisibility);
elements.showAArcs.addEventListener("change", updateVisibility);
elements.showCArcs.addEventListener("change", updateVisibility);

function animate(now) {
  if (playing) {
    const delta = Math.min((now - previousFrame) / 1000, 0.1);
    time = (time + delta * Number(elements.speed.value)) % Number(elements.time.max);
    updateGeometry(true);
  }
  previousFrame = now;
  requestAnimationFrame(animate);
}

async function initialize() {
  updateGeometry();
  try {
    const manifest = await loadManifest();
    models = manifest.models;
    for (const model of models) {
      const option = document.createElement("option");
      option.value = model.id;
      option.textContent = model.label;
      elements.model.append(option);
    }
    const preferred = models.find((model) => model.id.includes("256x16")) ?? models[0];
    elements.model.value = preferred.id;
    await selectModel();
  } catch (error) {
    elements.loading.classList.add("error");
    elements.loading.textContent = `Setup error · ${error.message}`;
  }
  requestAnimationFrame(animate);
}

void initialize();
