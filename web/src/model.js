import * as ort from "onnxruntime-web/webgpu";

export async function loadManifest() {
  const response = await fetch(`${import.meta.env.BASE_URL}models/manifest.json`);
  if (!response.ok) throw new Error(`model manifest: ${response.status}`);
  return response.json();
}

async function createSession(url, preferredBackend) {
  const attempts =
    preferredBackend === "wasm"
      ? [["wasm"]]
      : typeof navigator !== "undefined" && "gpu" in navigator
        ? [["webgpu"], ["wasm"]]
        : [["wasm"]];
  let lastError;
  for (const executionProviders of attempts) {
    try {
      const session = await ort.InferenceSession.create(url, {
        executionProviders,
        graphOptimizationLevel: "all",
      });
      return { session, backend: executionProviders[0] };
    } catch (error) {
      lastError = error;
      console.warn(`Could not initialize ${executionProviders[0]}`, error);
    }
  }
  throw lastError;
}

export class BrowserPredictor {
  constructor(onStatus = () => {}) {
    this.onStatus = onStatus;
    this.session = null;
    this.backend = "loading";
    this.generation = 0;
    this.busy = false;
    this.pending = null;
  }

  async load(model, preferredBackend = "webgpu") {
    const generation = ++this.generation;
    this.session = null;
    this.onStatus({ state: "loading", model: model.label });
    const url = `${import.meta.env.BASE_URL}models/${model.file}`;
    const loaded = await createSession(url, preferredBackend);
    if (generation !== this.generation) return;
    this.session = loaded.session;
    this.backend = loaded.backend;
    this.onStatus({ state: "ready", model: model.label, backend: loaded.backend });
  }

  request(condition, callback) {
    this.pending = { condition, callback, generation: this.generation };
    if (!this.busy) void this.#drain();
  }

  async #drain() {
    if (!this.session || !this.pending) return;
    this.busy = true;
    while (this.session && this.pending) {
      const request = this.pending;
      this.pending = null;
      const started = performance.now();
      try {
        const tensor = new ort.Tensor("float32", Float32Array.from(request.condition), [1, 6]);
        const output = await this.session.run({ condition: tensor });
        if (request.generation === this.generation) {
          request.callback(output.prediction.data, performance.now() - started);
        }
      } catch (error) {
        if (request.generation === this.generation) {
          this.onStatus({ state: "error", error });
        }
      }
    }
    this.busy = false;
  }
}
