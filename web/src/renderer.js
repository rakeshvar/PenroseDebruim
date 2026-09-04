import { rhombusVertices } from "./geometry.js";

export const AMETHYST = {
  color0: "#5A189A",
  color1: "#C77DFF",
  exact0: "#72EFDD",
  exact1: "#FFD166",
  aArc: "#00B4D8",
  cArc: "#FFB703",
  background: "#14001F",
  stroke: "#F8EFFF",
};

function hexToRgb(hex) {
  return [1, 3, 5].map((offset) => Number.parseInt(hex.slice(offset, offset + 2), 16));
}

function mixHex(left, right, amount) {
  const a = hexToRgb(left);
  const b = hexToRgb(right);
  const t = Math.max(0, Math.min(1, amount));
  return `rgb(${a.map((value, index) => Math.round(value + (b[index] - value) * t)).join(" ")})`;
}

export class RhombusRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.exact = null;
    this.prediction = null;
    this.showExact = true;
    this.showPrediction = true;
    this.showAArcs = false;
    this.showCArcs = false;
    this.resizeObserver = new ResizeObserver(() => this.draw());
    this.resizeObserver.observe(canvas);
  }

  setGeometry(exact, prediction) {
    this.exact = exact;
    this.prediction = prediction;
    this.draw();
  }

  setVisibility(
    showExact,
    showPrediction,
    showAArcs = false,
    showCArcs = false,
  ) {
    this.showExact = showExact;
    this.showPrediction = showPrediction;
    this.showAArcs = showAArcs;
    this.showCArcs = showCArcs;
    this.draw();
  }

  #fit(width, height) {
    const worldSize = 24;
    return {
      centerX: 0,
      centerY: 0,
      scale: Math.min(width, height) / worldSize,
    };
  }

  #polygon(vertices, view, width, height) {
    const context = this.context;
    context.beginPath();
    vertices.forEach((point, index) => {
      const [screenX, screenY] = this.#screenPoint(point, view, width, height);
      if (index === 0) context.moveTo(screenX, screenY);
      else context.lineTo(screenX, screenY);
    });
    context.closePath();
  }

  #screenPoint([x, y], view, width, height) {
    return [
      width / 2 + (x - view.centerX) * view.scale,
      height / 2 - (y - view.centerY) * view.scale,
    ];
  }

  #cornerArc(center, first, second, color, lineWidth) {
    const context = this.context;
    const start = [(center[0] + first[0]) / 2, (center[1] + first[1]) / 2];
    const end = [(center[0] + second[0]) / 2, (center[1] + second[1]) / 2];
    const radius = Math.hypot(first[0] - center[0], first[1] - center[1]) / 2;
    const startAngle = Math.atan2(start[1] - center[1], start[0] - center[0]);
    const endAngle = Math.atan2(end[1] - center[1], end[0] - center[0]);
    let delta = endAngle - startAngle;
    if (delta > Math.PI) delta -= 2 * Math.PI;
    if (delta < -Math.PI) delta += 2 * Math.PI;
    context.beginPath();
    context.arc(
      center[0],
      center[1],
      radius,
      startAngle,
      startAngle + delta,
      delta < 0,
    );
    context.strokeStyle = color;
    context.lineWidth = lineWidth;
    context.stroke();
  }

  #drawArcs(tiles, view, width, height, lineWidth) {
    const context = this.context;
    context.globalAlpha = 0.9;
    context.lineCap = "round";
    for (const tile of tiles) {
      const [a, b, c, d] = rhombusVertices(tile).map((point) =>
        this.#screenPoint(point, view, width, height),
      );
      if (this.showAArcs) {
        this.#cornerArc(a, b, d, AMETHYST.aArc, lineWidth);
      }
      if (this.showCArcs) {
        this.#cornerArc(c, b, d, AMETHYST.cArc, lineWidth);
      }
    }
  }

  draw() {
    const rect = this.canvas.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const pixelWidth = Math.round(rect.width * ratio);
    const pixelHeight = Math.round(rect.height * ratio);
    if (this.canvas.width !== pixelWidth || this.canvas.height !== pixelHeight) {
      this.canvas.width = pixelWidth;
      this.canvas.height = pixelHeight;
    }
    const context = this.context;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.fillStyle = AMETHYST.background;
    context.fillRect(0, 0, rect.width, rect.height);
    const view = this.#fit(rect.width, rect.height);
    const lineWidth = Math.max(0.75, Math.min(2, view.scale / 110));

    if (this.showPrediction && this.prediction) {
      context.globalAlpha = 0.8;
      context.lineJoin = "round";
      for (const tile of this.prediction) {
        this.#polygon(rhombusVertices(tile), view, rect.width, rect.height);
        context.fillStyle = mixHex(AMETHYST.color0, AMETHYST.color1, tile.color);
        context.fill();
        context.strokeStyle = AMETHYST.stroke;
        context.lineWidth = lineWidth * 0.45;
        context.stroke();
      }
    }

    if (this.showExact && this.exact) {
      context.globalAlpha = 0.95;
      context.lineJoin = "round";
      context.lineWidth = lineWidth;
      for (const tile of this.exact) {
        this.#polygon(rhombusVertices(tile), view, rect.width, rect.height);
        context.strokeStyle = tile.color ? AMETHYST.exact1 : AMETHYST.exact0;
        context.stroke();
      }
    }
    if (this.showAArcs || this.showCArcs) {
      const arcGeometry = this.showPrediction
        ? this.prediction
        : this.showExact
          ? this.exact
          : null;
      if (arcGeometry) {
        this.#drawArcs(
          arcGeometry,
          view,
          rect.width,
          rect.height,
          Math.max(1.25, lineWidth * 2.4),
        );
      }
    }
    context.globalAlpha = 1;
  }
}
