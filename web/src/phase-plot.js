const COLORS = {
  background: "#14001F",
  trail: "#C77DFF",
  point: "#72EFDD",
};

export class PhasePlot {
  constructor(canvas) {
    this.canvas = canvas;
    this.context = canvas.getContext("2d");
    this.points = [];
    this.current = [0, 0];
    this.resizeObserver = new ResizeObserver(() => this.draw());
    this.resizeObserver.observe(canvas);
  }

  reset() {
    this.points = [];
  }

  update(x, y, append = true) {
    this.current = [x, y];
    if (append) {
      const previous = this.points.at(-1);
      if (!previous || Math.hypot(x - previous[0], y - previous[1]) > 0.001) {
        this.points.push([x, y]);
        if (this.points.length > 240) this.points.shift();
      }
    }
    this.draw();
  }

  draw() {
    const rect = this.canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = Math.round(rect.width * ratio);
    this.canvas.height = Math.round(rect.height * ratio);
    const context = this.context;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.fillStyle = COLORS.background;
    context.fillRect(0, 0, rect.width, rect.height);

    const padding = 7;
    const width = rect.width - 2 * padding;
    const height = rect.height - 2 * padding;
    const screen = ([x, y]) => [
      padding + x * width,
      padding + (1 - y) * height,
    ];

    if (this.points.length > 1) {
      context.beginPath();
      this.points.forEach((point, index) => {
        const [x, y] = screen(point);
        if (index === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      });
      context.strokeStyle = COLORS.trail;
      context.globalAlpha = 0.6;
      context.lineWidth = 1.5;
      context.stroke();
      context.globalAlpha = 1;
    }

    const [x, y] = screen(this.current);
    context.beginPath();
    context.arc(x, y, 4, 0, 2 * Math.PI);
    context.fillStyle = COLORS.point;
    context.shadowColor = COLORS.point;
    context.shadowBlur = 8;
    context.fill();
    context.shadowBlur = 0;
  }
}
