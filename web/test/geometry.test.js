import assert from "node:assert/strict";
import test from "node:test";

import {
  TAU,
  TILE_COUNT,
  conditionsAt,
  debruijn,
  rhombusVertices,
  triangle01,
} from "../src/geometry.js";

const close = (actual, expected, tolerance = 1e-10) => {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);
};

test("triangle01 has the requested 0→1→0 period", () => {
  close(triangle01(0), 0);
  close(triangle01(Math.PI), 1);
  close(triangle01(TAU), 0);
  close(triangle01(-Math.PI), 1);
});

test("animated conditions remain in range and constrain gamma sum", () => {
  const condition = conditionsAt(7.25, [0.1, 0.2, 0.3, 0.4]);
  for (const value of [...condition.gamma, condition.thetaOffset]) {
    assert.ok(value >= 0 && value <= 1);
  }
  const sum = condition.gamma.reduce((total, value) => total + value, 0);
  close(sum, Math.round(sum));
});

test("theta offset can be held fixed", () => {
  close(conditionsAt(7.25, [0.1, 0.2, 0.3, 0.4], false).thetaOffset, 0.5);
});

test("JavaScript de Bruijn geometry matches the NumPy reference fixture", () => {
  const result = debruijn([0.125, 0.25, 0.375, 0.5, 0.75], 0.625);
  assert.equal(result.length, TILE_COUNT);
  const expected = [
    [-2.2232665796184907, -3.1495711050297523, 0.7068583470577035, 0],
    [-3.225144812481768, -0.73082308740484, -2.434734306532089, 0],
    [-2.6139769153909214, 1.814874413834307, 0.7068583470577035, 0],
  ];
  expected.forEach(([x, y, angle, color], index) => {
    close(result[index].x, x);
    close(result[index].y, y);
    close(result[index].angle, angle);
    assert.equal(result[index].color, color);
  });
});

test("rhombus vertices preserve their center", () => {
  const tile = { x: 1.2, y: -0.7, angle: 0.9, color: 1 };
  const vertices = rhombusVertices(tile);
  close(vertices.reduce((sum, [x]) => sum + x, 0) / 4, tile.x);
  close(vertices.reduce((sum, [, y]) => sum + y, 0) / 4, tile.y);
});
