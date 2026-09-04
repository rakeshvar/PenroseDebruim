export const PHI = (1 + Math.sqrt(5)) / 2;
export const E = Math.E;
export const TAU = 2 * Math.PI;
export const KMAX = 1;
export const TILE_COUNT = 10 * (2 * KMAX + 1) ** 2;
export const XY_SCALE = Math.sqrt((25 / 6) * KMAX * (KMAX + 1) + 5.65);
export const ANGLE_SCALE = Math.sqrt(3) / Math.PI;

export function mod(value, modulus = 1) {
  return ((value % modulus) + modulus) % modulus;
}

export function triangle01(value) {
  const phase = mod(value, TAU) / TAU;
  return 1 - Math.abs(2 * phase - 1);
}

export function randomOffsets(random = Math.random) {
  return Array.from({ length: 4 }, () => random());
}

export function conditionsAt(t, offsets, moveTheta = true) {
  if (offsets.length !== 4) throw new Error("offsets must contain a0…a3");
  const gamma = [
    triangle01(t + offsets[0]),
    triangle01(t / PHI + offsets[1]),
    (Math.sin(t + offsets[2]) + 1) / 2,
    (Math.sin(t / E + offsets[3]) + 1) / 2,
  ];
  gamma.push(mod(-gamma.reduce((sum, value) => sum + value, 0)));
  return { gamma, thetaOffset: moveTheta ? triangle01(t) : 0.5 };
}

export function wrapAngle(angle) {
  return mod(angle + Math.PI, TAU) - Math.PI;
}

function indexData(kmax) {
  const result = [];
  for (let i = 0; i < 5; i += 1) {
    for (let j = i + 1; j < 5; j += 1) {
      for (let lineI = -kmax; lineI <= kmax; lineI += 1) {
        for (let lineJ = -kmax; lineJ <= kmax; lineJ += 1) {
          result.push({ i, j, lineI, lineJ });
        }
      }
    }
  }
  return result;
}

const INDICES = indexData(KMAX);

export function debruijn(gamma, thetaOffset, kmax = KMAX) {
  if (gamma.length !== 5) throw new Error("gamma must contain five values");
  if (thetaOffset < 0 || thetaOffset > 1) {
    throw new Error("thetaOffset must lie in [0, 1]");
  }
  if (kmax !== KMAX) {
    throw new Error(`browser models are fixed to kmax=${KMAX}`);
  }

  const theta0 = (thetaOffset - 0.5) * (Math.PI / 5);
  const directions = Array.from({ length: 5 }, (_, family) => {
    const theta = (TAU / 5) * family + theta0;
    return [Math.cos(theta), Math.sin(theta)];
  });

  return INDICES.map(({ i, j, lineI, lineJ }) => {
    const ui = directions[i];
    const uj = directions[j];
    const determinant = ui[0] * uj[1] - ui[1] * uj[0];
    const ci = lineI - gamma[i];
    const cj = lineJ - gamma[j];
    const x = (ci * uj[1] - cj * ui[1]) / determinant;
    const y = (ui[0] * cj - uj[0] * ci) / determinant;
    const regions = directions.map(
      ([dx, dy], family) => Math.ceil(x * dx + y * dy + gamma[family]),
    );
    regions[i] = lineI;
    regions[j] = lineJ;

    const axisX = ui[0] + uj[0];
    const axisY = ui[1] + uj[1];
    let centerX = axisX / 2;
    let centerY = axisY / 2;
    for (let family = 0; family < 5; family += 1) {
      centerX += regions[family] * directions[family][0];
      centerY += regions[family] * directions[family][1];
    }
    const baseAngle = Math.atan2(axisY, axisX);
    const parity = mod(regions.reduce((sum, value) => sum + value, 0), 2) === 1;
    const color = j - i === 2 || j - i === 3 ? 1 : 0;
    return {
      x: centerX,
      y: centerY,
      angle: wrapAngle(parity ? baseAngle : baseAngle + Math.PI),
      color,
    };
  });
}

export function decodePrediction(data) {
  if (data.length !== TILE_COUNT * 4) {
    throw new Error(`prediction has ${data.length} values; expected ${TILE_COUNT * 4}`);
  }
  return Array.from({ length: TILE_COUNT }, (_, index) => ({
    x: data[index * 4] * XY_SCALE,
    y: data[index * 4 + 1] * XY_SCALE,
    angle: wrapAngle(data[index * 4 + 2] / ANGLE_SCALE),
    color: (data[index * 4 + 3] + 1) / 2,
  }));
}

export function rhombusVertices(tile, side = 1) {
  const topAngle = ((3 - 2 * tile.color) * Math.PI) / 5;
  const short = side * Math.cos(topAngle / 2);
  const long = side * Math.sin(topAngle / 2);
  const along = [Math.cos(tile.angle) * long, Math.sin(tile.angle) * long];
  const across = [-Math.sin(tile.angle) * short, Math.cos(tile.angle) * short];
  return [
    [tile.x + along[0], tile.y + along[1]],
    [tile.x + across[0], tile.y + across[1]],
    [tile.x - along[0], tile.y - along[1]],
    [tile.x - across[0], tile.y - across[1]],
  ];
}

export function geometryBounds(...geometries) {
  const points = geometries
    .filter(Boolean)
    .flatMap((tiles) => tiles.flatMap((tile) => rhombusVertices(tile)));
  if (points.length === 0) return { minX: -1, maxX: 1, minY: -1, maxY: 1 };
  const xs = points.map(([x]) => x);
  const ys = points.map(([, y]) => y);
  return {
    minX: Math.min(...xs),
    maxX: Math.max(...xs),
    minY: Math.min(...ys),
    maxY: Math.max(...ys),
  };
}
