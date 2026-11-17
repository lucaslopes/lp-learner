/** A small seeded random number generator (mulberry32), so that demos can be replayed and compared on the same luck. */
export class Rng {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0 || 0x9e3779b9;
  }

  /** Uniform in [0, 1). */
  next(): number {
    let t = (this.state = (this.state + 0x6d2b79f5) >>> 0);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Geometric number of trials until the first success, on {1, 2, ...}. */
  geometric(p: number): number {
    if (p >= 1) return 1;
    let k = 1;
    while (this.next() >= p) k++;
    return k;
  }

  /** In-place Fisher-Yates shuffle. */
  shuffle<T>(xs: T[]): T[] {
    for (let i = xs.length - 1; i > 0; i--) {
      const j = Math.floor(this.next() * (i + 1));
      [xs[i], xs[j]] = [xs[j], xs[i]];
    }
    return xs;
  }

  /** A child generator with its own stream. */
  fork(salt: number): Rng {
    return new Rng((Math.floor(this.next() * 4294967296) ^ Math.imul(salt + 1, 0x9e3779b1)) >>> 0);
  }
}
