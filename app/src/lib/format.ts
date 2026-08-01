// Display helpers. Kept out of components so the same number never renders two ways.

export const pct = (value: number, digits = 1) => `${(value * 100).toFixed(digits)} %`;

export const risk = (value: number) => (value < 0.001 ? '<0.1 %' : pct(value));

export const signed = (value: number, digits = 2) =>
  `${value >= 0 ? '+' : ''}${value.toFixed(digits)}`;

// Anchors have genuinely one-sided predicates, and they arrive as null because JSON has no
// infinity. "in [3.09, null]" is how an engineer reads a bug, not a rule.
export function bound(lower: number | null, upper: number | null, digits = 3): string {
  const low = lower !== null && Number.isFinite(lower);
  const high = upper !== null && Number.isFinite(upper);
  if (low && high) return `${trim(lower!, digits)} to ${trim(upper!, digits)}`;
  if (low) return `at least ${trim(lower!, digits)}`;
  if (high) return `at most ${trim(upper!, digits)}`;
  return 'any value';
}

export const trim = (value: number, digits = 3) => Number(value.toPrecision(digits)).toString();

// Charts plot the logit. On a probability axis every alarm pins to the top and a real 16-logit
// improvement is invisible, because a risky shot sits where the sigmoid is flat.
const CLAMP = 1e-9;
export const logit = (p: number) => {
  const bounded = Math.min(Math.max(p, CLAMP), 1 - CLAMP);
  return Math.log(bounded / (1 - bounded));
};

export const label = (name: string) => name.replace(/_/g, ' ').replace(/\b(c|s|ms|mpa|pct)$/, '');

// Alarms per hour, the unit ISA-18.2 is written in. A rate in percent means nothing to an
// operator; a count per hour is the thing they can say yes or no to.
export const SECONDS_PER_SHOT = 60;
export const alarmsPerHour = (rate: number) => (rate * 3600) / SECONDS_PER_SHOT;
