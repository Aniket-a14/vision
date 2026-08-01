// Typed wrappers over the FastAPI routes.
//
// A 422 carries the twin's own validation message ("pour_temp_c=2000 is outside the machine
// limits"), which is far more use to an operator than "Bad Request", so it is unwrapped rather
// than swallowed.

import type {
  AuditState,
  Explanation,
  Health,
  Override,
  Parameter,
  Prescription,
  ReasonCode,
  Readings,
  Score,
} from './types';

export const BASE = import.meta.env.VITE_API ?? 'http://127.0.0.1:8000';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!response.ok) throw new Error(await detail(response));
  return (await response.json()) as T;
}

async function detail(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body.detail === 'string') return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((d: { msg: string }) => d.msg).join('; ');
  } catch {
    // A non-JSON error body is still an error; fall through to the status line.
  }
  return `${response.status} ${response.statusText}`;
}

const post = <T,>(path: string, body: unknown) =>
  request<T>(path, { method: 'POST', body: JSON.stringify(body) });

export const health = () => request<Health>('/health');
export const parameters = () => request<Parameter[]>('/parameters');
export const reasons = () => request<ReasonCode[]>('/reasons');
export const audit = () => request<AuditState>('/audit');

export const score = (readings: Readings) => post<Score>('/score', { readings });
export const explain = (readings: Readings) => post<Explanation>('/explain', { readings });
export const prescribe = (readings: Readings) => post<Prescription>('/prescribe', { readings });

export interface OverrideInput {
  audit_hash: string;
  defective: boolean;
  reason: string;
  note: string;
  explanation_shown: string;
}

export const override = (input: OverrideInput) => post<Override>('/override', input);
