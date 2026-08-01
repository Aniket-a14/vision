// Mirrors the FastAPI response models. Kept in one file so a schema change breaks compilation
// in one place rather than in six components.

export type Readings = Record<string, number>;

export interface Shot {
  shot_index: number;
  lot_id: number;
  die_id: number;
  shift_id: number;
  risk: number;
  flagged: boolean;
  prediction_set: number[];
  abstained: boolean;
  readings: Readings;
}

export interface Score {
  risk: number;
  threshold: number;
  flagged: boolean;
  prediction_set: number[];
  abstained: boolean;
  model_version: string;
  audit_hash: string;
}

// JSON has no infinity, so an unbounded side of a predicate arrives as null rather than as
// Infinity. Typing it as `number` would be a lie that happens to render correctly.
export interface Predicate {
  parameter: string;
  lower: number | null;
  upper: number | null;
}

export interface Explanation {
  rule: string;
  prediction: number;
  precision: number;
  coverage: number;
  predicates: Predicate[];
}

export interface Action {
  parameter: string;
  current: number;
  proposed: number;
  delta: number;
  unit: string;
}

export interface Prescription {
  actions: Action[];
  risk_before: number;
  risk_after: number;
  margin_gain: number;
  stability: number;
}

export interface Parameter {
  name: string;
  unit: string;
  nominal: number;
  lower: number;
  upper: number;
  actionability: 'immediate' | 'slow' | 'lot_level' | 'maintenance';
  ramp_limit: number | null;
}

export interface ReasonCode {
  code: string;
  label: string;
  note_required: boolean;
}

export interface Health {
  status: string;
  model_version: string;
  audit_entries: number;
}

export interface AuditState {
  intact: boolean;
  entries: number;
  broken_at: number | null;
  head: string;
}

export interface Override {
  audit_hash: string;
  overrides: string;
}
