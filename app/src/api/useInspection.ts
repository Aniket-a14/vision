// Everything fetched about one selected shot.
//
// The three calls are independent and are fired together rather than in sequence: prescribe is
// the slow one, and waiting for it before asking why would leave the panel blank for no reason.
//
// Selecting a shot also re-scores it through /score, which is what puts the decision in the
// audit chain and yields the hash an override has to be signed against. The stream deliberately
// does not audit -- a shot nobody looked at is not a decision anyone made.

import { useCallback, useEffect, useState } from 'react';
import * as api from './client';
import type { Explanation, Prescription, Readings, Score } from './types';

interface Flags {
  explain: boolean;
  prescribe: boolean;
}

interface Errors {
  explain: string | null;
  prescribe: string | null;
}

export function useInspection(readings: Readings | null) {
  const [score, setScore] = useState<Score | null>(null);
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [advice, setAdvice] = useState<Prescription | null>(null);
  const [loading, setLoading] = useState<Flags>({ explain: false, prescribe: false });
  const [errors, setErrors] = useState<Errors>({ explain: null, prescribe: null });

  const clear = useCallback(() => {
    setScore(null);
    setExplanation(null);
    setAdvice(null);
    setErrors({ explain: null, prescribe: null });
  }, []);

  useEffect(() => {
    if (!readings) {
      clear();
      return;
    }
    let live = true;
    setLoading({ explain: true, prescribe: true });
    setErrors({ explain: null, prescribe: null });

    api.score(readings).then((result) => live && setScore(result), () => undefined);

    api.explain(readings).then(
      (result) => live && finish(setExplanation, setLoading, 'explain', result),
      (problem: Error) => live && fail(setErrors, setLoading, 'explain', problem),
    );

    api.prescribe(readings).then(
      (result) => live && finish(setAdvice, setLoading, 'prescribe', result),
      (problem: Error) => live && fail(setErrors, setLoading, 'prescribe', problem),
    );

    return () => {
      live = false;
    };
  }, [readings, clear]);

  return { score, explanation, advice, loading, errors };
}

function finish<T>(
  set: (value: T) => void,
  setLoading: (update: (prior: Flags) => Flags) => void,
  key: keyof Flags,
  value: T,
) {
  set(value);
  setLoading((prior) => ({ ...prior, [key]: false }));
}

function fail(
  setErrors: (update: (prior: Errors) => Errors) => void,
  setLoading: (update: (prior: Flags) => Flags) => void,
  key: keyof Flags,
  problem: Error,
) {
  setErrors((prior) => ({ ...prior, [key]: problem.message }));
  setLoading((prior) => ({ ...prior, [key]: false }));
}
