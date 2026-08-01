// Reference data that never changes while the server is up: machine limits, reason codes,
// model version. Fetched once so every panel reads the same copy.

import { useEffect, useState } from 'react';
import * as api from './client';
import type { Health, Parameter, ReasonCode } from './types';

export function useReference() {
  const [parameters, setParameters] = useState<Parameter[]>([]);
  const [reasons, setReasons] = useState<ReasonCode[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    Promise.all([api.parameters(), api.reasons(), api.health()])
      .then(([specs, codes, status]) => {
        if (!live) return;
        setParameters(specs);
        setReasons(codes);
        setHealth(status);
      })
      .catch((problem: Error) => live && setError(problem.message));
    return () => {
      live = false;
    };
  }, []);

  return { parameters, reasons, health, error };
}
