// SSE subscription to the running line.
//
// EventSource, not fetch: it reconnects on its own and the browser handles the framing. The
// buffer is capped because a view that keeps every shot of a long shift eventually kills the
// tab -- an operator screen is a window, not an archive. The archive is Power BI.

import { useCallback, useEffect, useRef, useState } from 'react';
import { BASE } from './client';
import type { Shot } from './types';

export const BUFFER = 240;

export type Status = 'connecting' | 'live' | 'stopped' | 'error';

export function useStream(interval: number) {
  const [shots, setShots] = useState<Shot[]>([]);
  const [status, setStatus] = useState<Status>('stopped');
  const [running, setRunning] = useState(false);
  const source = useRef<EventSource | null>(null);

  const disconnect = useCallback(() => {
    source.current?.close();
    source.current = null;
  }, []);

  useEffect(() => {
    if (!running) {
      disconnect();
      setStatus('stopped');
      return;
    }
    setStatus('connecting');
    const events = new EventSource(`${BASE}/stream?interval=${interval}`);
    source.current = events;
    events.addEventListener('shot', (event) => {
      setStatus('live');
      setShots((previous) => append(previous, JSON.parse((event as MessageEvent).data) as Shot));
    });
    // EventSource retries by itself, so an error is a state to display, not one to recover from.
    events.onerror = () => setStatus('error');
    return disconnect;
  }, [running, interval, disconnect]);

  const clear = useCallback(() => setShots([]), []);
  return { shots, status, running, setRunning, clear };
}

function append(previous: Shot[], shot: Shot): Shot[] {
  const next = [...previous, shot];
  return next.length > BUFFER ? next.slice(next.length - BUFFER) : next;
}
