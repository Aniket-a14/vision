// Streaming risk chart.
//
// uPlot is imperative and owns its canvas, so React must not re-render it. The plot is created
// once and fed with setData; recreating it on every shot would drop a frame per second and leak
// a canvas each time.
//
// The y-axis is the logit, for the reason given on `logit` -- the same trap the SPC charts hit.

import { useEffect, useRef } from 'react';
import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';
import type { Shot } from '../api/types';
import { logit } from '../lib/format';

const HEIGHT = 220;

interface Props {
  shots: Shot[];
  threshold: number;
}

export function RiskChart({ shots, threshold }: Props) {
  const holder = useRef<HTMLDivElement>(null);
  const plot = useRef<uPlot | null>(null);

  useEffect(() => {
    if (!holder.current) return;
    plot.current = new uPlot(options(holder.current.clientWidth, threshold), empty(), holder.current);
    const resize = () => plot.current?.setSize({ width: holder.current!.clientWidth, height: HEIGHT });
    window.addEventListener('resize', resize);
    return () => {
      window.removeEventListener('resize', resize);
      plot.current?.destroy();
      plot.current = null;
    };
  }, [threshold]);

  useEffect(() => {
    plot.current?.setData(series(shots, threshold));
  }, [shots, threshold]);

  return <div ref={holder} style={{ width: '100%', height: HEIGHT }} />;
}

function series(shots: Shot[], threshold: number): uPlot.AlignedData {
  const x = shots.map((shot) => shot.shot_index);
  const y = shots.map((shot) => logit(shot.risk));
  const limit = shots.map(() => logit(threshold));
  return [x, y, limit];
}

const empty = (): uPlot.AlignedData => [[], [], []];

function options(width: number, threshold: number): uPlot.Options {
  return {
    width,
    height: HEIGHT,
    padding: [12, 12, 0, 0],
    legend: { show: false },
    cursor: { drag: { x: false, y: false } },
    scales: { x: { time: false } },
    axes: [
      { label: 'shot', stroke: '#868e96', grid: { stroke: '#e9ecef' } },
      { label: 'risk (logit)', stroke: '#868e96', grid: { stroke: '#e9ecef' } },
    ],
    series: [
      {},
      { label: 'risk', stroke: '#1c7ed6', width: 1.5, points: { show: false } },
      {
        label: `threshold ${threshold.toFixed(3)}`,
        stroke: '#e03131',
        width: 1,
        dash: [6, 4],
        points: { show: false },
      },
    ],
  };
}
