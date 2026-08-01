// What-if: move the setpoints and re-score.
//
// Slider bounds come from /parameters, so the UI cannot allow a setpoint the machine cannot
// reach. Lot-level and maintenance parameters are shown but locked -- an operator cannot change
// the alloy chemistry mid-shift, and a sandbox that pretends otherwise teaches the wrong lesson.

import { Badge, Button, Group, NumberInput, Paper, Slider, Stack, Text } from '@mantine/core';
import { useState } from 'react';
import type { Parameter, Readings, Score } from '../api/types';
import { label, risk } from '../lib/format';

const LOCKED = new Set(['lot_level', 'maintenance']);

interface Props {
  parameters: Parameter[];
  readings: Readings;
  scored: Score | null;
  busy: boolean;
  onScore: (readings: Readings) => void;
  onReset: () => void;
}

export function Sandbox({ parameters, readings, scored, busy, onScore, onReset }: Props) {
  const [draft, setDraft] = useState<Readings>(readings);
  const set = (name: string, value: number) => setDraft((prior) => ({ ...prior, [name]: value }));

  return (
    <Stack gap="sm">
      <Group justify="space-between">
        <Verdict scored={scored} />
        <Group gap="xs">
          <Button
            size="xs"
            variant="light"
            onClick={() => {
              setDraft(readings);
              onReset();
            }}
          >
            Reset
          </Button>
          <Button size="xs" loading={busy} onClick={() => onScore(draft)}>
            Score
          </Button>
        </Group>
      </Group>
      {parameters.map((parameter) => (
        <Knob
          key={parameter.name}
          parameter={parameter}
          value={draft[parameter.name] ?? parameter.nominal}
          onChange={(value) => set(parameter.name, value)}
        />
      ))}
    </Stack>
  );
}

function Verdict({ scored }: { scored: Score | null }) {
  if (!scored) return <Text size="sm" c="dimmed">Not scored yet.</Text>;
  return (
    <Group gap="xs">
      <Badge color={scored.flagged ? 'red' : 'green'} variant="light">
        {scored.flagged ? 'flag' : 'pass'}
      </Badge>
      <Text size="sm">{risk(scored.risk)}</Text>
      <Text size="xs" c="dimmed">
        threshold {scored.threshold.toFixed(3)}
      </Text>
    </Group>
  );
}

interface KnobProps {
  parameter: Parameter;
  value: number;
  onChange: (value: number) => void;
}

function Knob({ parameter, value, onChange }: KnobProps) {
  const locked = LOCKED.has(parameter.actionability);
  const step = (parameter.upper - parameter.lower) / 200;
  return (
    <Paper withBorder p="xs">
      <Group justify="space-between" mb={4}>
        <Group gap={6}>
          <Text size="sm">{label(parameter.name)}</Text>
          {locked && (
            <Badge size="xs" color="gray" variant="light">
              {parameter.actionability.replace('_', ' ')}
            </Badge>
          )}
        </Group>
        <NumberInput
          size="xs"
          w={110}
          value={value}
          disabled={locked}
          min={parameter.lower}
          max={parameter.upper}
          step={step}
          suffix={` ${parameter.unit}`}
          onChange={(next) => onChange(Number(next))}
        />
      </Group>
      <Slider
        size="sm"
        disabled={locked}
        value={value}
        min={parameter.lower}
        max={parameter.upper}
        step={step}
        label={null}
        marks={[{ value: parameter.nominal, label: 'nominal' }]}
        onChange={onChange}
      />
    </Paper>
  );
}
