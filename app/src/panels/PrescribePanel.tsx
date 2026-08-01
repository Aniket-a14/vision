// Setpoint advice for the selected shot, with the honest caveats attached.

import { Alert, Badge, Group, Loader, Paper, Stack, Table, Text } from '@mantine/core';
import type { Prescription } from '../api/types';
import { label, pct, risk, signed, trim } from '../lib/format';

interface Props {
  advice: Prescription | null;
  loading: boolean;
  error: string | null;
}

export function PrescribePanel({ advice, loading, error }: Props) {
  if (loading) return <Loader size="sm" />;
  if (error) return <Alert color="red">{error}</Alert>;
  if (!advice) return <Text c="dimmed" size="sm">Select a shot for advice.</Text>;
  if (!advice.actions.length) return <NoAdvice advice={advice} />;

  return (
    <Stack gap="sm">
      <Gain advice={advice} />
      <Table>
        <Table.Thead>
          <Table.Tr>
            <Table.Th>Parameter</Table.Th>
            <Table.Th>Now</Table.Th>
            <Table.Th>Set to</Table.Th>
            <Table.Th>Change</Table.Th>
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {advice.actions.map((action) => (
            <Table.Tr key={action.parameter}>
              <Table.Td>{label(action.parameter)}</Table.Td>
              <Table.Td>{trim(action.current)}</Table.Td>
              <Table.Td fw={500}>
                {trim(action.proposed)} {action.unit}
              </Table.Td>
              <Table.Td>{signed(action.delta)}</Table.Td>
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
      <Stability rate={advice.stability} />
    </Stack>
  );
}

const SATURATED = 0.001;

// The headline is the margin, not the probability drop. For the shots worth advising, the
// sigmoid is flat: a 20-logit improvement can leave the probability reading 100 % both before
// and after. Showing "100 % to 100 %" next to a large gain looks like a bug, so the saturated
// case says what is happening instead of hiding it.
function Gain({ advice }: { advice: Prescription }) {
  const moved = advice.risk_before - advice.risk_after;
  return (
    <Stack gap={2}>
      <Group gap="xs">
        <Badge variant="filled" color="blue">
          {signed(advice.margin_gain, 1)} logits of margin
        </Badge>
        <Text size="sm" c="dimmed">
          {risk(advice.risk_before)} to {risk(advice.risk_after)}
        </Text>
      </Group>
      {moved < SATURATED && (
        <Text size="xs" c="dimmed">
          The probability is saturated at this risk, so it barely moves. The margin is the honest
          measure of the improvement.
        </Text>
      )}
    </Stack>
  );
}

// The improvement is quoted in logits because probability saturates: a shot at risk 0.9999999
// that improves by 16 logits still reads as a probability change of roughly zero.
function NoAdvice({ advice }: { advice: Prescription }) {
  return (
    <Alert color="gray" variant="light">
      <Text size="sm">
        No change recommended. At {risk(advice.risk_before)} this shot is already below the level
        where a ramp-limited move would earn its disruption.
      </Text>
    </Alert>
  );
}

// Stated with its reason. Every recommendation worsens at least one failure mechanism, so a
// different weighting could in principle flip it; it survives because the improvement dominates
// the worsening. The claim is "the advice does not depend on the weights for a shot this far
// from nominal", not "the advice is verified".
function Stability({ rate }: { rate: number }) {
  return (
    <Paper withBorder p="xs" bg="var(--mantine-color-gray-0)">
      <Text size="xs">
        <Text span fw={500}>
          Survives {pct(rate, 0)}
        </Text>{' '}
        of simulators with their mechanism weights perturbed by up to 50 %. Each move is capped at
        the parameter&apos;s ramp limit, so it is one shift&apos;s adjustment, not a re-qualification.
      </Text>
    </Paper>
  );
}
