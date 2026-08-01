// The live line: the chart, the running alarm rate, and the shot list.

import { Badge, Button, Group, Paper, ScrollArea, Stack, Table, Text } from '@mantine/core';
import type { Shot } from '../api/types';
import type { Status } from '../api/useStream';
import { alarmsPerHour, risk } from '../lib/format';
import { RiskChart } from './RiskChart';

const TONE: Record<Status, string> = {
  live: 'green',
  connecting: 'yellow',
  stopped: 'gray',
  error: 'red',
};

interface Props {
  shots: Shot[];
  status: Status;
  running: boolean;
  threshold: number;
  selected: number | null;
  onSelect: (shot: Shot) => void;
  onToggle: () => void;
  onClear: () => void;
}

export function LineView(props: Props) {
  const { shots, status, running, threshold, selected } = props;
  const flagged = shots.filter((shot) => shot.flagged).length;
  const rate = shots.length ? flagged / shots.length : 0;

  return (
    <Stack gap="sm">
      <Group justify="space-between">
        <Group gap="xs">
          <Badge color={TONE[status]} variant="light">
            {status}
          </Badge>
          <Text size="sm" c="dimmed">
            {shots.length} shots buffered
          </Text>
        </Group>
        <Group gap="xs">
          <Button size="xs" variant="light" onClick={props.onClear} disabled={!shots.length}>
            Clear
          </Button>
          <Button size="xs" color={running ? 'red' : 'blue'} onClick={props.onToggle}>
            {running ? 'Stop line' : 'Start line'}
          </Button>
        </Group>
      </Group>

      <Paper withBorder p="sm">
        <RiskChart shots={shots} threshold={threshold} />
      </Paper>

      <AlarmRate flagged={flagged} rate={rate} total={shots.length} />

      <Paper withBorder>
        <ScrollArea h={280}>
          <Table stickyHeader highlightOnHover striped>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Shot</Table.Th>
                <Table.Th>Lot</Table.Th>
                <Table.Th>Risk</Table.Th>
                <Table.Th>Verdict</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {[...shots].reverse().map((shot) => (
                <Row
                  key={shot.shot_index}
                  shot={shot}
                  selected={shot.shot_index === selected}
                  onSelect={props.onSelect}
                />
              ))}
            </Table.Tbody>
          </Table>
        </ScrollArea>
      </Paper>
    </Stack>
  );
}

function Row({
  shot,
  selected,
  onSelect,
}: {
  shot: Shot;
  selected: boolean;
  onSelect: (shot: Shot) => void;
}) {
  return (
    <Table.Tr
      onClick={() => onSelect(shot)}
      bg={selected ? 'var(--mantine-color-blue-light)' : undefined}
      style={{ cursor: 'pointer' }}
    >
      <Table.Td>{shot.shot_index}</Table.Td>
      <Table.Td>{shot.lot_id}</Table.Td>
      <Table.Td>{risk(shot.risk)}</Table.Td>
      <Table.Td>
        <Verdict shot={shot} />
      </Table.Td>
    </Table.Tr>
  );
}

// Abstention is shown as its own state rather than folded into "ok". A conformal set holding
// both classes is the model declining to choose, which is a different instruction to an operator
// than a confident pass.
function Verdict({ shot }: { shot: Shot }) {
  if (shot.abstained) {
    return (
      <Badge color="yellow" variant="light">
        abstain
      </Badge>
    );
  }
  return (
    <Badge color={shot.flagged ? 'red' : 'green'} variant="light">
      {shot.flagged ? 'flag' : 'pass'}
    </Badge>
  );
}

// ISA-18.2 puts a sustainable operator load at 6-12 alarms/hour. Showing the rate in those units
// rather than as a percentage is what makes it a number anyone can act on.
function AlarmRate({ flagged, rate, total }: { flagged: number; rate: number; total: number }) {
  const perHour = alarmsPerHour(rate);
  const tone = perHour > 12 ? 'red' : perHour < 6 ? 'blue' : 'green';
  return (
    <Paper withBorder p="sm">
      <Group justify="space-between">
        <Text size="sm">
          {flagged} flagged of {total}
        </Text>
        <Group gap="xs">
          <Badge color={tone} variant="light">
            {perHour.toFixed(1)} alarms/hour
          </Badge>
          <Text size="xs" c="dimmed">
            ISA-18.2 band 6-12
          </Text>
        </Group>
      </Group>
    </Paper>
  );
}
