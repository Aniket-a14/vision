// Everything known about one shot: the verdict, the rule behind it, the advice, the override.

import { Accordion, Badge, Group, Stack, Table, Text } from '@mantine/core';
import type { Explanation, Prescription, ReasonCode, Shot } from '../api/types';
import { label, risk, trim } from '../lib/format';
import { AnchorPanel } from './AnchorPanel';
import { OverridePanel } from './OverridePanel';
import { PrescribePanel } from './PrescribePanel';

interface Props {
  shot: Shot | null;
  auditHash: string | null;
  explanation: Explanation | null;
  advice: Prescription | null;
  reasons: ReasonCode[];
  loading: { explain: boolean; prescribe: boolean };
  errors: { explain: string | null; prescribe: string | null };
}

export function ShotInspector(props: Props) {
  const { shot } = props;
  if (!shot) {
    return (
      <Text c="dimmed" size="sm">
        Click a shot in the line to inspect it.
      </Text>
    );
  }

  return (
    <Stack gap="sm">
      <Header shot={shot} />
      <Accordion multiple defaultValue={['why', 'fix']} variant="separated">
        <Accordion.Item value="why">
          <Accordion.Control>Why</Accordion.Control>
          <Accordion.Panel>
            <AnchorPanel
              explanation={props.explanation}
              loading={props.loading.explain}
              error={props.errors.explain}
            />
          </Accordion.Panel>
        </Accordion.Item>
        <Accordion.Item value="fix">
          <Accordion.Control>What to change</Accordion.Control>
          <Accordion.Panel>
            <PrescribePanel
              advice={props.advice}
              loading={props.loading.prescribe}
              error={props.errors.prescribe}
            />
          </Accordion.Panel>
        </Accordion.Item>
        <Accordion.Item value="readings">
          <Accordion.Control>Readings</Accordion.Control>
          <Accordion.Panel>
            <ReadingTable shot={shot} />
          </Accordion.Panel>
        </Accordion.Item>
        <Accordion.Item value="override">
          <Accordion.Control>Disagree</Accordion.Control>
          <Accordion.Panel>
            <OverridePanel
              auditHash={props.auditHash}
              flagged={shot.flagged}
              reasons={props.reasons}
              explanationShown={props.explanation?.rule ?? ''}
            />
          </Accordion.Panel>
        </Accordion.Item>
      </Accordion>
    </Stack>
  );
}

function Header({ shot }: { shot: Shot }) {
  return (
    <Group justify="space-between">
      <Group gap="xs">
        <Text fw={500}>Shot {shot.shot_index}</Text>
        <Text size="xs" c="dimmed">
          lot {shot.lot_id} · die {shot.die_id} · shift {shot.shift_id}
        </Text>
      </Group>
      <Group gap="xs">
        {shot.abstained && (
          <Badge color="yellow" variant="light">
            abstain
          </Badge>
        )}
        <Badge color={shot.flagged ? 'red' : 'green'} variant="light">
          {risk(shot.risk)}
        </Badge>
      </Group>
    </Group>
  );
}

function ReadingTable({ shot }: { shot: Shot }) {
  return (
    <Table>
      <Table.Tbody>
        {Object.entries(shot.readings).map(([name, value]) => (
          <Table.Tr key={name}>
            <Table.Td>{label(name)}</Table.Td>
            <Table.Td ta="right">{trim(value, 4)}</Table.Td>
          </Table.Tr>
        ))}
      </Table.Tbody>
    </Table>
  );
}
