// Shell: the live line on the left, the inspector on the right, the sandbox behind a tab.

import { Alert, AppShell, Badge, Grid, Group, Paper, Tabs, Text, Title } from '@mantine/core';
import { useCallback, useMemo, useState } from 'react';
import * as api from './api/client';
import type { Parameter, Readings, Score, Shot } from './api/types';
import { useInspection } from './api/useInspection';
import { useReference } from './api/useReference';
import { useStream } from './api/useStream';
import { LineView } from './panels/LineView';
import { Sandbox } from './panels/Sandbox';
import { ShotInspector } from './panels/ShotInspector';

const INTERVAL = 1.0;
const FALLBACK_THRESHOLD = 0.3;

export default function App() {
  const reference = useReference();
  const { shots, status, running, setRunning, clear } = useStream(INTERVAL);
  const [selected, setSelected] = useState<Shot | null>(null);
  const inspection = useInspection(selected?.readings ?? null);

  const threshold = inspection.score?.threshold ?? FALLBACK_THRESHOLD;

  return (
    <AppShell header={{ height: 56 }} padding="md">
      <AppShell.Header>
        <Header version={reference.health?.model_version} status={status} />
      </AppShell.Header>
      <AppShell.Main>
        {reference.error && <Offline detail={reference.error} />}
        <Grid>
          <Grid.Col span={{ base: 12, md: 7 }}>
            <LineView
              shots={shots}
              status={status}
              running={running}
              threshold={threshold}
              selected={selected?.shot_index ?? null}
              onSelect={setSelected}
              onToggle={() => setRunning(!running)}
              onClear={() => {
                clear();
                setSelected(null);
              }}
            />
          </Grid.Col>
          <Grid.Col span={{ base: 12, md: 5 }}>
            <RightPane selected={selected} inspection={inspection} reference={reference} />
          </Grid.Col>
        </Grid>
      </AppShell.Main>
    </AppShell>
  );
}

function Offline({ detail }: { detail: string }) {
  return (
    <Alert color="red" mb="md" title="No API">
      <Text size="sm">
        Cannot reach {api.BASE}. Start it with{' '}
        <Text span ff="monospace">
          defectlab serve
        </Text>
        . ({detail})
      </Text>
    </Alert>
  );
}

function Header({ version, status }: { version?: string; status: string }) {
  return (
    <Group h="100%" px="md" justify="space-between">
      <Group gap="xs">
        <Title order={4}>DefectLab</Title>
        <Text size="xs" c="dimmed">
          live casting gate
        </Text>
      </Group>
      <Group gap="xs">
        {version && (
          <Badge variant="light" color="gray">
            {version}
          </Badge>
        )}
        <Badge variant="dot" color={status === 'live' ? 'green' : 'gray'}>
          {status}
        </Badge>
      </Group>
    </Group>
  );
}

type Inspection = ReturnType<typeof useInspection>;
type Reference = ReturnType<typeof useReference>;

function RightPane({
  selected,
  inspection,
  reference,
}: {
  selected: Shot | null;
  inspection: Inspection;
  reference: Reference;
}) {
  return (
    <Paper withBorder p="sm">
      <Tabs defaultValue="inspect">
        <Tabs.List mb="sm">
          <Tabs.Tab value="inspect">Inspect</Tabs.Tab>
          <Tabs.Tab value="sandbox">Sandbox</Tabs.Tab>
        </Tabs.List>
        <Tabs.Panel value="inspect">
          <ShotInspector
            shot={selected}
            auditHash={inspection.score?.audit_hash ?? null}
            explanation={inspection.explanation}
            advice={inspection.advice}
            reasons={reference.reasons}
            loading={inspection.loading}
            errors={inspection.errors}
          />
        </Tabs.Panel>
        <Tabs.Panel value="sandbox">
          <SandboxTab parameters={reference.parameters} seed={selected?.readings ?? null} />
        </Tabs.Panel>
      </Tabs>
    </Paper>
  );
}

// Scores on demand rather than on every slider move: a request per drag frame would hammer the
// API and tell the user nothing they could read at that speed.
function SandboxTab({ parameters, seed }: { parameters: Parameter[]; seed: Readings | null }) {
  const [scored, setScored] = useState<Score | null>(null);
  const [busy, setBusy] = useState(false);
  const start = useMemo(() => seed ?? nominal(parameters), [seed, parameters]);

  const run = useCallback(async (readings: Readings) => {
    setBusy(true);
    try {
      setScored(await api.score(readings));
    } finally {
      setBusy(false);
    }
  }, []);

  if (!parameters.length) {
    return (
      <Text size="sm" c="dimmed">
        Waiting for machine limits.
      </Text>
    );
  }

  return (
    <Sandbox
      parameters={parameters}
      readings={start}
      scored={scored}
      busy={busy}
      onScore={run}
      onReset={() => setScored(null)}
    />
  );
}

function nominal(parameters: Parameter[]): Readings {
  return Object.fromEntries(parameters.map((parameter) => [parameter.name, parameter.nominal]));
}
