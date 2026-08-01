// The anchor rule for the selected shot.

import { Alert, Badge, Group, List, Loader, Stack, Text } from '@mantine/core';
import type { Explanation } from '../api/types';
import { bound, label, pct } from '../lib/format';

interface Props {
  explanation: Explanation | null;
  loading: boolean;
  error: string | null;
}

export function AnchorPanel({ explanation, loading, error }: Props) {
  if (loading) return <Loader size="sm" />;
  if (error) return <Alert color="red">{error}</Alert>;
  if (!explanation) return <Text c="dimmed" size="sm">Select a shot to explain it.</Text>;

  return (
    <Stack gap="xs">
      <Group gap="xs">
        <Badge color={explanation.prediction === 1 ? 'red' : 'green'} variant="light">
          {explanation.prediction === 1 ? 'defect' : 'ok'}
        </Badge>
        <Text size="xs" c="dimmed">
          holds {pct(explanation.precision)} of the time, covers {pct(explanation.coverage)} of the
          line
        </Text>
      </Group>
      {explanation.predicates.length ? (
        <Predicates explanation={explanation} />
      ) : (
        <EmptyAnchor precision={explanation.precision} />
      )}
    </Stack>
  );
}

function Predicates({ explanation }: { explanation: Explanation }) {
  return (
    <>
      <Text size="sm" fw={500}>
        This part is called {explanation.prediction === 1 ? 'defective' : 'good'} because:
      </Text>
      <List size="sm" spacing={4}>
        {explanation.predicates.map((predicate) => (
          <List.Item key={predicate.parameter}>
            <Text span fw={500}>
              {label(predicate.parameter)}
            </Text>{' '}
            is {bound(predicate.lower, predicate.upper)}
          </List.Item>
        ))}
      </List>
    </>
  );
}

// Not a failure. At the served threshold most of the line passes, so the empty rule already
// beats the precision target and there is nothing to add. Saying "no rule needed" is honest;
// inventing conditions to fill the panel would not be.
function EmptyAnchor({ precision }: { precision: number }) {
  return (
    <Alert color="gray" variant="light">
      <Text size="sm">
        No rule needed. The verdict holds {pct(precision)} of the time across the whole line, so
        nothing about this shot in particular is driving it.
      </Text>
    </Alert>
  );
}
