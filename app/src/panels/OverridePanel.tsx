// Operator override. The disagreement goes on the record with the explanation that was on
// screen when it was made -- re-deriving it later would log what the model says now.

import { Alert, Button, Group, Select, Stack, Text, Textarea } from '@mantine/core';
import { useState } from 'react';
import { override } from '../api/client';
import type { ReasonCode } from '../api/types';

interface Props {
  auditHash: string | null;
  flagged: boolean;
  reasons: ReasonCode[];
  explanationShown: string;
}

export function OverridePanel({ auditHash, flagged, reasons, explanationShown }: Props) {
  const [code, setCode] = useState<string | null>(null);
  const [note, setNote] = useState('');
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const chosen = reasons.find((reason) => reason.code === code);
  const needsNote = chosen?.note_required ?? false;
  const ready = Boolean(auditHash && code && (!needsNote || note.trim()));

  const submit = async () => {
    if (!auditHash || !code) return;
    setError(null);
    try {
      const saved = await override({
        audit_hash: auditHash,
        defective: !flagged,
        reason: code,
        note,
        explanation_shown: explanationShown,
      });
      setResult(saved.audit_hash);
      setNote('');
      setCode(null);
    } catch (problem) {
      setError((problem as Error).message);
    }
  };

  if (!auditHash) {
    return (
      <Text c="dimmed" size="sm">
        Score a shot before overriding it.
      </Text>
    );
  }

  return (
    <Stack gap="xs">
      <Text size="sm">
        The gate says <Text span fw={500}>{flagged ? 'defect' : 'ok'}</Text>. Recording a
        disagreement marks this part <Text span fw={500}>{flagged ? 'ok' : 'defect'}</Text>.
      </Text>
      <Select
        size="xs"
        placeholder="Reason"
        data={reasons.map((reason) => ({ value: reason.code, label: reason.label }))}
        value={code}
        onChange={setCode}
      />
      {needsNote && (
        <Textarea
          size="xs"
          autosize
          minRows={2}
          placeholder="Required for 'other'"
          value={note}
          onChange={(event) => setNote(event.currentTarget.value)}
        />
      )}
      <Group justify="space-between">
        <Text size="xs" c="dimmed">
          Signed against {auditHash.slice(0, 8)}
        </Text>
        <Button size="xs" disabled={!ready} onClick={submit}>
          Record override
        </Button>
      </Group>
      {error && <Alert color="red">{error}</Alert>}
      {result && (
        <Alert color="green" variant="light">
          <Text size="xs">Recorded as {result.slice(0, 8)} in the audit chain.</Text>
        </Alert>
      )}
    </Stack>
  );
}
