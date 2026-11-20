# Reviewing what happens to a human contribution

This experimental review aid asks a narrow process question: **after a person
introduces a criterion, interpretation, or reason, what observable evidence shows
how the agent handled it in subsequent work?**

Confirmation exchanges and final task performance answer different questions.
Acknowledging a message does not establish that its meaning reached the shared
work. Retaining a decision with a sound explanation can also be substantive.
This aid complements the existing analyses; it does not change their scores or
define a validated measure of agency, human benefit, or learning.

## Prepare a packet

From the repository root, using Python 3.11 or later:

```sh
python scripts/prepare_contribution_review.py \
  --event-log path/to/event_log.jsonl --human-role user \
  --output review.json
```

Use the exact human role in the log; repeat `--human-role` for multiple people.
The script needs only the Python standard library. It exports the entire event
sequence, including failed actions and later revisions, and creates **blank**
review records for successful human events. Both messages and workspace actions
are candidates. They are not automatically classified as meaningful reasoning.
Unknown roles produce an error rather than an apparently empty assessment.

Each event retains its action, status, timestamp, and public observation, where
available. Indices count JSONL events from zero, ignoring blank lines. The source
file's SHA-256 identifies the exact input. Private observations, repeated chat
history, and unrelated fields are omitted. A missing public observation is `null`,
not evidence of unchanged work. Actions and public task material can still contain
sensitive information; inspect a packet before sharing it. The script will not
overwrite an existing output.

## Annotate evidence, not conversational style

1. Read the task context and the complete sequence. Select one consequential
   contribution and copy its exact span into `contribution_span`. Split compound
   contributions into separate review records when their handling differs.
2. Follow it through the subsequent actions and public workspace states, including
   later human changes of direction. Cite zero-based `evidence_event_indices`.
   Attribute human edits to the human; a later agent acknowledgment is not proof
   the agent made the change. Do not assume private knowledge was available to it.
3. Assign one provisional disposition below and explain the grounds and limits in
   `explanation`. These labels are an annotation convention, not quality rankings.

| Disposition | Evidence needed |
| --- | --- |
| `implemented` | The relevant subsequent work embodies the selected contribution. A claim that an edit succeeded is not enough when the recorded state contradicts it. |
| `reasoned_response` | The agent engages the grounds and explains a consequential retention, qualification, or disagreement. Agreement and reversal are not required. |
| `not_implemented` | Observable subsequent work contradicts or omits the selected contribution without a substantive response. Cite the contradiction, not just absence of another question. |
| `not_applicable` | The selected event introduces no consequential judgment to trace, such as routine delegation or closure. This is not a failure. |
| `insufficient_evidence` | The record cannot establish the disposition, for example because it ends before a revision or omits the relevant state. Do not turn a truncated log into a failure. |

If one part is implemented and another omitted, annotate each span separately.
When the selected contribution is reflected in the subsequent work, prefer
`implemented`, even if the agent also explains its reasoning. Use
`reasoned_response` when the requested change is not made but its grounds receive
a substantive response, or when a reasoned retention needs no work revision.
Keeping a decision while correcting its recorded rationale can be `implemented`:
the meaningful change is in the reason, not necessarily the chosen option.
Use `insufficient_evidence` for unresolved cases rather than forcing a label.
Do not infer motive or human satisfaction from these dispositions. Record
unnecessary questions, imposed framing, repair effort, or appropriate closure in
the explanation when observable. More participation, questions, or edits need not
mean better collaboration. Missing task context can make the whole case unratable.
An observed handling is not proof that the contribution caused the outcome; the
same work might have resulted without it.

## Calibration cases

`tests/fixtures/contribution_review/` contains eight **authored synthetic** logs.
They are neither participant records nor observed system runs. Their event fields
follow `TaskEnvNode.add_record_to_event_log`; the shared editor uses the tabular
analysis environment's `result_editor` observation. The short task context is
included in the initial message.

```sh
python scripts/prepare_contribution_review.py \
  --event-log tests/fixtures/contribution_review/01_implemented.jsonl \
  --human-role user --output calibration-review.json
python -m unittest discover -s tests -p test_contribution_review.py -v
```

Reviewers should label exported packets independently before opening
`reference_annotations.json`. Reference labels are proposed interpretations for
calibration, not empirical ground truth. Cases cover an applied criterion, an
acknowledgment followed by contradictory work, reasoned retention, reasoned
disagreement, routine closure, a truncated record, a failed edit, and a human edit
that is subsequently overwritten. Tests verify packet integrity and reference
links; they cannot establish the validity of the judgments.

## Small validation step

Before proposing an automated evaluator, have two reviewers independently annotate
the calibration cases, record disagreements, and refine ambiguous definitions.
Then use a prespecified small sample of complete collaboration episodes with task
context, keeping the sample and exclusions explicit. Report agreement, unresolved
cases, and evidence for individual judgments; do not turn a convenience sample
into a prevalence estimate or a leaderboard. Compare useful handling and effort
separately from output quality. These observations alone cannot establish improved
human judgment or learning.

Related discussion: [intervention purpose and CollabSkill](https://github.com/SALT-NLP/collaborative-gym/issues/15).
