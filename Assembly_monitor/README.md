# Fix: Stage_01 being skipped, and display disagreeing with the system

## The bug you described, reproduced

```
start                            -> current_stage = None, needs body_a x1 + bushing x1
show body_a + BOTH bushings      -> current_stage = Stage_02
```

Stage_01 was **never confirmed**. The system jumped straight to
Stage_02. So in practice the first thing it waited for was TWO
bushings - exactly what you said. This was in code I wrote.

**Cause:** while the current stage was still unconfirmed, evidence that
happened to match the NEXT stage was allowed to advance the sequence.
If both bushings were in view at the start - or the detector
double-detected one - Stage_01 was bypassed entirely.

**Fix:** a stage can no longer be skipped. While the current stage is
unconfirmed, matching the next stage does nothing. Each stage must be
confirmed in turn, in order.

```
both bushings at start  -> stays unconfirmed, still asking for Stage_01
body_a + 1 bushing      -> Stage_01
then 2nd bushing        -> Stage_02
```

## Display and system now read from one source

The panel used to print a worded hint while the state machine tested a
different dictionary, so any mismatch was invisible on screen. The
panel now prints the live requirement straight from the state machine:

```
Stage:  Stage_01
needs:  body_a x1, bushing x1        <-- read directly from the logic
Do this now: Install one more bushing
```

| Panel shows | System checks |
|---|---|
| Stage_01 - body_a x1, bushing x1 | identical |
| Stage_02 - body_a x1, bushing x2 | identical |
| Stage_03 - body_a x1, body_b x1, bushing x2 | identical |
| Stage_04 - + block_pin x1 | identical |
| Stage_05 - + nut x1 | identical |
| Stage_06 - knob x1, threaded_pin x1 | identical |
| Stage_07 - all seven | identical |

They cannot disagree now - both come from `current_requirement`.

## Regression suite, 10/10

| Check | Result |
|---|---|
| Both bushings at start does not skip Stage_01 | PASS |
| Genuine assembly reaches Stage_07 | PASS |
| One bushing held forever stays Stage_01 | PASS |
| Detector glitch does not advance | PASS |
| False Stage_02 gets revoked | PASS |
| Foreign part -> WRONG_PART | PASS |
| Skip ahead -> WRONG_ORDER | PASS |
| Blank camera, 400 frames, no drift | PASS |
| Slow pace + heavy occlusion reaches Stage_07 | PASS |
| Manual stage control + reset | PASS |

## Also check this on your machine

The state machine loads the sequence from a JSON file if it exists, and
only uses the built-in order if it does not:

```
D:\Thesis\Blender\synthetic_dataset\subassembly\assembly_sequence.json
```

**If that file exists, it overrides the code entirely.** If its order is
wrong, the running system uses the wrong order no matter what the code
says. Check it with:

```powershell
Get-Content "D:\Thesis\Blender\synthetic_dataset\subassembly\assembly_sequence.json"
```

If it is missing, the console prints a warning at startup and the
correct built-in order is used.

## Still open - the detection problem

None of the above makes YOLO detect the nut, threaded_pin or knob. If
the detector never outputs `nut`, the nut stage cannot confirm, because
the signal never arrives.

Consistent with your training numbers: `threaded_pin` had the lowest
mAP50 of all seven classes (0.898), and bushing, nut and threaded_pin
all had far fewer training instances than expected - the
occlusion-aware labelling correctly skipped them when hidden, so they
got less training signal. Exactly these three classes.

**Diagnostic worth 20 minutes before generating any new data**, because
the fix differs completely:

Put a nut alone on the table, well lit, nothing else in view. Watch the
"Detected components" list.

- **Detected alone but not during assembly** - the model is fine; the
  problem is occlusion and scale once it is installed. More of the same
  synthetic images will not help. You need images of the nut in its
  installed position, partially hidden.
- **Not detected even alone** - genuine model weakness. Retraining is
  the right call, and generation should be weighted toward nut,
  threaded_pin and knob rather than spread evenly over all seven
  classes.

Same test for threaded_pin and knob.
