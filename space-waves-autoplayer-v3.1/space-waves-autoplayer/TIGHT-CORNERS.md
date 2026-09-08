# Detection and tight-corner update

Run START.bat as before. F8 starts and F9 stops. No new dependencies.

- Detection scans a smaller horizontal strip; dark arrow outlines are searched
  locally. Positions are not smoothed, avoiding extra lag at reversals.
- Fast mode uses a more accurate 5x5 distance approximation for diagonal gaps.
- First-turn decisions interpolate costs at the actual fractional arrow height.
- Latency prediction includes the full pre-planning frame age, and no longer
  caps forward prediction at 28 pixels at high speeds.
- Planning intervals round up to the capture interval; the live loop targets
  120 Hz when processing and screen capture can keep up (not guaranteed FPS).

Validation: existing suite: 26 tests, 3 skipped OCR screenshot groups because
the OCR dependency was unavailable. Four additional regressions pass, including
a narrow zigzag with one-frame delayed input and swept collision checks.

Synthetic 800x526 detection benchmark, 200 measured frames after 20 warmups,
single OpenCV thread: median 9.465 ms before / 8.382 ms after; p95 10.628 /
9.308 ms. These are this machine's synthetic measurements, not live-game FPS.
Windows input, real hard-level completion, and menu clicks were not tested.

Run tests: `python -m unittest test_planner test_tight_corners -q`.
