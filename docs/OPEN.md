# Open

- Windows and macOS builds are validated only by the CI workflow, not on a real machine.
- Answers cannot yet: group by two things at once (a cross-tab), link keys written differently
  (`#10234` against `10234`), treat survey answers as ordered, or say which month was over budget.
- Times with a UTC offset are bucketed in UTC.
- A saturating fit on near-exact data can report "did not settle" although its parameters are right.
- Under Flatpak the sandbox has its own process ids, so cache leases from the host may be misjudged.
- Previews read the live project from worker threads (a brief wrong preview at worst; nothing is stored).

Design and extension of the answer engine: `docs/ANSWERS.md`. Earlier plans and review worklists
(all items done): `docs/history/`.
