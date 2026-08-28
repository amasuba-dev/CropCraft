"""The unattended runner's failure handling.

The whole point of `ggssvt.run_all` is that it survives a night alone, so the
behaviour worth pinning is not that it runs things, it is what it does when one
of them fails. An optional failure must not cost the remaining hours, and a
required failure must not let later steps score a stale cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ggssvt.run_all import Step, _fresh, _tail, programme


def _fail(key: str, *, required: bool) -> Step:
    return Step(key, key, [sys.executable, "-c", "raise SystemExit(3)"],
                None, required)


def _pass(key: str) -> Step:
    return Step(key, key, [sys.executable, "-c", "pass"], None, False)


def test_an_optional_failure_does_not_stop_the_run(monkeypatch, tmp_path):
    """The mesh arm without scikit-image must not cost the campaign."""
    import ggssvt.run_all as mod

    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(mod, "programme",
                        lambda *a, **k: (_fail("optional_boom", required=False),
                                         _pass("after")))
    monkeypatch.setattr(mod, "_summarise",
                        lambda outcomes, *a: seen.extend(
                            (o.key, o.status) for o in outcomes))

    mod.run(work_dir=tmp_path, device="cpu")
    assert dict(seen) == {"optional_boom": "failed", "after": "done"}


def test_a_required_failure_abandons_the_rest(monkeypatch, tmp_path):
    """Nothing downstream may run against a cache that was never rebuilt."""
    import ggssvt.run_all as mod

    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(mod, "programme",
                        lambda *a, **k: (_fail("required_boom", required=True),
                                         _pass("after")))
    monkeypatch.setattr(mod, "_summarise",
                        lambda outcomes, *a: seen.extend(
                            (o.key, o.status) for o in outcomes))

    code = mod.run(work_dir=tmp_path, device="cpu")
    assert dict(seen) == {"required_boom": "failed", "after": "blocked-by"}
    assert code == 1


def test_a_current_artefact_is_skipped_so_a_night_resumes(tmp_path):
    """Re-running after an interruption must not redo eleven hours."""
    artefact = tmp_path / "done.json"
    artefact.write_text("{}", encoding="utf-8")
    step = Step("k", "k", ["true"], artefact)

    assert _fresh(step, artefact.stat().st_mtime - 1)
    # An artefact older than the ground truth it was fitted to is not current.
    assert not _fresh(step, artefact.stat().st_mtime + 1)


def test_a_missing_artefact_always_runs(tmp_path):
    step = Step("k", "k", ["true"], tmp_path / "never_written.json")
    assert not _fresh(step, 0.0)


def test_every_step_key_is_unique():
    """--skip and --only address steps by key, so duplicates would be silent."""
    keys = [s.key for s in programme(Path("wd"), device="cpu", plan="core",
                                     batch_size=2, workers=4)]
    assert len(keys) == len(set(keys))


def test_the_required_steps_are_the_ones_others_read():
    """A step is required exactly when a later step consumes its output."""
    steps = {s.key: s for s in programme(Path("wd"), device="cpu", plan="core",
                                         batch_size=2, workers=4)}
    for key in ("inspect", "preprocess", "gate", "fuse", "baselines"):
        assert steps[key].required, key
    for key in ("mesh", "dino_probe", "posefree", "campaign"):
        assert not steps[key].required, key


def test_tail_survives_a_missing_log(tmp_path):
    assert _tail(tmp_path / "absent.log") == ""
