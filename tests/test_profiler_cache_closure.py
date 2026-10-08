"""Pin the actual sibling profiler implementation without arbitrary traversal."""

import importlib.metadata

import pytest

from triton_evaluation.runtime.profiler_runtime import MANIFEST_ENV
from triton_evaluation.runtime.reference_cache import (
    CacheDeployment,
    CacheMeasurement,
    ReferenceCacheError,
    sha256,
)


def deployment(tmp_path, helper_key):
    engine = tmp_path / "engine"
    engine.mkdir()
    names = ["verify.py", "benchmark.py", "_common_utils.py"]
    for name in names:
        (engine / name).write_text("controlled implementation")
    helper = tmp_path / "runtime" / "profiler_runtime.py"
    helper.parent.mkdir()
    helper.write_text("controlled runtime helper")
    marker = tmp_path / "runtime-version"
    marker.write_text("controlled runtime")
    model = CacheDeployment(
        generation="controlled-cpu",
        runtime_versions={"pytest": importlib.metadata.version("pytest")},
        runtime_files={str(marker): sha256(marker.read_bytes())},
        algorithm_files={
            **{name: sha256((engine / name).read_bytes()) for name in names},
            **({helper_key: sha256(helper.read_bytes())} if helper_key is not None else {}),
        },
        device_conditions={"0": {"hardware": "controlled"}},
        measurement=CacheMeasurement(method="profiler", warmup=5, repeats=20, cv_threshold=0.3),
    )
    return model, engine, helper


def test_only_named_sibling_helper_is_allowed_and_live_hash_is_checked(tmp_path):
    model, engine, helper = deployment(tmp_path, "../runtime/profiler_runtime.py")
    model.check_runtime(engine)
    helper.write_text("changed actual helper")
    with pytest.raises(ReferenceCacheError, match="algorithm changed"):
        model.check_runtime(engine)


@pytest.mark.parametrize(
    "key", [
        "../other.py", "../../profiler_runtime.py", "/profiler_runtime.py", "..",
        "../profiler_runtime.py", "../runtime/other.py",
    ]
)
def test_arbitrary_traversal_still_refused(tmp_path, key):
    with pytest.raises(ValueError, match="basenames"):
        deployment(tmp_path, key)


def test_configured_runtime_cannot_use_old_cache_without_helper_pin(tmp_path, monkeypatch):
    model, engine, _helper = deployment(tmp_path, None)
    monkeypatch.setenv(MANIFEST_ENV, str(tmp_path / "profiler-manifest.json"))
    with pytest.raises(ReferenceCacheError, match="must pin"):
        model.check_runtime(engine)


def test_named_helper_cannot_be_a_symlink_outside_service_root(tmp_path):
    model, engine, helper = deployment(tmp_path, "../runtime/profiler_runtime.py")
    outside = tmp_path / "outside.py"
    outside.write_bytes(helper.read_bytes())
    helper.unlink()
    helper.symlink_to(outside)
    with pytest.raises(ReferenceCacheError, match="stay in service runtime package"):
        model.check_runtime(engine)


def test_engine_root_cannot_redirect_fixed_helper_through_symlink(tmp_path):
    model, engine, _helper = deployment(tmp_path, "../runtime/profiler_runtime.py")
    alias = tmp_path / "engine-alias"
    alias.symlink_to(engine, target_is_directory=True)
    with pytest.raises(ReferenceCacheError, match="stay in service runtime package"):
        model.check_runtime(alias)
