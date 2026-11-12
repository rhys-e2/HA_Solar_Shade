"""Provider modules must be imported off the event loop.

Home Assistant flags blocking calls such as importlib.import_module() made on
the loop ("Detected blocking call to import_module ..."), so the lazy provider
registry is loaded in the executor via async_load_providers().
"""

import asyncio
import importlib
import threading
import time
from pathlib import Path

import pytest

from custom_components.solar_shade import elevation_provider as ep

_PROVIDER_FILES = sorted(
    p.stem for p in Path(ep.__file__).parent.glob("*_provider.py")
)


class _Hass:
    """Minimal stand-in: async_add_executor_job runs in a worker thread."""

    async def async_add_executor_job(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(None, fn, *args)


@pytest.fixture
def fresh_registry_state():
    """Force the lazy loader to run again, then restore the real state."""
    saved = ep._PROVIDERS_LOADED
    ep._PROVIDERS_LOADED = False
    yield
    ep._PROVIDERS_LOADED = saved


def _recording_import(monkeypatch, delay=0.0):
    """Patch importlib.import_module; return the list of (name, thread) calls."""
    real = importlib.import_module
    calls = []

    def recording(name, package=None):
        calls.append((name, threading.get_ident()))
        if delay:
            time.sleep(delay)
        return real(name, package)

    monkeypatch.setattr(importlib, "import_module", recording)
    return calls


def test_providers_are_imported_off_the_event_loop(fresh_registry_state, monkeypatch):
    calls = _recording_import(monkeypatch)
    loop_thread = []

    async def run():
        loop_thread.append(threading.get_ident())
        await ep.async_load_providers(_Hass())

    asyncio.run(run())

    assert len(calls) == len(_PROVIDER_FILES) >= 4
    assert all(t != loop_thread[0] for _, t in calls), "import ran on the event loop"
    assert ep._PROVIDERS_LOADED is True
    assert {"ign", "nrw", "pdok", "swisstopo", "usgs"} <= {p for p, _ in ep.list_providers()}


def test_already_loaded_registry_does_no_work(monkeypatch):
    ep._ensure_providers_loaded()  # make the precondition explicit (order-independent)
    assert ep._PROVIDERS_LOADED
    calls = _recording_import(monkeypatch)
    asyncio.run(ep.async_load_providers(_Hass()))
    assert calls == []


def test_overlapping_loads_both_see_a_complete_registry(fresh_registry_state, monkeypatch):
    """A second caller must wait for the first load, not see a half-built registry."""
    calls = _recording_import(monkeypatch, delay=0.05)  # widen the race window
    seen = []

    async def caller():
        await ep.async_load_providers(_Hass())
        seen.append((len(calls), ep._PROVIDERS_LOADED))

    async def run():
        await asyncio.gather(caller(), caller())

    asyncio.run(run())

    assert len(calls) == len(_PROVIDER_FILES)  # loaded exactly once
    # Each caller only resumed after *all* provider modules had been imported.
    assert seen == [(len(_PROVIDER_FILES), True)] * 2


def test_try_download_loads_providers_before_detecting():
    """The async call site must preload before the first sync registry access."""
    src = (Path(ep.__file__).parent / "__init__.py").read_text()
    body = src[src.index("async def _try_download("):]
    body = body[: body.index("\nasync def ", 10)]
    assert "await async_load_providers(hass)" in body
    assert body.index("await async_load_providers(hass)") < body.index("detect_provider(")
    assert body.index("await async_load_providers(hass)") < body.index("get_provider(provider_id)")
