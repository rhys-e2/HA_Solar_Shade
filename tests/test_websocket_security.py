"""Security regression tests for the Solar Shade websocket API.

The test harness replaces ``homeassistant`` and ``voluptuous`` with mocks (see
conftest.py), so these tests inspect the source and the exported validation
patterns rather than exercising the real decorators.
"""

import ast
import re
from pathlib import Path

import pytest

from custom_components.solar_shade import websocket_api as ws

_SOURCE = Path(ws.__file__).read_text()
_TREE = ast.parse(_SOURCE)


def _decorator_names(fn: ast.AST) -> list[str]:
    """Return decorator names outermost-first, e.g. 'websocket_api.require_admin'."""
    names = []
    for dec in fn.decorator_list:
        node = dec.func if isinstance(dec, ast.Call) else dec
        names.append(ast.unparse(node))
    return names


def _command_functions():
    return [
        n
        for n in ast.walk(_TREE)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and "websocket_api.websocket_command" in _decorator_names(n)
    ]


class TestAdminRequired:
    def test_commands_were_found(self):
        # Guards against the AST walk silently matching nothing.
        assert len(_command_functions()) >= 13

    @pytest.mark.parametrize("fn", _command_functions(), ids=lambda f: f.name)
    def test_every_command_requires_admin(self, fn):
        names = _decorator_names(fn)
        assert "websocket_api.require_admin" in names, (
            f"{fn.name} is reachable by any logged-in user; add "
            "@websocket_api.require_admin"
        )
        # require_admin must wrap websocket_command (be listed above it).
        assert names.index("websocket_api.require_admin") < names.index(
            "websocket_api.websocket_command"
        )

    def test_every_registered_command_is_a_checked_command(self):
        """A command registered without the decorator would bypass the check."""
        registered = {
            ast.unparse(call.args[1])
            for call in ast.walk(_TREE)
            if isinstance(call, ast.Call)
            and ast.unparse(call.func) == "websocket_api.async_register_command"
        }
        decorated = {f.name for f in _command_functions()}
        assert registered == decorated


class TestZoneFieldPatterns:
    ID = re.compile(ws.ZONE_ID_PATTERN)
    COLOR = re.compile(ws.ZONE_COLOR_PATTERN)

    @pytest.mark.parametrize(
        "zone_id",
        ["front_lawn", "front_lawn_2", "a", "zone-1", "A1_b2", "", "x" * 128],
    )
    def test_id_accepts_panel_generated_values(self, zone_id):
        assert self.ID.match(zone_id)

    @pytest.mark.parametrize(
        "zone_id",
        [
            'x"><img src=x onerror=alert(1)>',
            "a b",
            "a'b",
            "a<b",
            "a&b",
            "front_lawn\n",  # `$` would accept a trailing newline; \Z must not
            "x" * 129,
            "é",
        ],
    )
    def test_id_rejects_markup_and_oversize(self, zone_id):
        assert not self.ID.match(zone_id)

    @pytest.mark.parametrize(
        "color",
        ["#2196F3", "#4CAF50", "#fff", "#FFFFFF", "#ffffff80", "#abc"],
    )
    def test_color_accepts_hex(self, color):
        assert self.COLOR.match(color)

    @pytest.mark.parametrize(
        "color",
        [
            "red",
            "#fff;background:url(//evil/x)",
            'red;"><img src=x onerror=alert(1)>',
            "#ff",
            "#fffffffff",
            "#fff\n",
            "",
        ],
    )
    def test_color_rejects_non_hex(self, color):
        assert not self.COLOR.match(color)

    def test_panel_palette_is_accepted(self):
        panel = (Path(ws.__file__).parent / "frontend" / "panel.html").read_text()
        block = re.search(r"const ZONE_COLORS = \[(.*?)\];", panel, re.S).group(1)
        colors = re.findall(r"'(#[0-9A-Fa-f]+)'", block)
        assert len(colors) >= 10
        for c in colors:
            assert self.COLOR.match(c), c


def test_limits_are_sane():
    assert ws.MAX_ZONES >= 20
    assert ws.MAX_ZONE_VERTICES >= 100
    # Panel flushes at 1000 strokes; the server cap must exceed that.
    assert ws.MAX_ERASER_STROKES > 1000
    assert ws.MAX_HOUR_OFFSET >= 24
