"""Site colors that admins can change from the Admin console, without touching code.

One accent color is turned into the full palette (buttons, header gradient, chips, page tint) for both light and
dark mode. Values use CSS light-dark(), so a theme switch recolors the page instantly, with no reload.
"""

import colorsys
import json
import os
import re

from src import config

PRESETS = {
    "🔵 Ocean blue": "#2563eb",
    "🟣 Royal violet": "#7c3aed",
    "🟢 Emerald": "#059669",
    "🟠 Sunset": "#ea580c",
    "🔴 Rose": "#e11d48",
    "🟡 Amber": "#d97706",
    "⚫ Graphite": "#475569",
}
CUSTOM = "🎨 Custom"
DEFAULTS = {
    "user": os.getenv("USER_THEME_COLOR", "#2563eb"),   # employee portal
    "admin": os.getenv("ADMIN_THEME_COLOR", "#7c3aed"),  # admin console
}
_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


# --- color math --------------------------------------------------------------------


def is_hex(value) -> bool:
    return isinstance(value, str) and bool(_HEX_RE.match(value))


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _hex(rgb) -> str:
    return "#" + "".join(f"{max(0, min(255, round(c))):02x}" for c in rgb)


def mix(a: str, b: str, t: float) -> str:
    """Blend color a toward color b by t (0 = a, 1 = b)."""
    ra, rb = _rgb(a), _rgb(b)
    return _hex(x + (y - x) * t for x, y in zip(ra, rb))


def tint(c: str, t: float) -> str:
    return mix(c, "#ffffff", t)


def shade(c: str, t: float) -> str:
    return mix(c, "#000000", t)


def alpha(c: str, a: float) -> str:
    r, g, b = _rgb(c)
    return f"rgba({r},{g},{b},{a})"


def hue_shift(c: str, degrees: float) -> str:
    r, g, b = (x / 255 for x in _rgb(c))
    h, lightness, s = colorsys.rgb_to_hls(r, g, b)
    r, g, b = colorsys.hls_to_rgb((h + degrees / 360) % 1, lightness, s)
    return _hex((r * 255, g * 255, b * 255))


def luminance(c: str) -> float:
    def channel(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (channel(x) for x in _rgb(c))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _ld(light: str, dark: str) -> str:
    return f"light-dark({light}, {dark})"


def palette(accent: str) -> dict[str, str]:
    """CSS variables (without the --hd- prefix) for one accent color, for light and dark mode."""
    if not is_hex(accent):
        accent = DEFAULTS["user"]
    accent = accent.lower()
    dark_accent = tint(accent, 0.12)
    page_dark = mix("#0b0e16", accent, 0.05)
    return {
        # surfaces: cool neutrals, faintly tinted with the accent so each portal has its own feel
        "bg": _ld(mix("#f5f7fb", accent, 0.015), page_dark),
        "bar": _ld("rgba(255,255,255,.88)", alpha(page_dark, 0.88)),
        "surface": _ld("#ffffff", mix("#141926", accent, 0.05)),
        "surface-2": _ld(mix("#f3f5f9", accent, 0.03), mix("#10141f", accent, 0.05)),
        "border": _ld("rgba(15,23,42,.09)", "rgba(148,163,184,.15)"),
        "border-strong": _ld("rgba(15,23,42,.18)", "rgba(148,163,184,.3)"),
        "muted": _ld("#5d6677", "#98a1b2"),
        "text": _ld("#0f172a", "#e6e9ef"),
        "shadow": f"0 1px 2px {_ld('rgba(15,23,42,.05)', 'rgba(0,0,0,.35)')}",
        "pop-shadow": f"0 12px 32px -12px {_ld('rgba(15,23,42,.25)', 'rgba(0,0,0,.6)')}",
        # the welcome banner: the portal color, deepened a little toward the bottom
        "panel": f"linear-gradient(165deg, {accent} 0%, {shade(accent, 0.3)} 100%)",
        # accent
        "accent": _ld(accent, dark_accent),
        "accent-hover": _ld(shade(accent, 0.12), accent),
        "accent-soft": _ld(alpha(accent, 0.09), alpha(dark_accent, 0.2)),
        "accent-text": _ld(shade(accent, 0.22), tint(accent, 0.55)),
        "on-accent": "#111827" if luminance(accent) > 0.45 else "#ffffff",
        "panel-shadow": f"0 18px 40px -24px {alpha(shade(accent, 0.3), 0.7)}",
        "brand": f"linear-gradient(150deg, {tint(accent, 0.15)}, {shade(accent, 0.3)})",
        # Streamlit's slider is painted in the config.toml blue; shift it to the accent's hue
        "slider-filter": _slider_filter(accent),
    }


def _slider_filter(accent: str) -> str:
    base_h, _, base_s = colorsys.rgb_to_hls(*(x / 255 for x in _rgb("#2563eb")))
    h, _, s = colorsys.rgb_to_hls(*(x / 255 for x in _rgb(accent)))
    return f"hue-rotate({round((h - base_h) * 360)}deg) saturate({round(s / base_s, 2)})"


def css_variables(accent: str) -> str:
    return ":root{" + "".join(f"--hd-{k}:{v};" for k, v in palette(accent).items()) + "}"


def preview_html(accent: str, label: str) -> str:
    """A small mock of the portal in this color, for the Admin console."""
    accent = accent if is_hex(accent) else DEFAULTS["user"]
    on = "#111827" if luminance(accent) > 0.45 else "#ffffff"
    hero = f"linear-gradient(165deg, {accent} 0%, {shade(accent, 0.3)} 100%)"
    return (
        f'<div class="hd-preview" style="background:{mix("#ffffff", accent, 0.03)}">'
        f'<div class="pv-bar"><span class="pv-logo" style="background:linear-gradient(135deg,{shade(accent, .35)},'
        f'{tint(accent, .2)})">🛠️</span><b>IT Helpdesk</b>'
        f'<span class="pv-tab" style="background:{alpha(accent, .1)};color:{shade(accent, .22)};'
        f'border-color:{accent}">Chat</span><span class="pv-tab">Tickets</span></div>'
        f'<div class="pv-hero" style="background:{hero}">{label}</div>'
        f'<div class="pv-row"><span class="pv-btn" style="background:{accent};color:{on}">Primary button</span>'
        f'<span class="pv-chip" style="background:{alpha(accent, .1)};color:{shade(accent, .22)}">Cited source</span>'
        f'<span class="pv-bubble" style="background:{accent};color:{on}">Your message</span></div></div>'
    )


# --- saved settings ----------------------------------------------------------------


def _settings_file():
    return config.LOG_DIR / "appearance.json"


_cache: dict = {"mtime": None, "data": {}}


def load() -> dict[str, str]:
    """The saved colors, falling back to the defaults (re-read only when the file changes)."""
    path = _settings_file()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return dict(DEFAULTS)
    if _cache["mtime"] != mtime:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        _cache.update(mtime=mtime, data=data if isinstance(data, dict) else {})
    data = _cache["data"]
    return {role: data[role] if is_hex(data.get(role)) else DEFAULTS[role] for role in DEFAULTS}


def save(colors: dict[str, str], updated_by: str = "") -> None:
    clean = {role: colors[role].lower() for role in DEFAULTS if is_hex(colors.get(role))}
    path = _settings_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**clean, "updated_by": updated_by}, indent=2), encoding="utf-8")


def reset() -> None:
    try:
        _settings_file().unlink()
    except OSError:
        pass
    _cache.update(mtime=None, data={})


def preset_name(color: str) -> str:
    return next((name for name, value in PRESETS.items() if value == (color or "").lower()), CUSTOM)
