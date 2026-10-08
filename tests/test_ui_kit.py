"""UI kit consistency: every step has an icon and every category a colour, and no emoji glyphs sneak in."""
from dancr.core.registry import registry
from dancr.ui.icons import NODE_ICONS
from dancr.ui.theme import CATEGORY_COLORS


def test_every_step_has_an_icon_and_every_category_a_colour():
    missing_icons = [nt.key for nt in registry.all() if nt.key not in NODE_ICONS]
    missing_colours = sorted({nt.category for nt in registry.all()} - set(CATEGORY_COLORS))
    assert not missing_icons, f"steps with no icon (they show a dashed circle): {missing_icons}"
    assert not missing_colours, f"categories with no colour: {missing_colours}"


def test_no_step_icon_is_an_emoji():
    emoji = [(nt.key, nt.icon) for nt in registry.all()
             if nt.icon and any(ord(ch) > 0x1F000 for ch in nt.icon)]
    assert not emoji, f"emoji used as step icons: {emoji}"
