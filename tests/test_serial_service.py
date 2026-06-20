import datetime as dt

from core.serial_service import (
    build_counter_key,
    build_preview_serial,
    default_serial_template,
    generate_serial,
    normalize_serial_part,
)


NOW = dt.datetime(2026, 6, 5, 8, 9, 10)


def test_default_template_matches_legacy_serial_and_counter_shape() -> None:
    counters = {}

    result = generate_serial(
        default_serial_template(),
        recipe_name="default_recipe",
        run_id="abcdef123456",
        counters=counters,
        now=NOW,
    )

    assert result.serial == "20260605-default_recipe-001"
    assert result.counter_day == "20260605"
    assert result.counter_key == "default_recipe"
    assert counters == {"20260605": {"default_recipe": 1}}


def test_preview_uses_fixed_seq_without_mutating_counter_input() -> None:
    result = build_preview_serial(default_serial_template(), recipe_name="demo", now=NOW)

    assert result.serial == "20260605-demo-001"
    assert result.seq_text == "001"


def test_counter_key_uses_enabled_distinguishing_fields() -> None:
    customer_template = {
        "separator": "-",
        "custom_values": {"customer": " 客户A "},
        "fields": [
            {"type": "system", "key": "date", "enabled": True},
            {"type": "custom", "key": "customer", "enabled": True},
            {"type": "system", "key": "seq", "enabled": True},
        ],
    }
    date_only_template = {
        "separator": "-",
        "fields": [
            {"type": "system", "key": "date", "enabled": True},
            {"type": "system", "key": "seq", "enabled": True},
        ],
    }

    assert build_counter_key(default_serial_template(), recipe_name="recipe one", now=NOW) == "recipe_one"
    assert build_counter_key(customer_template, recipe_name="ignored", now=NOW) == "客户A"
    assert build_counter_key(date_only_template, recipe_name="ignored", now=NOW) == "20260605"


def test_field_order_disabled_text_and_chinese_values_render() -> None:
    template = {
        "separator": "_",
        "custom_values": {"customer": "客户A", "batch": "B01"},
        "fields": [
            {"type": "custom", "key": "customer", "enabled": True},
            {"type": "text", "key": "text", "value": "FRP", "enabled": True},
            {"type": "system", "key": "recipe", "enabled": False},
            {"type": "custom", "key": "batch", "enabled": True},
            {"type": "system", "key": "seq", "enabled": True},
        ],
    }

    result = generate_serial(template, recipe_name="recipe", run_id="abcdef", counters={}, now=NOW)

    assert result.serial == "客户A_FRP_B01_001"


def test_template_without_seq_appends_run_id_suffix() -> None:
    template = {
        "separator": "-",
        "custom_values": {"customer": "客户A"},
        "fields": [
            {"type": "system", "key": "date", "enabled": True},
            {"type": "custom", "key": "customer", "enabled": True},
        ],
    }
    counters = {}

    result = generate_serial(template, recipe_name="recipe", run_id="a4f91cabcdef", counters=counters, now=NOW)

    assert result.serial == "20260605-客户A__A4F91C"
    assert result.seq is None
    assert result.suffix_added is True
    assert counters == {}


def test_normalize_serial_part_keeps_chinese_and_sanitizes_counter_values() -> None:
    assert normalize_serial_part("  客户 A<>:/B  ") == "客户_A_B"
    assert normalize_serial_part("a___b") == "a_b"
    assert normalize_serial_part("x" * 40) == "x" * 24
