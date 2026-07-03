#!/usr/bin/env python3
"""Report Python-to-Rust translation surface drift.

This is intentionally name-oriented. It does not prove behavioral parity, but it
keeps the 1:1 item mapping visible while the port is being normalized.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Item:
    name: str
    path: str
    kind: str
    public: bool = True


def snake_case(name: str) -> str:
    name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return name.lower()


def rust_struct_name(py_class: str) -> str:
    overrides = {
        "CSPModule": "CSPModule",
        "CSP": "CSP",
        "SPP": "SPP",
        "PSABlock": "PSABlock",
        "PSA": "PSA",
        "DFL": "DFL",
        "YOLO": "YOLO",
        "EMA": "EMA",
        "QFL": "QFL",
        "VFL": "VFL",
    }
    return overrides.get(py_class, py_class)


def python_items() -> list[Item]:
    items: list[Item] = []
    for path in sorted((ROOT / "YOLOv11-pt").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                items.append(Item(node.name, rel, "function"))
            elif isinstance(node, ast.ClassDef):
                items.append(Item(node.name, rel, "class"))
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        if child.name == "__init__":
                            method = f"{node.name}.new"
                        elif child.name == "__getitem__":
                            method = f"{node.name}.get_item"
                        elif child.name == "__len__":
                            method = f"{node.name}.len"
                        elif child.name == "__call__":
                            method = f"{node.name}.call"
                        else:
                            method = f"{node.name}.{child.name}"
                        items.append(Item(method, rel, "method"))
    return items


def strip_cfg_test_modules(text: str) -> str:
    """Remove `#[cfg(test)]` items so fixtures don't count as translated Rust."""
    lines = text.splitlines(keepends=True)
    output: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == "#[cfg(test)]":
            start = i
            i += 1
            while i < len(lines) and (
                not lines[i].strip()
                or lines[i].lstrip().startswith("#[")
            ):
                i += 1
            if i >= len(lines):
                break

            depth = 0
            seen_open = False
            while i < len(lines):
                depth += lines[i].count("{") - lines[i].count("}")
                if "{" in lines[i]:
                    seen_open = True
                i += 1
                if seen_open and depth <= 0:
                    break
            output.extend("\n" for _ in range(i - start))
            continue
        output.append(lines[i])
        i += 1
    return "".join(output)


def rust_items() -> list[Item]:
    items: list[Item] = []
    for path in sorted((ROOT / "src").rglob("*.rs")):
        rel = path.relative_to(ROOT).as_posix()
        text = strip_cfg_test_modules(path.read_text())
        for match in re.finditer(r"(?m)^(pub\s+)?(?:struct|enum)\s+([A-Za-z_][A-Za-z0-9_]*)", text):
            items.append(Item(match.group(2), rel, "type", bool(match.group(1))))
        for match in re.finditer(r"(?m)^(pub\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)", text):
            items.append(Item(match.group(2), rel, "function", bool(match.group(1))))

        impl_type: str | None = None
        depth = 0
        for line in text.splitlines():
            if re.match(r"\s*impl\b", line):
                header = line.split("{", 1)[0].strip()
                if header.startswith("impl<"):
                    depth_angle = 0
                    close = -1
                    for index, char in enumerate(header):
                        if char == "<":
                            depth_angle += 1
                        elif char == ">":
                            depth_angle -= 1
                            if depth_angle == 0:
                                close = index
                                break
                    target = header[close + 1 :].strip() if close >= 0 else ""
                else:
                    target = header.removeprefix("impl").strip()
                impl_match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", target)
                impl_type = impl_match.group(1) if impl_match else None
                depth = line.count("{") - line.count("}")
                continue
            if impl_type is not None:
                method_match = re.match(r"\s*(pub\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)", line)
                if method_match:
                    items.append(
                        Item(
                            f"{impl_type}.{method_match.group(2)}",
                            rel,
                            "method",
                            bool(method_match.group(1)),
                        )
                    )
                depth += line.count("{") - line.count("}")
                if depth <= 0:
                    impl_type = None
    return items


def expected_rust_names(items: list[Item]) -> Counter[str]:
    names: Counter[str] = Counter()
    for item in items:
        if item.kind == "class":
            names[rust_struct_name(item.name)] += 1
        elif item.kind == "method":
            cls, method = item.name.split(".", 1)
            names[f"{rust_struct_name(cls)}.{snake_case(method)}"] += 1
        else:
            names[snake_case(item.name)] += 1
    return names


def mapped_targets() -> list[tuple[str, str, str]]:
    targets: list[tuple[str, str, str]] = []
    path = ROOT / "TRANSLATION_MAP.md"
    if not path.exists():
        return targets

    rows: list[tuple[str, str, str]] = []
    for line in path.read_text().splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 3:
            continue
        python_item, rust_target, status = cells
        if status.strip("`") == "glue":
            continue
        match = re.match(r"`([^`]+)`", rust_target)
        if not match:
            continue
        rows.append((python_item.strip("`"), match.group(1), status))

    class_paths: dict[str, str] = {}
    for python_item, target, _ in rows:
        if "." in python_item:
            continue
        if not target.startswith("src/") or "::" not in target:
            continue
        rust_path, rust_name = target.split("::", 1)
        class_paths[rust_name] = rust_path

    for python_item, target, _ in rows:
        if target.startswith("src/") and "::" in target:
            rust_path, rust_name = target.split("::", 1)
            targets.append((python_item, rust_path, rust_name))
            continue
        if "::" in target:
            rust_type, rust_method = target.split("::", 1)
            rust_path = class_paths.get(rust_type)
            if rust_path is not None:
                targets.append((python_item, rust_path, f"{rust_type}.{rust_method}"))
    return targets


def main() -> int:
    py = python_items()
    rust = rust_items()
    expected = expected_rust_names(py)
    actual = Counter(item.name for item in rust)
    actual_public = Counter(item.name for item in rust if item.public)
    ignored_extras = {
        "Batch",
        "HeadOutput",
        "YOLOOutput",
        "Sample",
        "Label",
        "AugmentParams",
        # Public glue needed by Rust callers even though the Python code returns tensors/lists.
        # Rust enum used to represent Python activation objects in Conv.
        "Activation",
        # Rust data/config/CLI glue around translated functions.
        "Config",
        "Config.load",
        "Config.num_classes",
        "Config.box_gain",
        "Config.cls_gain",
        "Config.dfl_gain",
        "Config.to_augment_params",
    }

    missing = sorted(
        name
        for name, count in expected.items()
        if actual[name] < count
    )
    actual_by_path = {(item.path, item.name) for item in rust}
    target_mismatches = sorted(
        (python_item, rust_path, rust_name)
        for python_item, rust_path, rust_name in mapped_targets()
        if (rust_path, rust_name) not in actual_by_path
    )
    overcounted_public = sorted(
        (name, actual_public[name], count)
        for name, count in expected.items()
        if actual_public[name] > count and name not in ignored_extras
    )
    undercounted = sorted(
        (name, count, actual[name])
        for name, count in expected.items()
        if actual[name] < count
    )
    extras = sorted(
        item.name
        for item in rust
        if (
            item.path.startswith("src/model/")
            or item.path.startswith("src/data/")
            or item.path.startswith("src/train/")
        )
        and item.public
        and item.name not in expected
        and item.name not in ignored_extras
    )

    print(f"Python items: {len(py)}")
    print(f"Expected Rust items: {sum(expected.values())}")
    print(f"Rust items: {len(rust)}")
    print()

    print("Missing mapped Rust items:")
    for name, expected_count, actual_count in undercounted:
        suffix = "" if expected_count == 1 else f" ({actual_count}/{expected_count})"
        print(f"  {name}{suffix}")
    print()

    print("Mapped Rust target path mismatches:")
    for python_item, rust_path, rust_name in target_mismatches:
        print(f"  {python_item} -> {rust_path}::{rust_name}")
    print()

    print("Overcounted public mapped Rust item names:")
    for name, actual_count, expected_count in overcounted_public:
        print(f"  {name} ({actual_count}/{expected_count})")
    print()

    print("Extra public translated-layer Rust items:")
    for name in extras:
        print(f"  {name}")

    return 1 if missing or target_mismatches or overcounted_public or extras else 0


if __name__ == "__main__":
    raise SystemExit(main())
