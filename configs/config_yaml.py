#!/usr/bin/env python3
"""
config_yaml.py — 轻量版 YAML 字段批量修改 + 运行脚本

用法:  python config_yaml.py --mirror config_mirror.txt --script run.sh

流程:
  1. 从 {yaml}_mirror.txt 读取模式标记
  2. 展开 {1-3, 5} 等语法，生成所有组合
  3. 每次硬修改原始 yaml，运行一次 .sh
  4. 全部跑完后恢复原始 yaml

依赖: pyyaml (pip install pyyaml)
"""

import argparse
import itertools
import os
import re
import shutil
import subprocess
import sys

_PATTERN_RE = re.compile(r"\{([^}]+)\}")


def _parse_val(token):
    token = token.strip()
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        pass
    low = token.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "none", "~"):
        return None
    return token


def expand_pattern(value_str):
    m = _PATTERN_RE.search(value_str)
    if not m:
        return [_parse_val(value_str)]
    parts = [p.strip() for p in m.group(1).split(",")]
    expanded = []
    for part in parts:
        if "-" in part and not part.startswith("-"):
            try:
                l, r = part.split("-", 1)
                start, end = int(l.strip()), int(r.strip())
                step = 1 if end >= start else -1
                expanded.extend(range(start, end + step, step))
                continue
            except ValueError:
                pass
        expanded.append(_parse_val(part))
    return expanded


def find_patterns(text):
    results = []
    lines = text.splitlines()
    path = []
    for i, line in enumerate(lines, 1):
        indent = len(line) - len(line.lstrip())
        level = indent // 2
        path = path[:level]
        clean = line.split("#")[0].strip()
        if not clean:
            continue
        kv = re.match(r"^(\s*)([\w_]+)\s*:\s*(.*)", line)
        if kv:
            key = kv.group(2).strip()
            raw = kv.group(3).strip()
            path.append(key)
            fp = ".".join(path)
            if "{" in raw and "}" in raw:
                results.append((i, fp, raw))
    return results


def set_field(raw_yaml, field_path, new_value):
    keys = field_path.split(".")
    lines = raw_yaml.split("\n")
    path = []
    for i, line in enumerate(lines):
        indent = len(line) - len(line.lstrip())
        level = indent // 2
        path = path[:level]
        clean = line.split("#")[0].strip()
        if not clean:
            continue
        kv = re.match(r"^(\s*)([\w_]+)\s*:\s*(.*)", line)
        if kv:
            key = kv.group(2).strip()
            path.append(key)
            if ".".join(path) == field_path:
                comment = ""
                cm = re.search(r"(#.*)$", kv.group(3))
                if cm:
                    comment = " " + cm.group(1)
                lines[i] = f"{kv.group(1)}{key}: {new_value}{comment}"
                break
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="批量修改 yaml 字段并运行 .sh 脚本"
    )
    parser.add_argument("--mirror", "-m", required=True,
        help="镜像文件，命名须为 {yaml名}_mirror.txt")
    parser.add_argument("--script", "-s", required=True,
        help="每次修改后要运行的 .sh 脚本")
    args = parser.parse_args()

    mirror_path = os.path.abspath(args.mirror)
    script_path = os.path.abspath(args.script)

    # 从 mirror 文件名推导 yaml 路径
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 解析 mirror 路径：相对路径以 .py 所在目录为准
    mirror_path = args.mirror
    if not os.path.isabs(mirror_path):
        mirror_path = os.path.join(script_dir, mirror_path)
    mirror_path = os.path.abspath(mirror_path)

    # 解析 script 路径：相对路径以 .py 所在目录为准
    script_path = args.script
    if not os.path.isabs(script_path):
        script_path = os.path.join(script_dir, script_path)
    script_path = os.path.abspath(script_path)

    # yaml 文件与 mirror 文件放在同一目录
    base = os.path.basename(mirror_path)
    if not base.endswith("_mirror.txt"):
        print("错误: mirror 文件必须以 _mirror.txt 结尾")
        sys.exit(1)
    yaml_name = base[: -len("_mirror.txt")]
    yaml_path = os.path.join(os.path.dirname(mirror_path), f"{yaml_name}.yaml")

    for fpath, label in [(mirror_path, "mirror"),
                          (yaml_path, "yaml"),
                          (script_path, "script")]:
        if not os.path.isfile(fpath):
            print(f"错误: {label} 文件不存在: {fpath}")
            sys.exit(1)

    # 读取 mirror 并展开
    with open(mirror_path, "r", encoding="utf-8") as f:
        mirror_text = f.read()

    patterns = find_patterns(mirror_text)
    if not patterns:
        print("mirror 中未找到 {pattern} 标记，无需运行。")
        return

    field_map = {}
    for _, fp, raw in patterns:
        field_map[fp] = expand_pattern(raw)

    field_names = list(field_map.keys())
    value_lists = [field_map[f] for f in field_names]
    combos = list(itertools.product(*value_lists))

    print(f"\n发现 {len(patterns)} 个模式字段，共 {len(combos)} 种组合:\n")
    for fp, vals in field_map.items():
        print(f"  {fp}: {vals}")
    print()

    for i, combo in enumerate(combos, 1):
        labels = [f"{k}={v}" for k, v in zip(field_names, combo)]
        print(f"  [{i}/{len(combos)}] " + ", ".join(labels))

    # 备份原始 yaml
    with open(yaml_path, "r", encoding="utf-8") as f:
        original_yaml = f.read()
    backup_path = yaml_path + ".bak"
    shutil.copy2(yaml_path, backup_path)

    try:
        for i, combo in enumerate(combos, 1):
            labels = [f"{k}={v}" for k, v in zip(field_names, combo)]
            label = ", ".join(labels)
            print(f"\n=== [{i}/{len(combos)}] {label} ===")

            # 硬修改 yaml
            current = original_yaml
            for fp, val in zip(field_names, combo):
                current = set_field(current, fp, val)
            with open(yaml_path, "w", encoding="utf-8") as f:
                f.write(current)

            # 运行脚本
            env = os.environ.copy()
            env["CONFIG_PATH"] = yaml_path
            env["RUN_INDEX"] = str(i)
            result = subprocess.run(
                ["bash", script_path],
                cwd=os.path.dirname(script_path),
                env=env,
                capture_output=False,
            )
            status = ("OK" if result.returncode == 0
                      else f"FAIL (exit {result.returncode})")
            print(f"=== [{i}/{len(combos)}] {label} -> {status} ===")

    finally:
        shutil.copy2(backup_path, yaml_path)
        os.remove(backup_path)
        print(f"\n已恢复原始 yaml: {yaml_path}")

    print(f"\n完成: {len(combos)} 次实验运行完毕\n")


if __name__ == "__main__":
    main()
