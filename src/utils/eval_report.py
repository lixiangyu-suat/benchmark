"""Plain-text tree formatting shared by evaluation console output and logs."""

from unicodedata import east_asian_width


def _display_width(text):
    return sum(2 if east_asian_width(char) in {"W", "F"} else 1 for char in str(text))


def format_tree(title, sections):
    """Format ordered (section, [(label, value), ...]) pairs without tabs."""
    width = max((_display_width(label) for _, rows in sections for label, _ in rows),
                default=0)
    lines = [str(title)]
    for section_idx, (section, rows) in enumerate(sections):
        last_section = section_idx == len(sections) - 1
        lines.append(f"{'└──' if last_section else '├──'} {section}")
        prefix = "    " if last_section else "│   "
        for row_idx, (label, value) in enumerate(rows):
            branch = "└──" if row_idx == len(rows) - 1 else "├──"
            padding = " " * (width - _display_width(label))
            value_lines = str(value).splitlines() or [""]
            lines.append(f"{prefix}{branch} {label}{padding} : {value_lines[0]}")
            continuation = "    " if row_idx == len(rows) - 1 else "│   "
            for value_line in value_lines[1:]:
                lines.append(f"{prefix}{continuation}{' ' * (width + 3)}{value_line}")
    return "\n".join(lines)
