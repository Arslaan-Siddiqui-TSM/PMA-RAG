from __future__ import annotations

from datetime import datetime


def _fmt_timestamp(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value or "")


def build_project_context(
    *,
    active_project: dict,
    all_projects: list[dict],
    max_projects: int = 20,
) -> str:
    active_name = str(active_project.get("name") or "")
    active_description = str(active_project.get("description") or "")
    active_created_at = _fmt_timestamp(active_project.get("created_at"))
    active_updated_at = _fmt_timestamp(active_project.get("updated_at"))

    lines: list[str] = [
        "Active project details:",
        f"- name: {active_name}",
        f"- description: {active_description}",
        f"- created_at: {active_created_at}",
        f"- updated_at: {active_updated_at}",
        "",
        f"Project catalog (up to {max_projects} active projects):",
    ]

    for idx, project in enumerate(all_projects[:max_projects], start=1):
        name = str(project.get("name") or "")
        description = str(project.get("description") or "")
        created_at = _fmt_timestamp(project.get("created_at"))
        updated_at = _fmt_timestamp(project.get("updated_at"))
        lines.append(
            f"{idx}. name: {name}; description: {description}; "
            f"created_at: {created_at}; updated_at: {updated_at}"
        )

    if len(lines) == 7:
        lines.append("(none)")
    return "\n".join(lines)
