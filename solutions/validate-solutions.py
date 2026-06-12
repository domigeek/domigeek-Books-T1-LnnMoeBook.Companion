from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


MOJIBAKE_PATTERNS = (
    chr(0x00C3),
    chr(0x00C2),
    chr(0x00E2) + chr(0x20AC),
    chr(0xFFFD),
)

TARGETED_CORRUPTION_PATTERNS = tuple(
    item.replace("{q}", chr(0x003F))
    for item in (
        "R{q}seaux",
        "g{q}n{q}ration",
        "g{q}n{q}r{q}e",
        "{q} relire",
        "d{q}p{q}t",
        "m{q}me",
        "annonc{q}",
        "peut {q}tre",
        "p{q}dagogique",
        "na{q}f",
        "d{q}taill{q}es",
        "associ{q}",
        "diff{q}rente",
        "v{q}rification",
        "r{q}g{q}n{q}r",
        "R{q}sum",
        "Port{q}e",
        "pr{q}sente",
        "d{q}pend",
        "contr{q}le",
    )
)


@dataclass
class ValidationIssue:
    code: str
    message: str
    path: str | None = None


@dataclass
class ValidationReport:
    ok: bool = True
    issues: list[ValidationIssue] = field(default_factory=list)
    metrics: dict[str, int | str] = field(default_factory=dict)

    def fail(self, code: str, message: str, path: Path | str | None = None) -> None:
        self.ok = False
        self.issues.append(
            ValidationIssue(
                code=code,
                message=message,
                path=str(path) if path is not None else None,
            )
        )


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def first_line(path: Path) -> str:
    with path.open("r", encoding="utf-8") as stream:
        return stream.readline().rstrip("\n\r")


def scan_text_file(path: Path, report: ValidationReport, root: Path) -> None:
    text = path.read_text(encoding="utf-8")

    for pattern in MOJIBAKE_PATTERNS:
        if pattern in text:
            report.fail(
                "mojibake",
                f"Detected mojibake marker {pattern!r}.",
                path.relative_to(root),
            )

    for pattern in TARGETED_CORRUPTION_PATTERNS:
        if pattern in text:
            report.fail(
                "targeted-corruption",
                f"Detected targeted corruption pattern {pattern!r}.",
                path.relative_to(root),
            )


def validate_markdown_links(
    markdown_path: Path,
    base_dir: Path,
    report: ValidationReport,
    root: Path,
    metric_prefix: str,
) -> None:
    text = markdown_path.read_text(encoding="utf-8")
    links = re.findall(r"\]\(([^)]+)\)", text)
    missing = []
    for target in links:
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        target_path = (base_dir / target).resolve()
        try:
            target_path.relative_to(root.resolve())
        except ValueError:
            report.fail(
                "external-relative-link",
                f"Markdown link leaves repository root: {target}",
                markdown_path.relative_to(root),
            )
            continue

        if not target_path.exists():
            missing.append(target)

    for target in missing:
        report.fail(
            "broken-markdown-link",
            f"Markdown link target does not exist: {target}",
            markdown_path.relative_to(root),
        )

    report.metrics[f"{metric_prefix}_links"] = len(links)
    report.metrics[f"{metric_prefix}_broken_links"] = len(missing)


def validate_catalogue_links(solutions_dir: Path, report: ValidationReport, root: Path) -> None:
    catalogue = solutions_dir / "catalogue-solutions.md"
    if not catalogue.exists():
        report.fail("missing-catalogue", "Catalogue file is missing.", catalogue.relative_to(root))
        return

    validate_markdown_links(
        catalogue,
        solutions_dir,
        report,
        root,
        metric_prefix="catalogue",
    )


def validate_chapter_readmes(root: Path, data: dict[str, Any], report: ValidationReport) -> None:
    missing = 0
    broken_links = 0
    link_count = 0

    for chapter_key in sorted(data.get("chapters", {}), key=lambda key: int(key[2:])):
        readme = root / "solutions" / chapter_key / "README.md"
        if not readme.exists():
            missing += 1
            report.fail("missing-chapter-readme", "Chapter README is missing.", readme.relative_to(root))
            continue

        before_issue_count = len(report.issues)
        validate_markdown_links(
            readme,
            readme.parent,
            report,
            root,
            metric_prefix=f"{chapter_key}_readme",
        )
        link_count += int(report.metrics.get(f"{chapter_key}_readme_links", 0))
        broken_links += len(report.issues) - before_issue_count
        report.metrics.pop(f"{chapter_key}_readme_links", None)
        report.metrics.pop(f"{chapter_key}_readme_broken_links", None)

    report.metrics["chapter_readmes"] = len(data.get("chapters", {})) - missing
    report.metrics["chapter_readmes_missing"] = missing
    report.metrics["chapter_readme_links"] = link_count
    report.metrics["chapter_readme_broken_links"] = broken_links


def validate_solution_files(root: Path, data: dict[str, Any], report: ValidationReport) -> None:
    exercises = data.get("exercises", [])
    report.metrics["index_exercises"] = len(exercises)

    missing_declared = 0
    missing_long = 0
    missing_support = 0
    bad_headings = 0
    short_markdown = 0
    supporting_code = 0

    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()

    for exercise in exercises:
        exercise_id = exercise.get("id", "")
        if exercise_id in seen_ids:
            duplicate_ids.add(exercise_id)
        seen_ids.add(exercise_id)

        declared = exercise.get("declared_solution_file")
        if declared and not (root / declared).exists():
            missing_declared += 1
            report.fail("missing-declared-file", "Declared solution file is missing.", declared)

        long_solution = exercise.get("long_solution_file")
        if not long_solution:
            report.fail("missing-long-solution-field", "Exercise has no long_solution_file.", exercise_id)
            continue

        long_path = root / long_solution
        if not long_path.exists():
            missing_long += 1
            report.fail("missing-long-solution", "Long Markdown solution is missing.", long_solution)
        else:
            if first_line(long_path) != f"# {exercise_id} - {exercise.get('title', '')}":
                actual = first_line(long_path)
                if not actual.startswith(f"# {exercise_id} - "):
                    bad_headings += 1
                    report.fail("bad-heading", f"Unexpected first heading: {actual}", long_solution)

            if long_path.stat().st_size < 500:
                short_markdown += 1
                report.fail("short-markdown", "Long Markdown solution looks too short.", long_solution)

        support = exercise.get("supporting_code_file")
        if support:
            supporting_code += 1
            support_path = root / support
            if not support_path.exists():
                missing_support += 1
                report.fail("missing-supporting-code", "Supporting code file is missing.", support)
            elif support_path.stat().st_size < 300:
                report.fail("short-supporting-code", "Supporting code file looks too short.", support)

    for duplicate_id in sorted(duplicate_ids):
        report.fail("duplicate-id", f"Duplicate exercise id in index: {duplicate_id}")

    counts = data.get("counts", {})
    if counts.get("appendix_exercises") != len(exercises):
        report.fail(
            "count-mismatch",
            f"appendix_exercises={counts.get('appendix_exercises')} but exercises={len(exercises)}.",
            "solutions/index-solutions.json",
        )

    if counts.get("generated_long_solutions") != len(exercises) - missing_long:
        report.fail(
            "long-solution-count-mismatch",
            "generated_long_solutions does not match actual present Markdown solutions.",
            "solutions/index-solutions.json",
        )

    if counts.get("supporting_code_generated") != supporting_code - missing_support:
        report.fail(
            "supporting-code-count-mismatch",
            "supporting_code_generated does not match actual present code files.",
            "solutions/index-solutions.json",
        )

    report.metrics["missing_declared_files"] = missing_declared
    report.metrics["missing_long_solutions"] = missing_long
    report.metrics["missing_supporting_code"] = missing_support
    report.metrics["bad_headings"] = bad_headings
    report.metrics["short_markdown"] = short_markdown
    report.metrics["supporting_code_files"] = supporting_code


def validate_encoding(root: Path, solutions_dir: Path, report: ValidationReport) -> None:
    scanned = 0
    for path in solutions_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".md", ".json", ".cs", ".py"}:
            scanned += 1
            scan_text_file(path, report, root)

    report.metrics["encoding_scanned_files"] = scanned


def build_report(root: Path) -> ValidationReport:
    solutions_dir = root / "solutions"
    index_path = solutions_dir / "index-solutions.json"
    report = ValidationReport()

    if not index_path.exists():
        report.fail("missing-index", "solutions/index-solutions.json is missing.", index_path)
        return report

    data = read_json(index_path)
    validate_solution_files(root, data, report)
    validate_catalogue_links(solutions_dir, report, root)
    validate_chapter_readmes(root, data, report)
    validate_encoding(root, solutions_dir, report)

    report.metrics["quality_status"] = str(data.get("quality_report", {}).get("status"))
    return report


def print_text_report(report: ValidationReport) -> None:
    status = "ok" if report.ok else "failed"
    print(f"solutions validation: {status}")

    for key in sorted(report.metrics):
        print(f"{key}: {report.metrics[key]}")

    if report.issues:
        print("")
        print("issues:")
        for issue in report.issues:
            location = f" [{issue.path}]" if issue.path else ""
            print(f"- {issue.code}{location}: {issue.message}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Tome I solutions folder.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of text.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    report = build_report(root)

    if args.json:
        print(
            json.dumps(
                {
                    "ok": report.ok,
                    "metrics": report.metrics,
                    "issues": [issue.__dict__ for issue in report.issues],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print_text_report(report)

    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
