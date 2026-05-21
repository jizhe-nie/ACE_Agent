from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from agent_core.schemas import SupervisorReport

_BS_PLACEHOLDER = "\x00BS\x00"


def _latex_escape(text: str) -> str:
    """Escape special LaTeX characters. Uses a placeholder for backslash
    to prevent double-escaping of LaTeX commands introduced by other
    replacements (e.g. ``\\{`` → ``\\textbackslash{}\\{``)."""
    text = str(text)
    text = text.replace("\\", _BS_PLACEHOLDER)
    text = text.replace("{", "\\{")
    text = text.replace("}", "\\}")
    text = text.replace("&", "\\&")
    text = text.replace("%", "\\%")
    text = text.replace("$", "\\$")
    text = text.replace("#", "\\#")
    text = text.replace("_", "\\_")
    text = text.replace("~", "\\textasciitilde{}")
    text = text.replace("^", "\\textasciicircum{}")
    text = text.replace(_BS_PLACEHOLDER, "\\textbackslash{}")
    return text


def _strip_emoji(text: str) -> str:
    """Remove emoji and other non-LaTeX-renderable Unicode symbols."""
    result = []
    for ch in str(text):
        cp = ord(ch)
        # Keep ASCII printables, CJK ranges, and common punctuation
        if cp < 0x7F:
            result.append(ch)
        elif (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or
              0x2000 <= cp <= 0x206F or 0x3000 <= cp <= 0x303F or
              0xFF00 <= cp <= 0xFFEF or 0x0080 <= cp <= 0x00FF):
            result.append(ch)
        elif unicodedata.category(ch)[0] in ("L", "N", "P", "Z"):
            # Letter, Number, Punctuation, Separator
            result.append(ch)
        else:
            pass  # drop emoji, symbols like ≤ ≥ etc.
    return "".join(result)


def _md_summary_to_latex(text: str) -> str:
    """Convert LLM Markdown summary to LaTeX-friendly plain text."""
    text = _strip_emoji(str(text))

    # Phase 1: Convert Markdown → LaTeX (before escaping, so regexes match)
    text = re.sub(r'^####\s+(.+)$', r'\\paragraph{\1}', text, flags=re.MULTILINE)
    text = re.sub(r'^###\s+(.+)$', r'\\subsubsection*{\1}', text, flags=re.MULTILINE)
    text = re.sub(r'^##\s+(.+)$', r'\\subsection*{\1}', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', text)
    text = re.sub(r'^---\s*$', r'\\medskip\\hrule\\medskip', text, flags=re.MULTILINE)
    text = re.sub(r'^>\s+', r'\\quad\\textit{', text, flags=re.MULTILINE)
    text = re.sub(r'`([^`]+)`', r'\\texttt{\1}', text)
    text = text.replace("---", "\\hrulefill")

    # Phase 2: Escape remaining LaTeX-special chars.
    # We use placeholders to protect the commands inserted in Phase 1.
    _protected: list[str] = []
    def _protect(m: re.Match) -> str:
        _protected.append(m.group(0))
        return f"\x01CMD{len(_protected) - 1}\x01"
    text = re.sub(r'\\[a-zA-Z]+\*?\{[^}]*\}', _protect, text)
    text = re.sub(r'\\[a-zA-Z]+', _protect, text)

    # Now escape special chars in unprotected text
    for _ch in ("_", "&", "%", "$", "#", "{", "}", "~", "^"):
        text = text.replace(_ch, "\\" + _ch)

    # Restore protected commands
    for _i, _cmd in enumerate(_protected):
        text = text.replace(f"\x01CMD{_i}\x01", _cmd)

    return text


class LatexReportGenerator:
    def generate(self, report: SupervisorReport) -> Path:
        # P0.5-C: CODE_EXAMPLE 类型不生成 LaTeX 报告，直接跳过
        if report.response_type == "CODE_EXAMPLE":
            raise ValueError("CODE_EXAMPLE 类型不生成 LaTeX 报告。")
        tex_path = report.output_dir / "ace_report.tex"
        _ds_plot_path = getattr(report, "dataset_plot_path", None)
        dataset_plot = _ds_plot_path.name.replace("\\", "/") if _ds_plot_path and _ds_plot_path.name else ""
        rows = []
        for item in report.ranking[:5]:
            rows.append(
                " & ".join(
                    [
                        _latex_escape(item.expert_label),
                        _latex_escape(item.algorithm_name),
                        f"{float(item.metrics.get('score') or 0.0):.3f}",
                        _format_optional(item.metrics.get("ami")),
                        _format_optional(item.metrics.get("silhouette")),
                    ]
                )
                + r" \\"
            )

        figures = []
        _out_dir = Path(report.output_dir) if report.output_dir else Path(".")
        for item in report.ranking[:3]:
            _p = getattr(item, "plot_path", None)
            if not _p:
                continue
            _p_str = str(_p)
            # Make path relative to the output_dir so \includegraphics can find it
            try:
                _rel = Path(_p_str).relative_to(_out_dir)
                plot_name = _rel.as_posix()
            except ValueError:
                # Not under output_dir — use as-is or just the filename
                plot_name = Path(_p_str).name.replace("\\", "/") if _p_str else ""
            if not plot_name:
                continue
            figures.append(
                "\n".join(
                    [
                        r"\begin{figure}[H]",
                        r"\centering",
                        rf"\includegraphics[width=0.7\linewidth]{{{plot_name}}}",
                        rf"\caption{{{_latex_escape(item.expert_label)} - {_latex_escape(item.algorithm_name)}}}",
                        r"\end{figure}",
                    ]
                )
            )

        # ---- audit section (if available) ----
        audit_lines: list[str] = []
        audit = report.audit_report
        if audit and isinstance(audit, dict):
            audit_lines = _build_audit_section(audit)

        tex = "\n".join(
            [
                r"\documentclass[11pt]{article}",
                r"\usepackage[a4paper,margin=1in]{geometry}",
                r"\usepackage{xeCJK}",
                r"\usepackage{graphicx}",
                r"\usepackage{float}",
                r"\usepackage{booktabs}",
                r"\title{ACE 智能体聚类分析报告}",
                r"\author{自动生成演示}",
                r"\date{\today}",
                r"\begin{document}",
                r"\maketitle",
                r"\section*{概览}",
                _md_summary_to_latex(report.executive_summary),
                r"\section*{数据集说明}",
                _latex_escape(str(report.dataset.description)[:2000]),
                *(
                    [
                        r"\begin{figure}[H]",
                        r"\centering",
                        rf"\includegraphics[width=0.65\linewidth]{{{dataset_plot}}}",
                        rf"\caption{{数据集预览：{_latex_escape(report.dataset.display_name)}}}",
                        r"\end{figure}",
                    ]
                    if dataset_plot else []
                ),
                r"\section*{路由决策轨迹}",
                r"\begin{itemize}",
                *[rf"\item {_latex_escape(item)}" for item in report.routing.trace],
                r"\end{itemize}",
                r"\section*{最佳运行结果}",
                r"\begin{tabular}{lllll}",
                r"\toprule",
                r"专家 & 算法 & 综合得分 & AMI & 轮廓系数 \\",
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}",
                r"\section*{可视化图表}",
                *figures,
                *audit_lines,
                r"\end{document}",
            ]
        )
        tex_path.write_text(tex, encoding="utf-8")
        return tex_path


def _format_optional(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def _build_audit_section(audit: dict) -> list[str]:  # type: ignore[type-arg]
    """Build LaTeX lines for the independent Critic audit section."""
    endorsement = str(audit.get("endorsement", "N/A"))
    confidence = float(audit.get("confidence_level", 0))
    stability = float(audit.get("stability_score", 0))
    hopkins = float(audit.get("hopkins", 0))
    overfitting = str(audit.get("overfitting_risk", "unknown"))
    k_consistency = bool(audit.get("winner_k_consistency", False))
    findings = audit.get("findings", [])
    recommendation = str(audit.get("recommendation", ""))

    endorsement_label = {
        "endorsed": "通过",
        "qualified": "有条件通过",
        "qualified_with_warning": "需要关注",
    }.get(endorsement, "未知")

    lines: list[str] = [
        r"\section*{独立审计结论 (Critic Audit)}",
        r"\begin{tabular}{@{}ll@{}}",
        r"\toprule",
        rf"审计裁决 & {_latex_escape(endorsement_label)} ({_latex_escape(endorsement)}) \\",
        r"\midrule",
        rf"综合置信度 & {confidence:.0%} \\",
        rf"Bootstrap 稳定性 & {stability:.2f} \\",
        rf"Hopkins 聚类趋势 & {hopkins:.2f} \\",
        rf"过拟合风险 & {_latex_escape(overfitting)} \\",
        rf"聚类数一致性 & {'一致' if k_consistency else '与 CVI 共识不一致'} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]

    if findings:
        lines.append(r"\subsection*{审计发现}")
        lines.append(r"\begin{itemize}")
        for f_text in findings:
            lines.append(rf"\item {_latex_escape(str(f_text))}")
        lines.append(r"\end{itemize}")

    if recommendation:
        lines.append(r"\subsection*{建议}")
        lines.append(_latex_escape(recommendation))

    return lines
