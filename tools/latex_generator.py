from __future__ import annotations

from pathlib import Path

from ACE_Agent.agent_core.schemas import SupervisorReport


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    escaped = text
    for key, value in replacements.items():
        escaped = escaped.replace(key, value)
    return escaped


class LatexReportGenerator:
    def generate(self, report: SupervisorReport) -> Path:
        # P0.5-C: CODE_EXAMPLE 类型不生成 LaTeX 报告，直接跳过
        if report.response_type == "CODE_EXAMPLE":
            raise ValueError("CODE_EXAMPLE 类型不生成 LaTeX 报告。")
        tex_path = report.output_dir / "ace_report.tex"
        dataset_plot = report.dataset_plot_path.name.replace("\\", "/")
        rows = []
        for item in report.ranking[:5]:
            rows.append(
                " & ".join(
                    [
                        _latex_escape(item.expert_label),
                        _latex_escape(item.algorithm_name),
                        f"{float(item.metrics.get('score', 0.0)):.3f}",
                        _format_optional(item.metrics.get("ami")),
                        _format_optional(item.metrics.get("silhouette")),
                    ]
                )
                + r" \\"
            )

        figures = []
        for item in report.ranking[:3]:
            plot_name = item.plot_path.name.replace("\\", "/")
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
                _latex_escape(report.executive_summary),
                r"\section*{数据集说明}",
                _latex_escape(report.dataset.description),
                r"\begin{figure}[H]",
                r"\centering",
                rf"\includegraphics[width=0.65\linewidth]{{{dataset_plot}}}",
                rf"\caption{{数据集预览：{_latex_escape(report.dataset.display_name)}}}",
                r"\end{figure}",
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
    return f"{float(value):.3f}"


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
