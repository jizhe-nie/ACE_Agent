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
                r"\end{document}",
            ]
        )
        tex_path.write_text(tex, encoding="utf-8")
        return tex_path


def _format_optional(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"
