# 使用 Python 3.11 作为基础镜像
FROM python:3.11-slim

# 安装系统依赖，包括 LaTeX 编译环境（为了支持报告生成）
RUN apt-get update && apt-get install -y \
    texlive-latex-extra \
    texlive-fonts-recommended \
    dvipng \
    cm-super \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 暴露 Streamlit 默认端口
EXPOSE 8501

# 设置环境变量，确保 Python 能找到 ACE_Agent 模块
ENV PYTHONPATH=/app

# 默认启动命令：运行 Streamlit 演示
CMD ["streamlit", "run", "web_demo.py"]
