# subtype/ — 癌症亚型发现新核心（W1 起）

战略转向后的聚焦新核心（见 `docs/RESEARCH_DIRECTION_2026-06-22.md`）。
核心命题：**IP-1 稳定性认证子型 + IP-3 可解释多原型/medoid 子型**，数据用 **TCGA-BRCA**。
铁律（lessons §4 底座优先）：**数据体检 → 确定性内核 → 诚实评估 → 才裹 agent**，不可颠倒。

## W1：数据体检（当前阶段）

`01_data_health_check.py` 回答三问：① 数据能不能聚（Hopkins，高维下会低估，故有标签时以 ARI 为主判据）；
② 簇是否对应已知生物学亚型（ARI/NMI vs PAM50，越高越好）；③ 簇是不是其实在按批次分（ARI/NMI vs 批次，越低越好）。

```bash
# 现在就能跑的合成演示（看懂机制，无需生物数据）
python subtype/01_data_health_check.py --demo          # 正常：ARI_亚型→1, ARI_批次→0
python subtype/01_data_health_check.py --demo-batchy   # 坏例：ARI_批次>>ARI_亚型 → 红灯

# 真实数据
python subtype/01_data_health_check.py \
    --expr data/brca/expression.tsv --pheno data/brca/phenotype.tsv \
    --subtype-col PAM50Call_RNAseq --batch-col TSS
```

输出：stdout 三项检查 + 结论；`subtype/outputs/health_check_pca.png`（PCA 散点，分别按亚型/批次着色——一眼看出点是按生物学还是按批次成团）。

## 如何获取 TCGA-BRCA（推荐 UCSC Xena，对 CS 新手最友好）

> 数据较大、需联网，请你在本机下载（会话里用 `! <命令>` 也行）。下载到 `data/brca/`（已 gitignore，不会入库）。

1. 打开 **UCSC Xena**：https://xenabrowser.net/datapages/ → 选 cohort **「TCGA Breast Cancer (BRCA)」**。
2. 下两个矩阵：
   - **基因表达**：`IlluminaHiSeq` RNAseq（gene expression，log2(norm_count+1)）→ 存为 `expression.tsv`。
   - **表型/临床**（phenotype/clinical matrix）→ 存为 `phenotype.tsv`。其中含 **PAM50 亚型**列
     （列名常见 `PAM50Call_RNAseq` 或 `PAM50 mRNA`，用 `--subtype-col` 指定）。
3. **批次列**：TCGA 没有现成的 batch 列，但**样本条码（barcode）里编码了批次信息**：
   `TCGA-A8-A07B-01A-11R-A034-07` → 第 2 段 `A8` = **TSS（组织来源中心/医院）**，第 6 段 `A034` 含 plate。
   先用 TSS 当批次代理（`--batch-col TSS`；若表型表无此列，下一步我帮你写"从条码解析 TSS"的小函数）。

替代来源：cBioPortal（curated，含 PAM50）、LinkedOmics、GDC/`TCGAbiolinks`(R)。Xena 胜在直接下 TSV、零门槛。

## W1 验收标准（绿灯条件）
- 可聚类：Hopkins>0.6 **或** ARI(聚类,PAM50)>0.2（高维下后者更可信）。
- 有生物学意义：ARI(聚类,PAM50) 明显>0（哪怕 0.2–0.4，因为我们用的是粗探针 KMeans，最终方法会更高）。
- 批次可控：ARI(聚类,批次) 明显低于 ARI(聚类,PAM50)。若反过来 → 先做批次校正（ComBat/Harmony）再进 W3。

## 名词速查（详见对话里的「生物学第 1 课」）
- **组学(omics)**：一层分子测量。表达=基因活跃度；甲基化=表观开关；CNV=基因拷贝数增减。
- **PAM50**：50 基因的乳腺癌分子分型金标准（Luminal A/B、HER2-enriched、Basal-like、Normal-like）——我们的"弱标签/验证锚"。
- **批次效应(batch effect)**：不同中心/平台/试剂的技术指纹，会被聚类误当成生物学差异。
