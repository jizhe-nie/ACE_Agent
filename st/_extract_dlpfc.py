"""提取 DLPFC12.zip 中全部 12 样本所需文件(filtered h5 + spatial + gt)，跳过 tif/raw/__MACOSX。"""
import zipfile

SAMPLES = ["151507", "151508", "151509", "151510", "151669", "151670",
           "151671", "151672", "151673", "151674", "151675", "151676"]
z = zipfile.ZipFile("data/dlpfc/DLPFC12.zip")
want = []
for n in z.namelist():
    if "__MACOSX" in n or n.endswith(".tif") or "raw_feature" in n:
        continue
    for s in SAMPLES:
        if n.startswith(f"DLPFC12/{s}/") and (
            "filtered_feature_bc_matrix.h5" in n or "/spatial/" in n or "/gt/" in n
        ):
            want.append(n)
            break
z.extractall("data/dlpfc/extracted", members=want)
print(f"extracted {len(want)} members for {len(SAMPLES)} samples")
