"""
load_ptbxl_data.py
=====================
يحمّل بيانات PTB-XL الخام (ملفات WFDB .dat/.hea) من ptbxl_selection/<فئة>/،
يستخرج الإشارة الخام لكل قطب مدعوم (LeadI, aVR, V2, V6)، يمرّرها بخط
المعالجة الحالي (denoise -> remove_baseline -> detect R -> extract beats)،
ويُرجع كل النبضات مع (patient_id, label, lead) لاستخدامها بالتدريب والتقييم.
"""

from __future__ import annotations
import re
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ecg_pipeline.preprocessing import process_raw_signal, denoise, remove_baseline, detect_r_peaks

REFERENCE_LEAD = "II"  # القطب المرجعي لاكتشاف مواقع R (الأوضح غالباً سريرياً)

# مطابقة اسم مجلد الفئة بـPTB-XL مع رمز الفئة المستخدم بالمشروع الحالي
CATEGORY_TO_LABEL = {
    "NORM": "Normal",
    "AMI": "A",
    "ASMI": "AS",
    "ILMI": "IL",
    "IPLMI": "IPL",
    # "IMI" (احتشاء سفلي بدون تحديد وحشي) غير مستخدَم بمخطط الفئات الحالي - يُتجاهَل
}

# مطابقة اسم القطب بالمشروع مع اسم القناة الفعلي بملفات WFDB
LEAD_NAME_MAP = {
    "LeadI": "I",
    "aVR": "AVR",
    "V2": "V2",
    "V6": "V6",
}

SUPPORTED_LEADS = list(LEAD_NAME_MAP.keys())


def load_metadata(database_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(database_csv)
    return df.set_index("ecg_id")


def extract_ecg_id(hea_path: Path) -> int:
    """يستخرج ecg_id من اسم الملف (بعد إعادة تسمية .dat/.hea لمطابقة محتوى الheader)."""
    m = re.match(r"0*(\d+)_hr", hea_path.stem)
    return int(m.group(1))


def load_all_beats(selection_dir: Path, database_csv: Path,
                    pre: int = 100, post: int = 300, verbose: bool = True) -> pd.DataFrame:
    """
    يرجع DataFrame بعمود لكل: patient_id, ecg_id, label, lead, beat (np.ndarray)
    لكل نبضة مكتشفة، لكل قطب مدعوم، لكل سجل ضمن الفئات المستخدمة حالياً.
    """
    meta = load_metadata(database_csv)
    rows = []
    skipped_categories = set()

    for cat_dir in sorted(selection_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        label = CATEGORY_TO_LABEL.get(cat_dir.name)
        if label is None:
            skipped_categories.add(cat_dir.name)
            continue

        hea_files = sorted(cat_dir.glob("*.hea"))
        for hea in hea_files:
            ecg_id = extract_ecg_id(hea)
            if ecg_id not in meta.index:
                continue
            patient_id = meta.loc[ecg_id, "patient_id"]

            try:
                record = wfdb.rdrecord(str(hea.with_suffix("")))
            except Exception as e:
                print(f"  ⚠️ تعذّرت قراءة {hea.name} (ملف تالف على الأرجح) — تم تجاوزه: {e}")
                continue
            fs = record.fs
            sig_names = record.sig_name

            # اكتشاف مواقع R مرة واحدة من القطب المرجعي (Lead II)، بمقاومة
            # لانعكاس القطبية — تُستخدم نفس المواقع لاستخراج كل الأقطاب
            # الأربعة المدعومة، فتُضمَن محازاة فعلية بين الأقطاب لنفس النبضة.
            if REFERENCE_LEAD not in sig_names:
                continue
            ref_raw = record.p_signal[:, sig_names.index(REFERENCE_LEAD)]
            ref_clean = remove_baseline(denoise(ref_raw, fs), degree=6)
            ref_r_locs = detect_r_peaks(ref_clean, fs, polarity_robust=True)

            for lead, wfdb_name in LEAD_NAME_MAP.items():
                if wfdb_name not in sig_names:
                    continue
                raw = record.p_signal[:, sig_names.index(wfdb_name)]
                beats, r_locs = process_raw_signal(raw, fs, pre=pre, post=post,
                                                    external_r_locs=ref_r_locs)
                for b in beats:
                    rows.append({
                        "patient_id": patient_id, "ecg_id": ecg_id,
                        "label": label, "lead": lead, "beat": b,
                    })

        if verbose:
            print(f"  فئة {cat_dir.name} ({label}): {len(hea_files)} سجل")

    if skipped_categories and verbose:
        print(f"  فئات تم تجاهلها (غير مستخدمة بمخطط الفئات الحالي): {sorted(skipped_categories)}")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    selection_dir = Path("/home/claude/ptbxl_data/ptbxl_selection")
    database_csv = Path("/mnt/user-data/uploads/ptbxl_database.csv")
    df = load_all_beats(selection_dir, database_csv)
    print(f"\nإجمالي النبضات المستخرجة: {len(df)}")
    print(df.groupby(["lead", "label"])["patient_id"].nunique())
    df.to_pickle("/home/claude/ecg_app/ptbxl_beats.pkl")
    print("\n✅ حُفظت كل النبضات بـ ptbxl_beats.pkl")
