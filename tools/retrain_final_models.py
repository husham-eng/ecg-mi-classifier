"""
retrain_final_models.py
==========================
يعيد تدريب النماذج النهائية (نموذج مستقل لكل قطب) على **كل** بيانات PTB-XL
المتاحة (289 مريضاً صالحاً)، باستخدام:
  - محازاة موحّدة عبر Lead II كمرجع لمواقع R (بدل اكتشاف مستقل لكل قطب)
  - كشف R مقاوم لانعكاس القطبية (مهم بشكل خاص لـaVR)
  - بناء مرجع طبيعي مع استبعاد الشواذ (trim_fraction=0.1)
  - Random Forest منظَّم (max_depth=12, min_samples_leaf=4)

هذا النموذج النهائي "للنشر" — يُدرَّب على كل البيانات المتاحة دفعة واحدة
(بعد أن أثبتت تجربة Cross-Validation المنفصلة صحة المنهجية وفائدة إصلاح
المحازاة تحديداً). يحفظ النماذج في models/<lead>/.
"""

from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ecg_pipeline.classifier import train_lead_model, MODELS_DIR

LEAD_CATEGORIES = {
    "LeadI": ["A", "AS", "IL", "IPL"],
    "aVR": ["AS", "IL", "IPL"],
    "V2": ["AS", "IL", "IPL"],
    "V6": ["AS", "IL", "IPL"],
}


def main():
    df = pd.read_pickle("/home/claude/ecg_app/ptbxl_beats.pkl")  # مُستخرَجة بمحازاة Lead II

    for lead, cats in LEAD_CATEGORIES.items():
        print(f"=== تدريب نموذج قطب {lead} (بيانات مُحاذاة، كل المرضى) ===")
        sub = df[df["lead"] == lead]

        normal_beats = np.stack(sub[sub["label"] == "Normal"]["beat"].values)
        pathological_beats = {c: np.stack(sub[sub["label"] == c]["beat"].values) for c in cats}

        model = train_lead_model(
            lead, normal_beats, pathological_beats,
            k_std=2.5, n_estimators=300,
            max_depth=12, min_samples_leaf=4, max_features="sqrt",
        )
        model.save(MODELS_DIR / lead)

        n_total = sum(len(b) for b in pathological_beats.values())
        n_patients = sub["patient_id"].nunique()
        print(f"  مرضى: {n_patients} | نبضات مرضية: {n_total} | نبضات طبيعية: {len(normal_beats)}")
        print(f"  الفئات: {model.classes}")
        depths = [e.get_depth() for e in model.clf.estimators_]
        print(f"  متوسط عمق الشجرة: {np.mean(depths):.1f} (كانت 69-122 قبل التنظيم)")
        print()

    print("✅ انتهى تدريب كل النماذج النهائية المُحدَّثة. الملفات محفوظة في:", MODELS_DIR)


if __name__ == "__main__":
    main()
