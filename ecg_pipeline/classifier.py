"""
ecg_pipeline.classifier
==========================
تدريب وحفظ وتحميل نماذج التصنيف (نموذج مستقل لكل قطب مدعوم)، بالإضافة
لدالة التنبؤ الموحّدة على مستوى مريض واحد (نبضة واحدة أو أكثر).
"""

from __future__ import annotations
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .reference import build_normal_envelope, stem_beat, is_likely_normal
from .weighting import weight_with_class_specificity, weight_single_beat, build_weight_lookup

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class LeadModel:
    """يجمع كل ما يلزم للتصنيف على قطب واحد: المرجع الطبيعي + النموذج + جداول الترجيح."""

    def __init__(self, lead: str, classes: list[str], window_len: int,
                 ref_min: np.ndarray, ref_max: np.ndarray,
                 idf_table: dict, specificity_table: dict,
                 clf: RandomForestClassifier):
        self.lead = lead
        self.classes = classes
        self.window_len = window_len
        self.ref_min = ref_min
        self.ref_max = ref_max
        self.idf_table = idf_table
        self.specificity_table = specificity_table
        self.clf = clf
        # جداول بحث مبنية مرة واحدة (استيفاء خطي) بدل إعادة بنائها بكل نبضة
        self._idf_lookup = build_weight_lookup(idf_table)
        self._specificity_lookup = build_weight_lookup(specificity_table)

    def predict_beat(self, beat: np.ndarray, normal_threshold: int = 50) -> dict:
        """
        يعيد قاموس الاحتمالات لكل الفئات (بما فيها Normal)، بعد تطبيق
        قاعدة "العينات المتبقية القليلة => طبيعي" أولاً كفلتر أولي سريع.
        """
        beat = np.asarray(beat, dtype=float)
        if len(beat) != self.window_len:
            beat = np.interp(np.linspace(0, 1, self.window_len),
                              np.linspace(0, 1, len(beat)), beat)

        stemmed = stem_beat(beat, self.ref_min, self.ref_max)

        if is_likely_normal(stemmed, threshold=normal_threshold):
            probs = {c: 0.0 for c in self.classes}
            probs["Normal"] = 1.0
            return probs

        weighted = weight_single_beat(stemmed, self._idf_lookup, self._specificity_lookup)
        proba = self.clf.predict_proba([weighted])[0]
        probs = dict(zip(self.clf.classes_, proba))
        probs.setdefault("Normal", 0.0)
        return probs

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", "wb") as f:
            pickle.dump(self.clf, f)
        np.save(path / "ref_min.npy", self.ref_min)
        np.save(path / "ref_max.npy", self.ref_max)
        meta = {
            "lead": self.lead, "classes": self.classes, "window_len": self.window_len,
            "idf_table": {str(k): v for k, v in self.idf_table.items()},
            "specificity_table": {str(k): v for k, v in self.specificity_table.items()},
        }
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path: Path) -> "LeadModel":
        with open(path / "model.pkl", "rb") as f:
            clf = pickle.load(f)
        ref_min = np.load(path / "ref_min.npy")
        ref_max = np.load(path / "ref_max.npy")
        with open(path / "meta.json") as f:
            meta = json.load(f)
        idf_table = {float(k): v for k, v in meta["idf_table"].items()}
        specificity_table = {float(k): v for k, v in meta["specificity_table"].items()}
        return cls(meta["lead"], meta["classes"], meta["window_len"],
                    ref_min, ref_max, idf_table, specificity_table, clf)


def _compute_idf_specificity(stemmed_beats: np.ndarray, labels: np.ndarray, decimals: int = 1):
    """يستخرج جداول IDF/الاختصاص من بيانات التدريب لحفظها واستخدامها لاحقاً وقت الاستدلال."""
    from collections import defaultdict
    n_beats = stemmed_beats.shape[0]
    bins = np.round(stemmed_beats, decimals)
    classes = sorted(set(labels))
    class_beat_counts = {c: (labels == c).sum() for c in classes}
    value_freq_per_class = defaultdict(lambda: defaultdict(int))
    doc_freq_total = defaultdict(int)
    for i, row in enumerate(bins):
        cls = labels[i]
        for v in set(v for v in row if v != 0):
            value_freq_per_class[v][cls] += 1
            doc_freq_total[v] += 1
    specificity, idf = {}, {}
    for v, per_class in value_freq_per_class.items():
        freqs = np.array([per_class.get(c, 0) / class_beat_counts[c] for c in classes])
        specificity[v] = float(freqs.max() / (freqs.sum() + 1e-9))
        idf[v] = float(np.log(n_beats / doc_freq_total[v]))
    return idf, specificity


def train_lead_model(lead: str, normal_beats: np.ndarray,
                      pathological_beats: dict[str, np.ndarray],
                      k_std: float = 2.5, n_estimators: int = 300,
                      max_depth: int | None = 12, min_samples_leaf: int = 4,
                      max_features: str | float = "sqrt") -> LeadModel:
    """
    يدرّب نموذج قطب واحد كامل من الصفر: بناء مرجع طبيعي، Stemming،
    حساب أوزان IDF/الاختصاص، ثم تدريب Random Forest.

    Parameters
    ----------
    normal_beats : مصفوفة نبضات طبيعية (لبناء المرجع فقط، ليست جزءاً من فئات التصنيف)
    pathological_beats : قاموس {اسم الفئة المرضية: مصفوفة نبضاتها}
    max_depth, min_samples_leaf, max_features : معاملات تنظيم (Regularization)
        لأشجار Random Forest. **مهم**: النماذج المنشورة سابقاً كانت بلا أي تقييد
        (max_depth=None الافتراضي)، وفحصها الفعلي بعد التدريب أظهر متوسط عمق
        شجرة بين 69-122 مستوى وأوراقاً بمتوسط أقل من نبضتين لكل ورقة — دليل
        قوي على Overfitting حاد للنبضات الفردية بدل تعلّم نمط عام. القيم
        الافتراضية الجديدة هنا (max_depth=12, min_samples_leaf=4) محافظة
        ومقترحة كنقطة بداية معقولة، لكن **يجب التحقق من أثرها الفعلي على
        Macro-F1 عبر نفس منهجية التقييم الصارمة على مستوى المريض** (Leave-
        One-Patient-Out) المستخدمة سابقاً بالمشروع، قبل اعتمادها للنشر —
        قد تحتاج تعديلاً إضافياً حسب حجم بيانات كل قطب على حدة.
    """
    window_len = normal_beats.shape[1]
    ref_min, ref_max = build_normal_envelope(normal_beats, k=k_std)

    X, y = [], []
    for cls, beats in pathological_beats.items():
        for b in beats:
            X.append(stem_beat(b, ref_min, ref_max))
            y.append(cls)
    X = np.array(X)
    y = np.array(y)

    idf_table, specificity_table = _compute_idf_specificity(X, y)
    X_weighted = weight_with_class_specificity(X, y)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X_weighted, y)

    classes = sorted(set(y)) + ["Normal"]
    return LeadModel(lead, classes, window_len, ref_min, ref_max, idf_table, specificity_table, clf)
