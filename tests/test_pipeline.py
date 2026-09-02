"""
اختبار سلامة أساسي (Sanity Test): يتأكد من أن كل نموذج مدرَّب يُحمَّل
بدون أخطاء ويعطي تصنيفاً صالحاً على إشارة تركيبية بسيطة.

التشغيل: pytest tests/test_pipeline.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from ecg_pipeline import classify_lead_signal, classify_patient, SUPPORTED_LEADS


def synthetic_ecg(fs=500, duration_s=6, amp=1.0, seed=0):
    """إشارة تركيبية بسيطة تحوي نبضات دورية واضحة، لأغراض اختبار سلامة الكود فقط."""
    n = int(fs * duration_s)
    t = np.arange(n)
    signal = np.zeros(n)
    for pos in range(int(0.5 * fs), n - int(0.3 * fs), int(0.8 * fs)):
        width = 15
        signal[pos - width:pos + width] += amp * 1500 * np.exp(
            -0.5 * ((np.arange(-width, width)) / (width / 3)) ** 2
        )
    rng = np.random.default_rng(seed)
    return signal + rng.normal(0, 20, n)


@pytest.mark.parametrize("lead", SUPPORTED_LEADS)
def test_model_loads_and_predicts(lead):
    sig = synthetic_ecg()
    result = classify_lead_signal(sig, fs=500.0, lead=lead)
    assert "error" not in result, f"فشل التصنيف على قطب {lead}: {result.get('error')}"
    assert "probabilities" in result
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize("lead", SUPPORTED_LEADS)
def test_multi_beat_aggregation(lead):
    """يتأكد أن التجميع الجديد (كل النبضات، بدل أول نبضة فقط) يعمل ويعيد
    beat_agreement ضمن المدى الصحيح [0, 1]."""
    sig = synthetic_ecg(duration_s=10)
    result = classify_lead_signal(sig, fs=500.0, lead=lead)
    assert result["n_beats_used"] == result["n_beats_detected"]
    assert 0.0 <= result["beat_agreement"] <= 1.0


@pytest.mark.parametrize("lead", SUPPORTED_LEADS)
def test_explicit_beat_index_still_works(lead):
    """يتأكد أن السلوك القديم (تصنيف نبضة واحدة محددة) لا يزال متاحاً صراحة."""
    sig = synthetic_ecg(duration_s=10)
    result = classify_lead_signal(sig, fs=500.0, lead=lead, beat_index=0)
    assert result["n_beats_used"] == 1


def test_classify_patient_weighted_fusion():
    """يتأكد أن دمج عدة أقطاب يستخدم التصويت المرجّح الجديد، والاحتمالات
    المدمجة تبقى موزّعة احتمالياً بشكل صحيح (تجمع إلى 1)."""
    signals = {lead: (synthetic_ecg(duration_s=8, seed=i), 500.0)
               for i, lead in enumerate(SUPPORTED_LEADS)}
    result = classify_patient(signals)
    assert "error" not in result
    assert sum(result["combined_probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert set(result["lead_weights_used"].keys()) == set(SUPPORTED_LEADS)
    # V2 يجب أن يحمل أعلى وزن حسب الأداء الموثّق بالـREADME
    assert result["lead_weights_used"]["V2"] == max(result["lead_weights_used"].values())


if __name__ == "__main__":
    for lead in SUPPORTED_LEADS:
        test_model_loads_and_predicts(lead)
        test_multi_beat_aggregation(lead)
        test_explicit_beat_index_still_works(lead)
    test_classify_patient_weighted_fusion()
    print("✅ كل الاختبارات نجحت")
