from big_brother.gating import GatingConfig, GatingEngine


def test_gating_skips_after_stable_windows() -> None:
    gate = GatingEngine(GatingConfig(stable_window_limit=1))
    d1 = gate.decide(motion=0.01, embedding_drift=0.01, audio_spike=0.01)
    d2 = gate.decide(motion=0.01, embedding_drift=0.01, audio_spike=0.01)
    assert d1.should_call_vlm is True
    assert d2.should_call_vlm is False
    assert d2.reason == "stable_extend_previous"

