"""신스(전자음) 트랙 전용 처리.

신스 소리는 피아노와 달리 음 인식 모델을 헷갈리게 하는 특징이 있습니다.

1. 리트리거/트레몰로: 패드·사이드체인 펌핑·LFO 때문에 길게 끄는 한 음이 여러 조각으로 끊겨 잡힘
   -> 같은 음이 아주 짧은 틈(기본 50ms)으로 이어지면 하나로 합침 (merge_retriggers)
2. 풍부한 배음: 톱니파·사각파는 배음이 매우 강해서 옥타브·5도 위 유령 음이 생김
   -> 더 센 음과 동시에 울리는 약한 배음 위치의 음 제거 (remove_harmonics)
3. 디튠/슈퍼소우: 살짝 어긋난 여러 발진기 때문에 반음 옆 유령 음이 생김
   -> 더 센 음과 겹치는 반음 위아래의 약한 음 제거 (remove_detune_ghosts)
4. 딜레이/에코: 같은 음이 점점 작아지며 반복되는 메아리가 새 음으로 잡힘
   -> 직전 같은 음보다 훨씬 약하게 곧바로 반복되는 음 제거 (remove_echoes)

악보 쪽에서는 --synth-mode 로 표기 방식을 고를 수 있습니다 (music2score.py 참고).
  full   : 정리된 음을 큰보표에 모두 표기
  lead   : 맨 윗선율(리드)만 한 줄로
  chords : 패드처럼 화음 위주일 때, 화음 이름을 인식해 읽기 쉬운 블록 화음으로 단순화
"""
from __future__ import annotations

from dataclasses import replace

HARMONIC_INTERVALS = {12, 19, 24, 28, 31, 34, 36}  # 2~8배음까지의 반음 간격


def merge_retriggers(notes, max_gap: float = 0.05):
    """같은 음높이가 max_gap초 이하 틈으로 다시 시작되면 하나의 긴 음으로 합침."""
    by_pitch: dict[int, list] = {}
    for n in sorted(notes, key=lambda n: n.start):
        lst = by_pitch.setdefault(n.pitch, [])
        if lst and n.start - lst[-1].end <= max_gap and n.velocity <= lst[-1].velocity * 1.15:
            prev = lst[-1]
            lst[-1] = replace(prev, end=max(prev.end, n.end))
        else:
            lst.append(n)
    out = [n for lst in by_pitch.values() for n in lst]
    return sorted(out, key=lambda n: (n.start, n.pitch))


def _overlaps(a, b, slack: float = 0.08) -> bool:
    return b.start - slack <= a.start < b.end


def remove_harmonics(notes, ratio: float = 1.3, max_velocity: int = 90):
    """더 센 음(ratio배 이상)의 배음 위치에서 동시에 시작한 음을 제거."""
    keep = []
    for n in notes:
        if n.velocity < max_velocity and any(
                m is not n and m.velocity >= n.velocity * ratio
                and (n.pitch - m.pitch) in HARMONIC_INTERVALS and _overlaps(n, m)
                for m in notes):
            continue
        keep.append(n)
    return keep


def remove_detune_ghosts(notes, ratio: float = 1.4):
    """더 센 음과 겹치는 반음 위/아래의 약한 음(디튠·피치 흔들림 잔상) 제거."""
    keep = []
    for n in notes:
        if any(m is not n and abs(n.pitch - m.pitch) == 1 and m.velocity >= n.velocity * ratio
               and _overlaps(n, m, 0.05) for m in notes):
            continue
        keep.append(n)
    return keep


def remove_echoes(notes, window: float = 0.6, ratio: float = 0.6):
    """직전의 같은 음이 끝난 뒤 window초 안에 ratio배보다 약하게 다시 울리면 메아리로 보고 제거."""
    last: dict[int, object] = {}
    keep = []
    for n in sorted(notes, key=lambda n: n.start):
        prev = last.get(n.pitch)
        if prev is not None and 0 <= n.start - prev.end <= window and n.velocity < prev.velocity * ratio:
            continue  # 메아리: last는 원래 음으로 유지해서 연속 메아리도 같이 지움
        last[n.pitch] = n
        keep.append(n)
    return keep


def clean_synth(notes):
    """신스 트랙 정리 파이프라인. (정리 전 개수, 단계별 제거 수) 로그용 dict도 함께 돌려줌."""
    stats = {}
    n0 = len(notes)
    notes = merge_retriggers(notes)
    stats["끊긴 음 합침"] = n0 - len(notes)
    n1 = len(notes)
    notes = remove_harmonics(notes)
    stats["배음"] = n1 - len(notes)
    n2 = len(notes)
    notes = remove_detune_ghosts(notes)
    stats["디튠"] = n2 - len(notes)
    n3 = len(notes)
    notes = remove_echoes(notes)
    stats["메아리"] = n3 - len(notes)
    return notes, stats
