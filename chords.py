"""화음(코드) 인식: 박자에 맞춘 음표들에서 구간별 코드 이름을 추정합니다.

각 구간(기본 2박)의 음높이 분포(pitch class 12개, 길이×세기로 가중)를
장/단 3화음, 7화음, sus4, dim 틀(template)과 비교해 가장 잘 맞는 코드를 고릅니다.
가장 낮은 음(베이스)이 근음이면 가산점을 줍니다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 이름 접미사, music21 ChordSymbol kind, 구성음(반음), 복잡도 패널티
TEMPLATES = [
    ("", "major", (0, 4, 7), 0.0),
    ("m", "minor", (0, 3, 7), 0.0),
    ("7", "dominant-seventh", (0, 4, 7, 10), 0.12),
    ("maj7", "major-seventh", (0, 4, 7, 11), 0.12),
    ("m7", "minor-seventh", (0, 3, 7, 10), 0.12),
    ("sus4", "suspended-fourth", (0, 5, 7), 0.10),
    ("dim", "diminished", (0, 3, 6), 0.15),
]

SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


@dataclass
class ChordLabel:
    offset: float  # 4분음표 단위
    length: float
    root: int  # pitch class 0~11
    suffix: str
    kind: str
    tones: tuple

    def name(self, use_flats: bool = False) -> str:
        names = FLAT_NAMES if use_flats else SHARP_NAMES
        return names[self.root] + self.suffix


def detect_chords(qnotes, total_length: float, segment: float = 2.0,
                  min_weight: float = 0.05) -> list[ChordLabel]:
    """qnotes: (offset, length, pitch, velocity) 목록 (4분음표 단위)."""
    labels: list[ChordLabel] = []
    if not qnotes:
        return labels
    total_energy = sum(d * v for _, d, _, v in qnotes) / max(total_length, 1e-9)
    t = 0.0
    while t < total_length - 1e-9:
        lo, hi = t, t + segment
        hist = np.zeros(12)
        lowest = None
        for s, d, p, v in qnotes:
            ov = min(s + d, hi) - max(s, lo)
            if ov <= 0:
                continue
            hist[p % 12] += ov * v
            if lowest is None or p < lowest:
                lowest = p
        weight = hist.sum()
        if weight < min_weight * total_energy * segment or lowest is None:
            t += segment
            continue  # 거의 무음 → 코드 없음
        h = hist / weight
        best = None
        for root in range(12):
            for suffix, kind, tones, penalty in TEMPLATES:
                pcs = {(root + i) % 12 for i in tones}
                inside = sum(h[pc] for pc in pcs)
                score = inside - 0.6 * (1 - inside) - penalty
                # 3화음 구성음이 모두 어느 정도 있어야 함
                if min(h[(root + i) % 12] for i in tones[:3]) < 0.03:
                    score -= 0.3
                if lowest % 12 == root:
                    score += 0.12
                if best is None or score > best[0]:
                    best = (score, root, suffix, kind, tones)
        _, root, suffix, kind, tones = best
        if labels and labels[-1].root == root and labels[-1].suffix == suffix \
                and abs(labels[-1].offset + labels[-1].length - lo) < 1e-6:
            labels[-1].length += segment
        else:
            labels.append(ChordLabel(lo, segment, root, suffix, kind, tones))
        t += segment
    return labels


def voice_chords(labels: list[ChordLabel], velocity: int = 70, low: int = 53):
    """코드 이름을 읽기 쉬운 블록 화음(가운데 도 근처, 기본위치)으로 펼침 → qnotes."""
    out = []
    for c in labels:
        root = c.root + 12 * ((low - c.root) // 12 + 1)  # low 이상 가장 가까운 근음
        if root - 12 >= low:
            root -= 12
        for i in c.tones:
            out.append((c.offset, c.length, root + i, velocity))
    return out
