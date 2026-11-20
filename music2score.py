#!/usr/bin/env python3
"""music2score: 오디오 파일(mp3/wav/flac/m4a...)을 MIDI와 악보(MusicXML, PDF)로 변환합니다.

파이프라인
  0. (선택) Demucs로 악기 분리: 보컬 / 드럼 / 베이스 / 기타 / 피아노 / 그 외(신스 등)
  1. Spotify basic-pitch(ONNX)로 오디오 -> 음표(onset/offset/pitch/velocity) 추출
  2. 잡음 정리 (신스 트랙은 synth.py의 전용 정리 적용)
  3. librosa로 템포/박 위치 추정 후, 음표를 박자 격자(기본 16분음표)에 맞춰 정렬(quantize)
  4. music21로 악보 구성 (악기별 보표, 조성 추정, 코드 이름 표기)
  5. MusicXML 저장, verovio로 PDF 악보 렌더링

사용 예
  python music2score.py song.mp3
  python music2score.py song.mp3 --mode melody --tempo 96
  python music2score.py song.mp3 --separate                 # 악기별 파트 악보
  python music2score.py song.mp3 --separate --synth-mode chords
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

import numpy as np  # noqa: E402

from chords import FLAT_NAMES, SHARP_NAMES, detect_chords, voice_chords  # noqa: E402
from synth import clean_synth  # noqa: E402


@dataclass
class Note:
    start: float  # seconds
    end: float  # seconds
    pitch: int  # MIDI number
    velocity: int


@dataclass
class Track:
    """악보의 한 파트(악기)."""
    name: str
    kind: str  # grand | treble | treble8vb | bass | bass8vb | melody
    program: int  # General MIDI 악기 번호 (0부터)
    notes: list = field(default_factory=list)  # 정리된 Note (초 단위)
    qnotes: list = field(default_factory=list)  # (offset, length, pitch, velocity) 4분음표 단위


@dataclass
class StemProfile:
    label: str
    kind: str
    program: int
    min_freq: float | None = None
    max_freq: float | None = None
    synth: bool = False
    mono: str | None = None  # "high"(멜로디) | "low"(베이스 라인)


# Demucs stem → 악보 파트 설정. 위에서 아래로 악보 순서.
STEM_PROFILES = {
    "vocals": StemProfile("Vocals", "melody", 53, 65, 1500, mono="high"),
    "other": StemProfile("Synth / Other", "grand", 81, synth=True),
    "guitar": StemProfile("Guitar", "treble8vb", 25, 75, None),
    "piano": StemProfile("Piano", "grand", 0),
    "bass": StemProfile("Bass", "bass8vb", 33, 30, 400, mono="low"),
}


# ---------------------------------------------------------------- 1. transcription
_MODEL = None


def _basic_pitch_model():
    global _MODEL
    if _MODEL is None:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from basic_pitch.inference import Model

        _MODEL = Model(ICASSP_2022_MODEL_PATH)
    return _MODEL


def transcribe(audio_path: Path, onset_threshold: float = 0.5, frame_threshold: float = 0.3,
               min_note_ms: float = 80.0, min_freq: float | None = None,
               max_freq: float | None = None) -> list[Note]:
    from basic_pitch.inference import predict

    _, _, note_events = predict(
        str(audio_path),
        _basic_pitch_model(),
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=min_note_ms,
        minimum_frequency=min_freq,
        maximum_frequency=max_freq,
        multiple_pitch_bends=False,
        melodia_trick=True,
    )
    notes = [Note(float(s), float(e), int(p), int(round(a * 127)))
             for s, e, p, a, *_ in note_events]
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


HARMONIC_INTERVALS = {12, 19, 24, 28, 31, 34, 36}  # 2~8배음까지의 반음 간격


def clean_notes(notes: list[Note], min_velocity: int) -> list[Note]:
    """잡음 제거: (1) 약하고 짧은 음, (2) 더 센 음과 동시에 울리는 약한 배음."""
    if min_velocity <= 0:
        return notes
    keep = []
    for n in notes:
        dur = n.end - n.start
        if n.velocity < min_velocity and dur < 0.2:
            continue
        if n.velocity < min_velocity:
            is_harmonic = any(
                m is not n and m.velocity > n.velocity * 1.4
                and (n.pitch - m.pitch) in HARMONIC_INTERVALS
                and m.start - 0.08 <= n.start < m.end
                for m in notes)
            if is_harmonic:
                continue
        keep.append(n)
    return keep


# ---------------------------------------------------------------- 2. tempo & grid
def estimate_tempo(audio_path: Path) -> tuple[float, float]:
    """Return (bpm, first_beat_time_seconds)."""
    import librosa

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    bpm = float(np.atleast_1d(tempo)[0])
    first_beat = float(beats[0]) if len(beats) else 0.0
    if len(beats) >= 8:
        # 박 위치들에 직선을 맞춰 템포를 더 정밀하게 추정 (누적 오차 방지)
        idx = np.arange(len(beats))
        slope, intercept = np.polyfit(idx, beats, 1)
        if slope > 0:
            bpm = 60.0 / slope
            first_beat = float(intercept)
    if not np.isfinite(bpm) or bpm <= 0:
        bpm = 120.0
    # 악보로 읽기 쉬운 범위(60~160)로 접기
    while bpm < 60:
        bpm *= 2
    while bpm > 160:
        bpm /= 2
    return round(bpm, 1), max(first_beat, 0.0)


def grid_origin(notes: list[Note], bpm: float, first_beat: float, beats_per_bar: int,
                subdivision: int) -> float:
    """격자 원점(초): 첫 박 기준, 그보다 앞선 음(못갖춘마디)이 있으면 마디 단위로 당김."""
    spb = 60.0 / bpm
    grid = 1.0 / subdivision
    origin = first_beat
    if notes:
        earliest = min(n.start for n in notes)
        while earliest < origin - 0.5 * grid * spb:
            origin -= beats_per_bar * spb
    return origin


def quantize(notes: list[Note], bpm: float, origin: float, subdivision: int):
    """Seconds -> (offset_in_quarters, length_in_quarters, pitch, velocity) on a grid."""
    spb = 60.0 / bpm
    grid = 1.0 / subdivision  # in quarter notes
    out = []
    for n in notes:
        s = round(((n.start - origin) / spb) / grid) * grid
        e_q = (n.end - origin) / spb
        # 1박 이상 긴 음은 끝을 8분음표 단위로 맞춰 점·쉼표 조각을 줄임
        end_grid = grid * 2 if (e_q - s) >= 1.0 else grid
        e = round(e_q / end_grid) * end_grid
        if e - s < grid:
            e = s + grid
        out.append((s, e - s, n.pitch, n.velocity))
    # 같은 위치·같은 음 중복 제거
    seen, uniq = set(), []
    for q in sorted(out):
        key = (q[0], q[2])
        if key not in seen:
            seen.add(key)
            uniq.append(q)
    return uniq


def legato(qnotes, max_gap: float):
    """음 끝과 다음 음 시작 사이의 짧은 틈(<= max_gap 4분음표)을 메워 쉼표 조각을 줄임."""
    import bisect

    onsets = sorted({q[0] for q in qnotes})
    out = []
    for s, d, p, v in qnotes:
        e = s + d
        i = bisect.bisect_right(onsets, s + 1e-9)
        if i < len(onsets):
            nxt = onsets[i]
            if e < nxt and nxt - e <= max_gap:
                e = nxt
        out.append((s, e - s, p, v))
    return out


def _truncate_overlaps(mono):
    out = []
    for i, (s, d, p, v) in enumerate(mono):
        if i + 1 < len(mono):
            d = min(d, mono[i + 1][0] - s)
        if d > 0:
            out.append((s, d, p, v))
    return out


def skyline(qnotes, contour_drop: int = 9):
    """단선율(멜로디)만 남김.

    1) 각 시점의 가장 높은 음을 후보로 하되, 앞선 멜로디 음이 아직 울리는 중이면
       그보다 낮은 반주 음은 멜로디로 보지 않음.
    2) 주변 멜로디 흐름(이동 중앙값)보다 크게 낮은 음은 반주/베이스로 보고 제거(쉼표).
    """
    by_start = {}
    for s, d, p, v in qnotes:
        if s not in by_start or p > by_start[s][2]:
            by_start[s] = (s, d, p, v)
    mono = []
    for k in sorted(by_start):
        s, d, p, v = by_start[k]
        if mono:
            ps, pd, pp, _ = mono[-1]
            if ps + pd > s + 1e-9 and p < pp - 2:
                continue  # 멜로디가 지속되는 중 아래에서 울린 반주 음
        mono.append((s, d, p, v))

    for _ in range(2):
        if len(mono) < 5:
            break
        pitches = np.array([m[2] for m in mono])
        keep = []
        for i, m in enumerate(mono):
            lo, hi = max(0, i - 6), min(len(mono), i + 7)
            ctx = np.delete(pitches[lo:hi], i - lo)
            if m[2] >= np.median(ctx) - contour_drop:
                keep.append(m)
        mono = keep
    return _truncate_overlaps(mono)


def bassline(qnotes):
    """베이스 라인: 각 시점의 가장 낮은 음만 남김 (옥타브 위 배음 오인식 제거 효과)."""
    by_start = {}
    for s, d, p, v in qnotes:
        if s not in by_start or p < by_start[s][2]:
            by_start[s] = (s, d, p, v)
    mono = [by_start[k] for k in sorted(by_start)]
    # 주변 베이스 흐름보다 한 옥타브 넘게 높은 음은 배음/다른 악기 번짐으로 보고 제거
    if len(mono) >= 5:
        pitches = np.array([m[2] for m in mono])
        keep = []
        for i, m in enumerate(mono):
            lo, hi = max(0, i - 6), min(len(mono), i + 7)
            ctx = np.delete(pitches[lo:hi], i - lo)
            if m[2] <= np.median(ctx) + 10:
                keep.append(m)
        mono = keep
    return _truncate_overlaps(mono)


# ---------------------------------------------------------------- 3. notation
def _spell(pc: int, use_flats: bool) -> str:
    name = (FLAT_NAMES if use_flats else SHARP_NAMES)[pc]
    return name.replace("b", "-") if len(name) > 1 else name  # music21 표기: B- = Bb


def build_score(tracks: list[Track], bpm: float, beats_per_bar: int, title: str,
                split_pitch: int = 60, chord_labels=None):
    from music21 import clef, harmony, layout, meter, metadata, note, stream, tempo

    def make_part(items, part_clef, name=None):
        raw = stream.Stream()
        for s, d, p, v in items:
            n = note.Note(p)
            n.quarterLength = d
            n.volume.velocity = v
            raw.insert(s, n)
        part = stream.Part()
        if name:
            part.partName = name
        if items:
            # chordify: 동시에 울리는 음을 화음으로 묶고 겹침을 붙임줄로 정리
            flat = raw.chordify()
            for el in flat.recurse().notesAndRests:
                part.insert(el.getOffsetBySite(flat), el)
        part.insert(0, part_clef)
        part.insert(0, meter.TimeSignature(f"{beats_per_bar}/4"))
        return part

    score = stream.Score()
    score.insert(0, metadata.Metadata())
    score.metadata.title = title
    score.metadata.composer = "Transcribed by music2score"

    multi = len(tracks) > 1
    parts, groups = [], []
    for t in tracks:
        q = t.qnotes
        label = t.name if multi else None
        if t.kind == "grand":
            rh = make_part(legato([x for x in q if x[2] >= split_pitch], 0.5), clef.TrebleClef(), label)
            lh = make_part(legato([x for x in q if x[2] < split_pitch], 0.5), clef.BassClef())
            parts += [rh, lh]
            groups.append((rh, lh))
        else:
            if t.kind == "melody":
                pitches = sorted(x[2] for x in q)
                median = pitches[len(pitches) // 2] if pitches else 72
                # 낮은 음역(남성 보컬 등)은 낮은음자리표로
                c = clef.BassClef() if median < 57 else clef.TrebleClef()
            else:
                c = {"treble": clef.TrebleClef(), "treble8vb": clef.Treble8vbClef(),
                     "bass": clef.BassClef(), "bass8vb": clef.Bass8vbClef()}[t.kind]
            parts.append(make_part(legato(q, 0.5), c, label))

    # 모든 보표의 길이를 마디 단위로 맞춤
    all_q = [x for t in tracks for x in t.qnotes]
    total = max([x[0] + x[1] for x in all_q], default=beats_per_bar)
    bar = float(beats_per_bar)
    total = (int(total // bar) + (1 if total % bar else 0)) * bar
    for p in parts:
        end = p.highestTime
        if end < total:
            p.insert(end, note.Rest(quarterLength=total - end))
        p.makeRests(fillGaps=True, inPlace=True)

    # 템포 표기는 일반 글꼴(♩)로 써서 어떤 뷰어/PDF 렌더러에서도 깨지지 않게 함
    mm = tempo.MetronomeMark(text=f"♩ = {round(bpm)}", number=bpm)
    mm.numberImplicit = True
    mm.placement = "above"
    parts[0].insert(0, mm)

    # 조성 추정 후 조표 삽입
    try:
        k = stream.Score([p for p in parts]).analyze("key")
        k = k.asKey() if hasattr(k, "asKey") else k
        for p in parts:
            p.insert(0, k)
    except Exception:
        k = None

    # 조성에 맞게 이명동음 철자 정리 (예: F 장조에서 A# -> Bb), 불필요한 제자리표 제거
    scale_names = {pp.name for pp in k.getScale().getPitches()} if k is not None else set()
    for p in parts:
        for el in p.recurse().notes:
            for pt in el.pitches:
                if pt.accidental is not None and pt.accidental.name == "natural":
                    pt.accidental = None
                if scale_names and pt.name not in scale_names:
                    enh = pt.getEnharmonic()
                    if enh.name in scale_names and abs(enh.accidental.alter if enh.accidental else 0) <= 1:
                        pt.step, pt.accidental, pt.octave = enh.step, enh.accidental, enh.octave

    # 코드 이름(예: F#m7)을 맨 위 보표 위에 표기
    use_flats = k is not None and k.sharps < 0
    for c in chord_labels or []:
        cs = harmony.ChordSymbol(root=_spell(c.root, use_flats), kind=c.kind)
        parts[0].insert(c.offset, cs)

    for p in parts:
        p.makeMeasures(inPlace=True)
        p.makeTies(inPlace=True)
        p.makeAccidentals(inPlace=True, cautionaryNotImmediateRepeat=False)
        score.insert(0, p)

    for rh, lh in groups:
        score.insert(0, layout.StaffGroup([rh, lh], symbol="brace", barTogether=True))
    return score, k


KEY_KO = {"major": "장조", "minor": "단조"}


def key_label(k) -> str:
    if k is None:
        return "알 수 없음"
    name = k.tonic.name.replace("-", "b")
    return f"{name} {k.mode} ({name} {KEY_KO.get(k.mode, k.mode)})"


# ---------------------------------------------------------------- 4. export
def _fix_tempo_direction(xml_path: Path):
    """music21이 템포를 빈 <words>와 글자 <words> 두 개로 나눠 쓰는 것을 하나로 합침."""
    import re

    txt = xml_path.read_text(encoding="utf-8")
    pat = re.compile(
        r'<direction placement="above">\s*<direction-type>\s*<words />\s*</direction-type>\s*'
        r'(<sound tempo="[^"]+" />)\s*</direction>\s*<direction>\s*<direction-type>\s*'
        r'<words([^>]*)>([^<]*)</words>\s*</direction-type>\s*</direction>')
    txt = pat.sub(lambda m: (f'<direction placement="above"><direction-type><words{m.group(2)}>'
                             f'{m.group(3)}</words></direction-type>{m.group(1)}</direction>'),
                  txt, count=1)
    xml_path.write_text(txt, encoding="utf-8")


# 글자 안에 쓰인 SMuFL 기호(코드 이름의 #/b 등)를 일반 유니코드 글자로 바꿈.
# verovio는 이 기호를 SVG 안에 내장한 Leipzig 웹폰트로 그리는데, cairosvg는 내장 웹폰트를
# 읽지 못해 "NO GLYPH" 상자가 찍히기 때문.
_SMUFL_TEXT = {"\ue262": "#", "\ue260": "b", "\ue261": "", "\ue263": "x", "\ue264": "bb",
               "\uea66": "#", "\uea64": "b", "\uea65": "", "\uea67": "x", "\uea63": "bb"}


def _plain_text_glyphs(svg: str) -> str:
    import re

    def sub(m):
        size = int(m.group(1))
        text = "".join(_SMUFL_TEXT.get(ch, ch) for ch in m.group(2))
        return f'<tspan font-size="{int(size * 0.56)}px">{text}</tspan>'

    return re.sub(r'<tspan font-family="Leipzig" font-size="(\d+)px">([^<]*)</tspan>', sub, svg)


def render_pdf(musicxml_path: Path, pdf_path: Path) -> int:
    import io

    import cairosvg
    import verovio
    from pypdf import PdfReader, PdfWriter

    tk = verovio.toolkit()
    tk.setOptions({
        "pageWidth": 2100, "pageHeight": 2970,  # A4 (단위 0.1mm)
        "scale": 40, "adjustPageHeight": False,
        "footer": "none", "breaks": "auto",
    })
    if not tk.loadFile(str(musicxml_path)):
        raise RuntimeError("verovio가 MusicXML을 읽지 못했습니다")
    writer = PdfWriter()
    pages = tk.getPageCount()
    for i in range(1, pages + 1):
        svg = _plain_text_glyphs(tk.renderToSVG(i))
        pdf_bytes = cairosvg.svg2pdf(bytestring=svg.encode("utf-8"), background_color="white")
        for page in PdfReader(io.BytesIO(pdf_bytes)).pages:
            writer.add_page(page)
    with open(pdf_path, "wb") as f:
        writer.write(f)
    return pages


def write_midi(tracks: list[Track], bpm: float, path: Path, quantized: bool):
    """여러 트랙 MIDI. quantized=True면 악보와 같은 박자 정렬본, False면 원곡 타이밍 그대로."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    spq = 60.0 / bpm
    for t in tracks:
        inst = pretty_midi.Instrument(program=t.program, name=t.name)
        if quantized:
            for s, d, p, v in t.qnotes:
                inst.notes.append(pretty_midi.Note(velocity=int(v), pitch=int(p),
                                                   start=s * spq, end=(s + d) * spq))
        else:
            for n in t.notes:
                inst.notes.append(pretty_midi.Note(velocity=int(n.velocity), pitch=int(n.pitch),
                                                   start=n.start, end=n.end))
        pm.instruments.append(inst)
    pm.write(str(path))


# ---------------------------------------------------------------- pipeline
def _fmt_stats(stats: dict) -> str:
    return ", ".join(f"{k} {v}" for k, v in stats.items() if v)


def convert(audio: Path, outdir: Path, mode: str = "piano", bpm: float | None = None,
            beats_per_bar: int = 4, subdivision: int = 4, onset_threshold: float = 0.5,
            frame_threshold: float = 0.3, min_note_ms: float = 80.0,
            min_freq: float | None = None, max_freq: float | None = None,
            split_pitch: int = 60, min_velocity: int = 60, title: str | None = None,
            pdf: bool = True, separate: bool = False, demucs_model: str = "htdemucs_6s",
            stems: list[str] | None = None, synth: bool = False, synth_mode: str = "full",
            chords: bool | None = None, log=print) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    name = audio.stem
    title = title or name
    chords = separate if chords is None else chords
    steps = 5 if separate else 4
    step = 0

    def stage(msg):
        nonlocal step
        step += 1
        log(f"[{step}/{steps}] {msg}")

    tracks: list[Track] = []
    stem_info = {}
    if separate:
        from separation import separate as run_separation

        stage("악기 분리 중 (Demucs) — CPU에서는 곡 길이의 절반~2배 정도 걸립니다")
        stem_dir = outdir / f"{name}_stems"
        for st in run_separation(audio, stem_dir, model=demucs_model, log=log):
            stem_info[st.name] = st
            log(f"      {st.name:7s} 비중 {st.share * 100:5.1f}%  -> {st.path}")

        stage("악기별 음 인식 중 (basic-pitch)")
        wanted = stems or [s for s in STEM_PROFILES]
        for stem_name, prof in STEM_PROFILES.items():
            st = stem_info.get(stem_name)
            if stem_name not in wanted or st is None:
                continue
            if st.share < 0.02:
                log(f"      {prof.label}: 소리가 거의 없어 건너뜀")
                continue
            notes = transcribe(st.path, onset_threshold, frame_threshold, min_note_ms,
                               prof.min_freq or min_freq, prof.max_freq or max_freq)
            n_raw = len(notes)
            extra = ""
            if prof.synth:
                notes, stats = clean_synth(notes)
                extra = f" (신스 정리: {_fmt_stats(stats) or '해당 없음'})"
            notes = clean_notes(notes, min_velocity) or notes
            if len(notes) < 8:
                log(f"      {prof.label}: 인식된 음이 너무 적어 건너뜀")
                continue
            log(f"      {prof.label}: 음표 {n_raw}개 → 정리 후 {len(notes)}개{extra}")
            kind = prof.kind
            tracks.append(Track(prof.label, kind, prof.program, notes))
        if not tracks:
            raise RuntimeError("분리된 트랙에서 음을 찾지 못했습니다.")
    else:
        stage(f"음 인식 중 (basic-pitch): {audio.name}")
        notes = transcribe(audio, onset_threshold, frame_threshold, min_note_ms, min_freq, max_freq)
        if not notes:
            raise RuntimeError("인식된 음이 없습니다. --onset-threshold 값을 낮춰 보세요.")
        n_raw = len(notes)
        extra = ""
        if synth:
            notes, stats = clean_synth(notes)
            extra = f" (신스 정리: {_fmt_stats(stats) or '해당 없음'})"
        notes = clean_notes(notes, min_velocity) or notes
        log(f"      음표 {n_raw}개 인식 → 정리 후 {len(notes)}개{extra}")
        tracks.append(Track("Melody" if mode == "melody" else "Piano",
                            "melody" if mode == "melody" else "grand", 0, notes))

    stage("템포·박자 분석 및 정렬 중")
    est_bpm, first_beat = estimate_tempo(audio)
    bpm = bpm or est_bpm
    all_notes = [n for t in tracks for n in t.notes]
    origin = grid_origin(all_notes, bpm, first_beat, beats_per_bar, subdivision)
    for t in tracks:
        t.qnotes = quantize(t.notes, bpm, origin, subdivision)
        prof = next((p for p in STEM_PROFILES.values() if p.label == t.name), None)
        if t.kind == "melody":
            t.qnotes = skyline(t.qnotes)
        elif prof is not None and prof.mono == "low":
            t.qnotes = bassline(t.qnotes)
    log(f"      템포 {bpm:g} BPM, {beats_per_bar}/4 박자")

    # 코드 인식: 화음 악기(보컬 제외) 기준
    harmony_q = [x for t in tracks if t.kind != "melody" for x in t.qnotes] or \
                [x for t in tracks for x in t.qnotes]
    total_len = max((x[0] + x[1] for x in harmony_q), default=0.0)
    labels = detect_chords(harmony_q, total_len, segment=beats_per_bar / 2) if (chords or synth_mode == "chords") else []

    # 신스 표기 방식
    for t in tracks:
        prof = next((p for p in STEM_PROFILES.values() if p.label == t.name), None)
        if prof is None or not prof.synth or synth_mode == "full":
            continue
        if synth_mode == "lead":
            t.qnotes, t.kind, t.program = skyline(t.qnotes), "treble", 81
        elif synth_mode == "chords":
            own = detect_chords(t.qnotes, total_len, segment=beats_per_bar / 2)
            t.qnotes, t.kind, t.program = voice_chords(own, low=57), "treble", 89

    score_midi = outdir / f"{name}_score.mid"
    write_midi(tracks, bpm, score_midi, quantized=True)
    raw_midi = outdir / f"{name}.mid"
    write_midi(tracks, bpm, raw_midi, quantized=False)

    stage("악보 생성 중 (music21)")
    score, key = build_score(tracks, bpm, beats_per_bar, title, split_pitch,
                             labels if chords else None)
    xml_path = outdir / f"{name}.musicxml"
    score.write("musicxml", fp=str(xml_path))
    _fix_tempo_direction(xml_path)
    log(f"      조성 추정: {key_label(key)}  -> {xml_path}")
    if labels and chords:
        use_flats = key is not None and key.sharps < 0
        preview = " ".join(c.name(use_flats) for c in labels[:12])
        log(f"      코드 {len(labels)}개 인식: {preview}{' …' if len(labels) > 12 else ''}")

    result = {"midi": raw_midi, "score_midi": score_midi, "musicxml": xml_path,
              "bpm": bpm, "key": key_label(key), "notes": sum(len(t.notes) for t in tracks),
              "tracks": [t.name for t in tracks]}
    if separate:
        result["stems_dir"] = outdir / f"{name}_stems"
    if pdf:
        stage("PDF 악보 렌더링 중 (verovio)")
        pdf_path = outdir / f"{name}.pdf"
        pages = render_pdf(xml_path, pdf_path)
        result["pdf"] = pdf_path
        log(f"      {pages}쪽 -> {pdf_path}")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="music2score",
        description="오디오(mp3/wav 등)를 MIDI와 악보(MusicXML/PDF)로 변환합니다.")
    ap.add_argument("audio", type=Path, help="입력 오디오 파일")
    ap.add_argument("-o", "--outdir", type=Path, default=Path("output"), help="출력 폴더 (기본: output)")
    ap.add_argument("--mode", choices=["piano", "melody"], default="piano",
                    help="(분리 안 할 때) piano: 큰보표 악보, melody: 가장 높은 음만 남긴 단선율 악보")

    g = ap.add_argument_group("악기 분리 / 신스")
    g.add_argument("--separate", action="store_true",
                   help="Demucs로 악기를 분리해 악기별 파트 악보를 만듦 (requirements-separate.txt 필요)")
    g.add_argument("--demucs-model", default="htdemucs_6s",
                   help="htdemucs_6s(기본, 기타·피아노 분리) | htdemucs | htdemucs_ft(정확하지만 느림)")
    g.add_argument("--stems", default=None,
                   help="악보에 넣을 악기만 선택 (쉼표로 구분): vocals,other,guitar,piano,bass")
    g.add_argument("--synth", action="store_true",
                   help="(분리 안 할 때) 전자음악용 신스 정리 적용: 끊긴 음 합침, 배음·디튠·메아리 제거")
    g.add_argument("--synth-mode", choices=["full", "lead", "chords"], default="full",
                   help="(분리 시) 신스 파트 표기: full=모든 음, lead=윗선율만, chords=코드 블록 화음으로 단순화")
    g.add_argument("--chords", action=argparse.BooleanOptionalAction, default=None,
                   help="악보 위에 코드 이름 표기 (분리 시 기본 켜짐)")

    g2 = ap.add_argument_group("세부 설정")
    g2.add_argument("--tempo", type=float, default=None, help="템포(BPM) 직접 지정. 생략하면 자동 추정")
    g2.add_argument("--time-sig", type=int, default=4, help="한 마디의 4분음표 수 (기본 4 → 4/4)")
    g2.add_argument("--grid", type=int, choices=[2, 3, 4, 6, 8], default=4,
                    help="4분음표당 격자 수 (4=16분음표, 2=8분음표, 3/6=셋잇단)")
    g2.add_argument("--onset-threshold", type=float, default=0.5, help="음 시작 감도 (낮을수록 음이 많이 잡힘)")
    g2.add_argument("--frame-threshold", type=float, default=0.3, help="음 지속 감도")
    g2.add_argument("--min-note-ms", type=float, default=80.0, help="이보다 짧은 음은 무시 (ms)")
    g2.add_argument("--min-freq", type=float, default=None, help="이 주파수(Hz) 미만 음 무시")
    g2.add_argument("--max-freq", type=float, default=None, help="이 주파수(Hz) 초과 음 무시")
    g2.add_argument("--min-velocity", type=int, default=60,
                    help="이 세기(0~127)보다 약하고 짧은 음은 잡음으로 보고 제외 (0=끄기)")
    g2.add_argument("--split", type=int, default=60, help="큰보표 위/아래 분리 기준 MIDI 음번호 (60 = 가운데 도)")
    g2.add_argument("--title", default=None, help="악보 제목")
    g2.add_argument("--no-pdf", action="store_true", help="PDF 생성 생략")
    a = ap.parse_args(argv)

    if not a.audio.exists():
        ap.error(f"파일이 없습니다: {a.audio}")
    stems = [s.strip() for s in a.stems.split(",")] if a.stems else None
    if stems:
        bad = [s for s in stems if s not in STEM_PROFILES]
        if bad:
            ap.error(f"알 수 없는 악기: {', '.join(bad)} (가능: {', '.join(STEM_PROFILES)})")
    res = convert(a.audio, a.outdir, mode=a.mode, bpm=a.tempo, beats_per_bar=a.time_sig,
                  subdivision=a.grid, onset_threshold=a.onset_threshold,
                  frame_threshold=a.frame_threshold, min_note_ms=a.min_note_ms,
                  min_freq=a.min_freq, max_freq=a.max_freq, split_pitch=a.split,
                  min_velocity=a.min_velocity, title=a.title, pdf=not a.no_pdf,
                  separate=a.separate, demucs_model=a.demucs_model, stems=stems,
                  synth=a.synth, synth_mode=a.synth_mode, chords=a.chords)
    print("\n완료!")
    for k in ("midi", "score_midi", "musicxml", "pdf", "stems_dir"):
        if k in res:
            print(f"  {k:10s} {res[k]}")


if __name__ == "__main__":
    sys.exit(main())
