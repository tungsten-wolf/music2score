#!/usr/bin/env python3
"""music2score: 오디오 파일(mp3/wav/flac/m4a...)을 MIDI와 악보(MusicXML, PDF)로 변환합니다.

파이프라인
  1. Spotify basic-pitch(ONNX)로 오디오 -> 음표(onset/offset/pitch/velocity) 추출
  2. librosa로 템포/박 위치 추정 후, 음표를 박자 격자(기본 16분음표)에 맞춰 정렬(quantize)
  3. music21로 피아노 큰보표(높은음자리표/낮은음자리표) 또는 단선율 악보 구성, 조성 추정
  4. MusicXML 저장, verovio로 PDF 악보 렌더링

사용 예
  python music2score.py song.mp3
  python music2score.py song.mp3 -o out --mode melody --tempo 96
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

import numpy as np  # noqa: E402


@dataclass
class Note:
    start: float  # seconds
    end: float  # seconds
    pitch: int  # MIDI number
    velocity: int


# ---------------------------------------------------------------- 1. transcription
def transcribe(audio_path: Path, onset_threshold: float, frame_threshold: float,
               min_note_ms: float, min_freq: float | None, max_freq: float | None):
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict

    _, midi_data, note_events = predict(
        str(audio_path),
        ICASSP_2022_MODEL_PATH,
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
    return midi_data, notes


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


def quantize(notes: list[Note], bpm: float, first_beat: float, beats_per_bar: int,
             subdivision: int) -> list[tuple[float, float, int, int]]:
    """Seconds -> (offset_in_quarters, length_in_quarters, pitch, velocity) on a grid."""
    spb = 60.0 / bpm
    grid = 1.0 / subdivision  # in quarter notes
    # 격자 원점: 첫 박을 기준으로 하되, 그보다 앞선 음(못갖춘마디)이 있으면 마디 단위로 당김
    origin = first_beat
    if notes:
        earliest = min(n.start for n in notes)
        bar_sec = beats_per_bar * spb
        while earliest < origin - 0.5 * grid * spb:
            origin -= bar_sec
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
    onsets = sorted({q[0] for q in qnotes})
    import bisect
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


def skyline(qnotes, contour_drop: int = 9):
    """단선율 모드: 멜로디 한 줄만 남김.

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

    out = []
    for i, (s, d, p, v) in enumerate(mono):
        if i + 1 < len(mono):
            d = min(d, mono[i + 1][0] - s)
        if d > 0:
            out.append((s, d, p, v))
    return out


# ---------------------------------------------------------------- 3. notation
def build_score(qnotes, bpm: float, beats_per_bar: int, mode: str, title: str,
                split_pitch: int):
    from music21 import clef, meter, metadata, note, stream, tempo, instrument, layout

    def make_part(items, part_clef, name):
        raw = stream.Stream()
        for s, d, p, v in items:
            n = note.Note(p)
            n.quarterLength = d
            n.volume.velocity = v
            raw.insert(s, n)
        part = stream.Part()
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

    if mode == "melody":
        pitches = sorted(q[2] for q in qnotes)
        median = pitches[len(pitches) // 2] if pitches else 72
        # 낮은 음역(남성 보컬, 베이스 등)은 낮은음자리표로
        melody_clef = clef.BassClef() if median < 57 else clef.TrebleClef()
        parts = [make_part(legato(qnotes, 0.5), melody_clef, "Melody")]
    else:
        rh = legato([q for q in qnotes if q[2] >= split_pitch], 0.5)
        lh = legato([q for q in qnotes if q[2] < split_pitch], 0.5)
        parts = [make_part(rh, clef.TrebleClef(), "Piano RH"),
                 make_part(lh, clef.BassClef(), "Piano LH")]

    # 두 보표의 길이를 맞춤
    total = max([q[0] + q[1] for q in qnotes], default=beats_per_bar)
    bar = float(beats_per_bar)
    total = (int(total // bar) + (1 if total % bar else 0)) * bar
    for p in parts:
        end = p.highestTime
        if end < total:
            r = note.Rest(quarterLength=total - end)
            p.insert(end, r)
        p.makeRests(fillGaps=True, inPlace=True)

    # 템포 표기는 일반 글꼴(♩)로 써서 어떤 뷰어/PDF 렌더러에서도 깨지지 않게 함
    mm = tempo.MetronomeMark(text=f"♩ = {round(bpm)}", number=bpm)
    mm.numberImplicit = True
    mm.placement = "above"
    parts[0].insert(0, mm)

    # 조성 추정 후 조표 삽입
    try:
        all_notes = stream.Score([p for p in parts])
        k = all_notes.analyze("key")
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

    for p in parts:
        p.makeMeasures(inPlace=True)
        p.makeTies(inPlace=True)
        p.makeAccidentals(inPlace=True, cautionaryNotImmediateRepeat=False)
        score.insert(0, p)

    if len(parts) == 2:
        score.insert(0, layout.StaffGroup(parts, symbol="brace", barTogether=True))
    return score, k


KEY_KO = {"major": "장조", "minor": "단조"}


def key_label(k) -> str:
    if k is None:
        return "알 수 없음"
    name = k.tonic.name.replace("-", "b")
    return f"{name} {k.mode} ({name} {KEY_KO.get(k.mode, k.mode)})"


# ---------------------------------------------------------------- 4. export
def _fix_tempo_direction(xml_path: Path, bpm: float):
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


def render_pdf(musicxml_path: Path, pdf_path: Path) -> int:
    import cairosvg
    import verovio
    from pypdf import PdfWriter, PdfReader
    import io

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
        svg = tk.renderToSVG(i)
        pdf_bytes = cairosvg.svg2pdf(bytestring=svg.encode("utf-8"),
                                     background_color="white")
        for page in PdfReader(io.BytesIO(pdf_bytes)).pages:
            writer.add_page(page)
    with open(pdf_path, "wb") as f:
        writer.write(f)
    return pages


def write_score_midi(qnotes, bpm: float, path: Path):
    """악보와 동일하게 박자에 맞춘(quantized) MIDI."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(program=0, name="Piano")
    spq = 60.0 / bpm
    for s, d, p, v in qnotes:
        inst.notes.append(pretty_midi.Note(velocity=int(v), pitch=int(p),
                                           start=s * spq, end=(s + d) * spq))
    pm.instruments.append(inst)
    pm.write(str(path))


def convert(audio: Path, outdir: Path, mode: str = "piano", bpm: float | None = None,
            beats_per_bar: int = 4, subdivision: int = 4, onset_threshold: float = 0.5,
            frame_threshold: float = 0.3, min_note_ms: float = 80.0,
            min_freq: float | None = None, max_freq: float | None = None,
            split_pitch: int = 60, min_velocity: int = 60, title: str | None = None, pdf: bool = True,
            log=print) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    stem = audio.stem
    title = title or stem

    log(f"[1/4] 음 인식 중 (basic-pitch): {audio.name}")
    midi_data, notes = transcribe(audio, onset_threshold, frame_threshold, min_note_ms,
                                  min_freq, max_freq)
    if not notes:
        raise RuntimeError("인식된 음이 없습니다. --onset-threshold 값을 낮춰 보세요.")
    raw_midi = outdir / f"{stem}.mid"
    midi_data.write(str(raw_midi))
    n_raw = len(notes)
    notes = clean_notes(notes, min_velocity) or notes
    log(f"      음표 {n_raw}개 인식, 잡음 {n_raw - len(notes)}개 제거 -> {raw_midi}")

    log("[2/4] 템포·박자 분석 및 정렬 중")
    est_bpm, first_beat = estimate_tempo(audio)
    if bpm:
        first_beat = min(n.start for n in notes)
    bpm = bpm or est_bpm
    qnotes = quantize(notes, bpm, first_beat, beats_per_bar, subdivision)
    if mode == "melody":
        qnotes = skyline(qnotes)
    score_midi = outdir / f"{stem}_score.mid"
    write_score_midi(qnotes, bpm, score_midi)
    log(f"      템포 {bpm:g} BPM, {beats_per_bar}/4 박자")

    log("[3/4] 악보 생성 중 (music21)")
    score, key = build_score(qnotes, bpm, beats_per_bar, mode, title, split_pitch)
    xml_path = outdir / f"{stem}.musicxml"
    score.write("musicxml", fp=str(xml_path))
    _fix_tempo_direction(xml_path, bpm)
    log(f"      조성 추정: {key_label(key)}  -> {xml_path}")

    result = {"midi": raw_midi, "score_midi": score_midi, "musicxml": xml_path,
              "bpm": bpm, "key": key_label(key), "notes": len(notes)}
    if pdf:
        log("[4/4] PDF 악보 렌더링 중 (verovio)")
        pdf_path = outdir / f"{stem}.pdf"
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
                    help="piano: 큰보표(양손) 악보, melody: 가장 높은 음만 남긴 단선율 악보")
    ap.add_argument("--tempo", type=float, default=None, help="템포(BPM) 직접 지정. 생략하면 자동 추정")
    ap.add_argument("--time-sig", type=int, default=4, help="한 마디의 4분음표 수 (기본 4 → 4/4)")
    ap.add_argument("--grid", type=int, choices=[2, 3, 4, 6, 8], default=4,
                    help="4분음표당 격자 수 (4=16분음표, 2=8분음표, 3/6=셋잇단)")
    ap.add_argument("--onset-threshold", type=float, default=0.5, help="음 시작 감도 (낮을수록 음이 많이 잡힘)")
    ap.add_argument("--frame-threshold", type=float, default=0.3, help="음 지속 감도")
    ap.add_argument("--min-note-ms", type=float, default=80.0, help="이보다 짧은 음은 무시 (ms)")
    ap.add_argument("--min-freq", type=float, default=None, help="이 주파수(Hz) 미만 음 무시")
    ap.add_argument("--max-freq", type=float, default=None, help="이 주파수(Hz) 초과 음 무시")
    ap.add_argument("--min-velocity", type=int, default=60,
                    help="이 세기(0~127)보다 약하고 짧은 음은 잡음으로 보고 악보에서 제외 (0=끄기)")
    ap.add_argument("--split", type=int, default=60, help="양손 분리 기준 MIDI 음번호 (기본 60 = 가운데 도)")
    ap.add_argument("--title", default=None, help="악보 제목")
    ap.add_argument("--no-pdf", action="store_true", help="PDF 생성 생략")
    a = ap.parse_args(argv)

    if not a.audio.exists():
        ap.error(f"파일이 없습니다: {a.audio}")
    res = convert(a.audio, a.outdir, mode=a.mode, bpm=a.tempo, beats_per_bar=a.time_sig,
                  subdivision=a.grid, onset_threshold=a.onset_threshold,
                  frame_threshold=a.frame_threshold, min_note_ms=a.min_note_ms,
                  min_freq=a.min_freq, max_freq=a.max_freq, split_pitch=a.split,
                  min_velocity=a.min_velocity,
                  title=a.title, pdf=not a.no_pdf)
    print("\n완료!")
    for k in ("midi", "score_midi", "musicxml", "pdf"):
        if k in res:
            print(f"  {k:10s} {res[k]}")


if __name__ == "__main__":
    sys.exit(main())
