#!/usr/bin/env python3
"""music2score 웹 화면: 브라우저에서 음악 파일을 올리면 MIDI/악보를 내려받을 수 있습니다.

실행:  python app.py   →  http://127.0.0.1:7860 접속
"""
import importlib.util
import shutil
import tempfile
from pathlib import Path

import gradio as gr

from music2score import convert

HAS_DEMUCS = importlib.util.find_spec("demucs") is not None


def preview_svg(xml_path: Path) -> str:
    import verovio

    tk = verovio.toolkit()
    tk.setOptions({"pageWidth": 2100, "pageHeight": 2970, "scale": 40,
                   "adjustPageHeight": True, "footer": "none"})
    tk.loadFile(str(xml_path))
    svg = tk.renderToSVG(1).replace("<svg ", '<svg style="max-width:100%;height:auto" ', 1)
    return f'<div style="background:white;padding:8px;border-radius:8px;overflow:auto">{svg}</div>'


def run(audio_path, mode_label, separate, synth_label, chords, tempo, grid_label, sensitivity,
        title, progress=gr.Progress()):
    if not audio_path:
        raise gr.Error("음악 파일을 먼저 올려 주세요.")
    if separate and not HAS_DEMUCS:
        raise gr.Error("악기 분리에는 demucs가 필요합니다: pip install -r requirements-separate.txt")
    mode = "melody" if mode_label.startswith("멜로디") else "piano"
    grid = {"16분음표": 4, "8분음표": 2, "셋잇단(8분)": 3}[grid_label]
    onset = {"낮음 (음 적게)": 0.6, "보통": 0.5, "높음 (음 많이)": 0.35}[sensitivity]
    synth_mode = {"모든 음": "full", "윗선율만": "lead", "코드로 단순화": "chords"}[synth_label]
    outdir = Path(tempfile.mkdtemp(prefix="music2score_"))
    logs = []

    def log(msg):
        logs.append(msg)
        progress(min(0.95, len(logs) / 18), desc=msg.strip())

    res = convert(Path(audio_path), outdir, mode=mode, bpm=tempo or None, subdivision=grid,
                  onset_threshold=onset, title=title or None, separate=separate,
                  synth=not separate and synth_mode != "full", synth_mode=synth_mode,
                  chords=chords, log=log)
    files = [str(res[k]) for k in ("pdf", "musicxml", "midi", "score_midi") if k in res]
    if "stems_dir" in res:  # 분리된 악기 소리를 zip으로
        files.append(shutil.make_archive(str(outdir / "stems"), "zip", res["stems_dir"]))
    summary = (f"**템포** {round(res['bpm'])} BPM · **조성** {res['key']} · "
               f"**파트** {', '.join(res['tracks'])} · **음표** {res['notes']}개")
    return summary, files, preview_svg(res["musicxml"]), "\n".join(logs)


with gr.Blocks(title="music2score") as demo:
    gr.Markdown("# 🎼 music2score\n음악 파일(mp3, wav 등)을 올리면 <b>MIDI</b>와 "
                "<b>악보(PDF / MusicXML)</b>로 바꿔 드립니다.")
    with gr.Row():
        with gr.Column(scale=1):
            audio = gr.Audio(type="filepath", label="음악 파일")
            separate = gr.Checkbox(value=HAS_DEMUCS, interactive=HAS_DEMUCS,
                                   label="악기 분리 (보컬·신스·기타·피아노·베이스를 파트별 악보로)"
                                   + ("" if HAS_DEMUCS else " — demucs 설치 필요"))
            synth = gr.Radio(["모든 음", "윗선율만", "코드로 단순화"], value="모든 음",
                             label="신스 파트 표기 (분리 끔: '모든 음' 외 선택 시 신스 잡음 정리 적용)")
            chords = gr.Checkbox(value=True, label="코드 이름 표기 (예: F#m7)")
            mode = gr.Radio(["피아노 (양손 악보)", "멜로디만 (한 줄 악보)"], value="피아노 (양손 악보)",
                            label="악보 형태 (악기 분리를 끈 경우)")
            with gr.Accordion("세부 설정", open=False):
                grid = gr.Radio(["16분음표", "8분음표", "셋잇단(8분)"], value="16분음표", label="리듬 정렬 단위")
                sens = gr.Radio(["낮음 (음 적게)", "보통", "높음 (음 많이)"], value="보통", label="음 인식 감도")
                tempo = gr.Number(value=None, label="템포(BPM) — 비워두면 자동", precision=0)
                title = gr.Textbox(label="악보 제목 (선택)")
            btn = gr.Button("변환하기", variant="primary")
        with gr.Column(scale=2):
            summary = gr.Markdown()
            files = gr.File(label="결과 파일 (PDF · MusicXML · MIDI · 분리된 악기 소리)", file_count="multiple")
            preview = gr.HTML(label="악보 미리보기 (첫 쪽)")
            with gr.Accordion("처리 기록", open=False):
                log_box = gr.Textbox(lines=12, show_label=False)
    btn.click(run, [audio, mode, separate, synth, chords, tempo, grid, sens, title],
              [summary, files, preview, log_box])

if __name__ == "__main__":
    demo.launch()
