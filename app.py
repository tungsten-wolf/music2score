#!/usr/bin/env python3
"""music2score 웹 화면: 브라우저에서 음악 파일을 올리면 MIDI/악보를 내려받을 수 있습니다.

실행:  python app.py   →  http://127.0.0.1:7860 접속
"""
import tempfile
from pathlib import Path

import gradio as gr

from music2score import convert


def preview_svg(xml_path: Path) -> str:
    import verovio

    tk = verovio.toolkit()
    tk.setOptions({"pageWidth": 2100, "pageHeight": 2970, "scale": 40,
                   "adjustPageHeight": True, "footer": "none"})
    tk.loadFile(str(xml_path))
    svg = tk.renderToSVG(1).replace("<svg ", '<svg style="max-width:100%;height:auto" ', 1)
    return f'<div style="background:white;padding:8px;border-radius:8px;overflow:auto">{svg}</div>'


def run(audio_path, mode_label, tempo, grid_label, sensitivity, title, progress=gr.Progress()):
    if not audio_path:
        raise gr.Error("음악 파일을 먼저 올려 주세요.")
    mode = "melody" if mode_label.startswith("멜로디") else "piano"
    grid = {"16분음표": 4, "8분음표": 2, "셋잇단(8분)": 3}[grid_label]
    onset = {"낮음 (음 적게)": 0.6, "보통": 0.5, "높음 (음 많이)": 0.35}[sensitivity]
    outdir = Path(tempfile.mkdtemp(prefix="music2score_"))
    logs = []

    def log(msg):
        logs.append(msg)
        progress(min(0.95, len(logs) / 12), desc=msg.strip())

    res = convert(Path(audio_path), outdir, mode=mode, bpm=tempo or None, subdivision=grid,
                  onset_threshold=onset, title=title or None, log=log)
    files = [str(res[k]) for k in ("pdf", "musicxml", "midi", "score_midi") if k in res]
    summary = (f"**템포** {round(res['bpm'])} BPM · **조성** {res['key']} · "
               f"**인식된 음표** {res['notes']}개")
    return summary, files, preview_svg(res["musicxml"])


with gr.Blocks(title="music2score") as demo:
    gr.Markdown("# 🎼 music2score\n음악 파일(mp3, wav 등)을 올리면 <b>MIDI</b>와 <b>악보(PDF / MusicXML)</b>로 바꿔 드립니다.")
    with gr.Row():
        with gr.Column(scale=1):
            audio = gr.Audio(type="filepath", label="음악 파일")
            mode = gr.Radio(["피아노 (양손 악보)", "멜로디만 (한 줄 악보)"], value="피아노 (양손 악보)", label="악보 형태")
            grid = gr.Radio(["16분음표", "8분음표", "셋잇단(8분)"], value="16분음표", label="리듬 정렬 단위")
            sens = gr.Radio(["낮음 (음 적게)", "보통", "높음 (음 많이)"], value="보통", label="음 인식 감도")
            tempo = gr.Number(value=None, label="템포(BPM) — 비워두면 자동", precision=0)
            title = gr.Textbox(label="악보 제목 (선택)")
            btn = gr.Button("변환하기", variant="primary")
        with gr.Column(scale=2):
            summary = gr.Markdown()
            files = gr.File(label="결과 파일 (PDF · MusicXML · MIDI)", file_count="multiple")
            preview = gr.HTML(label="악보 미리보기 (첫 쪽)")
    btn.click(run, [audio, mode, tempo, grid, sens, title], [summary, files, preview])

if __name__ == "__main__":
    demo.launch()
