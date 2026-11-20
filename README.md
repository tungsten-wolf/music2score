# 🎼 music2score

음악 파일(mp3, wav, flac, m4a, ogg …)을 넣으면 **MIDI**와 **악보(PDF, MusicXML)**로 바꿔 주는 프로그램입니다.

## 어떻게 동작하나요?

1. **음 인식**: Spotify의 오픈소스 AI 모델 [basic-pitch](https://github.com/spotify/basic-pitch)가 오디오에서 음(시작·끝·높이·세기)을 찾아냅니다.
2. **잡음 정리**: 약하고 짧은 음, 큰 음에 딸려 나온 배음(옥타브 위 유령 음)을 걸러냅니다.
3. **박자 맞추기**: 템포를 자동 추정하고, 음을 16분음표(또는 8분/셋잇단) 격자에 맞춰 정렬합니다.
4. **악보 만들기**: music21로 조성(조표)을 추정하고 피아노 큰보표(양손) 또는 한 줄 멜로디 악보를 만듭니다.
5. **PDF 출력**: verovio로 악보를 그려 PDF로 저장합니다.

## 결과 파일

| 파일 | 설명 |
|---|---|
| `곡이름.pdf` | 인쇄용 악보 |
| `곡이름.musicxml` | MuseScore, Finale, Sibelius, Dorico 등에서 열어 **편집**할 수 있는 악보 |
| `곡이름.mid` | 원곡 연주의 미묘한 타이밍이 살아있는 MIDI (재생·DAW용으로 추천) |
| `곡이름_score.mid` | 악보와 똑같이 박자에 맞춘 MIDI |

## 설치 (한 번만)

Python **3.10 또는 3.11**이 필요합니다 (basic-pitch가 3.12 이상을 아직 지원하지 않습니다).

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

* PDF 생성에 쓰는 `cairosvg`는 Cairo 라이브러리가 필요합니다. macOS는 `brew install cairo`, Ubuntu는 `sudo apt install libcairo2`.
  (Cairo가 없으면 `--no-pdf`로 MusicXML만 만든 뒤 MuseScore에서 PDF로 내보내면 됩니다.)

## 사용법

### 1) 웹 화면 (추천)

```bash
python app.py
```

브라우저에서 <http://127.0.0.1:7860> 을 열고, 음악 파일을 올린 뒤 **변환하기**를 누르면 됩니다. 결과 파일 다운로드와 악보 미리보기가 함께 나옵니다.

### 2) 명령줄

```bash
python music2score.py 노래.mp3                       # output/ 폴더에 결과 저장
python music2score.py 노래.mp3 --mode melody          # 멜로디만 한 줄 악보로
python music2score.py 노래.mp3 --tempo 96 --grid 2    # 템포 지정, 8분음표 단위로 정렬
python music2score.py 노래.mp3 -o 결과 --title "내 곡"
```

주요 옵션

| 옵션 | 설명 |
|---|---|
| `--mode piano / melody` | 양손 피아노 악보(기본) / 가장 높은 음만 남긴 멜로디 악보 |
| `--tempo 숫자` | 템포(BPM)를 직접 지정 (자동 추정이 2배·절반으로 틀릴 때 유용) |
| `--time-sig 3` | 3/4 박자 (기본 4 → 4/4) |
| `--grid 2/3/4/6/8` | 리듬 정렬 단위 (4=16분음표 기본, 2=8분음표, 3=셋잇단) |
| `--onset-threshold 0.35~0.7` | 음 인식 감도. 낮추면 음이 더 많이, 높이면 더 적게 잡힘 |
| `--min-freq / --max-freq` | 이 주파수 범위 밖의 음 무시 (예: 베이스만 → `--max-freq 300`) |
| `--split 60` | 양손을 나누는 음 (60 = 가운데 도) |
| `--no-pdf` | PDF 생성 생략 |

## 정확도에 대해 (꼭 읽어 주세요)

* **잘 되는 것**: 피아노 독주, 기타 독주, 노래·플루트·바이올린 같은 단선율 악기. 음높이는 꽤 정확합니다.
* **어려운 것**: 밴드·오케스트라처럼 악기가 많이 섞인 곡. 모든 악기의 음이 한 악보에 섞여 나오고, 드럼·리버브 때문에 잡음 음표가 늘어납니다.
  → 이런 곡은 먼저 보컬/악기 분리 도구(예: Demucs, Ultimate Vocal Remover)로 원하는 파트만 뽑은 뒤 넣으면 훨씬 좋아집니다.
* **리듬·박자표**: 템포가 일정한 곡일수록 정확합니다. 템포가 자주 변하거나(루바토) 박자표가 4/4가 아니면 직접 옵션을 주는 게 좋습니다. 마디의 첫 박 위치는 추정이라 한두 박 어긋날 수 있습니다.
* 결과는 **초안**으로 생각해 주세요. MusicXML을 MuseScore(무료)에서 열어 다듬는 것을 권장합니다.
