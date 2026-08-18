# False Complete 자동 후보 추출

Rollout의 자동 evidence만으로 episode를 `false_complete`, `failure`, `not_failure`,
`uncertain`으로 나누고, 사람이 영상으로 확인하기 좋은 CSV를 만든다. Python 표준 라이브러리만
사용한다.

이 결과는 최종 정답이 아니라 **human review 우선순위용 후보**다. 기본 `robust` 정책은
Annotator A 예시와 비슷한 경향을 보였지만, 다른 rollout의 정확도는 별도 annotation으로
확인해야 한다.

## 가장 빠른 사용법

자동 detector evidence CSV가 있으면 다음 한 줄로 실행한다.

```bash
python scripts/false_complete_review/classify_false_complete.py \
  --input episode_evidence.csv
```

기본 출력은 입력 파일 옆에 자동 생성된다.

- `episode_evidence.false_complete_review.csv`
- `episode_evidence.false_complete_review.summary.json`

경로를 직접 지정하려면 `--output outputs/run1_review.csv`를 추가한다. 기존 파일은 덮어쓰지
않으므로 같은 이름이 있으면 새 경로를 지정해야 한다.

이미 suite별 detector analysis tree가 있다면 중간 CSV 없이 한 번에 처리할 수 있다.

```bash
python scripts/false_complete_review/classify_false_complete.py \
  --analysis-root /path/to/analysis_root \
  --output outputs/run1_review.csv
```

analysis tree는 각 suite 아래에 다음 파일을 가져야 한다.

```text
analysis_root/
  libero_object/
    taxonomy/episodes/episode_*.json
    failure_recovery/episodes/episode_*.json
    terminal_like/episodes/episode_*.json
  libero_goal/...
```

이 스크립트는 raw LeRobot dataset 자체에서 task completion이나 failure event를 새로 추정하지
않는다. 종료 직전 simulator task predicate와 detector event가 들어 있는 normalized CSV 또는
위 analysis tree가 필요하다. 필수 evidence가 없는 legacy rollout을 outcome으로 대신 판정하지
말고 `uncertain`으로 보존한다.

## CSV를 사람이 대조하는 방법

출력 CSV에서 `review_recommended=True`를 먼저 필터링하고, `suite`, `task_id`,
`episode_index`, `seed`로 해당 rollout 영상을 연다. 자동 열은 그대로 두고 오른쪽의 빈 열만
채운다.

- `human_label`: `false_complete`, `failure`, `not_failure`, `uncertain`
- `human_failure_type`: 자유 텍스트 또는 프로젝트 taxonomy
- `human_notes`: 판단 근거나 애매한 지점

자동 판정은 이 human 열들을 feature로 읽지 않는다. 자동 `classification`, `confidence`,
`classification_reason`과 human 열이 같은 행에 있어 Excel/LibreOffice에서 바로 비교할 수 있다.

## NAS rollout을 웹에서 보기

저장소의 기존 `hf_review_space`는 frozen 200개와 Annotator A assignment에 고정된 전용
viewer라 새 rollout에는 그대로 쓸 수 없다. `build_rollout_viewer.py`는 현재 review CSV와
NAS/local mount의 영상을 연결하는 범용 localhost viewer다. 영상 파일은 복사하지 않는다.

NAS가 `/mnt/nas/.../rollouts`로 보이는 머신에서 다음처럼 한 번에 생성하고 실행한다.

```bash
python scripts/false_complete_review/build_rollout_viewer.py build \
  --review-csv outputs/run1_review.csv \
  --video-root /mnt/nas/path/to/rollouts \
  --output-dir outputs/run1_viewer \
  --serve
```

표시된 `http://127.0.0.1:8765/`을 브라우저에서 열면 된다. Windows에서는 NAS를 SMB 드라이브로
마운트한 뒤 `--video-root Z:\\path\\to\\rollouts`처럼 지정할 수 있다. Viewer만 다시 실행할
때는 다음 명령을 쓴다.

```bash
python scripts/false_complete_review/build_rollout_viewer.py serve \
  --viewer-dir outputs/run1_viewer \
  --video-root /mnt/nas/path/to/rollouts
```

Viewer 기능:

- suite/task/episode, 자동 classification·confidence·reason을 영상 옆에 표시
- classification, review priority, 추천 여부, 미검토 상태 필터
- 여러 camera MP4를 한 episode에 함께 표시하고 동시 재생/정지
- `human_label`, `human_failure_type`, `human_notes`를 브라우저 localStorage에 저장
- human review CSV export 및 이전 export CSV import
- HTTP byte-range 지원으로 긴 NAS MP4 seek
- 기본 `127.0.0.1` bind, NAS 영상 복사 0건

CSV에 `video_paths`, `video_path`, `review_clip_path` 열이 있으면 그 경로를 우선 사용한다.
값은 `|`로 여러 camera를 구분할 수 있다. 경로 열이 없으면 `--video-root` 아래의
`episode_*.mp4`를 한 번 스캔해 `suite + episode_index`로 연결한다. 모호하면 추측하지 않고
중단하므로 CSV에 명시적 video path 열을 추가한다. Custom 열은 `--video-column`으로 지정한다.

Viewer도 blinded review 경계를 지켜 `outcome/reward/success/done/length/frame_count/similarity`
열이 포함된 CSV는 거부한다. NAS가 로컬 경로로 mount되어 있지 않으면 먼저 read-only mount를
준비해야 하며, 이 도구는 SSH secret을 읽거나 원격 파일을 내려받지 않는다.

## 판정 규칙

기본값은 아래 core를 쓰는 `--decision-policy robust`다.

```text
false_complete =
    task_incomplete
    AND failure_event_detected
    AND next_phase_entry
    AND NOT valid_recovery_attempt
```

`terminal_like`는 hard gate가 아니라 confidence 보조 신호다. 출력 점수는 확률이 아니라 규칙
근거 강도다.

- `robust` (기본): 가장 일반적인 자동 review 후보
- `strict`: `terminal_like=true`까지 필수인 보수적 참고 정책

`outcome`, `reward`, `success`, `done`, episode length/frame count, trajectory/action/
representation similarity는 False Complete feature나 proxy로 사용하지 않는다. 입력에 있어도
판정에서 무시하고 summary에 열 이름만 기록한다.

참고로 noisy Annotator A의 판정이 알려진 paired episode 163개에서 기본 robust의 결과는
TP/TN/FP/FN 46/97/7/13, precision 0.8679, recall 0.7797, F1 0.8214였다. 이는 보편적 정확도나
ground truth 성능이 아니다. 별도 completion-fixed 200개에는 human 정답이 없으며 robust는
65개를 review 후보로 냈다.

## 입력 CSV 필수 열

가장 단순한 normalized profile은 다음 열을 사용한다. Boolean evidence는 `true`, `false`,
`unknown` 중 하나다.

```text
review_id,suite,task_id,episode_index,seed,
task_incomplete,failure_event_detected,next_phase_entry,terminal_like,
valid_recovery_attempt,failure_types
```

기존 detector summary를 직접 쓸 때는 다음 열도 자동 인식한다.

```text
task_incomplete,
detector_failure_event_count,
detector_next_phase_state,
detector_terminal_like_state,
detector_valid_recovery_attempt_count,
detector_failure_types
```

## 테스트

```bash
cd scripts/false_complete_review
python -m unittest -v test_classify_false_complete.py test_build_rollout_viewer.py
```
