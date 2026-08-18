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
python -m unittest -v test_classify_false_complete.py
```
