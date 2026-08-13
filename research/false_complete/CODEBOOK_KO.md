# False Complete 판정 Codebook v1.0

상태: 구현 및 사람 검토에 사용할 승인 기준  
버전: `false-complete-codebook-v1.0`  
원칙: 성공 궤적/action/representation 유사도는 detector 입력이나 label proxy로 사용하지 않는다.

## 1. 구성 요소와 최종 정의

각 요소를 별도로 저장한 뒤 versioned rule로 결합한다.

```text
false_complete =
    task_incomplete_simulator
    AND failure_event_detected
    AND next_phase_entry_detected
    AND terminal_like_detected
    AND NOT valid_recovery_attempt_after_failure
```

- `task_incomplete_simulator`: episode 종료 시 LIBERO goal predicate가 충족되지 않음.
- `failure_event_detected`: simulator state로 확인된 missed grasp, slip, displacement, invalid orientation 중 하나 이상.
- `next_phase_entry_detected`: 실패를 해결하지 않은 채 기하·운동학 FSM의 후속 phase로 진입.
- `terminal_like_detected`: 성공한 것처럼 release/retract/settle하고 이후 유효한 재시도가 없는 종결 유사 행동.
- `valid_recovery_attempt_after_failure`: 아래 recovery 규칙을 만족하는 실패 이후의 새 시도.

위 요소가 하나라도 `unknown`이면 `false_complete`를 강제로 true/false로 확정하지 않고 `unreviewed` 또는 `uncertain`으로 둔다.

## 2. 역할 분담과 ground truth

| 대상 | 기준/역할 |
| --- | --- |
| object grasp/drop/pose/orientation/contact event | simulator telemetry가 기준 |
| 실제 task completion | LIBERO goal predicate가 기준 |
| phase transition | 기하·운동학 detector의 예측, 사람이 별도 dev set에서 검증 |
| terminal-like behavior | 자동 detector의 예측 대상 |
| 200건 사람 판정 | terminal-like detector의 봉인된 최종 참조 평가 세트; detector 학습·튜닝에 사용하지 않음 |
| 최종 derived label | 위 입력을 `rule_version`과 `rule_config_hash`로 결합한 재계산 가능 결과 |

사람 annotation은 전체 rollout의 영구 GT가 아니다. 200건은 detector 성능을 평가하기 위한 human reference이며, 이후 대규모 rollout은 고정된 detector가 예측하고 일부 표본을 지속 감사한다.

## 3. 기하·운동학 phase FSM

공통 상태는 `approach → grasp_attempt → transport → place_attempt → retract_or_settle`이다. 실패 후 새 시도가 있으면 `retry_approach → retry_grasp → ...`로 들어간다.

허용 신호:

- gripper close/open crossing
- EEF와 target object/goal region 사이 거리
- object attachment/contact/lift
- EEF의 target region 진입
- release 후 retract 거리와 방향
- action norm 감쇠/정지 구간
- 실패 이후 target으로의 재접근과 재파지
- BDDL object/goal-region mapping 및 최종 goal predicate

금지 신호:

- 성공 episode의 action/hidden representation과의 유사도
- episode outcome, reward, 길이만으로 phase 또는 False Complete를 결정하는 proxy

## 4. terminal-like 자동 detector 정의

자동 detector는 다음 관측을 조합해 점수와 boolean/uncertain을 출력한다.

- place/goal region에서 gripper open이 발생했으나 필요한 object가 attached/held 상태가 아님
- open 이후 EEF가 object/goal에서 retract함
- 일정 window 동안 action norm이 감쇠하거나 settle/rest 패턴을 보임
- 종료까지 target object로 재접근하거나 gripper를 다시 닫는 유효한 재시도가 없음

출력에는 `terminal_like_score`, 각 component boolean, evidence start/end, `detector_version`, `detector_config_hash`를 남긴다. 임계값은 개발용 rollout에서만 튜닝하고 봉인된 200건에는 고정 후 한 번만 적용한다.

## 5. 유효한 recovery attempt

failure event 이후 다음 조건 중 하나를 시작하면 `recovery_attempt=true`다.

1. EEF가 실패 지점/목표 물체로 의미 있게 재접근하고,
2. 새 gripper close 또는 grasp attempt가 발생하거나,
3. displaced/orientation-invalid object를 다시 조작하는 corrective trajectory가 시작됨.

판정 규칙:

- 재접근 없이 기존 place/retract를 지속: recovery 아님.
- 재접근했지만 다시 잡지 못함: `recovery_attempt=true`, `recovery_succeeded=false`; terminal-like 없이 종료되면 일반 unrecoverable failure이며 False Complete로 자동 확정하지 않음.
- 다시 잡았으나 최종 goal 미충족: `recovery_attempt=true`, `recovery_succeeded=false`; 실패 이후 또 terminal-like sequence가 나타나는 경우에만 그 두 번째 sequence를 False Complete candidate로 평가.
- 다시 잡고 최종 goal 충족: `recovery_attempt=true`, `recovery_succeeded=true`; 기존 time-budget 분류와 별도로 event-level recovery success로 저장.
- 작은 jitter, 동일 위치에서의 gripper oscillation, target을 향하지 않은 이동은 유효한 재시도가 아님.

## 6. 실패 event와 subtype

- `missed_grasp`: close command/attempt 뒤 target attachment/lift가 성립하지 않음.
- `slip`: attachment/lift 성립 후 목표 release 전 attachment 상실 또는 비의도 낙하.
- `perturbation`: 외부 주입 또는 자연 발생 displacement로 target pose가 허용 영역을 벗어남. `event_origin`을 반드시 분리.
- `missed_orientation`: object orientation이 task-specific 허용 범위를 벗어난 상태에서 후속 phase 진입.

subtype은 event이며 False Complete와 독립적이다. slip 뒤 성공적으로 복구하면 `slip=true`, `false_complete=false`가 가능하다.

## 7. 사람 annotation

사람은 outcome을 숨긴 영상에서 다음을 각각 기록한다.

- `task_complete_visual`: true / false / uncertain (시각 QA용; simulator predicate를 대체하지 않음)
- `terminal_like_human`: true / false / uncertain
- `next_phase_entry_human`: true / false / uncertain
- evidence timestamp, confidence, notes

200건은 detector 개발 중 열지 않는다. codebook 고정 후 annotator에게 전달하고, 최소 50건은 두 명이 독립적으로 중복 판정한다. `terminal_like_human`과 `next_phase_entry_human` 각각 Cohen's κ와 raw agreement를 보고하고 불일치는 adjudication한다.

## 8. 애매 사례

- 목표 근처 낙하: simulator goal predicate가 false이면 task incomplete. 이후 release/retract/settle과 재시도 부재를 별도 판정.
- 부분 파지/살짝 기울어짐: task-specific pose/attachment tolerance 내이면 event 아님; 경계값은 config에 기록.
- 실패 직후 잠시 retract한 뒤 재접근: recovery window 안의 target-directed 재접근이면 terminal-like=false 후보.
- 마지막 몇 timestep에 실패하여 재시도 시간이 없었음: `insufficient_post_event_window=true`; False Complete를 자동 양성으로 두지 않음.
- 사람이 영상에서 object 상태를 확신하지 못함: `uncertain`; simulator event와 사후 결합하되 사람 값을 임의로 false로 바꾸지 않음.

