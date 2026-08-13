# Perturbation 통제군·sham 검증 계획 v1.0

## matched conditions

각 task/seed/policy checkpoint에서 다음 조건을 pair로 수집한다.

1. `baseline`: perturbation 및 trigger hook 없음
2. `sham`: 동일 trigger detector/logging은 실행하지만 물리 state 변경 없음
3. `perturb_recovered`: 실제 perturbation 후 최종 성공
4. `perturb_failed`: 실제 perturbation 후 실패 또는 False Complete candidate
5. `natural_failure`: perturbation 없이 자연 발생한 실패

자연 발생과 주입 유도 사례는 별도 dataset 경로와 `event_origin=natural|injected`로 분리한다.

## sham 결정론 smoke

sham이 baseline을 교란하지 않는지 GPU 본실험 전에 최소 smoke로 검증한다.

- 동일 checkpoint, task, seed, initial state, deterministic inference 설정 사용
- baseline과 sham의 초기 observation hash 일치 확인
- sham hook 전후 Python/NumPy/Torch/env RNG state hash 기록
- sham trigger가 RNG를 소비하지 않도록 random sampling 금지
- action, observation/state, object/eef/gripper trajectory를 step별로 비교
- 완전 결정론 환경이면 bitwise equality 요구
- 비결정론 성분이 확인되면 사전 정의 tolerance와 divergence 최초 step을 기록하고 원인을 규명하기 전 sham을 control로 사용하지 않음
- sham과 baseline이 갈라지면 실제 perturbation collection을 시작하지 않음

## perturbation 기록

- `pair_id`, task, seed, checkpoint hash
- scheduled trigger predicate/time과 actual trigger step
- perturbation type/parameters/config hash
- trigger 전후 simulator state
- recovery attempt/success
- final task predicate와 time-budget outcome

이 matrix는 failure representation과 단순 perturbation visual trace를 분리하기 위한 최소 설계다.
