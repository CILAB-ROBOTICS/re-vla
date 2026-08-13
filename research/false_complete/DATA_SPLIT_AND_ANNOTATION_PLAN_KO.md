# False Complete 데이터 분리·annotation 계획 v1.0

## 봉인된 200건 평가 세트

- 현재 생성된 200개 블라인드 MP4는 `terminal_like` 및 phase detector의 최종 human-reference 평가 세트다.
- detector 구현자와 threshold 튜닝 과정에서는 영상과 사람 label을 열지 않는다.
- 기존 200건은 object/contact telemetry가 없으므로 simulator event detector 개발용 GT로 사용하지 않는다.
- 사람 annotation은 codebook 확정 직후 detector 개발과 병렬로 진행하되, 결과 파일은 detector 고정 전까지 구현자에게 봉인한다.

## 이중 판정 배정

- 200건 전체를 1차 annotator가 판정한다.
- review ID의 SHA-256 결정적 순서에서 각 suite 12건씩 48건, 추가 2건을 전체에서 선택해 총 50건을 2차 annotator가 독립 판정한다.
- 중복 여부는 annotator에게 노출하지 않는다.
- Cohen's κ와 raw agreement를 `terminal_like_human`, `next_phase_entry_human`별로 계산한다.

## detector 개발 세트

- telemetry v2로 별도 수집하며 기존 200건과 seed/output path를 겹치지 않는다.
- 최소 구성: suite/task 분산을 가진 소량 baseline, 자연 실패, perturbation, recovery 사례.
- detector threshold와 FSM parameter는 이 dev set에서만 조정한다.
- final detector version/hash를 고정한 뒤 200건을 한 번만 평가한다.

## base rate와 표집

- 현재 200건은 네 suite의 전체 main rollout이므로 해당 collection 내부의 자연 발생 prevalence 추정에 사용 가능하다.
- detector precision/recall을 위한 추가 감사 세트는 predicted positive/negative를 층화 표집하고 selection probability와 sample weight를 저장한다.
- 첫 human pass에서 False Complete 양성 수를 확인한 후 목표 confidence interval에 맞춰 표본 수를 다시 계산한다.
- 자연 발생과 perturbation 유도 데이터는 `event_origin`과 별도 dataset namespace로 분리한다.

