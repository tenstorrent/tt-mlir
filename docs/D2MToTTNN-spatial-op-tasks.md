# d2m.spatial 처리 시 필요한 작업 (하이레벨)

리팩토링 방향: region 안의 d2m.generic을 모두 ttnn.generic으로 먼저 변환한 뒤, d2m.spatial 처리 시 변환된 ttnn.generic을 모아서 하나의 통합 ttnn.generic으로 합친다.

**리소스 공유 정리:**
- Input IO: 여러 region 간 공유 가능
- Output IO: region 간 공유 안 함
- CB: region 간 공유 안 함

---

## 1. 내부 ttnn.generic에서 리소스 수집

각 region의 이미 변환된 ttnn.generic에서 다음을 수집한다.

- IO(텐서) 목록
- CB 디스크립터/피연산자
- semaphore 관련 인자/디스크립터

## 2. 통합 리소스 번호 부여

수집한 항목에 대해 하나의 ttnn.generic에서 쓸 **겹치지 않는** 인덱스 체계를 정한다.

- **IO 인덱스:** input은 공유 가능하므로 동일 input은 하나의 인덱스로 통일, output은 region별로 서로 다른 인덱스 부여
- **CB 인덱스:** region별로 별도 인덱스 부여 (region 간 공유 없음)
- **Semaphore 인덱스:** 겹치지 않게 부여

## 3. 인덱스 매핑 자료 구성

통합 op의 operand/디스크립터 인덱스와 (원본 ttnn.generic, 그 안의 인덱스) 사이의 매핑을 만들어, descriptor/program 병합 시 참조를 갱신할 수 있게 한다.

## 4. 프로그램(description) 병합

region별 program(커널, core range 등)을 하나로 합치고, 그 안의 CB/semaphore/IO 인덱스 참조를 위 매핑에 맞게 통합 인덱스로 갱신한다.

## 5. 통합 operand 목록 구성

병합된 program이 참조하는 통합 인덱스에 맞춰, 단일 ttnn.generic용 IO 목록과 additional args(세마포어 등) 목록을 조립한다.

## 6. 통합 ttnn.generic op 생성 및 치환

통합된 operand 목록과 병합된 program으로 ttnn.generic 하나를 만들고, 기존 d2m.spatial(과 그 region/내부 ttnn.generic들)을 이 op로 치환한다.

## 7. 검증

통합 후 리소스 중복/누락이 없고, program 내 인덱스 참조가 모두 유효한지 확인한다.
