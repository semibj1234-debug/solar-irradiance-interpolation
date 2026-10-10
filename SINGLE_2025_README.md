# 2025 Enhanced SR-Kriging: single OIBC snapshot

검토일: 2026-10-11 KST. 이 문서는 2025년 점추정 알고리즘의 2D 적용을 설명한다.
기존 sr.py, data.py, audit.py, tests.py, config.json, README.md, run.py는 변경하지 않았다.
신규 파일은 single_2025.py(실행), single_2025_tests.py(추가 검증)이다.

## 실행

작업 디렉터리: E:/04_Project/01_OIBC 2025/code_SRKriging

```powershell
& 'C:/Users/user/anaconda3/python.exe' single_2025.py
```

테스트 11개 -> 기존 단일 스냅샷 로딩/검사 -> 20회 실제 데이터 smoke 갱신 ->
88개 전체 위치에서 기존 8-start fit_sr 학습 -> 독립 예측 검증 -> 평가 및 저장.
기본 출력은 results/single_2025_<UTC timestamp>이며 기존 폴더가 있으면 실패한다.
명시적 경로는 --output으로 지정할 수 있다. 모델/설정 변경 시 기존 결과를 재사용하지 않는다.
테스트만 실행: python single_2025_tests.py
실행 경로에는 Adaptive, CV, 1-SE, Bayesian Gibbs 및 다른 모델의 학습/평가가 없다.

## 원문 확인과 함수 대응

원문: Miao & Wang (2025), DOI 10.1061/AJRUA6.RUENG-1403.
로컬 PDF는 선행 연구 조사/miao-wang-2024-interpolation-of-sparse-measurements-and-quantification-of-measurement-error-using-enhanced-kriging-with.pdf.
파일명의 2024는 온라인 발표/저작권 연도이며 논문 서지는 2025, 11(1), 04024081이다.
전체 PDF 13쪽에 접근했으며 방법론 2-5쪽을 읽고, 4-5쪽 수식을 이미지로 대조했다.
PDF SHA256은 각 실행 manifest.json에 기록된다.

| 원문 식/위치 | 수학적 의미 | 실제 구현 | 판정 |
|---|---|---|---|
| (1), p2 | Cf = B diag(q) B.T, q=1/alpha | sr.grid_basis; audit.audit_fit의 cf | 기본 표현 일치. 실제 학습은 전체 Cf를 만들지 않음 |
| (2)-(3), p2 | A=Psi B, Cy=A diag(q) A.T+eta I | sr.observation_operator, design, state | 공분산식 일치. bilinear Psi는 별도 2D 선택 |
| (8), p3 | 조건부 예측평균 | sr.predict_sr | 타깃 설계행렬 H를 사용한 같은 식 |
| (9), p3 | 조건부 잠재장 공분산 | sr.predict_sr | 대각분산 계산; 전체 공분산을 만들 필요 없음 |
| (10)-(12), p4 | Gaussian marginal likelihood | sr.state | 정규화 상수 포함. 작은 예제로 coefficient-space 식 (12)와 대조 |
| (16), p4 | alpha_new=(1-alpha_old*S_ii)/mu_i^2 | sr.update | 동일한 observation-space 식 사용 |
| (20), p5 | eta_new=(residual.T residual+tr(A S A.T))/n | sr.update | 새 alpha, 기존 eta로 S와 mu를 재계산한 후 갱신 |
| p5, 종료조건 | 인접 log-likelihood 변화 <1e-8 | sr.fit_sr | abs(delta L)<1e-8, 기존 maxiter=10000 |
| (2)에서의 확장 | 새 noisy observation의 분산 | sr.predict_sr | latent_variance+eta; 독립적인 동일 noise를 가정한 적용 |

eta는 sigma_e^2, 즉 분산이다. sigma_e와 eta를 혼동하지 않는다.
코드의 gamma=q*h는 보조 벡터이며 2026년 Bayesian hyperparameter gamma가 아니다.
기존 공식을 수정할 수학적 이유가 발견되지 않아 핵심 함수는 재작성하지 않았다.
논문 p3의 희소성 설명에는 trivial alpha를 zero로 쓴 문구가 있으나, 식 (1)의
precision 정의에서는 사라지는 기여는 q=1/alpha ->0이다. 임의 cutoff는 도입하지 않았다.

### 관측공간 변환의 근거

Q=diag(q), C=A Q A.T+eta I,
S=(Q^-1+A.T A/eta)^-1=Q-Q A.T C^-1 A Q,
mu=Q A.T C^-1 y.
h_i=A_i.T C^-1 A_i, b=A.T C^-1 y이면
S_ii=q_i-q_i^2 h_i, mu_i=q_i b_i이다.
식 (16)은 q_new=q*b^2/h가 된다. 작은 양수 q에서 소거 오차가 있는
1-alpha*S_ii를 직접 계산하지 않아도 수학식은 동일하다.

새 Q에서 S, mu를 다시 계산하고
tr(A S A.T)=eta*sum(q*h)를 사용하면 기존 noise 업데이트가 식 (20)과 같다.
이때 eta는 아직 이전 값이다. 기저와 noise를 동시에 오래된 상태로 갱신하지 않는다.
독립 테스트는 역행렬을 직접 구성하여 한 번 및 여러 번의 갱신 결과를 비교한다.
실제 코드는 Cholesky solve를 사용한다. jitter, noise floor, variance clipping은 없다.
q=0은 alpha 무한대의 수치적 경계값으로 다루며, 열을 삭제하지 않는다.

## 2D 적용 선택과 배열 순서

2025년은 1D를 직접 제시한다. 16x16 격자, 256개 초기 기저, bilinear 관측,
학습 자료의 축별 min-max와 반응 표준화, 8개의 초기값은 OIBC 적용 선택이다.
2026년 논문 식 (4)의 tensor 구성과 비교하되, 계층 사전분포/Gibbs는 도입하지 않았다.
참고: https://link.springer.com/article/10.1007/s00477-025-03131-9

1D basis: b[j,r]=sqrt(2/n)*cos(pi*(j+1/2)*r/n), r=0 열은 1/sqrt(n).
B=kron(Bx,By); 행은 ix*ny+iy, 열은 fx*ny+fy.
F.shape=(nx,ny)에서 F.ravel(order="C")를 사용한다.
2026년 열 우선 vec의 kron(By,Bx)는 행/열 순열로 대응한다.
3x5 직사각형 예제에서 순열과 행렬 곱을 독립적으로 테스트했다.

스케일된 좌표 x에 대해 u=x*(n_axis-1), 해당 셀 네 꼭짓점에 bilinear 가중치를 준다.
격자점이면 row selection과 같고, off-grid에서는 네 성분의 가중 관측이다.
따라서 원문의 subvector 관측을 그대로 적용한 것은 아니다.
또한 off-grid에서는 Psi Psi.T != I일 수 있다. 측정오차는 관측 자체에
독립적으로 추가되므로 코드의 eta*I_n를 eta*Psi Psi.T로 바꾸면 다른 모형이 된다.
격자에서의 B는 정규직교지만, 불규칙 위치에서 A=Psi B의 열은 정규직교일 필요가 없다.
학습 사각형 밖의 예측은 정의하지 않는다. convex hull 밖 위치는 정확도 평가에서 제외한다.

반응은 z=(nins-training_mean)/training_population_SD이다. 원 단위 평균을 되돌리고
분산에는 SD^2을 곱한다. 이는 원 nins에 경험적 상수 평균을 사용하는 적용이며,
논문의 zero-mean 공분산식을 z에 적용한 것이다. 공간 추세를 별도로 적합하지 않는다.
학습 반응 표준화와 좌표 min-max에는 평가 레이블을 사용하지 않는다.

## 함수 입력과 출력

n=학습 위치 수, m=기저 수=nx*ny, k=예측 위치 수.

| 요소 | 함수/위치 | 입력 shape | 출력 shape |
|---|---|---|---|
| Basis | sr.py:15 grid_basis | nx,ny 정수 | B(m,m); 이번 실행은 전체 기저 |
| Observation | sr.py:33 observation_operator | x(n,2), nx,ny | Psi(n,m) |
| Design | sr.py:49 design | x(n,2), settings | A(n,m) |
| Covariance/likelihood | sr.py:56 state | A(n,m), y(n,), q(m,), eta scalar | C,L(n,n); mu,gamma,h,b(m,); ll scalar |
| One update | sr.py:70 update | A(n,m), y(n,), q(m,), eta scalar | newq(m,), neweta scalar |
| Estimation | sr.py:85 fit_sr | x(n,2), z(n,), settings, seed | fit dict: weights(m,), noise scalar, starts, trace |
| Prediction | sr.py:135 predict_sr | fit, x(n,2), z(n,), target(k,2), settings | mean,latent_variance,observation_variance 각각 (k,) |
| Preprocessing | single_2025.py:38 training_arrays | train DataFrame | x(n,2), z(n,), TrainingTransform |
| Evaluation | single_2025.py:49 evaluate | truth(k,), prediction(k,), mask(k,) | count,MAE,RMSE scalars |
| Entry point | single_2025.py:202 main | optional --output | 새로운 결과 디렉터리 |

## 검증과 실행 기록의 해석

테스트는 기존 SR 관련 직접 계산 6개와 신규 검사 5개로 구성된다.
허용오차는 기존 검사를 완화하지 않았고 신규 직접 갱신 검사는 rtol=1e-10, atol=1e-12다.
실제 audit의 rtol=atol=1e-7도 기존 그대로다.
smoke는 동일 88개와 256개 기저의 20회 업데이트를 확인하며 완전 수렴 결과라고 부르지 않는다.
최종 학습은 기존 8 starts, 10000 cap, 1e-8 tolerance 그대로 진행한다.

초기 q는 base=1/mean(sum(A*A,axis=1))에 U(0.5,1.5)를 곱한다.
초기 eta는 [0.01,0.5] 구간 log-uniform이다. seed는 manifest와 fit.json에 기록된다.
초기값 분포/다중 시작 예산은 적용 선택이며, 선택 기준은 학습 likelihood뿐이다.
다른 초기값이 서로 다른 수렴점에 도달할 수 있으므로 전역최적 또는 초기값 독립성을 보장하지 않는다.

예측 분산은 고정된 q,eta 및 학습 표준화에 조건부인 plug-in 분산이다.
모수 추정 불확실성은 적분하지 않는다.
좌표 평균의 원 관측 수가 1~8개로 다르지만 기존 iid noise 모형을 유지한다.
추정 eta를 순수한 단일 센서의 오차 분산으로 단정하지 않는다.
물리적 좌표 단위와 nins 단위는 캐시/전처리 코드에서 확인되지 않아
원래 coord 단위, nins 단위 및 nins 단위 제곱으로 보고한다.

결과 파일: manifest.json, tests.txt/json, data_audit.json, transform.json, smoke.json,
smoke_trace.csv, fit.json, starts.csv, likelihood_trace.csv, basis_variances.npy/csv,
prediction_audit.json, predictions_all.csv, evaluation.csv, result.json.
예외 발생 시 failure.json에 traceback을 보존하고 정상 완료로 처리하지 않는다.
기존 sr.fit_sr가 전부 미수렴으로 예외를 던지면 내부 전체 trace는 반환되지 않으며
failure.json의 예외에 restart 요약이 남는다. 이번 실행에서는 모든 시작이 수렴하여
모든 반복 trace가 저장됐다.


## 저장 위치 변경

신규 코드와 실행 결과는 code_SRKriging에 보관한다. 원래 검증된 sr.py와 공통 모듈, 데이터는 code_paper/refactored에서 읽기 전용으로 재사용한다. 새 실행의 기본 결과 경로는 이 폴더의 results이며 원래 프로젝트에 결과를 쓰지 않는다. 이전 실행 manifest의 경로와 소스 해시는 실행 당시 이력을 보존한다. 이동 후 변경은 relocation.json에 기록한다.
