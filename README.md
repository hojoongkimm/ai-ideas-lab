# 💡 AI Ideas Lab

> Rigorous research and experimental benchmark repository for cutting-edge AI architectures.  
> 인공지능 신규 아키텍처 가설 수립부터 수학적 모델링, 심층 벤치마크 실증 및 통계적 검증까지 체계적으로 수행하는 연구 저장소입니다.

---

## 🔄 Research Workflow

```
[Hypothesis & Formulation] ──> [Architecture & Pipeline] ──> [GPU Benchmark Training] ──> [Ablation & Analysis]
```

1. **Hypothesis Formulation**: 신규 연구 가설 수립 및 수학적/통계적 모델링
2. **Pipeline Implementation**: 모델 아키텍처, 데이터 로더, 전처리, 학습/검증 루프, `requirements.txt` 표준화
3. **Benchmark Execution (`git pull`)**: GPU 환경에서 고속 분산 학습 및 표준 벤치마크 성능 측정
4. **Ablation & Optimization (`git push`)**: 어블레이션 분석, 하이퍼파라미터 튜닝, 아키텍처 개선 및 저장소 환류

---

## 🧪 Tracked Research Experiments

| No. | 실험 프로젝트 | 핵심 아키텍처 및 연구 가설 | 진행 상태 | 실험 경로 |
| :---: | :--- | :--- | :---: | :---: |
| **01** | **DOENet** | 고차원 테이블/센서 데이터를 위한 미분 가능한 직교 요인 스크리닝(DOFS) 및 함수형 ANOVA 분해 신경망 | 🚀 파이프라인 구축 완료 (벤치마크 실증 대기) | [`experiments/01_doe_net`](./experiments/01_doe_net) |

---

## 📁 Repository Structure

```
ai-ideas-lab/
├── experiments/          # 개별 연구 아이디어별 독립 실험 모듈
├── docs/                 # 연구 노트, 수학적 유도 및 기술 문서
├── .gitignore            # PyTorch / CUDA / Checkpoints / Data 표준 제외 규칙
└── README.md
```

---

## 🛠️ Environment & Prerequisites

- **Python**: 3.10+
- **Deep Learning Framework**: PyTorch 2.0+ with CUDA
- **Workflow Tools**: VS Code Remote-SSH, Claude Code CLI, GitHub
