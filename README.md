# 💡 AI Ideas Lab

> Curated research prototyping and experimental pipeline repository.  
> 아이디어 구체화부터 베이스라인 아키텍처 구현, 연구실 GPU 서버 실증 및 Claude Code 기반 고도화까지 이어지는 신속 실험 파이프라인 저장소입니다.

---

## 🔄 Research Workflow

```
[Idea Formulation] ──> [Baseline Architecture & Pipeline] ──> [Lab GPU Server Execution] ──> [Claude Code Tuning & Analysis]
 (호중 님 & 피클 논의)           (Pickle 파이프라인 구현)               (호중 님 연구실 서버 실증)         (VSC 클로드 코드 고도화)
```

1. **아이디어 기획 및 타당성 검토**: 호중 님의 연구 가설 및 신규 아이디어 도출, 선행 연구 및 수식적/실험적 타당성 피드백
2. **베이스라인 구현 및 파이프라인 구축**: 모델 아키텍처, 데이터 로더, 전처리, 학습/검증 루프, `requirements.txt` 표준화
3. **연구실 서버 실증 (`git pull`)**: 명지대 DSLab GPU 서버에서 고속 분산 학습 및 벤치마크 실험 수행
4. **VS Code & Claude Code 최적화 (`git push`)**: 성능 분석, 하이퍼파라미터 튜닝, 아키텍처 개선 및 저장소 환류

---

## 📁 Repository Structure

```
ai-ideas-lab/
├── experiments/          # 개별 연구 아이디어별 독립 실험 모듈
├── docs/                 # 연구 노트, 수학적 유도, 아이디어 분석 보고서
├── .gitignore            # PyTorch / CUDA / Checkpoints / Data 표준 제외 규칙
└── README.md
```

---

## 🧪 Tracked Research Experiments

| No. | 실험 프로젝트 | 핵심 아키텍처 및 연구 가설 | 진행 상태 | 실험 경로 |
| :---: | :--- | :--- | :---: | :---: |
| **01** | **DOENet** | 고차원 테이블/센서 데이터를 위한 미분 가능한 직교 요인 스크리닝(DOFS) 및 함수형 ANOVA 분해 신경망 | 🚀 파이프라인 구축 완료 (서버 실증 대기) | [`experiments/01_doe_net`](./experiments/01_doe_net) |

---

## 🛠️ Environment & Prerequisites

- **Python**: 3.10+
- **Deep Learning Framework**: PyTorch 2.0+ with CUDA
- **Workflow Tools**: VS Code Remote-SSH, Claude Code CLI, GitHub
