# 강화 학습 2: Deep Reinforcement Learning
#### DQN부터 World Models까지, 딥러닝으로 완성하는 강화학습 핵심 가이드


<img src="https://beat-by-wire.gitbook.io/beat-by-wire/~gitbook/image?url=https%3A%2F%2F3055094660-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FYzxz4QeW9UTrhrpWwKiQ%252Fuploads%252FrV5fPICBRIFJ9oSFg5aw%252FRL2.png%3Falt%3Dmedia%26token%3Dfd71f18b-599d-44a6-8403-b348a368c85c&width=300&dpr=3&quality=100&sign=b54c5538&sv=2" width="500" height="707"/>


## 책 소개

강화학습을 처음 접하는 사람들이 공통적으로 경험하는 순간이 있다. AlphaGo가 이세돌을 이기는 장면을 보거나, 로봇이 스스로 걷는 법을 배우는 영상을 보며 "이걸 나도 만들 수 있을까"라는 질문을 품는 순간이다. 강화학습은 분명 매력적이다. 에이전트가 아무것도 모르는 상태에서 시작하여 시행착오만으로 최적의 전략을 스스로 터득하는 과정은, 기계가 학습한다는 것의 가장 순수한 형태처럼 보인다.

문제는 그 다음이다. 막상 공부를 시작하면 마르코프 결정 과정, 벨만 방정식, 정책 그래디언트, KL 발산 같은 개념들이 쏟아진다. 수식은 추상적이고, 이론과 코드 사이의 간격은 생각보다 넓다. 많은 독자들이 바로 그 지점에서 멈춘다. 이 책은 그 지점을 통과하기 위해 썼다.

이 책은 Deep Reinforcement Learning의 핵심 알고리즘을 이론과 구현 모두에서 체계적으로 다룬다. 단순히 알고리즘을 설명하는 것을 넘어, 각각의 방법론이 왜 등장했고, 어떤 문제를 해결하며, 이전 방법과 어떻게 연결되는지를 함께 짚는다. 코드는 PyTorch와 Gymnasium을 기반으로 작성되었으며, 독자가 직접 실행하고 실험할 수 있도록 설계했다.

책은 네 개의 파트로 구성된다.

Part 0은 이 책 전체의 출발점이다. Deep RL을 본격적으로 시작하기 전에 반드시 짚어야 할 수학적 기초와 환경 설정을 다룬다. 강화학습의 핵심 개념이 이미 익숙하다면 빠르게 훑고 넘어가도 좋고, 처음 접하는 독자라면 충분히 시간을 들여 읽기를 권한다. 이후의 모든 내용이 이 위에 세워진다.

Part 1: Advanced Deep Q-Networks에서는 DQN의 구조적 한계를 짚고, 이를 교정하기 위해 등장한 다섯 가지 핵심 개선 기법을 하나씩 탐구한다. Double DQN, Dueling DQN, Prioritized Experience Replay, Noisy DQN, N-step DQN, 그리고 Distributional DQN까지, 각각의 방법이 DQN의 서로 다른 약점을 어떻게 공략하는지를 살펴본다.

Part 2: Advanced DQN to SAC에서는 이산 행동 공간의 울타리를 넘어 연속 행동 공간으로 확장한다. NAF, DDPG, TD3, SAC를 거치며 Actor-Critic 구조가 어떻게 진화했는지를 따라가고, Hindsight Experience Replay로 희박한 보상 환경에서의 학습 전략도 다룬다.

Part 3: Policy Gradients에서는 정책 최적화의 핵심 문제인 "업데이트 보폭"을 정면으로 다룬다. Trust Region Methods의 이론적 토대 위에서 PPO와 TRPO가 어떻게 안정성을 확보하는지, GAE가 어드밴티지 추정을 어떻게 개선하는지를 살펴본다.

Part 4: World Models에서는 패러다임의 전환을 다룬다. 에이전트가 환경과 직접 부딪히는 대신, 환경이 어떻게 작동하는지를 내부적으로 학습하고 상상 속에서 계획하는 방법이다. VAE, MDN-RNN, RSSM의 핵심 이론을 쌓고, 직접 구현하며, DreamerV3와 Cosmos WFM 같은 최신 연구까지 조망한다.

이 책은 총 3편으로 기획된 AI Master: 강화학습 시리즈의 두 번째 편이다. 1편이 클래식 강화학습의 기초를 다졌다면, 이 책은 그 위에 딥러닝을 결합한 현대적 알고리즘들을 체계적으로 쌓아 올린다. 3편에서는 LLM 에이전트와 강화학습의 접점을 다룰 예정이다.
지금 우리는 에이전트의 시대를 살고 있다. ChatGPT, Claude 같은 대규모 언어 모델이 도구를 사용하고, 계획을 세우며, 복잡한 작업을 자율적으로 수행하는 에이전트 시스템의 핵심이 되었다. 그러나 LLM 단독으로는 장기적 목표를 위한 순차적 의사결정, 환경과의 상호작용을 통한 피드백 기반 학습이 불가능하다. 강화학습은 바로 그 빈자리를 채운다. RLHF가 LLM을 인간의 가치에 정렬하고, 강화학습 기반 에이전트가 복잡한 환경에서 장기 목표를 달성하는 시대가 바로 지금이다.

이 책이 다루는 알고리즘들은 그 에이전트 시대의 기반 기술이다. PPO는 LLM 정렬의 핵심 알고리즘이고, SAC는 로봇공학의 표준적 방법론이며, World Models는 다음 세대 AI의 인지적 토대로 부상하고 있다. 알고리즘의 이름을 아는 것과, 그 작동 원리를 손으로 구현해본 것은 전혀 다른 역량이다. 이 책을 마칠 때쯤이면, 새로운 논문을 읽고 직접 구현할 수 있는 토대가 만들어져 있을 것이다.



## 목 차

저자 소개

Table of Contents (목차)

서문: 들어가며

Part 0. Deep RL 들어가기 전에

Part 1. Advanced Deep Q-Networks

- Chapter 01: Deep Q-Networks 돌아보기
- Chapter 02: Double Deep-Q Networks
- Chapter 03: Dueling Deep Q-Networks
- Chapter 04: Prioritized Experience Replay (PER)
- Chapter 05: Noisy Deep Q-Networks
- Chapter 06: N-step Deep Q-Networks
- Chapter 07: Distributional Deep Q-Networks

Part 2. Advanced DQN to SAC

- Chapter 08: 정규화 어드밴티지 함수 (NAF)
- Chapter 09: DDPG (Deep Deterministic Policy Gradient)
- Chapter 10: TD3 - Twin Delayed DDPG
- Chapter 11: Soft Actor-Critic (SAC)
- Chapter 12: Hindsight Experience Replay (HER)

Part 3. Advanced RL: Policy Gradients 방법론

- Chapter 13: 정책 그래디언트 최적화: Trust Region Methods
- Chapter 14: 근접 정책 최적화 (PPO)
- Chapter 15: 일반화 어드밴티지 추정 (GAE)
- Chapter 16: 신뢰 영역 정책 최적화 (TRPO)

Part 4. World Models

- Chapter 17: World Models 월드 모델 개요
- Chapter 18: World Models 핵심 이론
- Chapter 19: World Models PyTorch 구현
- Chapter 20: World Model 최신 연구

결론: 마무리하며

References: 참고 문헌


## E-Book 구매

- Yes24: https://www.yes24.com/product/goods/178036563
- 교보문고: https://ebook-product.kyobobook.co.kr/dig/epd/ebook/E000012711886
- 알라딘: http://aladin.kr/p/RSQsp

## Github 코드: 

https://github.com/no-wave/deep-RL



