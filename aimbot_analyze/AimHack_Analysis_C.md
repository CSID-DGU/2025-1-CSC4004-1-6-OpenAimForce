
# C 분석

게임 내 부정행위에 해당하는 자동 조준 프로그램 C의 내부적인 **수학적 원리 및 구현 로직**을 중심으로 분석하였다.


## 자동 조준 및 조준 봇

### 기능
목표물과 사용자의 위치 차이를 기반으로 수평각과 수직각을 계산하여 목표물의 정확한 위치를 추적하고, 목표물의 각도 차이에 따라 조준 속도를 조절하며 목표물이 이동하는 경우에도 추적하여 정확히 맞출 수 있도록 설계되어 있음 

### 수학적 원리
목표물의 위치를 추적하기 위한 수평각과 수직각을  **arctan2** 함수와 **피타고라스 정리**를 사용해 계산

#### 수평각 (Horizontal Angle):
수평각은 목표물과 사용자 간의 X, Z 좌표 차이를 사용하여 계산되어 목표물이 수평면에서 어느 위치에 있는지 전달

```math
\theta_X = \text{arctan2}(EnemyX - MyX, EnemyZ - MyZ)
```

- **arctan2** 함수는 2D 평면에서 두 점 사이의 각도를 계산
- `EnemyX`, `EnemyZ`, `MyX`, `MyZ`는 목표물과 사용자의 X, Z 좌표

#### 수직각 (Vertical Angle):
수직각은 목표물의 Y 좌표와 사용자의 Y 좌표 차이를 이용하여 목표물이 위나 아래에 있을 때, 그 위치를 나타내는 각도를 전달

```math
\theta_Y = \text{arctan2}(MyY - EnemyY, \sqrt{(EnemyX - MyX)^2 + (EnemyZ - MyZ)^2})
```

- `MyY`, `EnemyY`는 각각 사용자와 목표물의 Y 좌표
- `arctan2` 함수는 목표물과 사용자의 상대적인 높이 차이를 기반으로 수직각을 계산


#### 속도 조정 공식:
각도 차이에 따른 조준 속도는 아래와 같은 방식으로 계산됨

```math
\text{Speed}_X = k \times \left| \theta_X - \text{CurrentAngle}_X \right|
```

```math
\text{Speed}_Y = k \times \left| \theta_Y - \text{CurrentAngle}_Y \right|
```

- `k`는 속도 조절을 위한 상수
- $\left| \theta_X - \text{CurrentAngle}_X \right|$는 목표물과 사용자의 X축에 대한 각도 차이



#### 목표물 예측 공식:
목표물의 예상 위치는 속도와 시간을 기반으로 계산됨

```math
\text{PredictedX} = \text{EnemyX} + v_X \times t
```

```math
\text{PredictedY} = \text{EnemyY} + v_Y \times t
```

- `v_X`, `v_Y`는 목표물의 속도
- `t`는 시간
- `PredictedX`, `PredictedY`는 예상되는 목표물의 위치
