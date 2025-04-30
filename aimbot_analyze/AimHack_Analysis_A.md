# 에임핵 'A' 분석 결과

다음 사항을 중점으로 분석하였음:

- 표적 지정, 벡터 계산 방식  
- 실제-화면 좌표변환  
- 적 위치 예측  
- 마우스 이동 방식

**익명성 보호를 위하여 실제 함수 이름을 사용하지 않고, 기능을 설명하는 임의의 가명을 붙여 서술하였음**

---

## 1. 좌표변환

적의 월드좌표가 다음과 같다고 가정한다:

$$
\mathbf{P}_\text{world} = \begin{bmatrix} x \\\\ y \\\\ z \\\\ 1 \end{bmatrix}
$$

에임핵 내에서 적의 월드좌표는 다음과 같은 과정을 통해 화면상의 좌표로 변환된다 (표준적인 뷰잉변환):

$$
\mathbf{P}_{a} =
$$

$$
\mathrm{Project}\left( \mathbf{V} \cdot \mathbf{P}_{\mathrm{world}} \right)
$$



단, 다음 행렬을 사용한다:

$$
\mathbf{V} \in \mathbb{R}^{4 \times 4}
$$

$$
(x_s, y_s) = \left( \frac{x'}{w'} \cdot \frac{W}{2} + \frac{W}{2},\ -\frac{y'}{w'} \cdot \frac{H}{2} + \frac{H}{2} \right)
$$

1080p의 화면, 즉 다음을 가정할 때:

$$
W = 1920,\quad H = 1080
$$

조준점은 다음과 같다:

$$
\mathbf{C} = \left( \frac{W}{2}, \frac{H}{2} \right)
$$

조준점과 적 사이의 유격 \( i \) 는 다음과 같이 정의된다:

$$
\Delta_i = \mathbf{P}_{\text{screen},i} - \mathbf{C}
$$

이를 거리로 환산하면:

$$
d_i = \| \Delta_i \|_2 = \sqrt{(x_s - x_c)^2 + (y_s - y_c)^2}
$$

---

## 2. 표적 지정

조건을 충족하는 적 플레이어의 집합 중, 조준점에 가장 가까운 적을 표적으로 지정한다:

$$
\text{Target} = \arg\min_{i \in \mathcal{E}} d_i \quad \text{such that} \quad d_i < \theta
$$

단, 다음 값을 사용한다:

$$
\theta = \texttt{Configurations.FieldOfView}
$$

조건은 다음과 같다:

$$
\texttt{Enemy} = \text{true},\quad \texttt{Alive} = \text{true},\quad \texttt{TEAM} \neq \text{self}
$$

---

## 3. 적 위치 예측

다음과 같은 1차 근사를 사용한다:

$$
\mathbf{v}_i = \mathbf{P}_i^\text{current} - \mathbf{P}_i^\text{previous}
$$

추정된 위치는 다음과 같다:

$$
\mathbf{P}_i^\text{predicted} = \mathbf{P}_i^\text{current} + \frac{5}{k} \cdot \mathbf{v}_i
$$

단, 다음 상수를 사용한다:

$$
k = \texttt{Configurations.PredictionMagnitude} > 0
$$

중력 반영 옵션이 활성화되었을 경우:

$$
v_{i,y} \leftarrow v_{i,y} \cdot \left( \frac{5 \cdot g}{k} \right)
$$

$$
g = \texttt{GravityForce}
$$

예측된 위치는 동일한 선형변환 파이프라인을 거쳐 화면에 투사된다.

---

## 4. 마우스 움직임

다음 정의를 사용한다:

$$
\Delta = (x, y) \in \mathbb{R}^2
$$

---

### 4.1 부드러운 마우스 움직임

틱당 이동 벡터는 다음과 같다:

$$
\delta = \left( \frac{x \cdot s}{h \cdot 8},\ \frac{y \cdot s}{h \cdot 8} \right)
$$

아래는 계수 정의이다:

$$
s, h = \mathrm{Configurations.AimSpeed}
$$


5ms마다 다음이 호출된다:

$$
\mathrm{MemoryEdit\_Mouse}(\delta_x, \delta_y)
$$

이 방식은 표적을 향한 **선형 보간 (linear interpolation)** 을 근사한다.

---

### 4.2 끊어치기

고정된 크기의 빠른 이동을 반복한다:

$$
\delta = \left( \frac{x \cdot f}{10},\ \frac{y \cdot f}{10} \right)
$$

$$
\text{for } i = 1 \text{ to } f: \quad \mathrm{MemoryEdit\_Mouse}(\delta_x, \delta_y)
$$

---

### 4.3 끊어치기 (빠른 샷 + 커서 복귀)

순간적으로 조준 후 다시 돌아오는 동작을 수행한다.

순방향 조준:

$$
\delta_{\text{flick}} = \left( \frac{x \cdot f}{10},\ \frac{y \cdot f}{10} \right)
$$

복귀 조준:

$$
\delta_{\text{return}} = -\delta_{\text{flick}}
$$

적용 순서:

$$
\text{for } i = 1 \text{ to } f: \quad \mathrm{MemoryEdit\_Mouse}(\delta_{\text{flick}})
$$

$$
\text{sleep}(12\ \text{ms})
$$

$$
\text{for } i = 1 \text{ to } f: \quad \mathrm{MemoryEdit\_Mouse}(\delta_{\text{return}})
$$

이는 숙련된 **빠른 샷 + 복귀 조준**을 모방한다.

---
