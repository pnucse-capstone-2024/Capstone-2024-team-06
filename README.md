# 강화학습기반 교차로 동적 신호제어시스템

&nbsp;

### 1. 프로젝트 소개

국내에서 가장 일반적으로 사용하는 교차로 신호제어 시스템은 "시간제어식 신호"이다. 이는 특정 상황별로 주기를 미리 지정해두고, 해당 상황이 발생하면 인위적인 조작을 통해 적용하는 방식이다. 하지만 기본적으로 미리 정해진 여러 옵션에서 단순히 선택을 하는 방식이기 때문에 정밀한 최적화가 힘들고, 인위적인 제어가 필요하기 때문에 실시간으로 변하는 교통량에 대응하는데 한계가 있다.

&nbsp;

따라서 이러한 문제를 해결하기 위해, 본 프로젝트에서는 **강화학습을 활용하여 동적인 신호제어 시스템**을 개발하였다.

- 불필요한 대기시간을 최소화하는 합리적 신호체계 구현
- 불규칙하게 증감하는 교통량에도 대응할 수 있는 신호체계 구현
- 동적인 신호순서를 적용하여 효율성 증대

&nbsp;
---------------------------------------
### 2. 소개 및 시연 영상

[![부산대학교 정보컴퓨터공학부 소개](http://img.youtube.com/vi/IAcQLQaV-dI/0.jpg)](https://www.youtube.com/watch?v=IAcQLQaV-dI)

&nbsp;
---------------------------------------

### 3. 시스템 구성

&nbsp;

<img width="800" alt="Tech Stack Diagram (Copy) (7)" src="https://github.com/user-attachments/assets/52787a84-3b3d-475c-b7ef-061ca538d35c">

&nbsp;

- Python
- PyTorch
- 교통 시뮬레이터 : [CARLA Simulator](https://carla.org/), [SUMO Simulator](https://eclipse.dev/sumo/)
- 객체 탐지 모델 : [YOLOv8](https://docs.ultralytics.com/models/yolov8/)
- 객체 추적 알고리즘 : [ByteTrack](https://github.com/ifzhang/ByteTrack)

&nbsp;
---------------------------------------

### 4. 설치 방법

#### 4.1. Carla Simulator 설치

**시스템 요구사항**

* x64 시스템
* 165GB 디스크 공간
* 최소 6GB 이상의 GPU
* 2000, 2001 TCP 포트

&nbsp;

**소프트웨어 요구사항**

* CMake: 빌드 파일을 생성하기 위해 필요. version >= 3.15
* Git: Carla repositories를 관리하기 위해 필요.
* Make: 실행 파일을 생성하기 위해 필요. version == 3.81
* 7Zip: asset 파일의 압축 해체에 필요.
* Python3 x64: 다른 버전이나 x32 버전은 제거 권장
* [Windows 8.1 SDK](https://developer.microsoft.com/ko-kr/windows/downloads/sdk-archive/)

&nbsp;

**Visual Studio 2019**
* C++를 사용한 데스크톱 개발
  * MSVC v140 -VS 2015 C++ 빌드 도구(v14.00)
* C++를 사용한 게임 개발
* x64 Visual C++ Toolset
* .Net framework 4.6.2 개발 도구

&nbsp;

**Python Dependencies**
```
pip3 -V
pip3 install --upgrade pip
pip3 install --user setuptools
pip3 install --user wheel
```

&nbsp;

**Unreal Engine**

먼저, 언리얼 엔진을 포크하여 다운로드하기 위해서는 Github 계정이 언리얼 엔진과 연동되어 있어야 한다. 자세한 내용은 아래 링크에서 확인할 수 있다.
[Accessing Unreal Engine Source](https://www.unrealengine.com/en-US/ue-on-github)

1. 언리얼 엔진 소스코드 클론
```
git clone --depth 1 -b carla https://github.com/CarlaUnreal/UnrealEngine.git .
```

언리얼 엔진의 경로는 C:\\\\로 하는것을 권장한다.

2. Configuration Script 실행
```
Setup.bat
GenerateProjectFiles.bat
```

3. 개조된 엔진 컴파일
   - Visual Studio 2019로 UE4.sln 열기
   - "Development Editor". "Win64", "UnrealBuildTool" 옵션을 선택
   - Solution Explorer에서 UE4를 우클릭 후 빌드

4. 컴파일이 성공적으로 되었는지 확인하려면 Engine\Binaries\Win64\UE4Editor.exe 을 실행

&nbsp;

**Carla 빌드**

1. Carla repository 클론
```
git clone https://github.com/carla-simulator/carla
```

2. 최신 버전의 Assets 다운로드

아래 Script 실행
```
Update.bat
```

3. Unreal Engine 환경 변수 설정

- 윈도우 시스템 환경 변수에 Unreal Engine 루트 경로를 **UE4_ROOT** 이름으로 새로 만든다.

4. Carla 빌드
**x64 Native Tools Command Prompt for VS 2019**를 실행한다.

&nbsp;

**Python API 클라이언트 컴파일**

Carla의 루트 경로에서 아래 명령어를 실행한다.

```
make PythonAPI
```

PythonAPI/carla/dist 폴더에 .egg 파일과 .whl 파일이 생성되었는지 확인한다.

&nbsp;

**서버 컴파일**

Carla의 루트 경로에서 아래 명령어를 실행한다.
```
make launch
```

Unreal Engine 4 Editor가 실행되면 성공적으로 설치된 것이다.

5. Import Assets

Capstone Repository의 Import 폴더 내의 파일들을 carla/Import 폴더에 복사 후 아래 명령어를 실행한다.

만약 Editor가 실행중이라면 종료 후 실행한다.

```
make import
```

Editor의 컨텐츠 브라우저의 루트 경로에, map_package가 생성되었다면 성공적으로 import된 것이다.

&nbsp;

#### 4.2. 기타 S/W 및 라이브러리 설치

- SUMO simulator <https://sumo.dlr.de/docs/Downloads.php>
- python 3.8.20
- sumo 1.20.0
- numpy 1.23.5
  
*이하는 PC환경 및 GPU에 따라 버전을 맞추어 설치해야함
(아래는 Windows10, Nvidia GPU 8.6 (RTX 3060) 환경의 경우)

- cuda 11.8
- cuDNN 8.9.3
- pytorch 2.4.1

&nbsp;
---------------------------------------

### 5. 사용법

#### 5.1. CARLA 시뮬레이터를 사용한 모델 학습 및 추론

**맵 지정**

컨텐츠 브라우저의 map_package의 cross_01.umap 파일을 연다.

맵을 연 상태에서, Alt+P 를 눌러, 시뮬레이션 환경을 실행한다.

&nbsp;

**모델 학습**

```
cd app
```

app 폴더로 이동한다.

```
python train.py --simulator carla
// or python train.py -s carla
```

실행 시 시뮬레이션 창에서 시뮬레이션이 시작되고, 카메라 뷰 창과 Total Reward 창이 켜진다.

- 시뮬레이션 창
  
   랜덤하게 차량들이 생성되고 중앙의 교차로로 진입한다. 3600 tick마다 시뮬레이터가 초기화되어 또 다른 랜덤한 환경이 재생된다.

- 카메라 뷰 창
  
   북, 동, 남, 서 방향의 교차로 진입지점의 실시간 카메라 뷰가 나타난다.

   각 뷰 하단에 현 교통 신호가 색깔로 나타나고 각 차선별 소요 프레임수의 총합이 출력된다.

   프레임 상에 관심구역(ROI)가 선으로 표현되어 있으며 진입하는 각 차량을 둘러싼 바운딩박스가 그려진다.

   각 차량별 id와, ROI 내부에 머물렀던 총 프레임수(소요 프레임 : lasted frames)가 바운딩박스에 달린 레이블에 출력된다.

- Total Reward 창
  
   Total Reward는 한 에피소드당 총 Reward(모든 차량의 소요프레임수의 합; 차량 대기시간과 비례)의 합에 음수를 취한 값이며, (이상적으로) 학습이 진행될수록 그래프가 서서히 증가 및 수렴한다.

Ctrl + C 시 학습 또는 추론이 중단되며, Total Reward 창을 닫으면 프로그램이 종료된다.

종료 시 학습이 완료된 모델(dqn_model.pth), Total Reward 그래프와, 해당 Total Reward들의 실제 값을 기록한 텍스트파일이 아래 위치에 저장된다.

- app > results > carla > training
   
&nbsp;

**모델 추론**

0. Pretrained Model
   
   아래 링크에서 미리 학습 완료한 Carla 강화학습 모델을 받을 수 있다.
   
   https://drive.google.com/file/d/1F__vD_Jpml5k_sgPX6YjsBXZSXZiMISq/view?usp=sharing
   
   받은 모델을 app > results > carla > training 폴더에 위치시키면 된다.
 

1. 학습한 모델로 추론
   
```
python control.py --simulator carla
// or python control.py -s carla
```

실행시 위의 모델 학습과 동일한 창들이 켜지며, 이번에는 미리 학습된 모델이 교차로의 신호를 제어하는 것을 볼 수 있다. 

이에 대한 Reward도 마찬가지로 Total Reward창에 나타난다.


2. 고정주기식 신호로 제어 (모델 미사용, 고정주기식 제어)
   
```
python control.py --simulator carla --fixed
// or python control.py -s carla -f
```

또는 학습한 모델과의 비교를 위해 고정주기식 신호로 테스트를 할 수 있다.

Ctrl + C 시 학습 또는 추론이 중단되며, Total Reward 창을 닫으면 프로그램이 종료된다.

종료 시 Total Reward 그래프와 해당 Total Reward들의 실제 값을 기록한 텍스트파일이 아래 위치에 저장된다.

- app > results > carla > inference

&nbsp;

#### 5.2. SUMO 시뮬레이터를 사용한 모델 학습 및 추론

**모델 학습**

```
cd app
```

app 폴더로 이동한다.

```
python train.py --simulator sumo
// or python train.py -s sumo
```

실행 시 시뮬레이션 창과 Total Reward 창이 켜진다.

- 시뮬레이션 창
   
   랜덤하게 차량들이 생성되고 중앙의 교차로로 진입한다. 1000 tick마다 시뮬레이터가 초기화되어 또다른 랜덤한 환경이 재생된다.

- Total Reward 창
   
   Total Reward는 한 에피소드당 총 Reward(모든 차량의 대기시간의 합)의 합에 음수를 취한 값이며, (이상적으로) 학습이 진행될수록 그래프가 서서히 증가 및 수렴한다.


Ctrl + C 시 학습 또는 추론이 중단되며, Total Reward 창을 닫으면 프로그램이 종료된다.

종료 시 학습이 완료된 모델(dqn_model.pth), Total Reward 그래프와, 해당 Total Reward들의 실제 값을 기록한 텍스트파일이 아래 위치에 저장된다.

- app > results > sumo > training
   
&nbsp;

**모델 추론**

1. 학습한 모델로 추론
   
```
python control.py --simulator sumo
// or python control.py -s sumo 
```

실행 시 위의 모델 학습과 동일한 창들이 켜지며, 이번에는 미리 학습된 모델이 교차로의 신호를 제어하는 것을 볼 수 있다. 

이에 대한 Reward도 마찬가지로 Total Reward창에 나타난다.

2. 고정 주기식 신호로 추론 (모델 미사용, 고정주기식 제어)
   
```
python control.py --simulator sumo --fixed
// or python control.py -s sumo -f
```

또는 학습한 모델과의 비교를 위해 고정주기식 신호로 테스트를 할 수 있다.

Ctrl + C 시 학습 또는 추론이 중단되며, Total Reward 창을 닫으면 프로그램이 종료된다.

종료 시 Total Reward 그래프와 해당 Total Reward들의 실제 값을 기록한 텍스트파일이 아래 위치에 저장된다.

- app > results > sumo > inference

&nbsp;
---------------------------------------

### 6. 팀 소개

**권오성 200628104 (gbkos@pusan.ac.kr)**
- DQN(Deep Q-Network)의 구조적 설계 담당

**이준표 202155652 (junpyo319@gmail.com)**
- CARLA 시뮬레이터로 교차로 환경을 구현하고 API 제작

**정하립 201924578 (jeonghalib@gmail.com)**
- SUMO 교차로 환경구현 및 객체탐지 모델을 활용한 Reward 알고리즘 제작
