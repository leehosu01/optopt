다음 업데이트 포함해야하는 목록: (2022.10 예상?)

## Variable_definer에서는
-----------------
1. (init value), (min / max init value), (custom init callable)
2. shift method ( add or mult(이 경우 log로 관리됨) )
3. min / max value
4. hp 랜덤 초기화는 env가 아닌 이것이 관리한다. env에서 호출을 요구한다.
5. 다만, set_value시점과 같이 Manager에서 set_hyperparameter을 하도록한다.
5. 현재의 set_value는 optuna검증비교 코드와의 호환을 위해 놔둔다 (min / max value 범주에서 관리)
6. set_value 대신 shift_value가 사용된다. shift 방식은 전부 이 클래스에서 관리
7. optopt.Manager에서 observation시 값을 갖고간다. shift method별로 관리되는 방식을 따른다. 

## env 관측값문제
-------------
1. env의 관측값에는 hyper parameter를 포함한다. 물론 정규화될 예정
2. 시작값은 전체가 결측이라는 문제를 갖고 있다. 이것을 input drop & is_null 로 분리해서 입력해주자. 

## optimizer 문제
----------------
논문에 포함된 feature을 구현하는건 어렵지 않지만? CAIO optimizer 대로 가야하는지에 대한 문제가 남긴한다.

논문에서도 action으로 관리하는 변수가 많지는 않기에 무관할수도 ... (beta2, epsilon, learning rate, weight decay)

> ### custom optimizer 
> -------------
> 1. std < eps 비율 같은 추적이 가능하다
2. 여하튼 cos norm 같은 것을 추적 가능
-1. optimizer 선택이 제한됨 (위 기능을 쓰고 싶었다면)
>
> ### 없는경우
> ------------
> -1. 추가 특성없다.
> 1. 선택은 자유롭다. (선택지가 줄었을뿐)

grad <-> momentum 의 cosnorm은 중요할 것 같긴 하다.

1. 이것은 optopt optimizer을 쓸지 말지를 선택 가능하게 한다.
2. 선택할 경우 추가된 특성값 전달 과정의 표준화를 위해 metric으로 넘겨야하는데 가능한가?

> * opt = optimizer_wrapper(subopt, model = None)# if model != None & model 은 model wrapper형 => metric 추가작업
> * model.compile(optimizer = opt)
> * optimizer_wrapper에서 get_slot으로 name을 모아서 model_wrapper로 해당 exp_moving_metric 생성을 요청함.
> 
> * 각 metric은 3종류의 momentum을 갖는다. 0.9, 0.99, 0.999. 
> * 나머지 함수는 놔두고, apply_gradients에서만 특성 추출을 위한 연산을 추가후 진행한다.
> * 추가로 변수를 만들지 않을 여지가 있는경우를 고려하여 (ex momentum), lazy 하게 추가 생성(& 관리) 하고 사용

>> momentum이 있으면 제공하고 없으면 포기한다. 또한, 모델이 알아서 어느정도 감안할테니, momentum 변화후 cosnorm한다. 

3. 오히려 metric wrapper에서 묶어주는게 구현상 효율적일수도 있다. (하지만, 중복 계산은 싫다)

설계는 optimzier만 최적화하는게 아니고, 모든 종류의 parameter을 조정하는것을 염두한다. (ex, label smoothing, dropout rate, ...)
tensorflow keras 구현상 rate 에 tf.Variables를 제공못한다는게 우습지만, 직접 layer로 구현하면 사용가능하게라도 할수있게 한다는 목표.

## agent문제
------------------------------
논문에서는 PPO를 사용하고, 이것에서는 TD3를 사용한다.

아직 PPO는 시도하지 못했고, SAC보다는 TD3가 압도적이었다. 평균적으로 50 episod전에 성능 향상이 시작된다.

1. TPU 미작동 원인 찾기
2. 여유가 된다면, PPO도 실험? 

## 실제 활용시
------------------------
1. 신규수집시 마다 1배치 학습
2. 기본적으로 100회에 random scheduling(dynamic) = 0.44, optuna (static) = 0.426, optopt(dynamic) 는 0.473까지 도달가능이지만, 제시된 구현으로 충분히(?) 0.49~0.50 까지 도달할것임

## 기타.
1. strategy는 사용자가 optopt 관련 코드 전체를 scope에 감싸면 되기때문에 strategy관련을 제거