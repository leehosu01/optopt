0.4.0
TPU compatible
> 1. dtype 자동 적응 -> 
>> * 아마도 metrics들 cast확인
>> * exp normal layer확인 필요
> 2. tpu에서 save 불가
>> * 기본적으로 cpu strategy 를 config 로


0.5.0
optimization with optuna
1. find initial condition by optuna
2. provide pre defined discontinuity informations to agent (units or do not controlled value)
3. optimize continuity values
