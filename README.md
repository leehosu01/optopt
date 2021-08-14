## optopt
optimizer optimizing tools

--------------------

motive by [LHOPT](https://arxiv.org/pdf/2106.00958.pdf)



## ToDo
------------------

[X] (apply at 0.3.2) add skipped reward at reset next step
[ ] (apply at 0.3.3) optimizer_metrics_wrapper에서 momentum / variance 가져오는 optimizer와 훈련에 사용되는  optimizer분리기능. ( ex, Lookahead(Radam) / mixed_precision.LossScaleOptimizer(optimizer) )

[ ] (apply at 0.4.0) manager <-> agent TCP로 분리
- [ ] TPU compatible
- [ ] mixed_float16 (GPU) compatible
- [ ] mixed_bfloat16 (TPU) compatible

[ ] (apply at 1.0.0) improve with optuna ? 
[ ] compare with optuna (static vs dynamic)
