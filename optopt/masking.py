import warnings
import tensorflow as tf
import time
def binominal_based_masking(inputs, prob):
    mask = tf.random.stateless_binomial(tf.shape(inputs), seed = [142, 332], probs = 1 - prob, counts = 1, output_dtype=inputs.dtype)
    return inputs * mask
    
def uniform_float16_based_masking(inputs, prob):
    mask = tf.greater(tf.random.uniform(tf.shape(inputs), 0, 1, dtype = tf.float16), prob)
    return inputs * tf.cast(mask, inputs.dtype)
    
def uniform_float32_based_masking(inputs, prob):
    mask = tf.greater(tf.random.uniform(tf.shape(inputs), 0, 1, dtype = tf.float32), prob)
    return inputs * tf.cast(mask, inputs.dtype)
    
def uniform_int_based_masking(inputs, prob):
    mask = tf.greater(tf.random.uniform(tf.shape(inputs), 0, int(100//prob), dtype = tf.int32), 100)
    return inputs * tf.cast(mask, inputs.dtype)

@tf.function
def tf_binominal_based_masking(inputs, prob):
    mask = tf.random.stateless_binomial(tf.shape(inputs), seed = [142, 332], probs = 1 - prob, counts = 1, output_dtype=inputs.dtype)
    return inputs * mask
@tf.function
def tf_uniform_float16_based_masking(inputs, prob):
    mask = tf.greater(tf.random.uniform(tf.shape(inputs), 0, 1, dtype = tf.float16), prob)
    return inputs * tf.cast(mask, inputs.dtype)
@tf.function
def tf_uniform_float32_based_masking(inputs, prob):
    mask = tf.greater(tf.random.uniform(tf.shape(inputs), 0, 1, dtype = tf.float32), prob)
    return inputs * tf.cast(mask, inputs.dtype)
@tf.function
def tf_uniform_int_based_masking(inputs, prob):
    mask = tf.greater(tf.random.uniform(tf.shape(inputs), 0, int(100//prob), dtype = tf.int32), 100)
    return inputs * tf.cast(mask, inputs.dtype)

def _masking(shape, dtype):
    def find_best_fn():
        def time_test(fn, inputs, prob):
            spend = []
            for _ in range(10):
                start = time.perf_counter()
                fn(inputs, prob)
                spend.append(time.perf_counter() - start)
            reduced = sorted(spend[3:])[2:-2]
            return sum(reduced) / len(reduced)
        if time.perf_counter() == time.perf_counter():
            warnings.warn("system have low time precision, we use default function")
            return _masking.default_fn
        inputs = tf.ones(shape, dtype = dtype)
        prob = 0.05
        best_time = float('inf')
        for FN in _masking.fns:
            est_time = time_test(FN, inputs, prob)
            print(f"{FN.__name__} takes {est_time:e} seconds")
            if est_time < best_time:
                best_time = est_time
                best_fn = FN
        print(f"selected: {best_fn.__name__} takes {best_time:e} seconds")
        return best_fn
    if (shape, dtype) not in _masking.cache:
        _masking.cache[(shape, dtype)] = find_best_fn()
    return _masking.cache[(shape, dtype)]
_masking.cache = {}
_masking.fns = [
      #tf_binominal_based_masking,
       tf_uniform_float16_based_masking,
       tf_uniform_float32_based_masking,
       tf_uniform_int_based_masking,
      #binominal_based_masking,
       uniform_float16_based_masking,
       uniform_float32_based_masking,
       uniform_int_based_masking]
_masking.default_fn = uniform_float16_based_masking
def masking(inputs, is_null__mask, masking_rate, provide_is_null = True, dynamic = False):
    #if null, mark as one
    #if no null in input, provide None 
    #if provide_is_null == False, provide "is not null infomation"
    if is_null__mask is None: is_not_null__mask = tf.ones_like(inputs, dtype = inputs.dtype)
    else: is_not_null__mask = 1 - is_null__mask
    inputs = inputs * is_not_null__mask
    if dynamic:
        try: mask_generate_fn = _masking(tuple(tf.shape(inputs).numpy()), inputs.dtype)#average = 1 - masking_rate
        except: mask_generate_fn = _masking.default_fn
    else: mask_generate_fn = _masking.default_fn
    mask = mask_generate_fn(inputs, masking_rate)
    masked_input = inputs * mask
    masked_not_null = is_not_null__mask * mask
    if provide_is_null: masked_null_method = 1 - masked_not_null
    else: masked_null_method = masked_not_null

    return tf.concat([masked_input, masked_null_method], axis = -1)