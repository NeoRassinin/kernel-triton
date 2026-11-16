# kernel-triton 

# 1. INT4 Weight Quantization with Triton

Реализация квантизации весов FP16/BF16 → INT4 с упаковкой в INT8 на Triton.

## Реализация

Используется симметричная per-row квантизация:

- Signed INT4, диапазон `[-7, 7]`.
- Два INT4-значения упаковываются в один INT8.
- Упаковка вдоль `in_features`: первый элемент в младшие 4 бита, второй — в старшие.
- Отрицательные значения хранятся в дополнительном коде.
- Scale вычисляется отдельно для каждой строки, zero-point = 0.

Формула масштабирования:

`scale[row] = max(abs(W[row, :])) / 7`

Для нулевых строк `scale = 1`.

Восстановление весов:

`W_dequant[row, k] = q[row, k] * scale[row]`

## Формат данных

```python
packed, scales = quantize_fp16_to_int4_packed(W)

# W:      float16 [out_features, in_features]
# packed: int8 [out_features, in_features // 2]
# scales: float32 [out_features]
```

Реализованы два варианта Triton-кернелей: `two_pass` и `fused`.

## Результаты

Тестирование на NVIDIA Tesla T4.

| Матрица | two_pass | fused |
|---|---:|---:|
| 2048×2048 | 0.092 мс | 0.097 мс |
| 8192×2048 | 0.333 мс | 0.380 мс |
| 8192×8192 | 1.377 мс | 2.494 мс |

- Сжатие весов: **3.984×** для матрицы 8192×2048 с учётом scales.
- Проверено побайтовое совпадение упаковки с PyTorch reference.
- Пройдены тесты FP16/BF16, нулевых строк и граничных значений.

## Запуск

Открыть `notebooks/01_quantization.ipynb` в Google Colab, выбрать GPU и выполнить все ячейки.

Результаты тестов и бенчмарков сохранены в `benchmarks/int4_task1_results.json`.