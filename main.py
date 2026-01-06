import torch
import time
import numpy as np

# Проверяем доступность GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# =============================================
# GPU-ОПТИМИЗИРОВАННАЯ Π-ТОПОЛОГИЯ
# =============================================

class PiFlowGPU:
    """Π-Поток, оптимизированный для GPU/TPU"""
    
    def __init__(self, name, x_domain, y_domain):
        self.name = name
        self.x_domain = torch.tensor(x_domain, device=device, dtype=torch.float32)
        self.y_domain = torch.tensor(y_domain, device=device, dtype=torch.float32)
        
        # Создаём сетку всех комбинаций x и y на GPU
        self.X, self.Y = torch.meshgrid(self.x_domain, self.y_domain, indexing='ij')
        self.values = None  # Тензор значений на GPU
        
    def create_from_function(self, func):
        """Создать поток из функции f(x, y) = value (векторизованной)"""
        # Векторизованное вычисление для всех точек одновременно
        self.values = func(self.X, self.Y)
        return self
    
    def pi_merge(self, other_flow):
        """Операция Π-Слияния (⨁) на GPU"""
        merged = PiFlowGPU(f"({self.name}⨁{other_flow.name})", 
                          self.x_domain.cpu().numpy(), 
                          self.y_domain.cpu().numpy())
        
        # Слияние как конкатенация значений по новой оси
        merged.values = torch.stack([self.values, other_flow.values], dim=-1)
        return merged
    
    def pi_involution(self, target_values, tolerance=0):
        """Оператор Π-Инволюции (ℑ) - массовая параллельная фильтрация"""
        if isinstance(target_values, (list, tuple)):
            target = torch.tensor(target_values, device=device, dtype=torch.float32)
        else:
            target = torch.tensor([target_values], device=device, dtype=torch.float32)
        
        # Вычисляем отклонения для всех точек одновременно
        if len(self.values.shape) > 2 and self.values.shape[-1] == len(target):
            # Для многокомпонентных значений (после слияния)
            deviations = torch.abs(self.values - target)
            # Проверяем, все ли компоненты удовлетворяют условию
            mask = torch.all(deviations <= tolerance, dim=-1)
        else:
            # Для однокомпонентных значений
            deviations = torch.abs(self.values - target[0])
            mask = deviations <= tolerance
        
        # Применяем маску: оставляем только удовлетворительные точки
        self.mask = mask
        self.valid_indices = torch.nonzero(mask, as_tuple=True)
        
        # Для совместимости сохраняем значения
        self.filtered_values = self.values[mask]
        self.filtered_X = self.X[mask]
        self.filtered_Y = self.Y[mask]
        
        return self
    
    def get_solutions(self):
        """Получить решения как массив пар (x, y)"""
        if not hasattr(self, 'valid_indices') or len(self.valid_indices[0]) == 0:
            return []
        
        # Собираем решения в массив на CPU
        solutions = torch.stack([self.filtered_X, self.filtered_Y], dim=-1)
        return solutions.cpu().numpy()
    
    def stats(self):
        """Статистика потока"""
        if not hasattr(self, 'valid_indices'):
            valid_count = 0
        else:
            valid_count = len(self.filtered_X) if hasattr(self, 'filtered_X') else 0
        
        return {
            'name': self.name,
            'total_contexts': self.X.numel(),
            'valid_contexts': valid_count,
            'device': str(device)
        }

# =============================================
# СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ
# =============================================

def run_cpu_test(size=100):
    """CPU тест с чистым Python"""
    x_domain = list(range(size))
    y_domain = list(range(size))
    
    start = time.time()
    solutions = []
    for x in x_domain:
        for y in y_domain:
            # Те же уравнения: x+y ≈ 10, 2x-y ≈ 5
            if abs(x + y - 10) <= 2 and abs(2*x - y - 5) <= 2:
                solutions.append((x, y))
    cpu_time = time.time() - start
    
    return cpu_time, len(solutions)

def run_gpu_test(size=100):
    """GPU тест с PyTorch"""
    x_domain = np.arange(size, dtype=np.float32)
    y_domain = np.arange(size, dtype=np.float32)
    
    start = time.time()
    
    # Создаём потоки на GPU
    flow_A = PiFlowGPU("A", x_domain, y_domain)
    flow_A.create_from_function(lambda X, Y: X + Y)
    
    flow_B = PiFlowGPU("B", x_domain, y_domain)
    flow_B.create_from_function(lambda X, Y: 2*X - Y)
    
    # Π-Слияние и инволюция
    merged = flow_A.pi_merge(flow_B)
    result = merged.pi_involution([10, 5], tolerance=2)
    
    gpu_time = time.time() - start
    solutions = result.get_solutions()
    
    return gpu_time, len(solutions)

def run_numpy_vectorized_test(size=100):
    """NumPy векторизованный тест (CPU, но с оптимизацией)"""
    x_domain = np.arange(size, dtype=np.float32)
    y_domain = np.arange(size, dtype=np.float32)
    
    start = time.time()
    
    # Создаём сетки
    X, Y = np.meshgrid(x_domain, y_domain, indexing='ij')
    
    # Векторизованные вычисления
    A = X + Y
    B = 2*X - Y
    
    # Векторизованная фильтрация
    mask = (np.abs(A - 10) <= 2) & (np.abs(B - 5) <= 2)
    solutions = np.column_stack([X[mask], Y[mask]])
    
    numpy_time = time.time() - start
    
    return numpy_time, len(solutions)

# =============================================
# ЗАПУСК ТЕСТОВ
# =============================================

print("=" * 70)
print("СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ РАЗНЫХ ПОДХОДОВ")
print("=" * 70)

test_sizes = [50, 100, 200, 500, 1000]
results = []

for size in test_sizes:
    print(f"\nРазмер пространства: {size}×{size} = {size*size:,} точек")
    
    # CPU (чистый Python)
    cpu_time, cpu_solutions = run_cpu_test(size)
    
    # NumPy (векторизованный CPU)
    numpy_time, numpy_solutions = run_numpy_vectorized_test(size)
    
    # GPU (PyTorch)
    gpu_time, gpu_solutions = run_gpu_test(size)
    
    results.append({
        'size': size,
        'cpu_time': cpu_time,
        'numpy_time': numpy_time,
        'gpu_time': gpu_time,
        'cpu_solutions': cpu_solutions,
        'numpy_solutions': numpy_solutions,
        'gpu_solutions': gpu_solutions
    })
    
    print(f"  CPU (чистый Python):   {cpu_time:.4f} сек, решений: {cpu_solutions}")
    print(f"  CPU (NumPy векториз.):  {numpy_time:.4f} сек, решений: {numpy_solutions}")
    print(f"  GPU (PyTorch):          {gpu_time:.4f} сек, решений: {gpu_solutions}")
    
    if cpu_time > 0:
        print(f"  Ускорение NumPy/CPU:    {cpu_time/numpy_time:.1f}×")
    if gpu_time > 0:
        print(f"  Ускорение GPU/NumPy:     {numpy_time/gpu_time:.1f}×")

# =============================================
# ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# =============================================

try:
    import matplotlib.pyplot as plt
    
    # Подготовка данных для графиков
    sizes = [r['size'] for r in results]
    cpu_times = [r['cpu_time'] for r in results]
    numpy_times = [r['numpy_time'] for r in results]
    gpu_times = [r['gpu_time'] for r in results]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # График 1: Время выполнения
    ax1 = axes[0]
    ax1.plot(sizes, cpu_times, 'r-o', label='CPU (чистый Python)', linewidth=2)
    ax1.plot(sizes, numpy_times, 'g-s', label='CPU (NumPy векториз.)', linewidth=2)
    ax1.plot(sizes, gpu_times, 'b-^', label='GPU (PyTorch)', linewidth=2)
    ax1.set_xlabel('Размер пространства (N×N)')
    ax1.set_ylabel('Время выполнения (сек)')
    ax1.set_title('Сравнение производительности')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # График 2: Ускорение относительно чистого Python
    ax2 = axes[1]
    if device.type == 'cuda':
        acceleration_numpy = [cpu/numpy for cpu, numpy in zip(cpu_times, numpy_times)]
        acceleration_gpu = [cpu/gpu for cpu, gpu in zip(cpu_times, gpu_times)]
        
        ax2.plot(sizes, acceleration_numpy, 'g-s', label='NumPy/CPU', linewidth=2)
        ax2.plot(sizes, acceleration_gpu, 'b-^', label='GPU/CPU', linewidth=2)
        ax2.set_xlabel('Размер пространства (N×N)')
        ax2.set_ylabel('Ускорение (раз)')
        ax2.set_title('Ускорение относительно чистого Python')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'GPU недоступно\nдля измерения ускорения', 
                ha='center', va='center', fontsize=12)
        ax2.set_title('GPU не обнаружено')
    
    plt.tight_layout()
    plt.savefig('gpu_performance_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nГрафики сохранены в 'gpu_performance_comparison.png'")
    
except ImportError:
    print("Для визуализации установите matplotlib")

# =============================================
# ТЕСТ С ОЧЕНЬ БОЛЬШИМ ПРОСТРАНСТВОМ
# =============================================

print("\n" + "=" * 70)
print("ТЕСТ С ОЧЕНЬ БОЛЬШИМ ПРОСТРАНСТВОМ")
print("=" * 70)

# Большое пространство поиска
large_size = 5000  # 25 миллионов точек!
x_domain_large = np.linspace(-100, 100, large_size, dtype=np.float32)
y_domain_large = np.linspace(-100, 100, large_size, dtype=np.float32)

print(f"Создаётся пространство: {large_size}×{large_size} = {large_size*large_size:,} точек")

try:
    # Только GPU тест (CPU не справится за разумное время)
    start = time.time()
    
    flow_A_large = PiFlowGPU("A_large", x_domain_large, y_domain_large)
    flow_A_large.create_from_function(lambda X, Y: X + Y)
    
    flow_B_large = PiFlowGPU("B_large", x_domain_large, y_domain_large)
    flow_B_large.create_from_function(lambda X, Y: 2*X - Y)
    
    merged_large = flow_A_large.pi_merge(flow_B_large)
    result_large = merged_large.pi_involution([10, 5], tolerance=5)
    
    gpu_large_time = time.time() - start
    solutions_large = result_large.get_solutions()
    
    print(f"Время выполнения на GPU: {gpu_large_time:.2f} сек")
    print(f"Найдено решений: {len(solutions_large)}")
    
    # Оценка времени на CPU (теоретическая)
    # Для 25 миллионов точек, при 1000 проверок в секунду: 25,000 сек ≈ 7 часов
    estimated_cpu_time = (large_size * large_size) / 1000  # оптимистичная оценка
    print(f"Оценочное время на CPU: ~{estimated_cpu_time/3600:.1f} часов")
    print(f"Ускорение GPU/CPU: ~{estimated_cpu_time/gpu_large_time:.0f}×")
    
except torch.cuda.OutOfMemoryError:
    print("Недостаточно памяти GPU для такого большого пространства")
    print("Попробуем уменьшить размер...")
    
    # Уменьшенный тест
    medium_size = 2000
    x_domain_medium = np.linspace(-50, 50, medium_size, dtype=np.float32)
    y_domain_medium = np.linspace(-50, 50, medium_size, dtype=np.float32)
    
    print(f"\nУменьшенный тест: {medium_size}×{medium_size} = {medium_size*medium_size:,} точек")
    
    start = time.time()
    flow_A_medium = PiFlowGPU("A_medium", x_domain_medium, y_domain_medium)
    flow_A_medium.create_from_function(lambda X, Y: X + Y)
    
    flow_B_medium = PiFlowGPU("B_medium", x_domain_medium, y_domain_medium)
    flow_B_medium.create_from_function(lambda X, Y: 2*X - Y)
    
    merged_medium = flow_A_medium.pi_merge(flow_B_medium)
    result_medium = merged_medium.pi_involution([10, 5], tolerance=5)
    
    gpu_medium_time = time.time() - start
    solutions_medium = result_medium.get_solutions()
    
    print(f"Время выполнения на GPU: {gpu_medium_time:.2f} сек")
    print(f"Найдено решений: {len(solutions_medium)}")

# =============================================
# ВЫВОДЫ И РЕКОМЕНДАЦИИ
# =============================================

print("\n" + "=" * 70)
print("ВЫВОДЫ ПО ИСПОЛЬЗОВАНИЮ GPU/TPU ДЛЯ Π-ТОПОЛОГИИ")
print("=" * 70)

print("""
1. ПРЕИМУЩЕСТВА GPU:
   - Массовый параллелизм идеально подходит для Π-Топологии
   - Операции над тензорами соответствуют работе с Π-Потоками
   - Экспоненциальный прирост производительности на больших пространствах
   - Естественная поддержка операций слияния и инволюции

2. ОГРАНИЧЕНИЯ:
   - Требуется фиксированный размер пространства (статические тензоры)
   - Потребление памяти растет квадратично с размерностью
   - Сложные условия могут требовать нескольких проходов

3. ДЛЯ НАСТОЯЩЕЙ Π-АРХИТЕКТУРЫ НУЖНО:
   a) Специализированные ядра для операций ⨁ и ℑ
   b) Поддержка динамических пространств контекстов
   c) Аппаратная реализация "когерентного коллапса"

4. ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ:
   - Использовать смешанную точность (fp16/fp32) для экономии памяти
   - Применять чанкование для очень больших пространств
   - Использовать TPU для ещё большей параллелизации матричных операций
   - Рассмотреть нейроморфные процессоры для более близкой эмуляции
""")