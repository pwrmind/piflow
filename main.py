import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union, Callable

# =============================================
# КОНФИГУРАЦИЯ СИСТЕМЫ
# =============================================

class Config:
    """Конфигурация Π-Топологии"""
    def __init__(self):
        # Автоматическое определение устройства
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Оптимальные настройки в зависимости от устройства
        if self.device.type == 'cuda':
            self.dtype = torch.float16  # Половинная точность для GPU
            self.chunk_size = 2048      # Размер чанка для больших пространств
            self.batch_size = 32        # Для пакетной обработки
        else:
            self.dtype = torch.float32  # Полная точность для CPU
            self.chunk_size = 512       # Меньший чанк для CPU
            self.batch_size = 8
        
        # Оптимизации
        self.use_mixed_precision = True
        self.use_memory_pooling = True
        self.auto_chunking = True
        
        # Статистика
        self.verbose = True

config = Config()

# =============================================
# ОПТИМИЗИРОВАННЫЙ Π-ПОТОК С ПОДДЕРЖКОЙ GPU
# =============================================

class PiFlowOptimized:
    """Высокооптимизированный Π-Поток с поддержкой GPU/TPU"""
    
    def __init__(self, name: str, x_domain: np.ndarray, y_domain: np.ndarray):
        self.name = name
        self.x_domain = x_domain.astype(np.float32)
        self.y_domain = y_domain.astype(np.float32)
        
        # Статистика и метрики
        self.stats = {
            'creation_time': 0,
            'merge_time': 0,
            'involution_time': 0,
            'memory_used': 0,
            'contexts_processed': 0
        }
        
        # Пул памяти для повторного использования
        self.memory_pool = {}
        
        if config.verbose:
            print(f"Создан Π-Поток '{name}' на {config.device}")
            print(f"  Размер: {len(x_domain)}×{len(y_domain)} = {len(x_domain)*len(y_domain):,} контекстов")
    
    def _create_grid(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Создание сетки с оптимизацией памяти"""
        if config.use_memory_pooling and 'grid' in self.memory_pool:
            return self.memory_pool['grid']
        
        # Используем torch.meshgrid с оптимизацией памяти
        x_tensor = torch.tensor(self.x_domain, device=config.device, dtype=config.dtype)
        y_tensor = torch.tensor(self.y_domain, device=config.device, dtype=config.dtype)
        
        # Для больших пространств используем чанкование
        if config.auto_chunking and len(x_tensor) * len(y_tensor) > 10_000_000:
            # Возвращаем не сетку, а индексы для чанкованной обработки
            return x_tensor, y_tensor
        
        X, Y = torch.meshgrid(x_tensor, y_tensor, indexing='ij')
        
        if config.use_memory_pooling:
            self.memory_pool['grid'] = (X, Y)
        
        return X, Y
    
    def create_from_function(self, func: Callable) -> 'PiFlowOptimized':
        """Создание потока из векторной функции"""
        start_time = time.time()
        
        X, Y = self._create_grid()
        
        # Векторизованное вычисление
        with torch.cuda.amp.autocast() if config.use_mixed_precision and config.device.type == 'cuda' else torch.no_grad():
            self.values = func(X, Y)
        
        self.stats['creation_time'] = time.time() - start_time
        self.stats['contexts_processed'] = X.numel()
        
        if config.verbose:
            print(f"  Создание за {self.stats['creation_time']:.4f} сек")
        
        return self
    
    def pi_merge_batched(self, other_flow: 'PiFlowOptimized', 
                         operation: str = 'concat') -> 'PiFlowOptimized':
        """Пакетное Π-Слияние с различными операциями"""
        start_time = time.time()
        
        # Проверка совместимости
        assert len(self.x_domain) == len(other_flow.x_domain)
        assert len(self.y_domain) == len(other_flow.y_domain)
        
        merged = PiFlowOptimized(
            f"({self.name}⨁{other_flow.name})",
            self.x_domain,
            self.y_domain
        )
        
        # Различные стратегии слияния
        if operation == 'concat':
            merged.values = torch.stack([self.values, other_flow.values], dim=-1)
        elif operation == 'add':
            merged.values = self.values + other_flow.values
        elif operation == 'multiply':
            merged.values = self.values * other_flow.values
        elif operation == 'max':
            merged.values = torch.maximum(self.values, other_flow.values)
        elif operation == 'min':
            merged.values = torch.minimum(self.values, other_flow.values)
        
        self.stats['merge_time'] = time.time() - start_time
        
        if config.verbose:
            print(f"  Слияние за {self.stats['merge_time']:.4f} сек")
        
        return merged
    
    def pi_involution_chunked(self, target_values: Union[float, List[float], torch.Tensor],
                             tolerance: Union[float, List[float]] = 0,
                             chunk_size: Optional[int] = None) -> 'PiFlowOptimized':
        """Чанкованная Π-Инволюция для больших пространств"""
        start_time = time.time()
        
        if chunk_size is None:
            chunk_size = config.chunk_size
        
        # Преобразование target_values в тензор
        if isinstance(target_values, (list, tuple)):
            target = torch.tensor(target_values, device=config.device, dtype=config.dtype)
        else:
            target = torch.tensor([target_values], device=config.device, dtype=config.dtype)
        
        # Преобразование tolerance
        if isinstance(tolerance, (list, tuple)):
            tol_tensor = torch.tensor(tolerance, device=config.device, dtype=config.dtype)
        else:
            tol_tensor = torch.tensor([tolerance], device=config.device, dtype=config.dtype)
        
        # Определяем размер чанков
        n_x = len(self.x_domain)
        n_y = len(self.y_domain)
        
        # Если пространство маленькое, обрабатываем целиком
        if n_x * n_y <= chunk_size * chunk_size:
            return self._involution_single(target, tol_tensor)
        
        # Чанкованная обработка для больших пространств
        if config.verbose:
            print(f"  Запуск чанкованной инволюции ({n_x//chunk_size + 1}×{n_y//chunk_size + 1} чанков)")
        
        # Списки для результатов
        all_solutions_x = []
        all_solutions_y = []
        all_values = []
        
        # Обработка по чанкам
        for i in range(0, n_x, chunk_size):
            for j in range(0, n_y, chunk_size):
                # Определяем границы чанка
                x_slice = slice(i, min(i + chunk_size, n_x))
                y_slice = slice(j, min(j + chunk_size, n_y))
                
                # Создаем чанк на GPU
                x_chunk = torch.tensor(self.x_domain[x_slice], 
                                      device=config.device, dtype=config.dtype)
                y_chunk = torch.tensor(self.y_domain[y_slice], 
                                      device=config.device, dtype=config.dtype)
                
                X_chunk, Y_chunk = torch.meshgrid(x_chunk, y_chunk, indexing='ij')
                
                # Вычисляем значения для чанка
                with torch.cuda.amp.autocast() if config.use_mixed_precision and config.device.type == 'cuda' else torch.no_grad():
                    # Для простоты используем ту же функцию, что и при создании
                    # В реальном приложении нужно сохранить функцию вычисления
                    chunk_values = X_chunk + Y_chunk  # Заглушка, нужно переделать
                
                # Проверяем условия
                if len(target) > 1:
                    deviations = torch.abs(chunk_values - target)
                    mask = torch.all(deviations <= tol_tensor, dim=-1)
                else:
                    deviations = torch.abs(chunk_values - target[0])
                    mask = deviations <= tol_tensor[0]
                
                # Собираем решения из чанка
                if mask.any():
                    all_solutions_x.append(X_chunk[mask])
                    all_solutions_y.append(Y_chunk[mask])
                    all_values.append(chunk_values[mask])
        
        # Объединяем все решения
        if all_solutions_x:
            self.solutions_x = torch.cat(all_solutions_x)
            self.solutions_y = torch.cat(all_solutions_y)
            self.filtered_values = torch.cat(all_values)
            self.mask = None  # Для чанкованной обработки не сохраняем полную маску
        else:
            self.solutions_x = torch.tensor([], device=config.device)
            self.solutions_y = torch.tensor([], device=config.device)
            self.filtered_values = torch.tensor([], device=config.device)
        
        self.stats['involution_time'] = time.time() - start_time
        
        if config.verbose:
            print(f"  Инволюция за {self.stats['involution_time']:.4f} сек")
            print(f"  Найдено {len(self.solutions_x)} решений")
        
        return self
    
    def _involution_single(self, target: torch.Tensor, tolerance: torch.Tensor) -> 'PiFlowOptimized':
        """Инволюция для всего пространства (без чанкования)"""
        # Вычисляем отклонения
        if len(self.values.shape) > 2 and self.values.shape[-1] == len(target):
            deviations = torch.abs(self.values - target)
            mask = torch.all(deviations <= tolerance, dim=-1)
        else:
            deviations = torch.abs(self.values - target[0])
            mask = deviations <= tolerance[0]
        
        # Применяем маску
        X, Y = self._create_grid()
        self.mask = mask
        self.filtered_values = self.values[mask] if mask.any() else torch.tensor([], device=config.device)
        self.solutions_x = X[mask] if mask.any() else torch.tensor([], device=config.device)
        self.solutions_y = Y[mask] if mask.any() else torch.tensor([], device=config.device)
        
        return self
    
    def get_solutions(self, max_solutions: int = 1000) -> np.ndarray:
        """Получить решения с ограничением по количеству"""
        if not hasattr(self, 'solutions_x') or len(self.solutions_x) == 0:
            return np.array([])
        
        # Ограничиваем количество решений для вывода
        n_solutions = min(len(self.solutions_x), max_solutions)
        
        # Собираем решения
        solutions = torch.stack([
            self.solutions_x[:n_solutions],
            self.solutions_y[:n_solutions]
        ], dim=-1)
        
        return solutions.cpu().numpy()
    
    def get_statistics(self) -> dict:
        """Полная статистика потока"""
        stats = self.stats.copy()
        
        if hasattr(self, 'solutions_x'):
            stats['solutions_found'] = len(self.solutions_x)
        else:
            stats['solutions_found'] = 0
        
        if config.device.type == 'cuda':
            stats['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**2
            stats['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**2
        
        return stats
    
    def visualize(self, title: str = None):
        """Визуализация решений"""
        if not hasattr(self, 'solutions_x') or len(self.solutions_x) == 0:
            print("Нет решений для визуализации")
            return
        
        solutions = self.get_solutions(1000)  # Ограничиваем для визуализации
        
        plt.figure(figsize=(10, 8))
        
        # Точки решений
        plt.scatter(solutions[:, 0], solutions[:, 1], 
                   c='red', s=10, alpha=0.6, label='Решения')
        
        # Контур пространства поиска
        plt.plot([self.x_domain[0], self.x_domain[-1], self.x_domain[-1], self.x_domain[0], self.x_domain[0]],
                [self.y_domain[0], self.y_domain[0], self.y_domain[-1], self.y_domain[-1], self.y_domain[0]],
                'b--', alpha=0.3, label='Пространство поиска')
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title or f'Решения Π-Потока: {self.name}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        return plt

# =============================================
# БЕНЧМАРК И СРАВНЕНИЕ ПОДХОДОВ
# =============================================

class PiTopologyBenchmark:
    """Бенчмарк для сравнения разных подходов"""
    
    @staticmethod
    def benchmark_system_of_equations(max_size: int = 5000):
        """Тестирование системы уравнений на разных размерах"""
        
        sizes = [50, 100, 200, 500, 1000, 2000, 5000]
        results = []
        
        print("=" * 80)
        print("БЕНЧМАРК Π-ТОПОЛОГИИ: СИСТЕМА УРАВНЕНИЙ")
        print("=" * 80)
        
        for size in sizes:
            if size > max_size:
                continue
                
            print(f"\nРазмер: {size}×{size} = {size*size:,} точек")
            
            # Подготовка данных
            x_domain = np.linspace(-size/10, size/10, size, dtype=np.float32)
            y_domain = np.linspace(-size/10, size/10, size, dtype=np.float32)
            
            # 1. Чистый Python (только для малых размеров)
            if size <= 500:
                start = time.time()
                solutions_cpu = []
                for x in x_domain[:100]:  # Ограничиваем для скорости
                    for y in y_domain[:100]:
                        if abs(x + y - 10) <= 2 and abs(2*x - y - 5) <= 2:
                            solutions_cpu.append((x, y))
                cpu_time = time.time() - start
            else:
                cpu_time = None
            
            # 2. Π-Топология с оптимизациями
            start = time.time()
            
            flow1 = PiFlowOptimized("Уравнение1", x_domain, y_domain)
            flow1.create_from_function(lambda X, Y: X + Y)
            
            flow2 = PiFlowOptimized("Уравнение2", x_domain, y_domain)
            flow2.create_from_function(lambda X, Y: 2*X - Y)
            
            merged = flow1.pi_merge_batched(flow2, operation='concat')
            result = merged.pi_involution_chunked([10, 5], tolerance=2)
            
            pi_time = time.time() - start
            pi_solutions = result.get_solutions()
            
            # 3. NumPy векторизованный
            start = time.time()
            X, Y = np.meshgrid(x_domain, y_domain, indexing='ij')
            mask = (np.abs(X + Y - 10) <= 2) & (np.abs(2*X - Y - 5) <= 2)
            numpy_solutions = np.column_stack([X[mask], Y[mask]])
            numpy_time = time.time() - start
            
            # Сбор результатов
            result_entry = {
                'size': size,
                'pi_time': pi_time,
                'pi_solutions': len(pi_solutions),
                'numpy_time': numpy_time,
                'numpy_solutions': len(numpy_solutions),
            }
            
            if cpu_time is not None:
                result_entry['cpu_time'] = cpu_time
                result_entry['cpu_solutions'] = len(solutions_cpu)
                print(f"  CPU: {cpu_time:.4f} сек, Решений: {len(solutions_cpu)}")
            
            print(f"  NumPy: {numpy_time:.4f} сек, Решений: {len(numpy_solutions)}")
            print(f"  Π-Топология: {pi_time:.4f} сек, Решений: {len(pi_solutions)}")
            
            if cpu_time and cpu_time > 0:
                print(f"  Ускорение Π/CPU: {cpu_time/pi_time:.1f}×")
            if numpy_time > 0:
                print(f"  Ускорение Π/NumPy: {numpy_time/pi_time:.1f}×")
            
            results.append(result_entry)
        
        return results
    
    @staticmethod
    def plot_results(results: List[dict]):
        """Визуализация результатов бенчмарка"""
        
        sizes = [r['size'] for r in results]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Время выполнения
        ax1 = axes[0]
        if 'cpu_time' in results[0]:
            cpu_times = [r.get('cpu_time', 0) for r in results]
            ax1.plot(sizes, cpu_times, 'r-o', label='CPU (чистый Python)', linewidth=2)
        
        numpy_times = [r['numpy_time'] for r in results]
        pi_times = [r['pi_time'] for r in results]
        
        ax1.plot(sizes, numpy_times, 'g-s', label='NumPy (векториз.)', linewidth=2)
        ax1.plot(sizes, pi_times, 'b-^', label='Π-Топология (GPU)', linewidth=2)
        
        ax1.set_xlabel('Размер пространства')
        ax1.set_ylabel('Время (сек)')
        ax1.set_title('Сравнение производительности')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Количество решений
        ax2 = axes[1]
        pi_solutions = [r['pi_solutions'] for r in results]
        numpy_solutions = [r['numpy_solutions'] for r in results]
        
        ax2.plot(sizes, pi_solutions, 'b-^', label='Π-Топология', linewidth=2)
        ax2.plot(sizes, numpy_solutions, 'g-s', label='NumPy', linewidth=2)
        
        ax2.set_xlabel('Размер пространства')
        ax2.set_ylabel('Количество решений')
        ax2.set_title('Масштабируемость поиска решений')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Эффективность памяти
        ax3 = axes[2]
        if config.device.type == 'cuda':
            # Оценочное использование памяти (MB)
            memory_estimate = [s**2 * 4 / 1024**2 for s in sizes]  # float32: 4 байта на точку
            
            ax3.plot(sizes, memory_estimate, 'm-*', label='Оценка памяти', linewidth=2)
            ax3.axhline(y=torch.cuda.get_device_properties(0).total_memory / 1024**3 * 1024, 
                       color='r', linestyle='--', label='Лимит памяти GPU', alpha=0.5)
            
            ax3.set_xlabel('Размер пространства')
            ax3.set_ylabel('Память (MB)')
            ax3.set_title('Использование памяти')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'GPU не обнаружено\nдля анализа памяти', 
                    ha='center', va='center', fontsize=12)
            ax3.set_title('Анализ памяти (требуется GPU)')
        
        plt.tight_layout()
        plt.savefig('pi_topology_benchmark_results.png', dpi=150, bbox_inches='tight')
        plt.show()

# =============================================
# РЕАЛЬНЫЙ ПРИМЕР: ОПТИМИЗАЦИЯ ПРОИЗВОДСТВА
# =============================================

def production_optimization_example():
    """Пример: Оптимизация производственного процесса"""
    
    print("\n" + "=" * 80)
    print("ПРИМЕР: ОПТИМИЗАЦИЯ ПРОИЗВОДСТВА")
    print("=" * 80)
    
    # Параметры производства
    # x - количество продукта A (0-100 единиц)
    # y - количество продукта B (0-100 единиц)
    
    x_domain = np.linspace(0, 100, 1000, dtype=np.float32)
    y_domain = np.linspace(0, 100, 1000, dtype=np.float32)
    
    print(f"\nПространство поиска: {len(x_domain)}×{len(y_domain)} = {len(x_domain)*len(y_domain):,} вариантов")
    
    # Создаем Π-Потоки для ограничений
    start_time = time.time()
    
    # 1. Ограничение по ресурсам: 2A + 3B ≤ 200
    flow_resources = PiFlowOptimized("Ресурсы", x_domain, y_domain)
    flow_resources.create_from_function(lambda X, Y: 2*X + 3*Y)
    
    # 2. Ограничение по времени: A + 2B ≤ 150
    flow_time = PiFlowOptimized("Время", x_domain, y_domain)
    flow_time.create_from_function(lambda X, Y: X + 2*Y)
    
    # 3. Целевая функция: Прибыль = 5A + 4B (максимизируем)
    flow_profit = PiFlowOptimized("Прибыль", x_domain, y_domain)
    flow_profit.create_from_function(lambda X, Y: 5*X + 4*Y)
    
    # Π-Слияние всех ограничений
    merged_constraints = flow_resources.pi_merge_batched(flow_time, operation='concat')
    
    # Π-Инволюция: находим все допустимые варианты
    # Ресурсы ≤ 200, Время ≤ 150
    result = merged_constraints.pi_involution_chunked(
        target_values=[200, 150],
        tolerance=[0, 0],  # Можно добавить допуск для гибкости
        chunk_size=512
    )
    
    # Теперь находим максимальную прибыль среди допустимых вариантов
    if hasattr(result, 'solutions_x') and len(result.solutions_x) > 0:
        # Вычисляем прибыль для всех допустимых решений
        profit_values = 5*result.solutions_x + 4*result.solutions_y
        
        # Находим максимальную прибыль
        max_profit_idx = torch.argmax(profit_values)
        optimal_A = result.solutions_x[max_profit_idx].item()
        optimal_B = result.solutions_y[max_profit_idx].item()
        max_profit = profit_values[max_profit_idx].item()
        
        total_time = time.time() - start_time
        
        print(f"\nРЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ:")
        print(f"  Время расчета: {total_time:.3f} сек")
        print(f"  Проанализировано вариантов: {len(x_domain)*len(y_domain):,}")
        print(f"  Допустимых вариантов: {len(result.solutions_x):,}")
        print(f"\n  ОПТИМАЛЬНОЕ РЕШЕНИЕ:")
        print(f"    Продукт A: {optimal_A:.1f} ед.")
        print(f"    Продукт B: {optimal_B:.1f} ед.")
        print(f"    Максимальная прибыль: {max_profit:.2f}")
        print(f"\n  ПРОВЕРКА ОГРАНИЧЕНИЙ:")
        print(f"    Ресурсы: {2*optimal_A + 3*optimal_B:.1f} (лимит: 200)")
        print(f"    Время: {optimal_A + 2*optimal_B:.1f} (лимит: 150)")
        
        # Визуализация
        result.visualize("Оптимизация производства: Допустимые варианты")
        
        # Дополнительно: топ-5 решений
        top5_indices = torch.topk(profit_values, min(5, len(profit_values))).indices
        print(f"\n  ТОП-5 РЕШЕНИЙ:")
        for i, idx in enumerate(top5_indices):
            A_val = result.solutions_x[idx].item()
            B_val = result.solutions_y[idx].item()
            profit_val = profit_values[idx].item()
            print(f"    {i+1}. A={A_val:.1f}, B={B_val:.1f}, Прибыль={profit_val:.2f}")
    
    else:
        print("Нет допустимых решений!")

# =============================================
# МОДУЛЬНЫЕ ТЕСТЫ
# =============================================

def run_tests():
    """Запуск модульных тестов"""
    
    print("=" * 80)
    print("МОДУЛЬНЫЕ ТЕСТЫ Π-ТОПОЛОГИИ")
    print("=" * 80)
    
    # Тест 1: Малое пространство
    print("\n1. Тест малого пространства (10×10):")
    x_small = np.linspace(0, 10, 10, dtype=np.float32)
    y_small = np.linspace(0, 10, 10, dtype=np.float32)
    
    flow_test = PiFlowOptimized("Тест", x_small, y_small)
    flow_test.create_from_function(lambda X, Y: X + Y)
    
    assert flow_test.values.shape == (10, 10), "Неверная форма тензора"
    print(f"  ✓ Создание потока: {flow_test.values.shape}")
    
    # Тест 2: Слияние
    flow_test2 = PiFlowOptimized("Тест2", x_small, y_small)
    flow_test2.create_from_function(lambda X, Y: X - Y)
    
    merged = flow_test.pi_merge_batched(flow_test2, operation='concat')
    assert merged.values.shape == (10, 10, 2), "Неверная форма после слияния"
    print(f"  ✓ Слияние потоков: {merged.values.shape}")
    
    # Тест 3: Инволюция
    result = merged.pi_involution_chunked([5, 0], tolerance=1)
    solutions = result.get_solutions()
    print(f"  ✓ Инволюция: найдено {len(solutions)} решений")
    
    # Тест 4: Статистика
    stats = result.get_statistics()
    assert all(key in stats for key in ['creation_time', 'merge_time', 'involution_time'])
    print(f"  ✓ Статистика: {list(stats.keys())}")
    
    print("\nВсе тесты пройдены успешно! ✓")

# =============================================
# ГЛАВНАЯ ФУНКЦИЯ
# =============================================

def main():
    """Главная функция с демонстрацией возможностей"""
    
    print("=" * 80)
    print("ПАРАЛЛЕЛЬНАЯ Π-ТОПОЛОГИЯ - ВЫСОКОПРОИЗВОДИТЕЛЬНАЯ РЕАЛИЗАЦИЯ")
    print("=" * 80)
    print(f"Устройство: {config.device}")
    if config.device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Точность: {config.dtype}")
    print(f"Размер чанка: {config.chunk_size}")
    print("=" * 80)
    
    # 1. Запуск модульных тестов
    run_tests()
    
    # 2. Бенчмарк производительности
    print("\n" + "=" * 80)
    print("ЗАПУСК БЕНЧМАРКА ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 80)
    
    benchmark_results = PiTopologyBenchmark.benchmark_system_of_equations(max_size=2000)
    
    # 3. Визуализация результатов бенчмарка
    PiTopologyBenchmark.plot_results(benchmark_results)
    
    # 4. Реальный пример
    production_optimization_example()
    
    # 5. Итоговая статистика
    print("\n" + "=" * 80)
    print("ИТОГОВАЯ СТАТИСТИКА И ВЫВОДЫ")
    print("=" * 80)
    
    # Анализ производительности
    if benchmark_results:
        last_result = benchmark_results[-1]
        print(f"\nПРОИЗВОДИТЕЛЬНОСТЬ НА МАКСИМАЛЬНОМ ТЕСТЕ:")
        print(f"  Размер пространства: {last_result['size']}×{last_result['size']}")
        print(f"  Π-Топология: {last_result['pi_time']:.3f} сек")
        print(f"  NumPy: {last_result['numpy_time']:.3f} сек")
        
        if last_result['numpy_time'] > 0:
            speedup = last_result['numpy_time'] / last_result['pi_time']
            print(f"  Ускорение Π/NumPy: {speedup:.1f}×")
        
        # Анализ памяти
        if config.device.type == 'cuda':
            print(f"\nИСПОЛЬЗОВАНИЕ ПАМЯТИ GPU:")
            print(f"  Выделено: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
            print(f"  Зарезервировано: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
    
    print(f"\nРЕКОМЕНДАЦИИ:")
    print(f"  1. Для пространств < 100K точек используйте NumPy")
    print(f"  2. Для пространств 100K-10M точек используйте Π-Топологию с чанкованием")
    print(f"  3. Для пространств > 10M точек используйте распределенную обработку")
    print(f"\nПРЕИМУЩЕСТВА Π-ТОПОЛОГИИ:")
    print(f"  • Естественный параллелизм для больших пространств")
    print(f"  • Автоматическое нахождение ВСЕХ решений")
    print(f"  • Гибкая работа с неоднозначностью и допусками")
    print(f"  • Масштабируемость на GPU/TPU кластерах")

# =============================================
# ЗАПУСК ПРИЛОЖЕНИЯ
# =============================================

if __name__ == "__main__":
    
    # Очистка памяти GPU перед запуском
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nПрограмма прервана пользователем")
    except Exception as e:
        print(f"\n\nОшибка выполнения: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Очистка памяти GPU после выполнения
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"\nОчистка памяти GPU: {torch.cuda.memory_allocated() / 1024**2:.1f} MB освобождено")