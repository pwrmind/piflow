import numpy as np

# =============================================
# КЛАССИЧЕСКИЙ ПОДХОД
# =============================================

def classical_solution_linear(A, b):
    """Решение системы линейных уравнений классическим методом"""
    return np.linalg.solve(A, b)

def classical_solution_search(domain_x, domain_y, equation1, equation2, tol=1e-10):
    """Решение перебором (для сравнения)"""
    solutions = []
    for x in domain_x:
        for y in domain_y:
            if abs(equation1(x, y)) < tol and abs(equation2(x, y)) < tol:
                solutions.append((x, y))
    return solutions

# =============================================
# Π-ТОПОЛОГИЧЕСКИЙ ПОДХОД
# =============================================

class PiFlow:
    """Класс Π-Потока"""
    def __init__(self, name):
        self.name = name
        self.values = {}  # контекст -> значение
        self.contexts = []  # список контекстов
        
    def create_from_function(self, contexts, func):
        """Создать поток из функции f(context) = value"""
        self.contexts = contexts
        self.values = {ctx: func(ctx) for ctx in contexts}
        return self
    
    def create_from_domain(self, domain_x, domain_y, func):
        """Создать поток для пар (x,y)"""
        contexts = [(x, y) for x in domain_x for y in domain_y]
        return self.create_from_function(contexts, func)
    
    def pi_merge(self, other_flow):
        """Операция Π-Слияния (⨁)"""
        # Проверяем, что пространства контекстов одинаковы
        assert set(self.contexts) == set(other_flow.contexts), "Пространства контекстов должны совпадать"
        
        merged = PiFlow(f"({self.name}⨁{other_flow.name})")
        merged.contexts = self.contexts
        
        # Слияние: создаём пары значений
        merged.values = {ctx: (self.values[ctx], other_flow.values[ctx]) 
                        for ctx in self.contexts}
        return merged
    
    def pi_involution(self, target_value, tolerance=0):
        """Оператор Π-Инволюции (ℑ)"""
        # "Коллапс" пространства: оставляем только контексты, 
        # где значение совпадает с целевым (с учётом допуска)
        result = PiFlow(f"ℑ({self.name})")
        
        if isinstance(target_value, tuple):
            # Если target_value - это пара (для слияния)
            result.contexts = [
                ctx for ctx in self.contexts
                if all(abs(self.values[ctx][i] - target_value[i]) <= tolerance 
                      for i in range(len(target_value)))
            ]
        else:
            # Если target_value - одиночное значение
            result.contexts = [
                ctx for ctx in self.contexts
                if abs(self.values[ctx] - target_value) <= tolerance
            ]
        
        result.values = {ctx: self.values[ctx] for ctx in result.contexts}
        return result
    
    def get_solutions(self):
        """Извлечь решения (проекция на исходные переменные)"""
        if not self.contexts:
            return []
        
        # Для задачи с двумя переменными
        solutions = list(set(self.contexts))  # уникальные решения
        return solutions
    
    def stats(self):
        """Статистика потока"""
        return {
            'name': self.name,
            'contexts_count': len(self.contexts),
            'solutions': self.get_solutions()
        }

# =============================================
# ТЕСТ: ТОЧНАЯ СИСТЕМА УРАВНЕНИЙ
# =============================================
print("=" * 60)
print("ТЕСТ 1: ТОЧНАЯ СИСТЕМА УРАВНЕНИЙ")
print("=" * 60)

# Определяем уравнения
# 1) x + y = 10
# 2) 2x - y = 5

# Классическое решение
print("\n1. КЛАССИЧЕСКОЕ РЕШЕНИЕ:")
A = np.array([[1, 1], [2, -1]])
b = np.array([10, 5])
solution_classical = classical_solution_linear(A, b)
print(f"   Аналитическое решение: x = {solution_classical[0]:.2f}, y = {solution_classical[1]:.2f}")

# Область поиска для Π-подхода (дискретизация)
x_domain = list(range(0, 11))  # x от 0 до 10
y_domain = list(range(0, 11))  # y от 0 до 10

# Создаём Π-Потоки
print("\n2. Π-ТОПОЛОГИЧЕСКИЙ ПОДХОД:")

# Поток A: x + y
flow_A = PiFlow("A")
flow_A.create_from_domain(x_domain, y_domain, lambda ctx: ctx[0] + ctx[1])

# Поток B: 2x - y
flow_B = PiFlow("B") 
flow_B.create_from_domain(x_domain, y_domain, lambda ctx: 2*ctx[0] - ctx[1])

print(f"   Создано контекстов: {len(flow_A.contexts)}")

# Π-Слияние
merged = flow_A.pi_merge(flow_B)
print(f"   После слияния (⨁): {len(merged.contexts)} комбинированных состояний")

# Π-Инволюция (целевые значения: 10 и 5)
target = (10, 5)
result = merged.pi_involution(target, tolerance=0)
print(f"   После инволюции (ℑ): {len(result.contexts)} согласованных состояний")

# Извлечение решений
solutions = result.get_solutions()
print(f"   Найдено решений: {len(solutions)}")
for sol in solutions:
    print(f"     x = {sol[0]}, y = {sol[1]}")

# =============================================
# ТЕСТ: СИСТЕМА С ПОГРЕШНОСТЬЮ
# =============================================
print("\n" + "=" * 60)
print("ТЕСТ 2: СИСТЕМА С ПОГРЕШНОСТЬЮ ±1")
print("=" * 60)

# Классический подход с перебором (для сравнения)
print("\n1. КЛАССИЧЕСКИЙ ПОДХОД (перебор):")

def eq1(x, y):
    return abs((x + y) - 10)  # отклонение от 10

def eq2(x, y):
    return abs((2*x - y) - 5)  # отклонение от 5

# Ищем решения с погрешностью ±1
classical_solutions = []
for x in x_domain:
    for y in y_domain:
        if eq1(x, y) <= 1 and eq2(x, y) <= 1:
            classical_solutions.append((x, y))

print(f"   Найдено решений перебором: {len(classical_solutions)}")
print("   Решения:", classical_solutions)

# Π-Топологический подход
print("\n2. Π-ТОПОЛОГИЧЕСКИЙ ПОДХОД:")

# Те же потоки A и B
# Π-Слияние
merged_with_tol = flow_A.pi_merge(flow_B)

# Π-Инволюция с допуском ±1
result_with_tol = merged_with_tol.pi_involution(target, tolerance=1)
print(f"   После инволюции с допуском ±1: {len(result_with_tol.contexts)} согласованных состояний")

# Извлечение решений
solutions_tol = result_with_tol.get_solutions()
print(f"   Найдено решений: {len(solutions_tol)}")
for sol in solutions_tol:
    deviation1 = abs((sol[0] + sol[1]) - 10)
    deviation2 = abs((2*sol[0] - sol[1]) - 5)
    print(f"     x = {sol[0]}, y = {sol[1]} (отклонения: {deviation1}, {deviation2})")

# =============================================
# ТЕСТ: ПРОИЗВОДИТЕЛЬНОСТЬ НА БОЛЬШОЙ ОБЛАСТИ
# =============================================
print("\n" + "=" * 60)
print("ТЕСТ 3: ПРОИЗВОДИТЕЛЬНОСТЬ НА БОЛЬШОЙ ОБЛАСТИ")
print("=" * 60)

import time

# Большая область поиска
big_x_domain = list(range(-50, 51))  # -50..50 (101 значение)
big_y_domain = list(range(-50, 51))  # -50..50 (101 значение)

print(f"Размер пространства поиска: {len(big_x_domain)} × {len(big_y_domain)} = {len(big_x_domain)*len(big_y_domain):,} контекстов")

# Классический перебор
print("\n1. КЛАССИЧЕСКИЙ ПЕРЕБОР:")
start = time.time()
classical_big_solutions = []
for x in big_x_domain:
    for y in big_y_domain:
        if abs((x + y) - 10) <= 2 and abs((2*x - y) - 5) <= 2:
            classical_big_solutions.append((x, y))
classical_time = time.time() - start
print(f"   Время: {classical_time:.4f} сек")
print(f"   Найдено решений: {len(classical_big_solutions)}")

# Π-Топологический подход (эмуляция)
print("\n2. Π-ТОПОЛОГИЧЕСКИЙ ПОДХОД (эмуляция):")
start = time.time()

# Создание потоков (можно оптимизировать, но для сравнения оставим так)
flow_A_big = PiFlow("A_big")
flow_A_big.create_from_domain(big_x_domain, big_y_domain, lambda ctx: ctx[0] + ctx[1])

flow_B_big = PiFlow("B_big")
flow_B_big.create_from_domain(big_x_domain, big_y_domain, lambda ctx: 2*ctx[0] - ctx[1])

# Слияние и инволюция
merged_big = flow_A_big.pi_merge(flow_B_big)
result_big = merged_big.pi_involution((10, 5), tolerance=2)
pi_time = time.time() - start

print(f"   Время: {pi_time:.4f} сек")
print(f"   Найдено решений: {len(result_big.get_solutions())}")
print(f"   Ускорение: {classical_time/pi_time:.2f}×")

# =============================================
# ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
# =============================================
try:
    import matplotlib.pyplot as plt
    
    print("\n" + "=" * 60)
    print("ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Пространство поиска
    ax1 = axes[0]
    X, Y = np.meshgrid(x_domain, y_domain)
    Z1 = X + Y  # Уравнение 1
    Z2 = 2*X - Y  # Уравнение 2
    
    # Отметим точное решение
    ax1.scatter([5], [5], color='red', s=100, marker='*', label='Точное решение')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Пространство поиска и точное решение')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Решения с погрешностью
    ax2 = axes[1]
    solutions_x = [sol[0] for sol in solutions_tol]
    solutions_y = [sol[1] for sol in solutions_tol]
    
    ax2.scatter(solutions_x, solutions_y, color='green', s=50, label='Решения с погрешностью ±1')
    ax2.scatter([5], [5], color='red', s=100, marker='*', label='Точное решение')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Множество решений с погрешностью')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Сравнение производительности
    ax3 = axes[2]
    methods = ['Классический\nперебор', 'Π-Топология\n(эмуляция)']
    times = [classical_time, pi_time]
    
    bars = ax3.bar(methods, times, color=['blue', 'orange'])
    ax3.set_ylabel('Время (сек)')
    ax3.set_title('Сравнение производительности')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Добавим значения на столбцы
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.3f} сек',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('pi_topology_comparison.png', dpi=150)
    print("Графики сохранены в 'pi_topology_comparison.png'")
    
except ImportError:
    print("\nДля визуализации установите matplotlib: pip install matplotlib")

# =============================================
# ВЫВОДЫ
# =============================================
print("\n" + "=" * 60)
print("ВЫВОДЫ И НАБЛЮДЕНИЯ")
print("=" * 60)
print("""
1. Π-ТОПОЛОГИЧЕСКИЙ ПОДХОД:
   - Работает со ВСЕМИ состояниями одновременно
   - Решение возникает как "коллапс" всего пространства возможностей
   - Естественным образом находит ВСЕ решения (а не только одно)
   - Легко обрабатывает неоднозначность и погрешности

2. КЛАССИЧЕСКИЙ ПОДХОД:
   - Линейный поиск решения (итерационный)
   - Находит обычно ОДНО решение
   - Для учёта погрешностей нужны дополнительные алгоритмы

3. КЛЮЧЕВОЕ ОТЛИЧИЕ:
   В Π-Топологии мы не ИЩЕМ решение, а ЗАДАЁМ условия, 
   и пространство само "схлопывается" к согласованным состояниям.
   
   Это похоже на принцип наименьшего действия в физике:
   система самопроизвольно находит состояние, удовлетворяющее
   всем заданным ограничениям одновременно.
""")