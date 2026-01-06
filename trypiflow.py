import numpy as np
import torch
import matplotlib.pyplot as plt
from main import PiFlow, config

# Патч для исправления метода pi_merge
def patched_pi_merge(self, other_flow: 'PiFlow') -> 'PiFlow':
    """Исправленный Π-Слияние потоков"""
    merged = PiFlow(
        f"({self.name}⨁{other_flow.name})",
        self.x_domain,
        self.y_domain
    )
    
    # Приводим оба тензора к 3D виду перед слиянием
    self_values = self.values if self.values.ndim == 3 else self.values.unsqueeze(-1)
    other_values = other_flow.values if other_flow.values.ndim == 3 else other_flow.values.unsqueeze(-1)
    
    # Объединяем по каналам
    merged.values = torch.cat([self_values, other_values], dim=-1)
    merged.func = None
    return merged

# Заменяем оригинальный метод
PiFlow.pi_merge = patched_pi_merge

# Теперь ваш код должен работать
def warehouse_distribution_optimization():
    """Распределение товаров по складам с минимальными затратами"""
    
    print("\n" + "="*80)
    print("ОПТИМИЗАЦИЯ РАСПРЕДЕЛЕНИЯ ТОВАРОВ ПО СКЛАДАМ")
    print("="*80)
    
    # ПАРАМЕТРЫ ЗАДАЧИ
    # Два склада: Склад1 и Склад2
    # Необходимо распределить товары между ними
    
    # Ограничения складов:
    # Склад1: макс. 100 тонн, стоимость хранения: 5 у.е./тонна
    # Склад2: макс. 150 тонн, стоимость хранения: 3 у.е./тонна
    
    # Ограничения транспортировки:
    # Доставка до Склада1: 2 у.е./тонна
    # Доставка до Склада2: 4 у.е./тонна
    
    # Общий объем товаров: 200 тонн
    
    # ЦЕЛЬ: минимизировать общие затраты = 
    #       (стоимость_хранения + стоимость_доставки) * количество
    
    # Создаем пространство поиска
    # x - количество товаров на Складе1 (от 0 до 200)
    # y - количество товаров на Складе2 (от 0 до 200)
    x_domain = np.linspace(0, 200, 400, dtype=np.float32)  # Склад1
    y_domain = np.linspace(0, 200, 400, dtype=np.float32)  # Склад2
    
    print(f"Пространство поиска: {len(x_domain)}×{len(y_domain)} = {len(x_domain)*len(y_domain):,} вариантов")
    
    start_time = time.time()
    
    # 1. Создаем потоки для ограничений
    
    # Общий объем товаров (должен равняться 200 тонн)
    flow_total = PiFlow("Общий объем", x_domain, y_domain)
    flow_total.create_from_function(lambda X, Y: X + Y)
    
    # Использование емкости Склада1 (не более 100 тонн)
    flow_capacity1 = PiFlow("Емкость Склада1", x_domain, y_domain)
    flow_capacity1.create_from_function(lambda X, Y: X)
    
    # Использование емкости Склада2 (не более 150 тонн)
    flow_capacity2 = PiFlow("Емкость Склада2", x_domain, y_domain)
    flow_capacity2.create_from_function(lambda X, Y: Y)
    
    # Затраты на хранение
    storage_cost1 = 5  # у.е./тонна на Складе1
    storage_cost2 = 3  # у.е./тонна на Складе2
    
    flow_storage_cost = PiFlow("Затраты на хранение", x_domain, y_domain)
    flow_storage_cost.create_from_function(lambda X, Y: storage_cost1*X + storage_cost2*Y)
    
    # Затраты на доставку
    delivery_cost1 = 2  # у.е./тонна до Склада1
    delivery_cost2 = 4  # у.е./тонна до Склада2
    
    flow_delivery_cost = PiFlow("Затраты на доставку", x_domain, y_domain)
    flow_delivery_cost.create_from_function(lambda X, Y: delivery_cost1*X + delivery_cost2*Y)
    
    # Общие затраты (целевая функция для минимизации)
    flow_total_cost = PiFlow("Общие затраты", x_domain, y_domain)
    flow_total_cost.create_from_function(lambda X, Y: 
        (storage_cost1 + delivery_cost1)*X + (storage_cost2 + delivery_cost2)*Y)
    
    # 2. Объединяем ограничения
    merged_constraints = flow_total.pi_merge(flow_capacity1).pi_merge(flow_capacity2)
    
    # 3. Применяем ограничения (Π-инволюция)
    result = merged_constraints.pi_involution([
        {'type': 'equals', 'target': 200, 'tolerance': 0.5},  # Всего 200 тонн
        {'type': 'less', 'target': 100},  # Склад1 ≤ 100 тонн
        {'type': 'less', 'target': 150}   # Склад2 ≤ 150 тонн
    ])
    
    # 4. Получаем все допустимые решения
    solutions = result.get_all_solutions()
    
    if len(solutions) == 0:
        print("Нет допустимых решений!")
        return
    
    # 5. Находим оптимальное решение (минимальные затраты)
    X_vals = torch.tensor(solutions[:, 0], device=config.device)
    Y_vals = torch.tensor(solutions[:, 1], device=config.device)
    
    # Вычисляем затраты для каждого варианта
    storage_costs = storage_cost1*X_vals + storage_cost2*Y_vals
    delivery_costs = delivery_cost1*X_vals + delivery_cost2*Y_vals
    total_costs = storage_costs + delivery_costs
    
    # Находим минимальные затраты
    min_cost_idx = torch.argmin(total_costs)
    optimal_x = X_vals[min_cost_idx].item()
    optimal_y = Y_vals[min_cost_idx].item()
    min_total_cost = total_costs[min_cost_idx].item()
    
    total_time = time.time() - start_time
    
    # 6. Выводим результаты
    print(f"\nРЕЗУЛЬТАТЫ:")
    print(f"  Время расчета: {total_time:.3f} сек")
    print(f"  Допустимых вариантов распределения: {len(solutions):,}")
    
    print(f"\n  ОПТИМАЛЬНОЕ РАСПРЕДЕЛЕНИЕ:")
    print(f"    Склад 1: {optimal_x:.1f} тонн")
    print(f"    Склад 2: {optimal_y:.1f} тонн")
    print(f"    Всего: {optimal_x + optimal_y:.1f} тонн")
    
    print(f"\n  ЗАТРАТЫ:")
    print(f"    Хранение на Складе 1: {storage_cost1} × {optimal_x:.1f} = {storage_cost1*optimal_x:.1f} у.е.")
    print(f"    Хранение на Складе 2: {storage_cost2} × {optimal_y:.1f} = {storage_cost2*optimal_y:.1f} у.е.")
    print(f"    Доставка на Склад 1: {delivery_cost1} × {optimal_x:.1f} = {delivery_cost1*optimal_x:.1f} у.е.")
    print(f"    Доставка на Склад 2: {delivery_cost2} × {optimal_y:.1f} = {delivery_cost2*optimal_y:.1f} у.е.")
    print(f"    ОБЩИЕ ЗАТРАТЫ: {min_total_cost:.1f} у.е.")
    
    # 7. Анализ чувствительности (топ-5 лучших вариантов)
    top5_idx = torch.topk(-total_costs, min(5, len(total_costs))).indices
    print(f"\n  ТОП-5 ВАРИАНТОВ:")
    for i, idx in enumerate(top5_idx):
        x = X_vals[idx].item()
        y = Y_vals[idx].item()
        cost = total_costs[idx].item()
        print(f"    {i+1}. Склад1={x:.1f}т, Склад2={y:.1f}т, Затраты={cost:.1f} у.е.")
    
    # 8. Визуализация
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # График 1: Допустимая область
        axes[0].scatter(solutions[:, 0], solutions[:, 1], 
                       c='blue', s=1, alpha=0.3, label='Допустимые варианты')
        axes[0].scatter([optimal_x], [optimal_y], 
                       c='red', s=200, marker='*', label='Оптимальное решение')
        axes[0].axvline(x=100, color='r', linestyle='--', alpha=0.5, label='Макс. Склад1')
        axes[0].axhline(y=150, color='g', linestyle='--', alpha=0.5, label='Макс. Склад2')
        axes[0].plot([0, 200], [200, 0], 'k-', alpha=0.3, label='Всего 200 тонн')
        axes[0].set_xlabel('Товары на Складе 1 (тонн)')
        axes[0].set_ylabel('Товары на Складе 2 (тонн)')
        axes[0].set_title('Допустимая область распределения')
        axes[0].legend(loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # График 2: Затраты по вариантам
        axes[1].scatter(solutions[:, 0], total_costs.cpu().numpy(),
                       c='purple', s=2, alpha=0.5)
        axes[1].scatter([optimal_x], [min_total_cost], 
                       c='red', s=100, marker='*')
        axes[1].set_xlabel('Товары на Складе 1 (тонн)')
        axes[1].set_ylabel('Общие затраты (у.е.)')
        axes[1].set_title('Зависимость затрат от распределения')
        axes[1].grid(True, alpha=0.3)
        
        # График 3: Структура затрат для оптимального решения
        cost_breakdown = {
            'Хранение С1': storage_cost1*optimal_x,
            'Хранение С2': storage_cost2*optimal_y,
            'Доставка С1': delivery_cost1*optimal_x,
            'Доставка С2': delivery_cost2*optimal_y
        }
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        axes[2].pie(cost_breakdown.values(), labels=cost_breakdown.keys(), 
                   colors=colors, autopct='%1.1f%%', startangle=90)
        axes[2].set_title(f'Структура затрат\nВсего: {min_total_cost:.1f} у.е.')
        
        plt.tight_layout()
        plt.savefig('warehouse_optimization.png', dpi=150, bbox_inches='tight')
        print(f"\n  Визуализация сохранена в 'warehouse_optimization.png'")
        
    except Exception as e:
        print(f"  Визуализация не удалась: {e}")
    
    return optimal_x, optimal_y, min_total_cost


def complex_warehouse_scenario():
    """Более сложный сценарий с тремя складами"""
    print("\n" + "="*80)
    print("СЛОЖНЫЙ СЦЕНАРИЙ: 3 СКЛАДА")
    print("="*80)
    
    # Параметры 3 складов
    capacities = [100, 150, 120]  # тонн
    storage_costs = [5, 3, 4]     # у.е./тонна
    delivery_costs = [2, 4, 3]    # у.е./тонна
    
    total_goods = 300  # тонн
    
    # Создаем сетку для двух складов (третий вычисляем)
    x_domain = np.linspace(0, min(total_goods, capacities[0]), 200, dtype=np.float32)
    y_domain = np.linspace(0, min(total_goods, capacities[1]), 200, dtype=np.float32)
    
    print("Анализируем распределение между Складами 1 и 2")
    print(f"(Склад 3 получает остаток от {total_goods} тонн)")
    
    flow_capacity1 = PiFlow("Склад1", x_domain, y_domain)
    flow_capacity1.create_from_function(lambda X, Y: X)
    
    flow_capacity2 = PiFlow("Склад2", x_domain, y_domain)
    flow_capacity2.create_from_function(lambda X, Y: Y)
    
    flow_capacity3 = PiFlow("Склад3", x_domain, y_domain)
    flow_capacity3.create_from_function(lambda X, Y: total_goods - X - Y)
    
    flow_total_cost = PiFlow("Общие затраты", x_domain, y_domain)
    flow_total_cost.create_from_function(lambda X, Y: 
        storage_costs[0]*X + storage_costs[1]*Y + storage_costs[2]*(total_goods - X - Y) +
        delivery_costs[0]*X + delivery_costs[1]*Y + delivery_costs[2]*(total_goods - X - Y))
    
    # Объединяем ограничения
    merged = flow_capacity1.pi_merge(flow_capacity2).pi_merge(flow_capacity3)
    
    # Применяем ограничения
    result = merged.pi_involution([
        {'type': 'greater', 'target': 0},  # Все склады должны получить ≥ 0
        {'type': 'less', 'target': capacities[0]},  # Склад1 ≤ 100
        {'type': 'less', 'target': capacities[1]},  # Склад2 ≤ 150
        {'type': 'range', 'target': [0, capacities[2]]}  # Склад3 от 0 до 120
    ])
    
    solutions = result.get_all_solutions()
    
    if len(solutions) > 0:
        X = torch.tensor(solutions[:, 0], device=config.device)
        Y = torch.tensor(solutions[:, 1], device=config.device)
        Z = total_goods - X - Y
        
        total_costs = (
            (storage_costs[0] + delivery_costs[0])*X +
            (storage_costs[1] + delivery_costs[1])*Y +
            (storage_costs[2] + delivery_costs[2])*Z
        )
        
        min_idx = torch.argmin(total_costs)
        
        print(f"\nОПТИМАЛЬНОЕ РАСПРЕДЕЛЕНИЕ (3 склада):")
        print(f"  Склад 1: {X[min_idx].item():.1f} тонн")
        print(f"  Склад 2: {Y[min_idx].item():.1f} тонн")
        print(f"  Склад 3: {Z[min_idx].item():.1f} тонн")
        print(f"  Общие затраты: {total_costs[min_idx].item():.1f} у.е.")
        print(f"  Допустимых вариантов: {len(solutions):,}")


if __name__ == "__main__":
    import time
    
    # Запускаем оптимизацию
    optimal_distribution = warehouse_distribution_optimization()
    
    # Запускаем сложный сценарий
    complex_warehouse_scenario()