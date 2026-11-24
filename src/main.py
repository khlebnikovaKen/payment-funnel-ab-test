"""Точка входа: запуск A/B-анализа и сохранение графиков в папку reports/."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.analyzer import (

    CONTROL,
    STEP_LABELS,
    TEST,
    audit_data_quality,
    build_user_funnel,
    load_raw_data,
    prepare_datasets,
    run_ab_test,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "date.xlsx"
REPORTS_DIR = BASE_DIR / "reports"
PALETTE = ["#033D9B", "#D30000"]


def plot_ab_conversion(res: dict, out_path: Path) -> None:
    """Строит и сохраняет график конверсии по группам с 95% доверительными интервалами."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5))

    groups = [f"Группа {CONTROL}\n(Контроль)", f"Группа {TEST}\n(Тест)"]
    rates = [res["p_a"] * 100, res["p_b"] * 100]
    err_low = [(res["p_a"] - res["ci_a"][0]) * 100, (res["p_b"] - res["ci_b"][0]) * 100]
    err_high = [(res["ci_a"][1] - res["p_a"]) * 100, (res["ci_b"][1] - res["p_b"]) * 100]

    bars = ax.bar(groups, rates, color=PALETTE, edgecolor="black", width=0.45, yerr=[err_low, err_high], capsize=6)

    for bar, rate, k, n in zip(bars, rates, [res["k_a"], res["k_b"]], [res["n_a"], res["n_b"]]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 4.8,
            f"{rate:.2f}%\n({k}/{n})",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_title(
        f"Конверсия в успешную оплату (Lift: {res['diff']*100:+.2f} п.п., p = {res['p_val']:.2e})",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    ax.set_ylabel("Конверсия, % (с 95% CI Уилсона)", fontsize=11)
    ax.set_ylim(0, max(rates) * 1.32)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_funnel(user_funnel: dict[str, list[int]], out_path: Path) -> None:
    """Строит и сохраняет сравнительный график воронки оплаты по шагам."""
    sns.set_theme(style="whitegrid")
    df_funnel = pd.DataFrame(user_funnel, index=STEP_LABELS)
    df_pct = df_funnel.div(df_funnel.iloc[0], axis=1) * 100

    fig, ax = plt.subplots(figsize=(10, 5.5))
    df_pct.plot(kind="bar", ax=ax, color=PALETTE, edgecolor="black", width=0.65)

    ax.set_title("Сравнение воронки оплаты по шагам (Группа A vs Группа B)", fontsize=13, fontweight="bold")
    ax.set_ylabel("% от Шага 1 (уникальные пользователи)", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylim(0, 114)
    plt.xticks(rotation=0)
    ax.legend(title="Группа", loc="upper right")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    users, payments = load_raw_data(DATA_PATH)

    print("=== 1. АУДИТ КАЧЕСТВА ДАННЫХ ===")
    for sev, label, cnt in audit_data_quality(users, payments):
        print(f"[{sev}] {label}: {cnt}")

    full_users, payments_clean = prepare_datasets(users, payments)

    print("\n=== 2. БАЛАНС ВЫБОРКИ ПОСЛЕ ОЧИСТКИ ===")
    balance = full_users.groupby("group").agg(
        users=("user_id", "count"),
        mean_age=("age", "mean"),
        median_age=("age", "median"),
    )
    print(balance.round(2))

    res = run_ab_test(full_users)
    print("\n=== 3. РЕЗУЛЬТАТЫ A/B-ТЕСТА (Z-ТЕСТ ДЛЯ ДОЛЕЙ) ===")
    print(f"Группа {CONTROL}: {res['p_a']:.2%} ({res['k_a']}/{res['n_a']})")
    print(f"Группа {TEST}: {res['p_b']:.2%} ({res['k_b']}/{res['n_b']})")
    print(f"Lift: {res['diff']*100:+.2f} п.п. | 95% CI: [{res['ci_lo']*100:+.2f}; {res['ci_hi']*100:+.2f}] п.п.")
    print(f"Z-stat = {res['z_stat']:.4f}, p-value = {res['p_val']:.4e}")

    user_funnel = build_user_funnel(payments_clean)
    print("\n=== 4. ВОРОНКА ПО УНИКАЛЬНЫМ ПОЛЬЗОВАТЕЛЯМ ===")
    print(pd.DataFrame(user_funnel, index=STEP_LABELS))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_ab_conversion(res, REPORTS_DIR / "ab_conversion.png")
    plot_funnel(user_funnel, REPORTS_DIR / "funnel_chart.png")
    print(f"\nГрафики сохранены в папку: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
