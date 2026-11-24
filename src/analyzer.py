"""Модуль аудита качества данных, очистки и статистического анализа A/B-теста."""

from pathlib import Path
import pandas as pd
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_confint,
    proportions_ztest,
)

CONTROL = "A"
TEST = "B"
STEP_COLS = ["step1_opened", "step2_entered", "step3_confirmed", "step4_success"]
STEP_LABELS = ["Шаг 1: открыт", "Шаг 2: введён", "Шаг 3: подтверждён", "Шаг 4: успех"]


def load_raw_data(filepath: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Загружает таблицы пользователей и платежей из Excel."""
    users = pd.read_excel(filepath, sheet_name="Пользователи")
    payments = pd.read_excel(filepath, sheet_name="Платежи")
    return users, payments


def audit_data_quality(users: pd.DataFrame, payments: pd.DataFrame) -> list[tuple[str, str, int]]:
    """Проверяет сырые логи на дубли, выбросы, сиротские записи и расхождение групп."""
    issues = []

    if users["user_id"].duplicated().any():
        issues.append(("ERROR", "Дубли user_id в users", int(users["user_id"].duplicated().sum())))

    invalid_age = ~users["age"].between(0, 100)
    if invalid_age.any():
        issues.append(("WARN", "Некорректный возраст (<0 или >100)", int(invalid_age.sum())))

    if users["group"].isna().any():
        issues.append(("WARN", "Пустой group в users", int(users["group"].isna().sum())))

    if payments["user_id"].isna().any():
        issues.append(("WARN", "Платежи без user_id", int(payments["user_id"].isna().sum())))

    orphans = set(payments["user_id"].dropna()) - set(users["user_id"])
    if orphans:
        issues.append(("WARN", "Orphan user_id в payments", len(orphans)))

    merged = payments.merge(
        users[["user_id", "group"]], on="user_id", how="inner", suffixes=("_pay", "_user")
    )
    mismatches = merged[
        merged["group_pay"].notna()
        & merged["group_user"].notna()
        & (merged["group_pay"] != merged["group_user"])
    ]
    if len(mismatches):
        issues.append(("ERROR", "Расхождение group payments vs users", len(mismatches)))

    multi_payers = int((payments.groupby("user_id").size() > 1).sum())
    if multi_payers:
        issues.append(("INFO", "Пользователей с >1 попыткой платежа", multi_payers))

    return issues


def prepare_datasets(
    users: pd.DataFrame, payments: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Очищает данные и формирует витрины на уровне уникальных пользователей и платежей."""
    users_clean = (
        users[users["age"].between(0, 100) & users["group"].isin([CONTROL, TEST])]
        .drop_duplicates(subset=["user_id"])
        .copy()
    )

    # Агрегация до уровня пользователя (единица рандомизации)
    per_user = (
        payments.groupby("user_id")
        .agg(reached_step4=("step4_success", lambda x: x.notnull().any()))
        .reset_index()
    )
    full_users = users_clean.merge(per_user, on="user_id", how="left")
    full_users["reached_step4"] = full_users["reached_step4"].fillna(False).astype(bool)

    # Платежи с эталонной группой из таблицы users
    payments_clean = payments.merge(
        users_clean[["user_id", "group"]],
        on="user_id",
        how="inner",
        suffixes=("_pay", "_user"),
    )

    return full_users, payments_clean


def run_ab_test(full_users: pd.DataFrame) -> dict:
    """Проводит двусторонний Z-тест для долей и вычисляет доверительные интервалы."""
    summary = full_users.groupby("group")["reached_step4"].agg(["count", "sum", "mean"])

    n_a, n_b = int(summary.loc[CONTROL, "count"]), int(summary.loc[TEST, "count"])
    k_a, k_b = int(summary.loc[CONTROL, "sum"]), int(summary.loc[TEST, "sum"])
    p_a, p_b = k_a / n_a, k_b / n_b

    z_stat, p_val = proportions_ztest([k_b, k_a], [n_b, n_a])
    ci_diff = confint_proportions_2indep(k_b, n_b, k_a, n_a, method="wald")

    if len(ci_diff) == 3:
        diff, ci_lo, ci_hi = ci_diff
    else:
        ci_lo, ci_hi = ci_diff
        diff = p_b - p_a

    return {
        "n_a": n_a,
        "n_b": n_b,
        "k_a": k_a,
        "k_b": k_b,
        "p_a": p_a,
        "p_b": p_b,
        "diff": diff,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "z_stat": z_stat,
        "p_val": p_val,
        "ci_a": proportion_confint(k_a, n_a, method="wilson"),
        "ci_b": proportion_confint(k_b, n_b, method="wilson"),
    }


def build_user_funnel(payments_clean: pd.DataFrame) -> dict[str, list[int]]:
    """Строит продуктовую воронку по уникальным пользователям с проверкой монотонности."""
    funnel = {}
    for grp in [CONTROL, TEST]:
        grp_df = payments_clean[payments_clean["group_user"] == grp]
        counts = [int(grp_df[grp_df[col].notna()]["user_id"].nunique()) for col in STEP_COLS]
        assert all(
            counts[i] >= counts[i + 1] for i in range(len(STEP_COLS) - 1)
        ), f"Воронка группы {grp} не монотонна"
        funnel[grp] = counts
    return funnel
