import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import Optional

st.set_page_config(page_title="Profit Simulator v1.3.1", layout="wide")
st.title("이익 시뮬레이터 v1.3.1")
st.caption("※ ₩ 기준(원화). 가격/목표매출/대출액 입력은 150,000 처럼 콤마 포함 가능")
st.caption("입력 후 Enter(또는 입력칸 밖 클릭) 시 값이 확정되어 화면에 반영됩니다. 환경에 따라 확정이 한 번 더 필요해 보일 수 있습니다.")

MONTHS = [f"{i}월" for i in range(1, 13)]

# -----------------------------
# Utils
# -----------------------------
def parse_currency(x) -> float:
    if x is None:
        return 0.0
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return 0.0
    s = s.replace("₩", "").replace("원", "").strip()
    s = re.sub(r"[,\s]", "", s)
    s = re.sub(r"[^0-9.\-]", "", s)
    try:
        return float(s) if s else 0.0
    except Exception:
        return 0.0

def fmt_won(v: float) -> str:
    return f"{v:,.0f}"

def normalize_weights(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    if len(s) == 0:
        return s
    total = float(s.sum())
    if total <= 0:
        return pd.Series([1.0/len(s)] * len(s), index=s.index)
    return s / total

def ensure_month_table(state_key: str, col_name: str, default_val: float = 1000.0):
    if state_key not in st.session_state:
        st.session_state[state_key] = pd.DataFrame({"Month": MONTHS, col_name: [default_val]*12})

def build_series_from_month_table(df: pd.DataFrame, col: str) -> pd.Series:
    out = df.copy()
    out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["Month"] = pd.Categorical(out["Month"], categories=MONTHS, ordered=True)
    out = out.sort_values("Month")
    return pd.Series(out[col].values, index=MONTHS)

def currency_input(label: str, key: str, default: float = 0.0, help_text: Optional[str] = None, container=None) -> float:
    """
    Text input that accepts commas and shows normalized ₩ formatting immediately.
    - container: st or st.sidebar (so it renders where you want)
    """
    if container is None:
        container = st

    raw_key = f"{key}_raw"
    if raw_key not in st.session_state:
        st.session_state[raw_key] = f"{default:,.0f}"

    raw = container.text_input(label, key=raw_key, help=help_text, placeholder="예: 150,000")
    val = parse_currency(raw)
    # sidebar에도 caption이 있어서 그대로 사용 가능
    container.caption(f"→ ₩ {val:,.0f}")
    return val

def qty_input(label: str, key: str, default: float = 0.0, help_text: Optional[str] = None, container=None) -> float:
    """
    Quantity input that accepts commas and shows normalized formatting immediately.
    """
    if container is None:
        container = st

    raw_key = f"{key}_raw"
    if raw_key not in st.session_state:
        st.session_state[raw_key] = f"{default:,.0f}"

    raw = container.text_input(label, key=raw_key, help=help_text, placeholder="예: 120,000")
    val = parse_currency(raw)
    container.caption(f"→ {val:,.0f} 개")
    return float(val)

# -----------------------------
# Defaults
# -----------------------------
if "products" not in st.session_state:
    st.session_state.products = pd.DataFrame([
        {"Category": "케이스", "Product": "케이스 A", "Price(₩)": "0", "WeightInCategory": 30.0},
        {"Category": "케이스", "Product": "케이스 B", "Price(₩)": "0", "WeightInCategory": 40.0},
        {"Category": "케이스", "Product": "케이스 C", "Price(₩)": "0", "WeightInCategory": 20.0},
        {"Category": "케이스", "Product": "케이스 D", "Price(₩)": "0", "WeightInCategory": 10.0},
        {"Category": "텀블러", "Product": "텀블러 A", "Price(₩)": "0", "WeightInCategory": 30.0},
        {"Category": "텀블러", "Product": "텀블러 B", "Price(₩)": "0", "WeightInCategory": 40.0},
        {"Category": "텀블러", "Product": "텀블러 C", "Price(₩)": "0", "WeightInCategory": 20.0},
        {"Category": "텀블러", "Product": "텀블러 D", "Price(₩)": "0", "WeightInCategory": 10.0},
    ])

ensure_month_table("monthly_total_units", "TotalUnits", 1000.0)
ensure_month_table("monthly_case_units", "CaseUnits", 700.0)

if "monthly_mom_growth" not in st.session_state:
    st.session_state.monthly_mom_growth = pd.DataFrame({
        "Month": MONTHS,
        "MoM_Growth_%": [0.0] * 12
    })

# -----------------------------
# 1) Product editor
# -----------------------------
st.subheader("1) 제품 설정 (이름/가격/비율/추가/삭제 가능)")

products = st.data_editor(
    st.session_state.products,
    num_rows="dynamic",
    use_container_width=True,
    key="products_editor",
    column_config={
        "Category": st.column_config.SelectboxColumn(options=["케이스", "텀블러"]),
        "Price(₩)": st.column_config.TextColumn(help="예: 150,000 또는 150000"),
        "WeightInCategory": st.column_config.NumberColumn(help="카테고리 내 비율(가중치). 합이 100일 필요 없음", format="%.2f"),
    },
)
for c in ["Category", "Product", "Price(₩)", "WeightInCategory"]:
    if c not in products.columns:
        products[c] = ""
products["WeightInCategory"] = pd.to_numeric(products["WeightInCategory"], errors="coerce").fillna(0.0)
products["PriceNum"] = products["Price(₩)"].apply(parse_currency)
st.session_state.products = products.drop(columns=["PriceNum"], errors="ignore").copy()

# -----------------------------
# Sidebar inputs (즉시 반영)
# -----------------------------
st.sidebar.header("설정 (즉시 반영)")

case_share_pct = st.sidebar.slider("케이스 비중(%)", 0.0, 100.0, 70.0, 1.0, key="case_share_pct")
case_share = case_share_pct / 100.0
tumbler_share = 1.0 - case_share
st.sidebar.caption(f"텀블러 비중(%) = {tumbler_share*100:.0f}% (자동)")

qty_base = st.sidebar.radio(
    "수량 기준",
    ["월별 총 판매량 기준", "월별 케이스 판매량 기준"],
    key="qty_base"
)

monthly_mode = st.sidebar.radio(
    "월별 수량 방식",
    ["월별 직접 입력", "모든 월 동일", "자동 생성(전월 기준)"],
    key="monthly_mode"
)

jan_units = qty_input("1월 기준 수량(개)", key="jan_units", default=1000, help_text="예: 120,000", container=st.sidebar)

autogen_mode = None
uniform_mom = 0.0
if monthly_mode == "자동 생성(전월 기준)":
    autogen_mode = st.sidebar.radio(
        "자동 생성 방식",
        ["전월과 동일", "전월대비 % 동일", "월별 전월대비 % 각각 입력"],
        key="autogen_mode"
    )
    if autogen_mode == "전월대비 % 동일":
        uniform_mom = st.sidebar.number_input("전월대비 증가율(%)", value=0.0, step=0.5, key="uniform_mom")
    elif autogen_mode == "월별 전월대비 % 각각 입력":
        st.sidebar.caption("월별 MoM%는 메인 화면의 표에서 입력 (1월은 0 추천)")

st.sidebar.divider()
st.sidebar.markdown("### 비용")
royalty_pct = st.sidebar.number_input("로열티율(매출%)", value=10.0, step=0.5, key="royalty_pct")
lid_cost = st.sidebar.number_input("증지/종지(?) 개당 비용(원)", value=5.0, step=1.0, key="lid_cost")

use_cogs = st.sidebar.checkbox("COGS(매출%) 사용", value=False, key="use_cogs")
cogs_pct = st.sidebar.number_input("COGS율(매출%)", value=30.0, step=1.0, key="cogs_pct") if use_cogs else 0.0

use_interest = st.sidebar.checkbox("이자비용 사용", value=False, key="use_interest")
loan_amount = currency_input("총 대출액(₩)", key="loan_amount", default=0, help_text="예: 100,000,000", container=st.sidebar) if use_interest else 0.0
annual_rate_pct = st.sidebar.number_input("연 이자율(%)", value=8.0, step=0.5, key="annual_rate_pct") if use_interest else 0.0

st.sidebar.divider()
st.sidebar.markdown("### 목표")
target_revenue = currency_input("목표 연 매출(₩)", key="target_revenue", default=0, help_text="예: 1,000,000,000", container=st.sidebar)

# -----------------------------
# 2) Monthly quantity configuration
# -----------------------------
st.subheader("2) 월별 수량 설정")

if qty_base == "월별 총 판매량 기준":
    base_key, base_col, base_label = "monthly_total_units", "TotalUnits", "월별 총 판매량(개)"
else:
    base_key, base_col, base_label = "monthly_case_units", "CaseUnits", "월별 케이스 판매량(개)"

with st.expander("월별 수량 입력/생성", expanded=True):
    st.caption(f"현재 기준: **{qty_base}** / 입력 대상: **{base_label}**")

    if monthly_mode == "월별 직접 입력":
        st.session_state[base_key] = st.data_editor(
            st.session_state[base_key],
            use_container_width=True,
            num_rows="fixed",
            key=f"{base_key}_editor",
            column_config={base_col: st.column_config.NumberColumn(format="%.0f", min_value=0.0)},
        )

    elif monthly_mode == "모든 월 동일":
        v = float(jan_units)
        st.session_state[base_key] = pd.DataFrame({"Month": MONTHS, base_col: [v]*12})
        st.info("모든 월 동일로 생성됨 (1월 기준 값)")

    else:
        jan = float(jan_units)

        if autogen_mode == "전월과 동일":
            vals = [jan] * 12

        elif autogen_mode == "전월대비 % 동일":
            g = 1.0 + float(uniform_mom)/100.0
            vals = []
            cur = jan
            for _ in range(12):
                vals.append(cur)
                cur *= g

        else:
            st.markdown("#### 월별 전월대비(%) 입력 (달마다 다르게 가능)")
            mom_df = st.data_editor(
                st.session_state.monthly_mom_growth,
                use_container_width=True,
                num_rows="fixed",
                key="mom_growth_editor",
                column_config={
                    "MoM_Growth_%": st.column_config.NumberColumn(help="전월대비 증가율(%)", step=0.5, format="%.2f"),
                }
            )
            st.session_state.monthly_mom_growth = mom_df.copy()

            mom_series = build_series_from_month_table(mom_df, "MoM_Growth_%")
            vals = [jan]
            for i in range(1, 12):
                g = 1.0 + float(mom_series.iloc[i])/100.0
                vals.append(vals[-1] * g)

        st.session_state[base_key] = pd.DataFrame({"Month": MONTHS, base_col: vals})
        st.success("자동 생성 완료")

# Build base DF
base_df = st.session_state[base_key].copy()
for c in ["Month", base_col]:
    if c not in base_df.columns:
        base_df[c] = 0
base_df[base_col] = pd.to_numeric(base_df[base_col], errors="coerce").fillna(0.0)
base_df["Month"] = pd.Categorical(base_df["Month"], categories=MONTHS, ordered=True)
base_df = base_df.sort_values("Month").reset_index(drop=True)

# -----------------------------
# 3) Convert base -> Total/Case/Tumbler monthly units
# -----------------------------
if qty_base == "월별 총 판매량 기준":
    monthly_total = base_df.rename(columns={base_col: "TotalUnits"})[["Month", "TotalUnits"]].copy()
    monthly_total["CaseUnits"] = monthly_total["TotalUnits"] * case_share
    monthly_total["TumblerUnits"] = monthly_total["TotalUnits"] * tumbler_share
else:
    monthly_case = base_df.rename(columns={base_col: "CaseUnits"})[["Month", "CaseUnits"]].copy()
    if case_share <= 0:
        st.error("케이스 비중이 0%라서 '케이스 기준 → 총량 역산'이 불가능합니다. 케이스 비중을 1% 이상으로 올려주세요.")
        st.stop()
    monthly_case["TotalUnits"] = monthly_case["CaseUnits"] / case_share
    monthly_case["TumblerUnits"] = monthly_case["TotalUnits"] - monthly_case["CaseUnits"]
    monthly_total = monthly_case[["Month", "TotalUnits", "CaseUnits", "TumblerUnits"]].copy()

monthly_total["Month"] = pd.Categorical(monthly_total["Month"], categories=MONTHS, ordered=True)
monthly_total = monthly_total.sort_values("Month").reset_index(drop=True)

# -----------------------------
# 4) Allocate to products + revenue
# -----------------------------
prod = products.copy()
prod["PriceNum"] = prod["Price(₩)"].apply(parse_currency)
prod["CatWeightNorm"] = 0.0
for cat in ["케이스", "텀블러"]:
    idx = prod["Category"] == cat
    prod.loc[idx, "CatWeightNorm"] = normalize_weights(prod.loc[idx, "WeightInCategory"])

rows = []
for _, mrow in monthly_total.iterrows():
    m = str(mrow["Month"])
    cat_units = {"케이스": float(mrow["CaseUnits"]), "텀블러": float(mrow["TumblerUnits"])}
    for _, p in prod.iterrows():
        cat = p["Category"]
        units = cat_units.get(cat, 0.0) * float(p["CatWeightNorm"])
        revenue = units * float(p["PriceNum"])
        rows.append({
            "Month": m,
            "Category": cat,
            "Product": p["Product"],
            "Units": units,
            "Price(₩)": p["Price(₩)"],
            "Revenue": revenue
        })

detail = pd.DataFrame(rows)

monthly = detail.groupby("Month", as_index=False).agg(
    TotalUnits=("Units", "sum"),
    Revenue=("Revenue", "sum"),
)
monthly["Month"] = pd.Categorical(monthly["Month"], categories=MONTHS, ordered=True)
monthly = monthly.sort_values("Month").reset_index(drop=True)

# Costs
royalty_rate = float(royalty_pct)/100.0
cogs_rate = float(cogs_pct)/100.0 if use_cogs else 0.0

monthly["Royalty"] = monthly["Revenue"] * royalty_rate
monthly["LidCost"] = monthly["TotalUnits"] * float(lid_cost)
monthly["COGS"] = monthly["Revenue"] * cogs_rate if use_cogs else 0.0

monthly_interest = 0.0
if use_interest:
    monthly_interest = float(loan_amount) * (float(annual_rate_pct)/100.0) / 12.0
monthly["Interest"] = monthly_interest

monthly["TotalCost"] = monthly["Royalty"] + monthly["LidCost"] + monthly["COGS"] + monthly["Interest"]
monthly["OperatingProfit"] = monthly["Revenue"] - monthly["TotalCost"]

# -----------------------------
# 5) P&L vertical
# -----------------------------
st.subheader("3) 월별 손익계산서 (세로 배치 + 우측 Total)")

midx = monthly.set_index("Month")
pnl_items = {
    "매출(Revenue)": midx["Revenue"],
    "로열티(Royalty)": midx["Royalty"],
    "증지/종지(LidCost)": midx["LidCost"],
}
if use_cogs:
    pnl_items["COGS"] = midx["COGS"]
if use_interest:
    pnl_items["이자비용(Interest)"] = midx["Interest"]

pnl_items["비용합계(TotalCost)"] = midx["TotalCost"]
pnl_items["영업이익(OperatingProfit)"] = midx["OperatingProfit"]

pnl = pd.DataFrame(pnl_items).T
pnl = pnl.reindex(columns=MONTHS)
pnl["Total"] = pnl.sum(axis=1)

pnl_display = pnl.applymap(lambda v: fmt_won(v))
st.dataframe(pnl_display, use_container_width=True)

# -----------------------------
# 6) Target revenue -> required units (annual)
# -----------------------------
st.subheader("4) 목표 연 매출 달성 시 필요한 판매량(자동 계산)")

annual_revenue = float(monthly["Revenue"].sum())
annual_total_units = float(monthly_total["TotalUnits"].sum())
annual_case_units = float(monthly_total["CaseUnits"].sum())

if target_revenue and target_revenue > 0:
    if annual_revenue <= 0:
        st.warning("현재 연매출이 0원이라 목표 매출 역산이 불가능합니다. 제품 가격(₩)을 1개 이상 입력해줘.")
    else:
        scale = float(target_revenue) / annual_revenue

        req_monthly_total = monthly_total.copy()
        req_monthly_total["Req_TotalUnits"] = req_monthly_total["TotalUnits"] * scale
        req_monthly_total["Req_CaseUnits"] = req_monthly_total["CaseUnits"] * scale
        req_monthly_total["Req_TumblerUnits"] = req_monthly_total["TumblerUnits"] * scale

        req_annual_total_units = annual_total_units * scale
        req_annual_case_units = annual_case_units * scale

        c1, c2, c3 = st.columns(3)
        c1.metric("현재 연 매출", f"₩ {fmt_won(annual_revenue)}")
        c2.metric("목표 연 매출", f"₩ {fmt_won(target_revenue)}")
        c3.metric("필요 판매량 배율", f"{scale:,.3f} x")

        st.write(
            f"- 목표 달성에 필요한 **연 총 판매량(추정)**: **{req_annual_total_units:,.0f}개**  \n"
            f"- 목표 달성에 필요한 **연 케이스 판매량(추정)**: **{req_annual_case_units:,.0f}개**"
        )

        with st.expander("월별 필요한 판매량(총/케이스/텀블러)", expanded=True):
            show = req_monthly_total[["Month", "Req_TotalUnits", "Req_CaseUnits", "Req_TumblerUnits"]].copy()
            show.columns = ["Month", "TotalUnits(필요)", "CaseUnits(필요)", "TumblerUnits(필요)"]
            show_fmt = show.copy()
            for col in ["TotalUnits(필요)", "CaseUnits(필요)", "TumblerUnits(필요)"]:
                show_fmt[col] = pd.to_numeric(show_fmt[col], errors="coerce").fillna(0).map(lambda x: f"{x:,.0f}")
            st.dataframe(show_fmt, use_container_width=True)

        detail_req = detail.copy()
        detail_req["Units_req"] = detail_req["Units"] * scale
        detail_req["Revenue_req"] = detail_req["Revenue"] * scale

        prod_year_req = detail_req.groupby(["Category","Product"], as_index=False).agg(
            AnnualUnitsRequired=("Units_req","sum"),
            AnnualRevenueRequired=("Revenue_req","sum")
        )

        with st.expander("제품별 연간 필요 판매량(현재 믹스/비중 유지 가정)", expanded=False):
            prod_year_req_fmt = prod_year_req.copy()
            prod_year_req_fmt["AnnualUnitsRequired"] = pd.to_numeric(prod_year_req_fmt["AnnualUnitsRequired"], errors="coerce").fillna(0).map(lambda x: f"{x:,.0f}")
            prod_year_req_fmt["AnnualRevenueRequired"] = pd.to_numeric(prod_year_req_fmt["AnnualRevenueRequired"], errors="coerce").fillna(0).map(lambda x: fmt_won(x))
            st.dataframe(prod_year_req_fmt, use_container_width=True)
else:
    st.caption("목표 연 매출(₩)을 입력하면, 현재 가격/분배를 유지한다는 가정 하에 필요한 판매량을 자동 계산합니다.")

# -----------------------------
# 7) Optional debug
# -----------------------------
with st.expander("추가 확인용(월별 총/케이스/텀블러 수량, 제품별 상세)", expanded=False):
    st.markdown("**월별 총/케이스/텀블러 수량**")
    monthly_total_fmt = monthly_total.copy()
    for col in ["TotalUnits", "CaseUnits", "TumblerUnits"]:
        monthly_total_fmt[col] = pd.to_numeric(monthly_total_fmt[col], errors="coerce").fillna(0).map(lambda x: f"{x:,.0f}")
    st.dataframe(monthly_total_fmt, use_container_width=True)

    st.markdown("**제품별 월별 상세(수량/매출)**")
    show_detail = detail.copy()
    show_detail["Revenue(₩)"] = show_detail["Revenue"]
    show_detail = show_detail.drop(columns=["Revenue"])
    show_detail_fmt = show_detail.copy()
    show_detail_fmt["Units"] = pd.to_numeric(show_detail_fmt["Units"], errors="coerce").fillna(0).map(lambda x: f"{x:,.2f}")
    show_detail_fmt["Revenue(₩)"] = pd.to_numeric(show_detail_fmt["Revenue(₩)"], errors="coerce").fillna(0).map(lambda x: fmt_won(x))
    st.dataframe(show_detail_fmt, use_container_width=True)
