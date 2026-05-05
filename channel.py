import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import base64


st.set_page_config(
    page_title="Dynamic Pricing Simulator",
    page_icon="⚡",
    layout="wide"
)


def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

img = get_base64_image("C:/Users/Mega Store/Downloads/360_F_432925015_F9ABhDbYB59L14rMt77rgl6gLNAw6jeC.jpg")

page_bg = f"""
<style>

[data-testid="stAppViewContainer"] {{
background-image:
linear-gradient(rgba(0,0,0,0.5),
rgba(0,0,0,0.5)),
url("data:image/jpg;base64,{img}");

background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stSidebar"] {{
background: rgba(0,0,0,0.3);
}}

h1, h2, h3, h4, h5, h6, p, label, div {{
color: white;
}}

</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)


BASE_RATE = 0.25
PEAK_RATE = 0.80
PENALTY_RATE = 1.20
DISCOUNT_RATE = 0.15


def generate_training_data(n=1000):
    np.random.seed(42)

    lamps = np.random.randint(1, 15, n)
    acs = np.random.randint(0, 6, n)
    washing = np.random.randint(0, 2, n)
    heavy_machines = np.random.randint(0, 5, n)
    occupants = np.random.randint(1, 8, n)
    house_size = np.random.randint(50, 300, n)

    baseline = (
        0.25 * lamps +
        1.1 * acs +
        1.0 * washing +
        1.4 * heavy_machines +
        0.35 * occupants +
        0.01 * house_size +
        np.random.normal(0, 0.4, n)
    )

    baseline = np.clip(baseline, 0.5, None)

    df = pd.DataFrame({
        "lamps": lamps,
        "acs": acs,
        "washing_machine": washing,
        "heavy_machines": heavy_machines,
        "occupants": occupants,
        "house_size": house_size,
        "historical_baseline_kwh": baseline
    })

    return df


@st.cache_resource
def train_model():
    df = generate_training_data()

    X = df.drop(columns=["historical_baseline_kwh"])
    y = df["historical_baseline_kwh"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, preds),
        "R2": r2_score(y_test, preds)
    }

    return model, metrics, df


def calculate_scenario_1(name, baseline, actual_usage):
    allowed_usage = baseline * 0.30

    if actual_usage <= allowed_usage:
        bill = actual_usage * BASE_RATE
        status = "Reduced enough"
    else:
        normal_part = allowed_usage * BASE_RATE
        penalty_part = (actual_usage - allowed_usage) * PENALTY_RATE
        bill = normal_part + penalty_part
        status = "Penalty applied"

    return {
        "Person": name,
        "Baseline kWh": baseline,
        "Actual Usage kWh": actual_usage,
        "Allowed After 70% Reduction": allowed_usage,
        "Bill": bill,
        "Status": status
    }


def calculate_scenario_2(name, baseline, actual_usage, stayed_below_peak):
    if actual_usage <= baseline:
        bill = actual_usage * BASE_RATE
        status = "Normal rate"
    else:
        normal_part = baseline * BASE_RATE
        extra_part = (actual_usage - baseline) * PENALTY_RATE
        bill = normal_part + extra_part
        status = "Only extra usage penalized"

    if stayed_below_peak:
        discount = bill * DISCOUNT_RATE
        bill_after_discount = bill - discount
    else:
        discount = 0
        bill_after_discount = bill

    return {
        "Person": name,
        "Personal Baseline kWh": baseline,
        "Actual Usage kWh": actual_usage,
        "Bill Before Discount": bill,
        "Discount": discount,
        "Final Bill": bill_after_discount,
        "Status": status
    }


def calculate_scenario_3(baseline, actual_usage, response_mode):
    above_baseline = max(actual_usage - baseline, 0)

    if response_mode == "Notify only":
        final_usage = actual_usage
        comfort = 100
    elif response_mode == "Auto shed non-critical loads":
        final_usage = baseline
        comfort = 75
    else:
        final_usage = actual_usage
        comfort = 100

    normal_usage = min(final_usage, baseline)
    premium_usage = max(final_usage - baseline, 0)

    bill = normal_usage * BASE_RATE + premium_usage * PEAK_RATE

    return final_usage, bill, comfort, premium_usage


def household_input(title, default_lamps, default_acs, default_washing, default_heavy, default_occupants, default_size):
    st.subheader(title)

    lamps = st.slider(f"{title} - Lamps", 1, 20, default_lamps)
    acs = st.slider(f"{title} - ACs", 0, 8, default_acs)
    washing = st.slider(f"{title} - Washing Machine", 0, 1, default_washing)
    heavy = st.slider(f"{title} - Heavy Machines", 0, 8, default_heavy)
    occupants = st.slider(f"{title} - Occupants", 1, 10, default_occupants)
    size = st.slider(f"{title} - House Size m²", 40, 400, default_size)

    return pd.DataFrame([{
        "lamps": lamps,
        "acs": acs,
        "washing_machine": washing,
        "heavy_machines": heavy,
        "occupants": occupants,
        "house_size": size
    }])


model, metrics, training_df = train_model()

st.title("⚡ Dynamic Pricing Simulator for Electrical Distribution")
st.markdown(
    """
This app simulates three dynamic pricing scenarios:

1. **Same 70% reduction rule for everyone**
2. **Personal historical baseline pricing**
3. **Pay-to-play comfort pricing**
"""
)

with st.sidebar:
    st.header("Model Performance")
    st.metric("MAE", f"{metrics['MAE']:.2f} kWh")
    st.metric("R² Score", f"{metrics['R2']:.2f}")

    st.divider()

    st.header("Electricity Prices")
    st.write(f"Normal rate: **{BASE_RATE} EGP/kWh**")
    st.write(f"Peak rate: **{PEAK_RATE} EGP/kWh**")
    st.write(f"Penalty rate: **{PENALTY_RATE} EGP/kWh**")
    st.write(f"Loyalty discount: **{int(DISCOUNT_RATE * 100)}%**")


col1, col2 = st.columns(2)

with col1:
    person_a = household_input(
        "Person A: Low Baseline Home",
        default_lamps=3,
        default_acs=0,
        default_washing=0,
        default_heavy=0,
        default_occupants=1,
        default_size=60
    )

with col2:
    person_b = household_input(
        "Person B: Heavy Usage Home",
        default_lamps=10,
        default_acs=4,
        default_washing=1,
        default_heavy=3,
        default_occupants=5,
        default_size=220
    )


baseline_a = float(model.predict(person_a)[0])
baseline_b = float(model.predict(person_b)[0])

st.divider()

m1, m2 = st.columns(2)
m1.metric("Predicted Baseline - Person A", f"{baseline_a:.2f} kWh")
m2.metric("Predicted Baseline - Person B", f"{baseline_b:.2f} kWh")

tab1, tab2, tab3, tab4 = st.tabs([
    "Scenario 1",
    "Scenario 2",
    "Scenario 3",
    "ML Dataset"
])


with tab1:
    st.header("Scenario 1: Same 70% Reduction Rule")

    st.markdown(
        """
In this scenario, the utility applies the same rule to everyone:

> During high district load, every household must reduce consumption by 70%,  
> otherwise the extra usage is charged at a high penalty rate.
"""
    )

    actual_a = st.slider("Person A actual usage during peak event", 0.1, 15.0, baseline_a, step=0.1)
    actual_b = st.slider("Person B actual usage during peak event", 0.1, 20.0, baseline_b, step=0.1)

    result_a = calculate_scenario_1("Person A", baseline_a, actual_a)
    result_b = calculate_scenario_1("Person B", baseline_b, actual_b)

    result_df = pd.DataFrame([result_a, result_b])

    st.dataframe(result_df, use_container_width=True)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=result_df["Person"],
        y=result_df["Baseline kWh"],
        name="Baseline"
    ))

    fig.add_trace(go.Bar(
        x=result_df["Person"],
        y=result_df["Allowed After 70% Reduction"],
        name="Allowed Usage"
    ))

    fig.add_trace(go.Bar(
        x=result_df["Person"],
        y=result_df["Actual Usage kWh"],
        name="Actual Usage"
    ))

    fig.update_layout(
        title="Scenario 1: Same Reduction Rule",
        barmode="group",
        yaxis_title="Energy kWh"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "This rule is unfair because Person A may be forced into near-darkness, "
        "while Person B can reduce luxury loads and still maintain useful comfort."
    )


with tab2:
    st.header("Scenario 2: Personalized Historical Baseline")

    st.markdown(
        """
Instead of comparing households to each other, each household is compared to its own normal behavior.

The ML model estimates a reasonable baseline using household features.
Only usage above this personal baseline is penalized.
"""
    )

    actual_a2 = st.slider("Person A usage", 0.1, 15.0, baseline_a, step=0.1, key="a2")
    actual_b2 = st.slider("Person B usage", 0.1, 20.0, baseline_b, step=0.1, key="b2")

    discount_a = st.checkbox("Person A stayed below baseline during repeated peak events", value=True)
    discount_b = st.checkbox("Person B stayed below baseline during repeated peak events", value=False)

    result_a2 = calculate_scenario_2("Person A", baseline_a, actual_a2, discount_a)
    result_b2 = calculate_scenario_2("Person B", baseline_b, actual_b2, discount_b)

    result_df2 = pd.DataFrame([result_a2, result_b2])

    st.dataframe(result_df2, use_container_width=True)

    fig2 = go.Figure()

    fig2.add_trace(go.Bar(
        x=result_df2["Person"],
        y=result_df2["Personal Baseline kWh"],
        name="Personal Baseline"
    ))

    fig2.add_trace(go.Bar(
        x=result_df2["Person"],
        y=result_df2["Actual Usage kWh"],
        name="Actual Usage"
    ))

    fig2.update_layout(
        title="Scenario 2: Personal Baseline Comparison",
        barmode="group",
        yaxis_title="Energy kWh"
    )

    st.plotly_chart(fig2, use_container_width=True)

    fig_bill = px.bar(
        result_df2,
        x="Person",
        y=["Bill Before Discount", "Discount", "Final Bill"],
        barmode="group",
        title="Bill and Discount Comparison"
    )

    st.plotly_chart(fig_bill, use_container_width=True)

    st.success(
        "This scenario is fairer because Person B is not forced down to Person A's low usage. "
        "Each user is judged against their own historical normal usage."
    )


with tab3:
    st.header("Scenario 3: Pay-to-Play System")

    st.markdown(
        """
The utility does not directly shut down your loads.

Instead, it broadcasts a peak price event:

> Usage above baseline between 5:00 PM and 7:00 PM costs more.

Your home automation system decides what to do.
"""
    )

    selected_person = st.radio("Choose household", ["Person A", "Person B"])

    if selected_person == "Person A":
        baseline = baseline_a
        default_usage = baseline_a + 1
    else:
        baseline = baseline_b
        default_usage = baseline_b + 3

    usage = st.slider(
        "Requested usage during 5:00 PM - 7:00 PM",
        0.1,
        25.0,
        float(default_usage),
        step=0.1
    )

    mode = st.selectbox(
        "Automation Response",
        [
            "Notify only",
            "Auto shed non-critical loads",
            "Take no action and pay premium"
        ]
    )

    final_usage, bill, comfort, premium_usage = calculate_scenario_3(
        baseline,
        usage,
        mode
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Baseline", f"{baseline:.2f} kWh")
    c2.metric("Final Usage", f"{final_usage:.2f} kWh")
    c3.metric("Premium Usage", f"{premium_usage:.2f} kWh")
    c4.metric("Comfort Level", f"{comfort}%")

    st.metric("Final Bill", f"${bill:.2f}")

    fig3 = go.Figure()

    fig3.add_trace(go.Bar(
        x=["Requested Usage", "Final Usage", "Baseline"],
        y=[usage, final_usage, baseline]
    ))

    fig3.update_layout(
        title="Pay-to-Play Usage Decision",
        yaxis_title="Energy kWh"
    )

    st.plotly_chart(fig3, use_container_width=True)

    if mode == "Auto shed non-critical loads":
        st.info(
            "The automation system reduced usage to the baseline, saving money but lowering comfort."
        )
    elif mode == "Notify only":
        st.info(
            "The system only notifies the user. The user still pays premium if usage exceeds baseline."
        )
    else:
        st.info(
            "The user keeps full comfort and pays the premium price for extra usage."
        )


with tab4:
    st.header("Synthetic ML Training Dataset")

    st.markdown(
        """
The model learns the relationship between household features and historical baseline energy usage.
In a real system, this dataset would come from smart meter readings.
"""
    )

    st.dataframe(training_df.head(100), use_container_width=True)

    fig_data = px.scatter(
        training_df,
        x="acs",
        y="historical_baseline_kwh",
        size="heavy_machines",
        color="occupants",
        title="Relationship Between ACs, Heavy Machines, Occupants, and Baseline Usage"
    )

    st.plotly_chart(fig_data, use_container_width=True)
