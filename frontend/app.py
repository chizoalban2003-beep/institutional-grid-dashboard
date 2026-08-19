"""Institutional Grid Policy Dashboard — Streamlit Frontend.

Cross-institutional governance evaluation:
Two panels viewing the same patient data with different risk postures.
"""
import streamlit as st
import requests
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Institutional Grid — Policy Dashboard",
    page_icon="🏥",
    layout="wide",
)

st.title("🏥 Institutional Grid — Policy Dashboard")
st.caption("Cross-Institutional Governance Evaluation • CQC-Compliant Audit Ledger")

# ─── Sidebar: Patient Selection ──────────────────────────────────────────────

with st.sidebar:
    st.header("Patient Selection")

    # Fetch available patients
    try:
        patients = requests.get(f"{BACKEND_URL}/patients", timeout=5).json()
    except:
        st.error("Backend unavailable. Is the container running?")
        st.stop()

    if patients:
        patient_ids = list(patients.keys())
        selected_pid = st.selectbox("Patient ID", patient_ids)
        pinfo = patients[selected_pid]
        st.info(f"**{pinfo['n_hours']} hours** • **{pinfo['n_positive']} sepsis-positive**")
    else:
        st.warning("No patients loaded")
        st.stop()

    st.divider()
    st.header("How to Read This Demo")
    st.markdown("""
**Panel A (ICU)**: Low risk tolerance for missed detections. 
Cells bid aggressively → early alerts.

**Panel B (Care Home)**: High cost/risk for false alarms.
Cells wait for stronger signals → later but more confident alerts.

Both panels see the **exact same patient data** — only governance differs.
    """)


# ─── Governance Sliders ─────────────────────────────────────────────────────

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### 🔴 Panel A: Aggressive ICU")
    icu_cost = st.slider("Cost Scale", 0.0, 2.0, 0.3, 0.1, key="icu_cost")
    icu_risk = st.slider("Risk Fine Scale", 0.0, 2.0, 0.2, 0.1, key="icu_risk")
    icu_bounty = st.slider("TP Bounty", 0.0, 5.0, 3.0, 0.5, key="icu_bounty")

with col_b:
    st.markdown("### 🔵 Panel B: Conservative Care Home")
    cc_cost = st.slider("Cost Scale", 0.0, 2.0, 1.0, 0.1, key="cc_cost")
    cc_risk = st.slider("Risk Fine Scale", 0.0, 2.0, 1.5, 0.1, key="cc_risk")
    cc_bounty = st.slider("TP Bounty", 0.0, 5.0, 1.0, 0.5, key="cc_bounty")

# ─── Run Simulation ──────────────────────────────────────────────────────────

if st.button("▶ Run Full Simulation", type="primary"):
    with st.spinner("Streaming patient data through both governance profiles..."):
        try:
            res_icu = requests.post(
                f"{BACKEND_URL}/stream",
                json={
                    "patient_id": selected_pid,
                    "governance": {
                        "cost_scale": icu_cost,
                        "risk_scale": icu_risk,
                        "bounty": icu_bounty,
                    }
                },
                timeout=60,
            ).json()

            res_cc = requests.post(
                f"{BACKEND_URL}/stream",
                json={
                    "patient_id": selected_pid,
                    "governance": {
                        "cost_scale": cc_cost,
                        "risk_scale": cc_risk,
                        "bounty": cc_bounty,
                    }
                },
                timeout=60,
            ).json()
        except Exception as e:
            st.error(f"Backend error: {e}")
            st.stop()

    traj_icu = res_icu['trajectory']
    traj_cc = res_cc['trajectory']
    n_hours = len(traj_icu)

    # ─── Find alert thresholds ─────────────────────────────────────────────
    def find_first_alert(trajectory, threshold=0.5):
        for t in trajectory:
            if t['prediction'] > threshold:
                return t['hour']
        return None

    alert_icu = find_first_alert(traj_icu)
    alert_cc = find_first_alert(traj_cc)

    # Find sepsis onset
    sepsis_onset = None
    for t in traj_icu:
        if t['true_label'] == 1:
            sepsis_onset = t['hour']
            break

    # ─── Summary Metrics ──────────────────────────────────────────────────
    st.divider()
    st.subheader("Governance Comparison")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ICU First Alert", f"Hour {alert_icu}" if alert_icu else "Never",
              f"{'⚡ ' + str(sepsis_onset - alert_icu) + 'h before' if alert_icu and sepsis_onset else ''}")
    m2.metric("Care Home First Alert", f"Hour {alert_cc}" if alert_cc else "Never",
              f"{'⚡ ' + str(sepsis_onset - alert_cc) + 'h before' if alert_cc and sepsis_onset else ''}")
    m3.metric("Alert Gap", f"{(alert_cc or 0) - (alert_icu or 0)}h" if alert_icu and alert_cc else "N/A")
    m4.metric("Sepsis Onset", f"Hour {sepsis_onset}" if sepsis_onset else "None")

    # ─── Main Chart: Predictions Over Time ────────────────────────────────
    st.divider()
    st.subheader("Prediction Trajectory")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4],
                        subplot_titles=("Prediction Probability", "Cell Bid Intensity"))

    hours = [t['hour'] for t in traj_icu]
    pred_icu = [t['prediction'] for t in traj_icu]
    pred_cc = [t['prediction'] for t in traj_cc]
    labels = [t['true_label'] for t in traj_icu]

    # Sepsis onset marker
    if sepsis_onset is not None:
        fig.add_vline(x=sepsis_onset, line_dash="dash", line_color="red",
                      row=1, col=1, annotation_text="Sepsis Onset")

    # Alert markers
    if alert_icu is not None:
        fig.add_vline(x=alert_icu, line_dash="dot", line_color="#e74c3c",
                      row=1, col=1, annotation_text="ICU Alert")
    if alert_cc is not None:
        fig.add_vline(x=alert_cc, line_dash="dot", line_color="#3498db",
                      row=1, col=1, annotation_text="Care Home Alert")

    fig.add_trace(go.Scatter(x=hours, y=pred_icu, name="ICU Prediction",
                             line=dict(color="#e74c3c", width=2), mode='lines'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hours, y=pred_cc, name="Care Home Prediction",
                             line=dict(color="#3498db", width=2), mode='lines'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hours, y=labels, name="True Label",
                             line=dict(color="gray", dash="dash"), mode='lines',
                             showlegend=False), row=1, col=1)

    # Bid intensity
    bid_icu = [t['top_bid'] for t in traj_icu]
    bid_cc = [t['top_bid'] for t in traj_cc]
    fig.add_trace(go.Scatter(x=hours, y=bid_icu, name="ICU Top Bid",
                             line=dict(color="#e74c3c", width=1), mode='lines', opacity=0.6), row=2, col=1)
    fig.add_trace(go.Scatter(x=hours, y=bid_cc, name="Care Home Top Bid",
                             line=dict(color="#3498db", width=1), mode='lines', opacity=0.6), row=2, col=1)

    fig.update_layout(height=600, showlegend=True, template="plotly_white")
    fig.update_yaxes(range=[0, 1.1], row=1, col=1)
    fig.update_yaxes(title_text="Bid Strength", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # ─── Jurisdiction Pie Charts ──────────────────────────────────────────
    st.divider()
    st.subheader("Jurisdiction Economic Posture (Final Hour)")

    # Get final hour detail
    try:
        final_icu = requests.post(
            f"{BACKEND_URL}/predict",
            json={
                "patient_id": selected_pid,
                "hour": n_hours - 1,
                "governance": {"cost_scale": icu_cost, "risk_scale": icu_risk, "bounty": icu_bounty}
            },
            timeout=10,
        ).json()

        final_cc = requests.post(
            f"{BACKEND_URL}/predict",
            json={
                "patient_id": selected_pid,
                "hour": n_hours - 1,
                "governance": {"cost_scale": cc_cost, "risk_scale": cc_risk, "bounty": cc_bounty}
            },
            timeout=10,
        ).json()

        jur_icu = final_icu['jurisdiction_pies']
        jur_cc = final_cc['jurisdiction_pies']

        cols = st.columns(5)
        for i, jur in enumerate(["VITALS", "HYDRATION", "MOBILITY", "COGNITIVE", "OPERATIONAL"]):
            with cols[i]:
                st.markdown(f"**{jur}**")
                col_j1, col_j2 = st.columns(2)
                with col_j1:
                    st.caption("ICU")
                    j = jur_icu[jur]
                    st.bar_chart({"Slice": [j['cost'], j['risk'], j['neutrality']]},
                                 height=120)
                    st.write(f"Cost: {j['cost']:.2f}")
                    st.write(f"Risk: {j['risk']:.2f}")
                    st.write(f"Neutral: {j['neutrality']:.2f}")
                with col_j2:
                    st.caption("Care Home")
                    j = jur_cc[jur]
                    st.bar_chart({"Slice": [j['cost'], j['risk'], j['neutrality']]},
                                 height=120)
                    st.write(f"Cost: {j['cost']:.2f}")
                    st.write(f"Risk: {j['risk']:.2f}")
                    st.write(f"Neutral: {j['neutrality']:.2f}")

        # ─── Top Bidding Cells ────────────────────────────────────────────
        st.divider()
        st.subheader("Top 5 Bidding Cells (Final Hour)")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**ICU Top Cells**")
            for ci in final_icu['top_cells']:
                st.write(f"`Cell {ci['cell_id']:3d}` | {ci['jurisdiction']:12s} | bid={ci['bid']:.3f} | "
                         f"cost={ci['pie']['cost']:.2f} risk={ci['pie']['risk']:.2f} neutral={ci['pie']['neutrality']:.2f}")
        with c2:
            st.markdown("**Care Home Top Cells**")
            for ci in final_cc['top_cells']:
                st.write(f"`Cell {ci['cell_id']:3d}` | {ci['jurisdiction']:12s} | bid={ci['bid']:.3f} | "
                         f"cost={ci['pie']['cost']:.2f} risk={ci['pie']['risk']:.2f} neutral={ci['pie']['neutrality']:.2f}")

    except Exception as e:
        st.warning(f"Could not load final-hour details: {e}")

    # ─── Audit Ledger ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("CQC Audit Ledger (Sample)")

    ledger_data = []
    for t in traj_icu:
        ledger_data.append({
            "Hour": t['hour'],
            "ICU Pred": f"{t['prediction']:.3f}",
            "Care Home Pred": f"{t['prediction']:.3f}",
            "True Label": "Sepsis" if t['true_label'] else "Stable",
            "ICU Alert": "🔴" if t['prediction'] > 0.5 else "",
            "Care Home Alert": "🔵" if t['prediction'] > 0.5 else "",
        })

    st.dataframe(ledger_data, use_container_width=True, height=300)
