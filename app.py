
import json
from pathlib import Path
from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.image_processing import load_image, process_image
from modules.classification import assess_image_quality
from modules.database import initialize_database, save_analysis, get_analyses


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Chest X-Ray | Medical AI",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

initialize_database()

PAGES = [
    "Home",
    "Dashboard",
    "Patient Registration & Analysis",
    "Image Processing Laboratory",
    "Classification Results",
    "Dataset Management",
    "Reports",
]


# ============================================================
# THEME
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #F4F8FC;
        --surface: #FFFFFF;
        --border: #DCE7F1;
        --text: #17324D;
        --muted: #647D95;
        --blue: #1769AA;
        --blue-dark: #104C82;
        --teal: #159E9A;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(255,255,255,.96);
    }

    [data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: var(--text);
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 0 22px;
    }

    .brand-icon, .module-icon {
        background: #E5F1FC;
        border: 1px solid #C9DFF2;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .brand-icon {
        width: 50px;
        height: 50px;
        font-size: 27px;
    }

    .brand-name {
        font-weight: 800;
        font-size: 17px;
        color: #153A5B;
    }

    .brand-sub {
        font-size: 9px;
        letter-spacing: 1.5px;
        color: #66819A;
        margin-top: 4px;
    }

    .hero {
        background: linear-gradient(120deg,#FFFFFF 0%,#EAF4FD 100%);
        border: 1px solid #D4E5F4;
        border-radius: 23px;
        padding: 38px;
        margin: 4px 0 25px;
        box-shadow: 0 10px 30px rgba(32,77,116,.06);
    }

    .eyebrow {
        color: #1769AA;
        font-weight: 800;
        font-size: 11px;
        letter-spacing: 1.8px;
        text-transform: uppercase;
    }

    .hero h1 {
        color: #153A5B;
        font-size: clamp(29px,4vw,43px);
        font-weight: 800;
        line-height: 1.2;
        margin: 13px 0;
    }

    .hero p {
        color: #58738D;
        line-height: 1.9;
        font-size: 14px;
        max-width: 780px;
    }

    .page-heading {
        color: #153A5B;
        font-size: 29px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .page-description {
        color: var(--muted);
        font-size: 13px;
        margin-bottom: 25px;
    }

    .module-icon {
        width: 43px;
        height: 43px;
        font-size: 22px;
        margin-bottom: 12px;
    }

    .module-card h4 {
        color: #153A5B;
        font-size: 16px;
        margin: 4px 0;
    }

    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #DCE7F1;
        border-radius: 15px;
        padding: 17px 19px;
        box-shadow: 0 5px 18px rgba(32,77,116,.035);
    }

    [data-testid="stMetricLabel"] {
        color: #647D95;
        font-size: 12px;
    }

    [data-testid="stMetricValue"] {
        color: #153A5B;
        font-weight: 800;
    }

    div.stButton > button {
        background: #FFFFFF;
        color: #195B91;
        border: 1px solid #C9DCEC;
        border-radius: 10px;
        min-height: 42px;
        font-weight: 600;
        transition: .2s;
    }

    div.stButton > button:hover {
        background: #EDF6FE;
        border-color: #1769AA;
        color: #104C82;
    }

    div.stButton > button[kind="primary"] {
        background: #1769AA;
        color: #FFFFFF;
        border: 1px solid #1769AA;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #104C82;
        color: #FFFFFF;
    }

    [data-testid="stForm"],
    [data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1px solid #DCE7F1;
        border-radius: 12px;
    }

    [data-testid="stFileUploader"] {
        background: #F8FBFE;
        border: 1px dashed #AFCBE3;
        border-radius: 12px;
        padding: 12px;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #DCE7F1;
        border-radius: 12px;
    }

    .result-fit {
        background: #E9F8F1;
        border: 1px solid #A6DFC4;
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        color: #16754F;
    }

    .result-unfit {
        background: #FFF0EF;
        border: 1px solid #F0C0BC;
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        color: #B33B39;
    }

    .result-label {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.5px;
    }

    .result-value {
        font-size: 38px;
        font-weight: 800;
        margin: 8px 0;
    }

    .status-chip {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 30px;
        background: #E5F1FC;
        color: #1769AA;
        border: 1px solid #C9DFF2;
        font-size: 11px;
        font-weight: 700;
    }

    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stDateInput"] input,
    [data-testid="stTextArea"] textarea {
        background: #FFFFFF;
        color: #17324D;
        border-color: #C9DCEC;
    }

    [data-baseweb="select"] > div {
        background: #FFFFFF;
        border-color: #C9DCEC;
    }

    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4 {
        color: #153A5B;
    }

    hr {
        border-color: #DCE7F1;
    }

    .footer {
        text-align: center;
        color: #7892A7;
        font-size: 11px;
        padding: 25px 0 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# NAVIGATION
# ============================================================

if "navigation" not in st.session_state:
    st.session_state["navigation"] = "Home"

if "next_navigation" in st.session_state:
    requested_page = st.session_state.pop("next_navigation")
    if requested_page in PAGES:
        st.session_state["navigation"] = requested_page


def go_to(page):
    st.session_state["next_navigation"] = page


def page_header(title, description):
    st.markdown(
        f'<div class="page-heading">{title}</div>'
        f'<div class="page-description">{description}</div>',
        unsafe_allow_html=True,
    )


def result_panel(status):
    if status == "Fit":
        css, symbol = "result-fit", "✓"
    else:
        css, symbol = "result-unfit", "!"

    st.markdown(
        f"""
        <div class="{css}">
            <div class="result-label">IMAGE QUALITY SCREENING RESULT</div>
            <div class="result-value">{symbol} {status}</div>
            <div>Preliminary image-quality assessment</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def navigation_buttons(previous_page, next_page, prefix):
    st.divider()
    left, right = st.columns(2)

    with left:
        st.button(
            f"← {previous_page}",
            key=f"{prefix}_previous",
            use_container_width=True,
            on_click=go_to,
            args=(previous_page,),
        )

    with right:
        st.button(
            f"{next_page} →",
            key=f"{prefix}_next",
            type="primary",
            use_container_width=True,
            on_click=go_to,
            args=(next_page,),
        )


# ============================================================
# DATABASE READ AND COLUMN MAPPING
# ============================================================

def read_records():
    records = get_analyses()

    columns = [
        "Record ID",
        "Case ID",
        "Patient Name",
        "Age",
        "Gender",
        "Image Name",
        "Reference Fit",
        "Predicted Fit",
        "Reference Medical Label",
        "Predicted Medical Label",
        "Measurements",
        "Analysis Date",
    ]

    if records is None or len(records) == 0:
        return pd.DataFrame(columns=columns)

    # get_analyses() returns tuples in this exact order.
    if isinstance(records, pd.DataFrame):
        df = records.copy()
    elif isinstance(records[0], dict):
        df = pd.DataFrame(records)
    else:
        df = pd.DataFrame(records, columns=columns)

    aliases = {
        "id": "Record ID",
        "case_id": "Case ID",
        "patient_name": "Patient Name",
        "age": "Age",
        "gender": "Gender",
        "image_name": "Image Name",
        "reference_fit": "Reference Fit",
        "predicted_fit": "Predicted Fit",
        "reference_medical_label": "Reference Medical Label",
        "predicted_medical_label": "Predicted Medical Label",
        "measurements": "Measurements",
        "analysis_date": "Analysis Date",
        "created_at": "Analysis Date",
    }

    for old, new in aliases.items():
        if new not in df.columns and old in df.columns:
            df[new] = df[old]

    for col in columns:
        if col not in df.columns:
            df[col] = None

    return df[columns]


def chart_layout(fig, title=None):
    fig.update_layout(
        title=title,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font_color="#17324D",
        margin=dict(l=15, r=15, t=55, b=15),
        legend=dict(bgcolor="rgba(255,255,255,0)"),
    )
    fig.update_xaxes(
        gridcolor="#E4ECF4",
        zerolinecolor="#DCE7F1",
    )
    fig.update_yaxes(
        gridcolor="#E4ECF4",
        zerolinecolor="#DCE7F1",
    )
    return fig


try:
    records_df = read_records()
except Exception as exc:
    st.error(f"Could not read the database: {exc}")
    records_df = pd.DataFrame(columns=[
        "Record ID", "Case ID", "Patient Name", "Age", "Gender",
        "Image Name", "Reference Fit", "Predicted Fit",
        "Reference Medical Label", "Predicted Medical Label",
        "Measurements", "Analysis Date",
    ])


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">🩻</div>
            <div>
                <div class="brand-name">Chest X-Ray AI</div>
                <div class="brand-sub">IMAGE ANALYSIS SYSTEM</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("WORKSPACE")

    st.radio(
        "Navigation",
        PAGES,
        key="navigation",
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**QUICK ACTIONS**")

    st.button(
        "＋ Start a New Analysis",
        key="sidebar_new_analysis",
        type="primary",
        use_container_width=True,
        on_click=go_to,
        args=("Patient Registration & Analysis",),
    )

    st.button(
        "⌂ Return to Home",
        key="sidebar_home",
        use_container_width=True,
        on_click=go_to,
        args=("Home",),
    )

    st.divider()
    st.caption("APPLICATION STATUS")
    st.markdown(
        '<span class="status-chip">● SYSTEM READY</span>',
        unsafe_allow_html=True,
    )
    st.caption("Academic medical imaging project")

page = st.session_state["navigation"]


# ============================================================
# HOME
# ============================================================

if page == "Home":
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">✦ Medical Imaging Intelligence</div>
            <h1>Chest X-Ray Image<br>Processing & Classification</h1>
            <p>
                A professional image-analysis workspace for chest X-ray
                processing, measurement visualization, Fit/Unfit
                image-quality screening, and structured case management.
            </p>
            <span class="status-chip">IMAGE PROCESSING</span>
            &nbsp;
            <span class="status-chip">QUALITY CLASSIFICATION</span>
            &nbsp;
            <span class="status-chip">DATA ANALYTICS</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Welcome to your analysis workspace")
    st.write(
        "Start a new case, inspect processing outputs, or explore "
        "your saved dataset and analytics."
    )

    st.button(
        "＋ Start a New Analysis",
        key="home_start_analysis",
        type="primary",
        use_container_width=True,
        on_click=go_to,
        args=("Patient Registration & Analysis",),
    )

    total = len(records_df)
    fit_count = int(records_df["Predicted Fit"].eq("Fit").sum())
    unfit_count = int(records_df["Predicted Fit"].eq("Unfit").sum())

    a, b, c = st.columns(3)
    a.metric("Total Cases", total)
    b.metric("Fit Results", fit_count)
    c.metric("Unfit Results", unfit_count)

    st.markdown("### Application Modules")

    modules = [
        ("🧑‍⚕️", "Patient Registration",
         "Register a patient and upload an X-ray.",
         "Patient Registration & Analysis"),
        ("🖼️", "Image Processing Laboratory",
         "Inspect processing technique outputs.",
         "Image Processing Laboratory"),
        ("📊", "Analytics Dashboard",
         "Explore classification counts and charts.",
         "Dashboard"),
        ("✓", "Classification Results",
         "Review saved Fit/Unfit results.",
         "Classification Results"),
        ("🗂️", "Dataset Management",
         "Search and export your saved dataset.",
         "Dataset Management"),
        ("📄", "Reports",
         "Download analysis summaries and records.",
         "Reports"),
    ]

    for start in range(0, len(modules), 3):
        cols = st.columns(3)

        for offset, col in enumerate(cols):
            idx = start + offset
            if idx >= len(modules):
                continue

            icon, title, description, destination = modules[idx]

            with col:
                with st.container(border=True):
                    st.markdown(
                        f"""
                        <div class="module-icon">{icon}</div>
                        <h4>{title}</h4>
                        <p style="color:#647D95;font-size:12px">
                            {description}
                        </p>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.button(
                        "Open Module →",
                        key=f"home_module_{idx}",
                        use_container_width=True,
                        on_click=go_to,
                        args=(destination,),
                    )

    st.info(
        "Fit/Unfit is a preliminary image-quality screening result. "
        "It is not a diagnosis of normality or disease."
    )

    navigation_buttons("Reports", "Dashboard", "home")


# ============================================================
# DASHBOARD
# ============================================================

elif page == "Dashboard":
    page_header(
        "Analytics Dashboard",
        "An overview of saved cases and image-quality screening results.",
    )

    total = len(records_df)
    fit_count = int(records_df["Predicted Fit"].eq("Fit").sum())
    unfit_count = int(records_df["Predicted Fit"].eq("Unfit").sum())
    known = int(records_df["Reference Fit"].isin(["Fit", "Unfit"]).sum())

    a, b, c, d = st.columns(4)
    a.metric("Total Cases", total)
    b.metric("Fit", fit_count)
    c.metric("Unfit", unfit_count)
    d.metric("Labelled Cases", known)

    if total:
        st.markdown("### Screening Distribution")

        left, right = st.columns(2)

        chart_df = pd.DataFrame({
            "Result": ["Fit", "Unfit"],
            "Cases": [fit_count, unfit_count],
        })

        with left:
            fig = px.pie(
                chart_df,
                names="Result",
                values="Cases",
                hole=0.65,
                color="Result",
                color_discrete_map={
                    "Fit": "#159E9A",
                    "Unfit": "#E45E62",
                },
            )
            st.plotly_chart(
                chart_layout(fig, "Fit / Unfit Distribution"),
                use_container_width=True,
            )

        with right:
            fig = px.bar(
                chart_df,
                x="Result",
                y="Cases",
                color="Result",
                text="Cases",
                color_discrete_map={
                    "Fit": "#159E9A",
                    "Unfit": "#E45E62",
                },
            )
            st.plotly_chart(
                chart_layout(fig, "Classification Case Count"),
                use_container_width=True,
            )

        st.markdown("### Recent Cases")
        st.dataframe(
            records_df[
                ["Case ID", "Patient Name", "Image Name",
                 "Predicted Fit", "Analysis Date"]
            ].head(10),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Register and process your first case to populate the dashboard.")

    navigation_buttons("Home", "Patient Registration & Analysis", "dashboard")


# ============================================================
# PATIENT REGISTRATION AND ANALYSIS
# ============================================================

elif page == "Patient Registration & Analysis":
    page_header(
        "Patient Registration & Analysis",
        "Enter case details, upload an X-ray, and run the existing analysis pipeline.",
    )

    with st.form("patient_analysis_form"):
        st.markdown("#### 01 · Patient Information")

        c1, c2 = st.columns(2)

        with c1:
            patient_name = st.text_input(
                "Patient Name *",
                placeholder="Enter patient name",
            )
            case_id = st.text_input(
                "Patient Reference / Case ID *",
                placeholder="CASE001",
            )
            age = st.number_input(
                "Age (optional)",
                min_value=0,
                max_value=120,
                value=0,
            )

        with c2:
            gender = st.selectbox(
                "Gender (optional)",
                ["Not provided", "Female", "Male", "Other"],
            )
            xray_date = st.date_input(
                "X-ray Date",
                value=date.today(),
            )

        st.markdown("#### 02 · X-Ray Upload")

        image_file = st.file_uploader(
            "Upload a chest X-ray image",
            type=["png", "jpg", "jpeg", "bmp"],
        )

        st.markdown("#### 03 · Reference Labels")

        c3, c4 = st.columns(2)

        with c3:
            reference_fit = st.selectbox(
                "Known Fit/Unfit Reference",
                ["Not known", "Fit", "Unfit"],
            )

        with c4:
            reference_medical_label = st.selectbox(
                "Known Medical Reference",
                ["Not known", "Normal", "Abnormal"],
            )

        submitted = st.form_submit_button(
            "Process Image & Classify",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not patient_name.strip():
            st.error("Please enter the patient name.")
        elif not case_id.strip():
            st.error("Please enter a patient reference or case ID.")
        elif image_file is None:
            st.error("Please upload a chest X-ray image.")
        else:
            try:
                rgb, gray = load_image(image_file)
                images, measurements = process_image(gray)

                status, reasons, quality_metrics = assess_image_quality(
                    gray, measurements
                )

                safe_id = "".join(
                    ch for ch in case_id.strip()
                    if ch.isalnum() or ch in ("-", "_")
                )

                image_path = UPLOAD_DIR / (
                    f"{safe_id}_{Path(image_file.name).name}"
                )
                image_path.write_bytes(image_file.getvalue())

                all_measurements = {
                    **measurements,
                    **quality_metrics,
                    "Patient Name": patient_name.strip(),
                    "Screening observations": reasons,
                    "X-ray date": str(xray_date),
                }

                # FIX: patient_name is saved as its own database field.
                save_analysis({
                    "case_id": case_id.strip(),
                    "patient_name": patient_name.strip(),
                    "age": int(age) if age else None,
                    "gender": gender,
                    "image_name": image_file.name,
                    "reference_fit": reference_fit,
                    "predicted_fit": status,
                    "reference_medical_label": reference_medical_label,
                    "predicted_medical_label": "Pending",
                    "measurements": json.dumps(all_measurements),
                })

                st.session_state["last_rgb"] = rgb
                st.session_state["last_images"] = images
                st.session_state["last_measurements"] = all_measurements
                st.session_state["last_status"] = status
                st.session_state["last_reasons"] = reasons
                st.session_state["last_case_id"] = case_id.strip()
                st.session_state["last_patient_name"] = patient_name.strip()
                st.session_state["last_image_name"] = image_file.name

                # Refresh records so other sections use the latest saved data.
                records_df = read_records()

                st.success("Image processed and analysis record saved.")

            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

    if "last_status" in st.session_state:
        st.divider()
        st.markdown("### Latest Analysis")

        st.write(
            f"**Patient:** {st.session_state['last_patient_name']}  "
            f"| **Case ID:** {st.session_state['last_case_id']}"
        )

        result_panel(st.session_state["last_status"])

        st.markdown("#### Original Chest X-Ray")
        st.image(
            st.session_state["last_rgb"],
            use_container_width=True,
        )

        st.markdown("#### Screening Observations")
        for reason in st.session_state["last_reasons"]:
            st.write(f"• {reason}")

        measurement_data = st.session_state["last_measurements"]

        numeric = {
            name: float(value)
            for name, value in measurement_data.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
        }

        st.markdown("### Image Measurements & Visualizations")

        if numeric:
            items = list(numeric.items())

            for start in range(0, len(items), 3):
                cols = st.columns(3)

                for offset, col in enumerate(cols):
                    idx = start + offset
                    if idx < len(items):
                        name, value = items[idx]
                        with col:
                            st.metric(
                                name.replace("_", " ").title(),
                                f"{value:.4f}",
                            )

            measurement_df = pd.DataFrame({
                "Measurement": list(numeric.keys()),
                "Value": list(numeric.values()),
            })

            fig = px.bar(
                measurement_df,
                x="Measurement",
                y="Value",
                color="Value",
                color_continuous_scale=[
                    "#B8D8F3", "#4E9BD4", "#1769AA"
                ],
                title="Image Measurement Comparison",
            )

            st.plotly_chart(
                chart_layout(fig),
                use_container_width=True,
            )

            with st.expander("View exact measurement values"):
                st.dataframe(
                    measurement_df,
                    use_container_width=True,
                    hide_index=True,
                )

            st.download_button(
                "Download Measurements CSV",
                data=measurement_df.to_csv(index=False).encode("utf-8"),
                file_name="image_measurements.csv",
                mime="text/csv",
                key="measurement_csv",
            )
        else:
            st.json(measurement_data)

        report = (
            "CHEST X-RAY IMAGE PROCESSING REPORT\n\n"
            f"Patient: {st.session_state['last_patient_name']}\n"
            f"Case ID: {st.session_state['last_case_id']}\n"
            f"Image: {st.session_state['last_image_name']}\n"
            f"Fit/Unfit result: {st.session_state['last_status']}\n\n"
            "Screening observations:\n"
            + "\n".join(
                f"- {item}" for item in st.session_state["last_reasons"]
            )
            + "\n\nMeasurements:\n"
            + "\n".join(
                f"- {key}: {value}"
                for key, value in measurement_data.items()
            )
            + "\n\nEducational screening only; not a medical diagnosis.\n"
        )

        st.download_button(
            "Download Case Report",
            data=report,
            file_name=f"{st.session_state['last_case_id']}_report.txt",
            mime="text/plain",
            key="case_report",
        )

    navigation_buttons(
        "Dashboard", "Image Processing Laboratory", "analysis"
    )


# ============================================================
# IMAGE PROCESSING LABORATORY
# ============================================================

elif page == "Image Processing Laboratory":
    page_header(
        "Image Processing Laboratory",
        "Compare the original image with the outputs of your processing techniques.",
    )

    if "last_images" not in st.session_state:
        st.info("Run an analysis first to view processed images.")
    else:
        st.write(
            f"**Patient:** {st.session_state.get('last_patient_name', 'N/A')} "
            f"| **Case ID:** {st.session_state['last_case_id']}"
        )

        images = st.session_state["last_images"]

        if not images:
            st.info("No processing images were returned.")
        else:
            names = list(images.keys())

            for start in range(0, len(names), 2):
                cols = st.columns(2)

                for offset, col in enumerate(cols):
                    idx = start + offset
                    if idx >= len(names):
                        continue

                    name = names[idx]

                    with col:
                        with st.container(border=True):
                            st.markdown(f"#### {name}")
                            st.image(
                                images[name],
                                caption=name,
                                use_container_width=True,
                                clamp=True,
                            )

    navigation_buttons(
        "Patient Registration & Analysis",
        "Classification Results",
        "processing",
    )


# ============================================================
# CLASSIFICATION RESULTS
# ============================================================

elif page == "Classification Results":
    page_header(
        "Classification Results",
        "Review Fit/Unfit image-quality screening and available reference labels.",
    )

    if records_df.empty:
        st.info("No saved classification records are available.")
    else:
        st.dataframe(
            records_df[
                ["Case ID", "Patient Name", "Image Name",
                 "Reference Fit", "Predicted Fit", "Analysis Date"]
            ],
            use_container_width=True,
            hide_index=True,
        )

        labelled = records_df[
            records_df["Reference Fit"].isin(["Fit", "Unfit"])
        ].copy()

        st.markdown("### Reference Label Evaluation")

        if labelled.empty:
            st.info(
                "No known Fit/Unfit reference labels are available. "
                "Provide reliable reference labels to evaluate agreement."
            )
        else:
            correct = (
                labelled["Reference Fit"] == labelled["Predicted Fit"]
            ).sum()

            agreement = 100 * correct / len(labelled)

            a, b = st.columns(2)
            a.metric("Labelled Cases", len(labelled))
            b.metric("Reference Agreement", f"{agreement:.1f}%")

            comparison = (
                labelled.groupby(["Reference Fit", "Predicted Fit"])
                .size()
                .reset_index(name="Cases")
            )

            fig = px.bar(
                comparison,
                x="Reference Fit",
                y="Cases",
                color="Predicted Fit",
                barmode="group",
                color_discrete_map={
                    "Fit": "#159E9A",
                    "Unfit": "#E45E62",
                },
            )

            st.plotly_chart(
                chart_layout(fig, "Reference vs Predicted Results"),
                use_container_width=True,
            )

            st.caption(
                "Reference agreement is calculated only on labelled records. "
                "It is not proof of clinical performance."
            )

    st.markdown("### Disease Classification")
    st.info(
        "The current system classifies image quality as Fit or Unfit. "
        "It does not diagnose disease or classify medical normality. "
        "A separately trained and validated medical classifier would "
        "be required for that function."
    )

    navigation_buttons(
        "Image Processing Laboratory",
        "Dataset Management",
        "classification",
    )


# ============================================================
# DATASET MANAGEMENT
# ============================================================

elif page == "Dataset Management":
    page_header(
        "Dataset Management",
        "Search, filter, inspect, and export your saved case dataset.",
    )

    if records_df.empty:
        st.info("No dataset records have been saved yet.")
    else:
        a, b = st.columns([2, 1])

        with a:
            search = st.text_input(
                "Search by patient, case ID, or image name",
                key="dataset_search",
            )

        with b:
            result_filter = st.selectbox(
                "Classification filter",
                ["All", "Fit", "Unfit"],
            )

        filtered = records_df.copy()

        if search.strip():
            query = search.strip()

            matches = (
                filtered["Case ID"].astype(str).str.contains(
                    query, case=False, na=False
                )
                | filtered["Patient Name"].astype(str).str.contains(
                    query, case=False, na=False
                )
                | filtered["Image Name"].astype(str).str.contains(
                    query, case=False, na=False
                )
            )

            filtered = filtered[matches]

        if result_filter != "All":
            filtered = filtered[
                filtered["Predicted Fit"] == result_filter
            ]

        st.caption(f"{len(filtered)} matching records")

        # Patient Name is now read from the patient_name database column.
        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download Filtered Dataset CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="Chest_Xray_Dataset.csv",
            mime="text/csv",
            type="primary",
        )

    navigation_buttons(
        "Classification Results", "Reports", "dataset"
    )


# ============================================================
# REPORTS
# ============================================================

elif page == "Reports":
    page_header(
        "Reports & Exports",
        "Review saved cases and download project records.",
    )

    if records_df.empty:
        st.info(
            "No reports are available. Process an image to create a case record."
        )
    else:
        total = len(records_df)
        fit_count = int(records_df["Predicted Fit"].eq("Fit").sum())
        unfit_count = int(records_df["Predicted Fit"].eq("Unfit").sum())

        a, b, c = st.columns(3)
        a.metric("Total Cases", total)
        b.metric("Fit Results", fit_count)
        c.metric("Unfit Results", unfit_count)

        st.markdown("### Report Preview")

        st.dataframe(
            records_df[
                ["Case ID", "Patient Name", "Image Name",
                 "Predicted Fit", "Analysis Date"]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download All Records CSV",
            data=records_df.to_csv(index=False).encode("utf-8"),
            file_name="Chest_Xray_Analysis_Records.csv",
            mime="text/csv",
            type="primary",
        )

    navigation_buttons("Dataset Management", "Home", "reports")


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        CHEST X-RAY IMAGE PROCESSING & CLASSIFICATION SYSTEM
        <br>
        Academic Project · Educational Screening Only · Not for Clinical Diagnosis
    </div>
    """,
    unsafe_allow_html=True,
)