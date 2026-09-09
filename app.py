import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)


# =========================================================
# LOAD MACHINE LEARNING MODEL
# =========================================================

model = joblib.load("diabetes_model.pkl")
features = joblib.load("diabetes_features.pkl")


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #f7f8ff 0%,
        #eef6ff 50%,
        #f6ffff 100%
    );
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* ================= HEADER ================= */

.main-header {
    background: linear-gradient(
        135deg,
        #5125d8,
        #743bea,
        #9b55ff
    );

    padding: 35px 45px;

    border-radius: 25px;

    margin-bottom: 30px;

    color: white;

    box-shadow: 0 12px 35px rgba(81,37,216,0.25);
}

.main-header h1 {
    font-size: 42px;
    font-weight: 800;
    margin: 0;
}

.main-header p {
    font-size: 18px;
    margin-top: 10px;
}


/* ================= CARDS ================= */

.card {
    background: white;

    padding: 28px;

    border-radius: 22px;

    border: 1px solid #eeeeff;

    box-shadow: 0 8px 25px rgba(0,0,0,0.06);

    margin-bottom: 20px;
}


/* ================= SECTION TITLE ================= */

.section-title {
    font-size: 25px;
    font-weight: 750;
    color: #29294d;
}

.section-text {
    color: #707089;
    font-size: 15px;
    margin-top: 5px;
    margin-bottom: 20px;
}


/* ================= INPUT LABELS ================= */

label {
    font-weight: 600 !important;
    color: #29294d !important;
}


/* ================= PREDICT BUTTON ================= */

div.stButton > button {
    width: 100%;
    height: 58px;

    border-radius: 15px;

    border: none;

    background: linear-gradient(
        90deg,
        #6332df,
        #8c4df5
    );

    color: white;

    font-size: 19px;

    font-weight: 700;

    box-shadow: 0 8px 20px rgba(99,50,223,0.25);

    transition: all 0.3s ease;
}

div.stButton > button:hover {
    transform: translateY(-2px);

    box-shadow:
        0 12px 28px rgba(99,50,223,0.35);
}


/* ================= POSITIVE RESULT ================= */

.result-positive {
    background: linear-gradient(
        135deg,
        #fff0f3,
        #ffe1e7
    );

    border: 2px solid #ffb5c1;

    padding: 35px;

    border-radius: 22px;

    text-align: center;

    margin-bottom: 20px;

    box-shadow:
        0 8px 25px rgba(220,60,80,0.10);
}


/* ================= NEGATIVE RESULT ================= */

.result-negative {
    background: linear-gradient(
        135deg,
        #edfff4,
        #dcf9e8
    );

    border: 2px solid #a4e6bc;

    padding: 35px;

    border-radius: 22px;

    text-align: center;

    margin-bottom: 20px;

    box-shadow:
        0 8px 25px rgba(30,140,70,0.10);
}


.result-icon {
    font-size: 55px;
}

.result-title {
    font-size: 31px;
    font-weight: 800;
    margin-top: 8px;
}

.result-probability {
    font-size: 19px;
    color: #45455c;
    margin-top: 10px;
}


/* ================= ADVICE CARD ================= */

.advice {
    background: white;

    padding: 25px;

    border-radius: 18px;

    border-left: 6px solid #7b3ff2;

    box-shadow:
        0 6px 18px rgba(0,0,0,0.06);

    margin-bottom: 18px;
}

.advice h3 {
    color: #29294d;
}

.advice p {
    color: #55556b;
    line-height: 1.7;
}

.advice li {
    color: #55556b;
    line-height: 1.8;
    margin-bottom: 5px;
}


/* ================= HEALTH TIPS ================= */

.tip {
    background: white;

    padding: 22px;

    border-radius: 18px;

    text-align: center;

    min-height: 155px;

    border: 1px solid #eeeeff;

    box-shadow:
        0 6px 18px rgba(0,0,0,0.05);
}

.tip-icon {
    font-size: 40px;
}

.tip-title {
    font-size: 18px;
    font-weight: 700;

    color: #29294d;

    margin-top: 8px;
}

.tip-text {
    color: #707089;

    font-size: 14px;

    line-height: 1.5;

    margin-top: 5px;
}


/* ================= DISCLAIMER ================= */

.disclaimer {
    background: #fff8e7;

    border: 1px solid #ffd98b;

    padding: 17px;

    border-radius: 15px;

    color: #6b5318;

    font-size: 14px;

    line-height: 1.6;

    margin-top: 20px;
}


/* ================= FOOTER ================= */

.footer {
    text-align: center;

    padding: 40px 20px 15px 20px;

    color: #77778c;

    font-size: 14px;
}

.footer-name {
    font-size: 17px;

    font-weight: 700;

    color: #6332df;

    margin-top: 10px;
}

.footer-line {
    width: 80px;

    height: 3px;

    background: linear-gradient(
        90deg,
        #6332df,
        #9b55ff
    );

    border-radius: 10px;

    margin: 15px auto;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.html("""
<div class="main-header">

    <h1>🩺 Diabetes Prediction System</h1>

    <p>
        Check your diabetes risk using Machine Learning
        and take a step towards a healthier lifestyle.
    </p>

</div>
""")


# =========================================================
# MAIN TWO-COLUMN LAYOUT
# =========================================================

left, right = st.columns(
    [1, 1],
    gap="large"
)


# =========================================================
# LEFT SIDE - PATIENT INFORMATION
# =========================================================

with left:

    st.html("""
    <div class="card">

        <div class="section-title">
            👤 Patient Information
        </div>

        <div class="section-text">
            Enter the patient's health information below
            to generate a prediction.
        </div>

    </div>
    """)

    col1, col2 = st.columns(2)


    # =====================================================
    # FIRST INPUT COLUMN
    # =====================================================

    with col1:

        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=2,
            step=1
        )

        glucose = st.number_input(
            "Glucose (mg/dL)",
            min_value=0.0,
            max_value=300.0,
            value=120.0,
            step=1.0
        )

        blood_pressure = st.number_input(
            "Blood Pressure (mm Hg)",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=1.0
        )

        skin_thickness = st.number_input(
            "Skin Thickness (mm)",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=1.0
        )


    # =====================================================
    # SECOND INPUT COLUMN
    # =====================================================

    with col2:

        insulin = st.number_input(
            "Insulin",
            min_value=0.0,
            max_value=1000.0,
            value=100.0,
            step=1.0
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=70.0,
            value=30.5,
            step=0.1
        )

        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.45,
            step=0.01
        )

        age = st.number_input(
            "Age (years)",
            min_value=1,
            max_value=120,
            value=35,
            step=1
        )


    st.write("")


    # =====================================================
    # PREDICTION BUTTON
    # =====================================================

    predict = st.button(
        "🔍  Predict Diabetes",
        use_container_width=True
    )


    # =====================================================
    # DISCLAIMER
    # =====================================================

    st.html("""
    <div class="disclaimer">

        ⚠️ <b>Important:</b>

        This application is developed for educational
        and demonstration purposes only.

        The prediction should not be considered a medical
        diagnosis. Please consult a qualified healthcare
        professional for medical advice.

    </div>
    """)


# =========================================================
# RIGHT SIDE - PREDICTION RESULT
# =========================================================

with right:

    st.html("""
    <div class="card">

        <div class="section-title">
            📊 Prediction Result
        </div>

        <div class="section-text">
            Your Machine Learning prediction will appear here.
        </div>

    </div>
    """)


    # =====================================================
    # RUN PREDICTION
    # =====================================================

    if predict:

        # -------------------------------------------------
        # CREATE INPUT DATAFRAME
        # -------------------------------------------------

        input_data = pd.DataFrame({

            "Pregnancies": [pregnancies],

            "Glucose": [glucose],

            "BloodPressure": [blood_pressure],

            "SkinThickness": [skin_thickness],

            "Insulin": [insulin],

            "BMI": [bmi],

            "DiabetesPedigreeFunction": [
                diabetes_pedigree
            ],

            "Age": [age]

        })


        # -------------------------------------------------
        # MAINTAIN ORIGINAL FEATURE ORDER
        # -------------------------------------------------

        input_data = input_data[features]


        # -------------------------------------------------
        # MAKE PREDICTION
        # -------------------------------------------------

        prediction = model.predict(
            input_data
        )[0]


        # -------------------------------------------------
        # GET PROBABILITY
        # -------------------------------------------------

        probability = model.predict_proba(
            input_data
        )[0][1]


        # =================================================
        # DIABETES PREDICTION
        # =================================================

        if prediction == 1:

            # Celebration animation
            st.balloons()


            # -------------------------------------------------
            # RESULT CARD
            # -------------------------------------------------

            st.html(f"""
            <div class="result-positive">

                <div class="result-icon">
                    ❤️
                </div>

                <div
                    class="result-title"
                    style="color:#d62839;"
                >
                    Diabetes Risk Detected
                </div>

                <div class="result-probability">

                    Estimated Probability:
                    <b>{probability:.2%}</b>

                </div>

            </div>
            """)


            # -------------------------------------------------
            # HEALTH MESSAGE
            # -------------------------------------------------

            st.html("""
            <div class="advice">

                <h3>
                    ❤️ Please Take Care of Your Health
                </h3>

                <p>
                    The model indicates a higher likelihood
                    of diabetes based on the information
                    provided.
                </p>

                <p>
                    This prediction does not confirm a
                    medical diagnosis. Please consult a
                    qualified healthcare professional for
                    proper testing and medical guidance.
                </p>

                <p>
                    Taking care of nutrition, physical
                    activity, sleep and regular health
                    monitoring can support better
                    long-term health.
                </p>

            </div>
            """)


            # -------------------------------------------------
            # DIET AND LIFESTYLE
            # -------------------------------------------------

            st.html("""
            <div class="advice">

                <h3>
                    🥗 Healthy Diet & Lifestyle Suggestions
                </h3>

                <ul>

                    <li>
                        Include plenty of vegetables
                        and high-fiber foods.
                    </li>

                    <li>
                        Choose whole grains and
                        balanced meals.
                    </li>

                    <li>
                        Include healthy protein sources
                        such as pulses, beans, eggs or fish.
                    </li>

                    <li>
                        Limit sugary drinks, sweets and
                        highly refined carbohydrates.
                    </li>

                    <li>
                        Pay attention to portion sizes.
                    </li>

                    <li>
                        Stay physically active according
                        to professional advice.
                    </li>

                    <li>
                        Maintain healthy sleep and
                        hydration habits.
                    </li>

                    <li>
                        Follow your healthcare professional's
                        recommendations for monitoring
                        and treatment.
                    </li>

                </ul>

            </div>
            """)


        # =================================================
        # NO DIABETES PREDICTION
        # =================================================

        else:

            # -------------------------------------------------
            # RESULT CARD
            # -------------------------------------------------

            st.html(f"""
            <div class="result-negative">

                <div class="result-icon">
                    🌱
                </div>

                <div
                    class="result-title"
                    style="color:#16803c;"
                >
                    No Diabetes Detected
                </div>

                <div class="result-probability">

                    Estimated Probability:
                    <b>{probability:.2%}</b>

                </div>

            </div>
            """)


            # -------------------------------------------------
            # POSITIVE MESSAGE
            # -------------------------------------------------

            st.html("""
            <div class="advice">

                <h3>
                    🌟 Great! Keep Taking Care of Yourself!
                </h3>

                <p>
                    Based on the information provided,
                    the model did not detect a diabetes
                    outcome.
                </p>

                <p>
                    Continue maintaining healthy habits,
                    eating balanced meals, staying active
                    and getting regular health check-ups.
                </p>

                <p>
                    💚 Small healthy choices today can
                    contribute to better health tomorrow.
                </p>

                <p>
                    Keep going — your health is worth
                    taking care of every day!
                </p>

            </div>
            """)


# =========================================================
# HEALTHY HABITS SECTION
# =========================================================

st.write("")
st.write("")

st.html("""
<div style="text-align:center; margin-bottom:25px;">

    <div class="section-title">
        💡 Simple Healthy Habits
    </div>

    <div class="section-text">
        Small healthy choices can make a positive difference.
    </div>

</div>
""")


tip1, tip2, tip3, tip4 = st.columns(4)


# =========================================================
# HEALTH TIP 1
# =========================================================

with tip1:

    st.html("""
    <div class="tip">

        <div class="tip-icon">
            🥗
        </div>

        <div class="tip-title">
            Eat Healthy
        </div>

        <div class="tip-text">
            Choose balanced meals with vegetables,
            fiber and nutritious foods.
        </div>

    </div>
    """)


# =========================================================
# HEALTH TIP 2
# =========================================================

with tip2:

    st.html("""
    <div class="tip">

        <div class="tip-icon">
            🏃
        </div>

        <div class="tip-title">
            Stay Active
        </div>

        <div class="tip-text">
            Regular physical activity can support
            overall health and wellbeing.
        </div>

    </div>
    """)


# =========================================================
# HEALTH TIP 3
# =========================================================

with tip3:

    st.html("""
    <div class="tip">

        <div class="tip-icon">
            😴
        </div>

        <div class="tip-title">
            Sleep Well
        </div>

        <div class="tip-text">
            Maintain a consistent sleep routine
            and give your body enough rest.
        </div>

    </div>
    """)


# =========================================================
# HEALTH TIP 4
# =========================================================

with tip4:

    st.html("""
    <div class="tip">

        <div class="tip-icon">
            💚
        </div>

        <div class="tip-title">
            Stay Positive
        </div>

        <div class="tip-text">
            Take small steps every day towards
            healthier and happier habits.
        </div>

    </div>
    """)


# =========================================================
# FINAL FOOTER WITH YOUR NAME
# =========================================================

st.html("""
<div class="footer">

    <div>
        🩺 <b>Diabetes Prediction System</b>
    </div>

    <div class="footer-line"></div>

    <div>
        Predict • Prevent • Live Better 💚
    </div>

    <div class="footer-name">
        Developed by Nuha Mushtaq
    </div>

    <br>

    <i>
        Machine Learning project developed
        for educational purposes.
    </i>

</div>
""")
