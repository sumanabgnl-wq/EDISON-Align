
import streamlit as st
import re
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EDISON-Align",
    page_icon="🎓",
    layout="wide"
)
if "analyze" not in st.session_state:
    st.session_state.analyze = False

# ============================================================
# STEP 2 - TEXT PREPROCESSING
# ============================================================

stop_words = {
    "the", "is", "are", "was", "were", "a", "an",
    "and", "or", "of", "to", "in", "for", "on",
    "with", "should", "have", "has", "be", "this"
}


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# ============================================================
# STEP 5 - REAL-WORLD DATA SCIENCE QUIZ
# ============================================================

questions = [

    {
        "question": "A dataset contains 10,000 customer records and some values are missing in the Age column. What is the most appropriate first step?",
        "options": [
            "Delete the entire dataset",
            "Check the percentage and pattern of missing values",
            "Convert Age into text",
            "Duplicate the missing records"
        ],
        "answer": 1,
        "skill": "Data Preprocessing"
    },

    {
        "question": "You have a Pandas DataFrame containing employee salaries. Which code calculates the average salary?",
        "options": [
            "df['salary'].mean()",
            "df['salary'].average()",
            "df.mean_salary()",
            "average(df['salary'])"
        ],
        "answer": 0,
        "skill": "Python/Pandas"
    },

    {
        "question": "A company wants to find customers who placed more than 5 orders. Which SQL approach is appropriate?",
        "options": [
            "WHERE COUNT(*) > 5",
            "GROUP BY customer and HAVING COUNT(*) > 5",
            "ORDER BY COUNT(*) > 5",
            "SELECT COUNT(*) WHERE customer > 5"
        ],
        "answer": 1,
        "skill": "SQL"
    },

    {
        "question": "A machine learning model performs very well on training data but poorly on unseen test data. What is the most likely problem?",
        "options": [
            "Underfitting",
            "Overfitting",
            "Data visualization",
            "Normalization"
        ],
        "answer": 1,
        "skill": "Machine Learning"
    },

    {
        "question": "You are building a model to predict whether a customer will leave a company: Yes or No. Which type of problem is this?",
        "options": [
            "Regression",
            "Classification",
            "Clustering",
            "Dimensionality reduction"
        ],
        "answer": 1,
        "skill": "Machine Learning"
    },

    {
        "question": "A dataset has income values ranging from 10,000 to 10,00,000. Which preprocessing technique can bring features to a comparable scale?",
        "options": [
            "Feature scaling",
            "Data duplication",
            "Row deletion",
            "Text tokenization"
        ],
        "answer": 0,
        "skill": "Data Preprocessing"
    },

    {
        "question": "A Data Scientist wants to understand the relationship between advertising expenditure and sales. Which visualization is most suitable?",
        "options": [
            "Scatter plot",
            "Pie chart",
            "Single-value card",
            "Histogram of names"
        ],
        "answer": 0,
        "skill": "Data Visualization"
    },

    {
        "question": "Which metric measures the proportion of actual positive cases correctly identified by a binary classification model?",
        "options": [
            "Recall",
            "Mean",
            "Variance",
            "R-squared"
        ],
        "answer": 0,
        "skill": "Model Evaluation"
    },

    {
        "question": "A company has thousands of customer records but no predefined customer categories. Which approach can discover natural groups?",
        "options": [
            "Clustering",
            "Linear regression",
            "Classification",
            "Time-series forecasting"
        ],
        "answer": 0,
        "skill": "Machine Learning"
    },

    {
        "question": "A Data Scientist wants to remove duplicate customer records from a Pandas DataFrame. Which function is appropriate?",
        "options": [
            "drop_duplicates()",
            "remove_rows()",
            "delete_duplicate()",
            "unique_rows()"
        ],
        "answer": 0,
        "skill": "Python/Pandas"
    }
]


# ============================================================
# STEP 6 - COURSE & PROJECT RECOMMENDATIONS
# ============================================================

recommendations = {

    "Python/Pandas": {
        "course": "Python for Data Science",
        "project": "Customer Data Analysis using Pandas"
    },

    "SQL": {
        "course": "SQL for Data Analysis",
        "project": "Sales Database Analysis"
    },

    "Machine Learning": {
        "course": "Machine Learning Fundamentals",
        "project": "House Price Prediction"
    },

    "Data Preprocessing": {
        "course": "Data Cleaning and Preprocessing",
        "project": "Real-World Dataset Cleaning"
    },

    "Data Visualization": {
        "course": "Data Visualization with Power BI",
        "project": "Business Sales Dashboard"
    },

    "Model Evaluation": {
        "course": "Machine Learning Model Evaluation",
        "project": "Classification Model Performance Analysis"
    }
}


# ============================================================
# TITLE
# ============================================================

st.title("🎓 EDISON-Align")

st.subheader(
    "Data-Driven Skill Verification and Course Recommendation System"
)

st.write(
    "Identify your skills, analyze skill gaps, "
    "verify your skills, and improve your career readiness."
)

st.divider()


# ============================================================
# STUDENT PROFILE
# ============================================================

st.header("👤 Student Profile")

name = st.text_input("Student Name")

education = st.selectbox(
    "Education",
    [
        "MCA",
        "BCA",
        "B.Sc",
        "B.Tech",
        "M.Tech",
        "Other"
    ]
)

job_role = st.selectbox(
    "Target Job Role",
    [
        "Data Analyst",
        "Data Scientist",
        "Machine Learning Engineer",
        "Business Analyst"
    ]
)

skills = st.multiselect(
    "Select Your Current Skills",
    [
        "Python",
        "SQL",
        "Statistics",
        "Machine Learning",
        "Pandas",
        "NumPy",
        "Power BI",
        "Excel",
        "Data Visualization"
    ]
)

# ============================================================
# RESUME UPLOAD
# ============================================================

st.divider()

st.header("📄 Upload Your Resume")

resume_file = st.file_uploader(
    "Upload your resume (PDF)",
    type=["pdf"]
)

resume_text = ""

if resume_file is not None:
    try:
        pdf_reader = PyPDF2.PdfReader(resume_file)

        for page in pdf_reader.pages:
            text = page.extract_text()

            if text:
                resume_text += text

        if resume_text.strip():
            st.success("✅ Resume uploaded and read successfully!")

            with st.expander("👀 View Extracted Resume Text"):
                st.write(resume_text)

        else:
            st.warning(
                "⚠️ Could not extract text from this PDF. "
                "Please upload a text-based PDF resume."
            )

    except Exception as e:
        st.error(f"❌ Error reading resume: {e}")
# ============================================================
# RESUME SKILL EXTRACTION
# ============================================================

skill_keywords = [
    "Python",
    "SQL",
    "Statistics",
    "Machine Learning",
    "Pandas",
    "NumPy",
    "Power BI",
    "Excel",
    "Data Visualization",
    "Deep Learning",
    "TensorFlow",
    "PyTorch",
    "Tableau",
    "NLP",
    "Data Analysis"
]

extracted_skills = []

if resume_text:
    resume_lower = resume_text.lower()

    # Remove spaces created inside words during PDF extraction
    normalized_resume = re.sub(r"\s+", "", resume_lower)

    for skill in skill_keywords:
        normalized_skill = skill.lower().replace(" ", "")

        if normalized_skill in normalized_resume:
            extracted_skills.append(skill)

    st.subheader("🧠 Skills Detected from Resume")

    if extracted_skills:
        for skill in extracted_skills:
            st.write("✅", skill)
    else:
        st.warning(
            "No matching skills detected in the uploaded resume."
        )
# ============================================================
# TARGET JOB ROLE REQUIREMENTS
# ============================================================

job_requirements = {

    "Data Analyst": [
        "Python",
        "SQL",
        "Statistics",
        "Pandas",
        "Excel",
        "Power BI",
        "Data Visualization"
    ],

    "Data Scientist": [
        "Python",
        "SQL",
        "Statistics",
        "Pandas",
        "NumPy",
        "Machine Learning",
        "Data Visualization",
        "Deep Learning"
    ],

    "Machine Learning Engineer": [
        "Python",
        "SQL",
        "NumPy",
        "Pandas",
        "Machine Learning",
        "Deep Learning",
        "TensorFlow",
        "PyTorch"
    ],

    "Business Analyst": [
        "SQL",
        "Excel",
        "Statistics",
        "Power BI",
        "Data Visualization",
        "Data Analysis"
    ]
}
# ============================================================
# RESUME + CURRENT SKILLS - JOB MATCHING
# ============================================================

if resume_text or skills:

    required_skills = job_requirements[job_role]

    # Combine manually selected skills and resume detected skills
    combined_skills = list(set(skills + extracted_skills))

    matched_skills = [
        skill for skill in required_skills
        if skill in combined_skills
    ]

    missing_job_skills = [
        skill for skill in required_skills
        if skill not in combined_skills
    ]

    resume_match_score = (
        len(matched_skills) / len(required_skills)
    ) * 100

    st.divider()

    st.header("🎯 Resume + Student Skills - Job Role Matching")

    st.write(
        f"**Target Job Role:** {job_role}"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Required Skills",
            len(required_skills)
        )

    with col2:
        st.metric(
            "Matched Skills",
            len(matched_skills)
        )

    with col3:
        st.metric(
            "Job Match Score",
            f"{resume_match_score:.2f}%"
        )

    st.subheader("👤 Student + Resume Skills")

    if combined_skills:
        for skill in combined_skills:
            st.write("✅", skill)

    st.subheader("✅ Matched Skills")

    if matched_skills:
        for skill in matched_skills:
            st.write("✅", skill)
    else:
        st.write("No matching skills found.")

    st.subheader("⚠️ Skills to Improve")

    if missing_job_skills:
        for skill in missing_job_skills:
            st.write("⚠️", skill)
    else:
        st.success(
            "🎉 You have all the required skills for this job role!"
        )
# ============================================================
# CURRICULUM + INDUSTRY DATA
# ============================================================

st.divider()

st.header("📚 Curriculum and Industry Data")

syllabus = st.text_area(
    "Enter University Syllabus Skills",
    "Python programming, SQL, Statistics, Machine Learning, "
    "Data Analysis and Data Visualization"
)

job_description = st.text_area(
    "Enter Data Science Job Requirements",
    "Python, SQL, Machine Learning, Data Analysis, "
    "Power BI, Statistics and Data Visualization"
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.divider()
if st.button("🔍 Analyze My Skills"):
    st.session_state.analyze = True

if st.session_state.analyze:

    if not name:
        st.warning("Please enter your name.")

    elif not skills:
        st.warning("Please select at least one current skill.")

    else:
        st.success("Student profile submitted successfully!")

    if not name:
        ...

    if not name:

        st.warning("Please enter your name.")

    elif not skills:

        st.warning(
            "Please select at least one current skill."
        )

    else:

        st.success(
            "Student profile submitted successfully!"
        )

        # ====================================================
        # PROFILE RESULT
        # ====================================================

        st.header("👤 Your Profile")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Name:**", name)

        with col2:
            st.write("**Education:**", education)

        with col3:
            st.write("**Target Role:**", job_role)

        st.write("**Current Skills:**")

        for skill in skills:
            st.write("✅", skill)


        # ====================================================
        # STEP 2 - PREPROCESSING
        # ====================================================

        st.divider()

        st.header("🧹 Step 2: Text Preprocessing")

        clean_syllabus = preprocess_text(syllabus)

        clean_job = preprocess_text(job_description)

        st.write("**Cleaned Syllabus:**")
        st.code(clean_syllabus)

        st.write("**Cleaned Job Requirements:**")
        st.code(clean_job)


        # ====================================================
        # STEP 3 - TF-IDF + COSINE SIMILARITY
        # ====================================================

        st.divider()

        st.header("📊 Step 3: Skill Analysis")

        vectorizer = TfidfVectorizer()

        tfidf_matrix = vectorizer.fit_transform(
            [
                clean_syllabus,
                clean_job
            ]
        )

        similarity = cosine_similarity(
            tfidf_matrix[0:1],
            tfidf_matrix[1:2]
        )[0][0]

        similarity_percentage = similarity * 100

        st.metric(
            "Curriculum–Job Match",
            f"{similarity_percentage:.2f}%"
        )

        st.write(
            "This score represents how closely the university "
            "syllabus matches the industry job requirements."
        )


        # ====================================================
        # STEP 4 - SKILL GAP
        # ====================================================

        st.divider()

        st.header("🔎 Step 4: Skill Gap Identification")

        syllabus_words = set(
            clean_syllabus.split()
        )

        job_words = set(
            clean_job.split()
        )

        missing_skills = job_words - syllabus_words

        common_skills = (
            job_words.intersection(
                syllabus_words
            )
        )

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("✅ Common Skills")

            if common_skills:

                for skill in sorted(common_skills):
                    st.write("✅", skill)

            else:

                st.write(
                    "No common skills found."
                )


        with col2:

            st.subheader("⚠️ Skill Gaps")

            if missing_skills:

                for skill in sorted(missing_skills):
                    st.write("⚠️", skill)

            else:

                st.success(
                    "No major text-based skill gaps identified."
                )


        # ====================================================
        # STEP 5 - QUIZ
        # ====================================================

        st.divider()

        st.header(
            "📝 Step 5: Data Science Skill Verification"
        )

        st.write(
            "Answer the following real-world Data Science "
            "questions to verify your current skills."
        )

        with st.form("data_science_quiz"):

            answers = []

            for i, q in enumerate(questions):

                st.write(
                    f"### Question {i + 1}"
                )

                st.write(
                    q["question"]
                )

                selected = st.radio(
                    "Select your answer:",
                    q["options"],
                    index=None,
                    key=f"quiz_{i}"
                )


                answers.append(selected)

            submit_quiz = st.form_submit_button(
                "📊 Submit Quiz"
            )


        # ====================================================
        # QUIZ RESULT
        # ====================================================

        if submit_quiz:

            quiz_score = 0

            skill_correct = {}

            skill_total = {}

            for i, q in enumerate(questions):

                skill = q["skill"]

                skill_total[skill] = (
                    skill_total.get(skill, 0) + 1
                )

                selected_answer = answers[i]

                correct_answer = q["options"][
                    q["answer"]
                ]

                if selected_answer == correct_answer:

                    quiz_score += 1

                    skill_correct[skill] = (
                        skill_correct.get(skill, 0) + 1
                    )


            # =================================================
            # OVERALL QUIZ SCORE
            # =================================================

            overall_quiz_score = (
                quiz_score / len(questions)
            ) * 100

            st.success(
                f"Skill Verification Score: "
                f"{overall_quiz_score:.2f}%"
            )


            # =================================================
            # SKILL-WISE VERIFICATION
            # =================================================

            st.subheader(
                "📊 Skill-wise Verification"
            )

            skill_scores = {}

            for skill in skill_total:

                correct = skill_correct.get(
                    skill,
                    0
                )

                total = skill_total[skill]

                score = (
                    correct / total
                ) * 100

                skill_scores[skill] = score

                st.write(
                    f"**{skill}:** {score:.2f}%"
                )


            # =================================================
            # STEP 6 - CAREER READINESS
            # =================================================

            st.divider()

            st.header(
                "🎯 Step 6: Career Readiness Score"
            )

            average_skill_score = (
                sum(skill_scores.values())
                / len(skill_scores)
                )

            career_readiness = (
                similarity_percentage
                + resume_match_score
                + overall_quiz_score
                )  / 3

            st.metric(
                "Career Readiness Score",
                f"{career_readiness:.2f}%"
            )


            if career_readiness >= 80:

                st.success(
                    "🟢 Job Ready"
                )

            elif career_readiness >= 60:

                st.info(
                    "🟡 Nearly Job Ready"
                )

            else:

                st.warning(
                    "🔴 Needs Skill Improvement"
                )


            # =================================================
            # COURSE + PROJECT RECOMMENDATIONS
            # =================================================

            st.subheader(
                "📚 Recommended Courses & Practical Projects"
            )

            weak_skills_found = False

            for skill, score in skill_scores.items():

                if score < 60:

                    weak_skills_found = True

                    st.write(
                        f"### ⚠️ {skill}"
                    )

                    st.write(
                        f"Current Score: "
                        f"**{score:.2f}%**"
                    )

                    if skill in recommendations:

                        st.write(
                            "📘 **Recommended Course:**",
                            recommendations[skill]["course"]
                        )

                        st.write(
                            "💻 **Recommended Practical Project:**",
                            recommendations[skill]["project"]
                        )


            if not weak_skills_found:

                st.success(
                    "🎉 No major weak skills identified. "
                    "Keep practicing and building projects."
                )

