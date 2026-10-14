import streamlit as st
import re
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

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
# TEXT PREPROCESSING
# ============================================================
stop_words = {
    "the", "is", "are", "was", "were", "a", "an", "and", "or",
    "of", "to", "in", "for", "on", "with", "should", "have",
    "has", "be", "this", "that", "from", "by", "as"
}


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)

@st.cache_data(show_spinner=False)
def extract_pdf_text_from_bytes(pdf_bytes):
    import io

    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    pages = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text)

    return "\n".join(pages).strip()


def extract_pdf_text(uploaded_file):
    """Extract and cache text from an uploaded PDF."""
    if uploaded_file is None:
        return ""

    try:
        return extract_pdf_text_from_bytes(uploaded_file.getvalue())
    except Exception as exc:
        st.error(f"Could not read the uploaded PDF: {exc}")
        return ""
# ============================================================
# ROLE CATALOGUE: IT AND NON-IT
# ============================================================
job_requirements = {
    # IT SECTOR
    "Data Analyst": [
        "Python", "SQL", "Statistics", "Pandas", "Excel",
        "Power BI", "Data Visualization"
    ],
    "Data Scientist": [
        "Python", "SQL", "Statistics", "Pandas", "NumPy",
        "Machine Learning", "Data Visualization", "Deep Learning"
    ],
    "Machine Learning Engineer": [
        "Python", "SQL", "NumPy", "Pandas", "Machine Learning",
        "Deep Learning", "TensorFlow", "PyTorch"
    ],
    "Business Analyst (IT)": [
        "SQL", "Excel", "Statistics", "Power BI",
        "Data Visualization", "Data Analysis", "Requirements Analysis"
    ],
    "Python Developer": [
        "Python", "SQL", "APIs", "Git", "Problem Solving"
    ],
    "Full Stack Developer": [
        "HTML", "CSS", "JavaScript", "Python", "SQL", "APIs"
    ],
    "Software Developer": [
        "Programming", "Python", "SQL", "Git", "Problem Solving",
        "Software Design"
    ],
    "Software Tester / QA Engineer": [
        "Software Testing", "Test Cases", "SQL", "Python",
        "Bug Reporting", "Automation Testing"
    ],
    "Cloud Engineer": [
        "Cloud Computing", "Linux", "Networking", "Security",
        "Automation"
    ],
    "Cybersecurity Analyst": [
        "Networking", "Cybersecurity", "Linux", "Security",
        "Risk Analysis"
    ],

    # NON-IT SECTOR
    "Financial Analyst": [
        "Excel", "Financial Analysis", "Accounting", "Statistics",
        "Reporting", "Data Analysis"
    ],
    "Banking Operations Executive": [
        "Banking Operations", "Excel", "Communication",
        "Documentation", "Customer Service"
    ],
    "Credit Analyst": [
        "Credit Analysis", "Financial Analysis", "Excel",
        "Risk Analysis", "Statistics"
    ],
    "HR Executive": [
        "Human Resources", "Communication", "Recruitment",
        "Documentation", "Excel"
    ],
    "HR Analyst": [
        "Human Resources", "Excel", "Statistics",
        "Data Analysis", "Reporting"
    ],
    "Talent Acquisition Specialist": [
        "Recruitment", "Communication", "Interviewing",
        "Candidate Sourcing", "Documentation"
    ],
    "Marketing Analyst": [
        "Marketing", "Excel", "Statistics", "Data Analysis",
        "Reporting", "Consumer Behaviour"
    ],
    "Market Research Analyst": [
        "Research Methods", "Survey Analysis", "Statistics",
        "Excel", "Reporting", "Consumer Behaviour"
    ],
    "Digital Marketing Executive": [
        "Digital Marketing", "SEO", "Social Media",
        "Content Marketing", "Analytics"
    ],
    "Operations Analyst": [
        "Operations Management", "Excel", "Statistics",
        "Process Analysis", "Reporting"
    ],
    "Supply Chain Analyst": [
        "Supply Chain", "Inventory Management", "Excel",
        "Forecasting", "Data Analysis"
    ],
    "Sales Executive": [
        "Sales", "Communication", "Customer Relationship",
        "Negotiation", "Reporting"
    ],
    "Research Assistant": [
        "Research Methods", "Statistics", "Data Analysis",
        "Documentation", "Reporting"
    ],
    "Administrative Executive": [
        "Administration", "Communication", "Documentation",
        "Excel", "Office Operations"
    ],
    "Quality Analyst": [
        "Quality Control", "Documentation", "Statistics",
        "Process Analysis", "Reporting"
    ],
}

sector_roles = {
    "IT Sector": [
        "Data Analyst",
        "Data Scientist",
        "Machine Learning Engineer",
        "Business Analyst (IT)",
        "Python Developer",
        "Full Stack Developer",
        "Software Developer",
        "Software Tester / QA Engineer",
        "Cloud Engineer",
        "Cybersecurity Analyst",
    ],
    "Non-IT Sector": [
        "Financial Analyst",
        "Banking Operations Executive",
        "Credit Analyst",
        "HR Executive",
        "HR Analyst",
        "Talent Acquisition Specialist",
        "Marketing Analyst",
        "Market Research Analyst",
        "Digital Marketing Executive",
        "Operations Analyst",
        "Supply Chain Analyst",
        "Sales Executive",
        "Research Assistant",
        "Administrative Executive",
        "Quality Analyst",
    ],
}

# The keywords below are used for basic text matching, not proof of competence.
skill_keywords = sorted({
    skill
    for required in job_requirements.values()
    for skill in required
} | {
    "Python", "SQL", "Statistics", "Machine Learning", "Pandas",
    "NumPy", "Power BI", "Excel", "Data Visualization",
    "Deep Learning", "TensorFlow", "PyTorch", "Tableau",
    "NLP", "Data Analysis", "HTML", "CSS", "JavaScript",
    "APIs", "Git", "Problem Solving", "Software Design",
    "Software Testing", "Test Cases", "Bug Reporting",
    "Automation Testing", "Cloud Computing", "Linux",
    "Networking", "Security", "Risk Analysis", "Accounting",
    "Financial Analysis", "Reporting", "Communication",
    "Customer Service", "Banking Operations", "Documentation",
    "Credit Analysis", "Human Resources", "Recruitment",
    "Interviewing", "Candidate Sourcing", "Marketing",
    "Consumer Behaviour", "Research Methods", "Survey Analysis",
    "Digital Marketing", "SEO", "Social Media", "Content Marketing",
    "Analytics", "Operations Management", "Process Analysis",
    "Supply Chain", "Inventory Management", "Forecasting",
    "Sales", "Customer Relationship", "Negotiation",
    "Administration", "Office Operations", "Quality Control"
}, key=str.lower)


def detect_skills(text):
    """Simple keyword-based extraction from resume or syllabus text."""
    if not text:
        return []
    normalized_text = re.sub(r"\s+", " ", text.lower())
    compact_text = re.sub(r"\s+", "", text.lower())
    found = []
    for skill in skill_keywords:
        skill_lower = skill.lower()
        if skill_lower in normalized_text or skill_lower.replace(" ", "") in compact_text:
            found.append(skill)
    # Preserve catalogue spelling and remove case-insensitive duplicates.
    unique = {}
    for skill in found:
        unique[skill.lower()] = skill
    return list(unique.values())


# ============================================================
# QUIZ: EXISTING DATA SCIENCE SKILL VERIFICATION
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
# PAGE HEADER
# ============================================================
st.title("🎓 EDISON-Align")
st.subheader("Data-Driven Skill Verification and Course Recommendation System")
st.write(
    "Compare your college curriculum and resume with industry job requirements, "
    "identify skill gaps, verify selected skills, and plan your career development."
)
st.divider()


# ============================================================
# STEP 1: STUDENT DETAILS
# ============================================================

st.header("👤 Step 1: Student Details")
st.caption("Enter your academic details before uploading your documents.")

with st.container(border=True):
    name = st.text_input(
        "Student Name *",
        placeholder="Enter your full name"
    )

    detail_col1, detail_col2 = st.columns(2)

    with detail_col1:
        age = st.number_input(
            "Age (years)",
            min_value=15,
            max_value=100,
            value=21,
            step=1
        )
        parent_name = st.text_input(
            "Parent / Guardian Name (optional)",
            placeholder="Enter parent or guardian name"
        )
        academic_year = st.selectbox(
            "Academic Year *",
            [
                "Select academic year", "1st Year", "2nd Year",
                "3rd Year", "Final Year", "Graduated", "Other"
            ]
        )
        college = st.text_input(
            "College / Institution *",
            placeholder="Enter your college or institution name"
        )
        course = st.selectbox(
            "Course *",
            [
                "Select course", "MCA", "BCA", "B.Sc", "B.Com",
                "BBA", "B.Tech", "M.Tech", "MBA", "M.Sc",
                "M.Com", "BA", "BE", "Other"
            ]
        )

    with detail_col2:
        parent_contact = st.text_input(
            "Parent / Guardian Contact (optional)",
            placeholder="Phone number"
        )
        parent_relation = st.selectbox(
            "Relationship (optional)",
            ["Select relationship", "Father", "Mother", "Guardian", "Other"]
        )

    course_specializations = {
        "B.Sc": [
            "Computer Science", "Statistics", "Mathematics",
            "Data Science", "Physics", "Chemistry",
            "Biotechnology", "Microbiology", "Electronics"
        ],
        "BCA": [
            "Computer Applications", "Data Science",
            "Artificial Intelligence", "Machine Learning",
            "Cybersecurity", "Cloud Computing", "Software Engineering"
        ],
        "B.Tech": [
            "Computer Science and Engineering", "Information Technology",
            "Artificial Intelligence", "Data Science",
            "Electronics and Communication", "Electrical Engineering",
            "Mechanical Engineering", "Civil Engineering",
            "Chemical Engineering", "Biotechnology"
        ],
        "MCA": [
            "Computer Applications", "Software Engineering",
            "Data Science", "Artificial Intelligence",
            "Machine Learning", "Cybersecurity", "Cloud Computing"
        ],
        "M.Tech": [
            "Computer Science", "Data Science", "Artificial Intelligence",
            "Cybersecurity", "VLSI Design", "Embedded Systems",
            "Software Engineering"
        ],
        "MBA": [
            "Finance", "Human Resources", "Marketing",
            "Business Analytics", "Operations Management",
            "International Business"
        ],
        "M.Sc": [
            "Statistics", "Mathematics", "Computer Science",
            "Data Science", "Physics", "Chemistry",
            "Biotechnology", "Microbiology"
        ],
        "M.Com": [
            "Accounting", "Finance", "Banking", "Taxation",
            "Business Management"
        ],
        "B.Com": [
            "Accounting", "Finance", "Banking", "Taxation",
            "Computer Applications"
        ],
        "BBA": [
            "Finance", "Marketing", "Human Resources",
            "Business Analytics", "Operations Management"
        ],
        "BA": [
            "Economics", "Psychology", "Sociology",
            "Political Science", "English", "History"
        ],
        "BE": [
            "Computer Science and Engineering", "Information Technology",
            "Electronics and Communication", "Electrical Engineering",
            "Mechanical Engineering", "Civil Engineering",
            "Chemical Engineering"
        ],
    }

    if course in course_specializations:
        specialization_options = course_specializations[course]
    elif course == "Other":
        specialization_options = ["Other / Not Listed"]
    else:
        specialization_options = []

    if specialization_options:
        specialization = st.selectbox(
            "Specialization *",
            options=specialization_options,
            index=None,
            placeholder="Select your specialization"
        )
    else:
        specialization = st.selectbox(
            "Specialization *",
            options=[],
            index=None,
            placeholder="Select course first"
        )

    city = st.text_input(
        "City",
        placeholder="Enter your city"
    )
    pincode = st.text_input(
        "PIN Code (optional)",
        max_chars=6,
        placeholder="6-digit PIN code"
    )

# ============================================================
# STEP 2: COLLEGE CURRICULUM / SYLLABUS PDF
# ============================================================
st.divider()
st.header("📚 Step 2: College Curriculum / Syllabus Upload")
st.write(
    "Upload the syllabus issued by your college. EDISON-Align will extract available "
    "text from the PDF and compare it with the selected job description."
)
syllabus_file = st.file_uploader(
    "Upload College Curriculum / Syllabus (PDF)",
    type=["pdf"],
    key="syllabus_pdf"
)
syllabus_text = extract_pdf_text(syllabus_file)

if syllabus_file is not None:
    if syllabus_text:
        st.success("College syllabus uploaded and text extracted successfully.")
        with st.expander("Preview extracted syllabus text"):
            st.text(syllabus_text[:12000])
    else:
        st.warning(
            "No selectable text was extracted. If this is a scanned PDF, "
            "please upload a text-based PDF version."
        )

# ============================================================
# STEP 3: RESUME UPLOAD
# ============================================================
st.divider()
st.header("📄 Step 3: Resume Upload")
st.write("Upload your current resume in PDF format.")
resume_file = st.file_uploader(
    "Upload Your Resume (PDF)",
    type=["pdf"],
    key="resume_pdf"
)
resume_text = extract_pdf_text(resume_file)

if resume_file is not None:
    if resume_text:
        st.success("Resume uploaded and text extracted successfully.")
        with st.expander("Preview extracted resume text"):
            st.text(resume_text[:12000])
    else:
        st.warning(
            "No selectable text was extracted. Please upload a text-based PDF resume."
        )

resume_skills = detect_skills(resume_text)
syllabus_skills = detect_skills(syllabus_text)

if resume_text:
    st.subheader("🧠 Skills Detected from Resume")
    if resume_skills:
        st.write(", ".join(resume_skills))
    else:
        st.info("No matching skills were detected using the current keyword list.")

if syllabus_text:
    st.subheader("📘 Skills / Topics Detected from Syllabus")
    if syllabus_skills:
        st.write(", ".join(syllabus_skills))
    else:
        st.info("No matching skills were detected using the current keyword list.")


# ============================================================
# STEP 4: CAREER SECTOR + JOB ROLE SELECTION
# ============================================================
st.divider()
st.header("🧭 Step 4: Choose Your Career Sector")
st.write("Choose a sector, then select your target job role.")

image_folder = Path(__file__).parent / "images"

if "career_sector_choice" not in st.session_state:
    st.session_state["career_sector_choice"] = "IT Sector"

if "show_sector_roles" not in st.session_state:
    st.session_state["show_sector_roles"] = None

it_col, non_it_col = st.columns(2)

with it_col:
    st.subheader("💻 IT SECTOR")
    st.image(
    "it_sector.jpg",
    use_container_width=True
)
    if st.button(
        "View IT Job Roles",
        key="choose_it",
        use_container_width=True
    ):
        st.session_state["career_sector_choice"] = "IT Sector"
        st.session_state["show_sector_roles"] = "IT Sector"

with non_it_col:
    st.subheader("🏢 NON-IT SECTOR")
    st.image(
        "non_it_sector.jpg",
        use_container_width=True
    )
    if st.button(
        "View Non-IT Job Roles",
        key="choose_non_it",
        use_container_width=True
    ):
        st.session_state["career_sector_choice"] = "Non-IT Sector"
        st.session_state["show_sector_roles"] = "Non-IT Sector"

sector = st.session_state["career_sector_choice"]
st.session_state["career_sector"] = sector

# Display selectable job roles in Step 4 itself
if st.session_state["show_sector_roles"]:
    selected_sector = st.session_state["show_sector_roles"]
    selected_roles = sector_roles[selected_sector]

    st.subheader(f"🎯 {selected_sector} — Select a Job Role")

    previous_role = st.session_state.get(
        f"job_role_{selected_sector}"
    )
    default_index = (
        selected_roles.index(previous_role)
        if previous_role in selected_roles
        else 0
    )

    job_role = st.radio(
        "Choose your target job role:",
        selected_roles,
        index=default_index,
        key=f"job_role_{selected_sector}"
    )

    st.session_state["selected_job_role"] = job_role
else:
    # Default role before the user chooses a sector button
    job_role = sector_roles[sector][0]
    st.session_state["selected_job_role"] = job_role

# Keep the selected role available to the remaining project code
job_role = st.session_state["selected_job_role"]

# ============================================================
# STEP 5: INDUSTRY JOB DESCRIPTION
# ============================================================
st.divider()
st.header("🧾 Step 5: Industry Job Description")
role_description_templates = {
    "Data Analyst": "Analyze structured data, clean datasets, write SQL queries, use Excel and Python/Pandas, build dashboards in Power BI or similar tools, identify trends, and communicate evidence-based findings to stakeholders.",
    "Data Scientist": "Collect and preprocess data, perform statistical analysis, engineer features, build and evaluate machine-learning models, use Python and SQL, visualize findings, and communicate model limitations and insights.",
    "Machine Learning Engineer": "Develop, test, deploy, and monitor machine-learning models; prepare data pipelines; use Python, ML frameworks, version control, APIs, and model evaluation practices; collaborate with software and data teams.",
    "Business Analyst (IT)": "Gather and document business requirements, communicate with stakeholders, map business processes, support user acceptance testing, analyze data using SQL and Excel, and prepare clear functional documentation.",
    "Python Developer": "Design, implement, test, debug, and maintain Python applications; work with APIs, databases, Git, reusable code, and automated tests; troubleshoot issues and collaborate with development teams.",
    "Full Stack Developer": "Build and maintain front-end and back-end web applications using HTML, CSS, JavaScript, server-side programming, APIs, and databases; test functionality, use version control, and resolve application issues.",
    "Software Developer": "Analyze requirements, design and implement software components, write maintainable code, test and debug applications, use version control, document solutions, and collaborate through the software development lifecycle.",
    "Software Tester / QA Engineer": "Prepare test plans and test cases, execute manual and automated tests, report and track defects, validate fixes, perform regression testing, and communicate software quality risks to development teams.",
    "Cloud Engineer": "Configure and maintain cloud infrastructure, support deployment and automation, monitor availability and performance, manage access and security, troubleshoot Linux/networking issues, and document infrastructure changes.",
    "Cybersecurity Analyst": "Monitor systems for security events, investigate alerts, assess vulnerabilities and risks, support incident response, maintain security documentation, and apply networking, operating-system, and security fundamentals.",
    "Financial Analyst": "Analyze financial statements and business performance, prepare budgets and forecasts, maintain spreadsheet models, interpret accounting and statistical data, produce reports, and explain financial trends to decision-makers.",
    "Banking Operations Executive": "Process banking transactions and customer requests accurately, verify documentation, follow compliance procedures, maintain records, resolve service issues, and protect confidential customer information.",
    "Credit Analyst": "Evaluate borrower creditworthiness using financial statements, repayment history, income and risk indicators; prepare credit reports, review supporting documents, and communicate evidence-based lending recommendations.",
    "HR Executive": "Support employee records, recruitment coordination, onboarding, attendance and leave administration, HR documentation, employee queries, and compliance with organizational policies.",
    "HR Analyst": "Analyze workforce and HR metrics, maintain accurate employee datasets, prepare reports and dashboards, identify staffing and retention trends, and communicate findings that support HR decisions.",
    "Talent Acquisition Specialist": "Manage recruitment activities including job postings, candidate sourcing, resume screening, interview coordination, applicant communication, recruitment records, and candidate pipeline reporting.",
    "Marketing Analyst": "Analyze campaign performance, customer and sales data, prepare reports, measure marketing KPIs, identify audience trends, use spreadsheet and analytics tools, and recommend evidence-based campaign improvements.",
    "Market Research Analyst": "Plan surveys and research studies, collect and analyze market and customer data, interpret consumer behaviour, summarize competitor trends, prepare research reports, and present actionable findings.",
    "Digital Marketing Executive": "Plan and execute digital campaigns, support SEO and social media activity, coordinate content, monitor web and campaign analytics, track engagement and conversion metrics, and report campaign performance.",
    "Operations Analyst": "Review operational data and workflows, measure performance indicators, identify process bottlenecks, prepare reports, use spreadsheet analysis, and recommend practical improvements to efficiency and service quality.",
    "Supply Chain Analyst": "Analyze inventory, demand, supplier, logistics and delivery data; support forecasting and replenishment; monitor supply-chain KPIs; identify delays or cost risks; and communicate improvement recommendations.",
    "Sales Executive": "Identify prospective customers, explain products or services, maintain customer relationships, manage leads and sales records, negotiate within approved policies, meet sales targets, and report pipeline progress.",
    "Research Assistant": "Support literature reviews, data collection, survey or experiment administration, statistical analysis, research documentation, data quality checks, and preparation of reports under a research supervisor.",
    "Administrative Executive": "Coordinate office activities, maintain records and schedules, prepare correspondence and reports, manage documentation, support meetings, communicate with stakeholders, and use office productivity tools.",
    "Quality Analyst": "Monitor product or service quality against defined standards, inspect records and processes, analyze defects and quality metrics, document non-conformities, support corrective actions, and prepare quality reports."
}
job_description = role_description_templates.get(job_role, "Perform the core responsibilities associated with the selected role, use the listed role skills, maintain accurate documentation, follow professional standards, and communicate results clearly.")
st.write(f"**Selected role:** {job_role}")
st.info("A predefined role description is shown automatically. It is maintained in the project and is not live-fetched from a specific employer's current vacancy.")
with st.container(border=True):
    st.write(job_description)
st.caption("The role description and required skills update when you change the target job role.")

# ============================================================
# ANALYSIS BUTTON
# ============================================================
st.divider()
analyze_clicked = st.button("🔍 Analyze My Skills", type="primary", use_container_width=True)
if analyze_clicked:
    st.session_state.analyze = True

if st.session_state.analyze:
    missing_details = []
    if not name.strip():
        missing_details.append("Student Name")
    if academic_year == "Select academic year":
        missing_details.append("Academic Year")
    if not college.strip():
        missing_details.append("College / Institution")
    if course == "Select course":
        missing_details.append("Course")
    if not syllabus_text:
        missing_details.append("a readable College Syllabus PDF")
    if not resume_text:
        missing_details.append("a readable Resume PDF")
    if not job_description.strip():
        missing_details.append("Industry Job Description")

    if missing_details:
        st.warning("Please complete the following before analysis: " + ", ".join(missing_details) + ".")
    else:
        st.success("Student details and documents received. Analysis is ready.")
        st.divider()
        st.header("👤 Student Profile")
        profile_col1, profile_col2, profile_col3 = st.columns(3)
        profile_col1.write(f"**Name:** {name}")
        profile_col2.write(f"**Education:** {course}")
        profile_col3.write(f"**Academic Year:** {academic_year}")
        st.write(f"**Age:** {age}")
        st.write(f"**College / Institution:** {college}")
        if parent_name.strip():
            st.write(f"**Parent / Guardian:** {parent_name}")
        if parent_contact.strip():
            st.write(f"**Parent / Guardian Contact:** {parent_contact}")
        if parent_relation != "Select relationship":
            st.write(f"**Relationship:** {parent_relation}")
        if specialization.strip():
            st.write(f"**Specialization:** {specialization}")
        if city.strip():
            st.write(f"**City:** {city}")
        if pincode.strip():
            st.write(f"**PIN Code:** {pincode}")
        st.write(f"**Career Sector:** {sector}")
        st.write(f"**Target Job Role:** {job_role}")

        # ----------------------------------------------------
        # Resume skills vs job requirements
        # ----------------------------------------------------
        
        st.divider()
        st.header("📄 Resume + Student Skills — Job Role Matching")

        required_skills = job_requirements[job_role]

        combined_skills_lower = {
            skill.lower() for skill in resume_skills
        }

        matched_skills = [
            skill for skill in required_skills
            if skill.lower() in combined_skills_lower
        ]

        missing_job_skills = [
            skill for skill in required_skills
            if skill.lower() not in combined_skills_lower
        ]

        resume_match_score = (
            len(matched_skills) / len(required_skills) * 100
            if required_skills else 0
        )

        match_col1, match_col2, match_col3 = st.columns(3)
        match_col1.metric("Required Skills", len(required_skills))
        match_col2.metric("Matched Skills", len(matched_skills))
        match_col3.metric("Resume / Self-Reported Match", f"{resume_match_score:.2f}%")

        st.caption(
            "This is a keyword-based preliminary match using resume text and self-reported skills. "
            "It does not independently prove proficiency."
        )
        result_col1, result_col2 = st.columns(2)
        with result_col1:
            st.subheader("✅ Matched Skills")
            if matched_skills:
                for skill in matched_skills:
                    st.write("✅", skill)
            else:
                st.write("No required skills matched yet.")
        with result_col2:
            st.subheader("⚠️ Skills to Improve")
            if missing_job_skills:
                for skill in missing_job_skills:
                    st.write("⚠️", skill)
            else:
                st.success("All listed role skills matched by keyword.")

        # ----------------------------------------------------
        # Curriculum vs job description: TF-IDF
        # ----------------------------------------------------
        st.divider()
        st.header("📊 Curriculum–Industry Skill Analysis")
        clean_syllabus = preprocess_text(syllabus_text)
        clean_job = preprocess_text(job_description)

        if not clean_syllabus.strip() or not clean_job.strip():
            st.warning("Not enough text was extracted to calculate curriculum similarity.")
            similarity_percentage = 0.0
        else:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([clean_syllabus, clean_job])
            similarity_percentage = (
                cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
            )

        st.metric("Curriculum–Job Text Similarity", f"{similarity_percentage:.2f}%")
        st.caption(
            "TF-IDF cosine similarity estimates textual similarity; it is not, by itself, "
            "a definitive measurement of competency or curriculum quality."
        )

        # ----------------------------------------------------
        # Skill gap analysis based on role skills
        # ----------------------------------------------------
        st.divider()
        st.header("🔎 Skill Gap Identification")
        syllabus_lower = {skill.lower() for skill in syllabus_skills}
        resume_lower = {skill.lower() for skill in resume_skills}
        covered_skills = [
            skill for skill in required_skills if skill.lower() in syllabus_lower
        ]
        resume_evidence_skills = [
            skill for skill in required_skills if skill.lower() in resume_lower
        ]
        absent_from_syllabus = [
            skill for skill in required_skills if skill.lower() not in syllabus_lower
        ]

        gap_col1, gap_col2 = st.columns(2)
        with gap_col1:
            st.subheader("📚 Required Skills Found in Syllabus")
            if covered_skills:
                for skill in covered_skills:
                    st.write("✅", skill)
            else:
                st.write("No target-role skills were detected in the syllabus text.")
        with gap_col2:
            st.subheader("⚠️ Required Skills Not Detected in Syllabus")
            if absent_from_syllabus:
                for skill in absent_from_syllabus:
                    st.write("⚠️", skill)
            else:
                st.success("All listed target-role skills were found by keyword.")

        st.subheader("🧾 Required Skills Detected in Resume")
        if resume_evidence_skills:
            st.write(", ".join(resume_evidence_skills))
        else:
            st.write("No target-role skills were detected in the resume text.")

        st.caption(
            "A skill not detected in a PDF may still be present if it is phrased differently, "
            "shown in an image, or not extractable from the PDF."
        )

        # ----------------------------------------------------
        # Role-specific skill verification quiz
        # ----------------------------------------------------
        st.divider()
        st.header(f"📝 {job_role} Skill Verification Quiz")
        st.caption("Questions are selected for the current role. Change the job role to load a different quiz.")
        role_quiz_bank = {
            "Data Analyst": [
                {"question":"In SQL, which clause filters grouped results after aggregation?", "options":["WHERE", "HAVING", "ORDER BY", "JOIN"], "answer":1, "skill":"SQL"},
                {"question":"In Excel, which feature is useful for summarizing and grouping a large table of data?", "options":["PivotTable", "WordArt", "Page Border", "Slide Master"], "answer":0, "skill":"Excel"},
                {"question":"Which chart is usually suitable for showing a trend across months?", "options":["Line chart", "Pie chart with many slices", "Word cloud", "Icon only"], "answer":0, "skill":"Data Visualization"}
            ],
            "Data Scientist": [
                {"question":"A model performs well on training data but poorly on unseen data. What is this commonly called?", "options":["Overfitting", "Sampling", "Sorting", "Encoding only"], "answer":0, "skill":"Machine Learning"},
                {"question":"Which Pandas operation removes duplicate rows?", "options":["drop_duplicates()", "delete_rows()", "remove_same()", "unique_rows()"], "answer":0, "skill":"Pandas"},
                {"question":"Which measure summarizes the average squared prediction error?", "options":["MSE", "Recall", "Accuracy only", "Support count"], "answer":0, "skill":"Model Evaluation"}
            ],
            "Machine Learning Engineer": [
                {"question":"Why should a model be evaluated on data not used to train it?", "options":["To estimate generalization", "To increase file size", "To remove labels", "To skip validation"], "answer":0, "skill":"Machine Learning"},
                {"question":"Which practice helps track code changes in a team?", "options":["Git version control", "Renaming files randomly", "Deleting history", "Screenshots only"], "answer":0, "skill":"Git"},
                {"question":"What is a common purpose of an API in a deployed ML service?", "options":["Allow software to send requests and receive results", "Replace all model testing", "Guarantee perfect accuracy", "Store only images"], "answer":0, "skill":"APIs"}
            ],
            "Business Analyst (IT)": [
                {"question":"What should a business analyst do when requirements are unclear?", "options":["Clarify them with stakeholders", "Assume without asking", "Skip documentation", "Start deployment immediately"], "answer":0, "skill":"Requirements Analysis"},
                {"question":"What is the main purpose of user acceptance testing?", "options":["Check whether the solution meets user needs", "Choose font colours only", "Replace all requirements", "Measure internet speed"], "answer":0, "skill":"Software Testing"},
                {"question":"Which tool is commonly used to query relational data?", "options":["SQL", "HTML", "CSS", "PNG"], "answer":0, "skill":"SQL"}
            ],
            "Python Developer": [
                {"question":"Which Python structure is commonly used to handle exceptions?", "options":["try/except", "if/print only", "for/else only", "import/return"], "answer":0, "skill":"Python"},
                {"question":"What is the main purpose of an API?", "options":["Enable software components to communicate", "Format a spreadsheet", "Replace a database", "Compile images"], "answer":0, "skill":"APIs"},
                {"question":"Why use Git in development?", "options":["Track and manage code changes", "Increase CPU temperature", "Automatically prove code is correct", "Replace testing"], "answer":0, "skill":"Git"}
            ],
            "Full Stack Developer": [
                {"question":"Which technology primarily structures the content of a web page?", "options":["HTML", "SQL", "Git", "Excel"], "answer":0, "skill":"HTML"},
                {"question":"Which technology is mainly used to style web page presentation?", "options":["CSS", "SQL", "NumPy", "Linux"], "answer":0, "skill":"CSS"},
                {"question":"What is a common purpose of a back-end API?", "options":["Exchange data between client and server", "Only change text colour", "Replace all testing", "Create spreadsheet formulas"], "answer":0, "skill":"APIs"}
            ],
            "Software Developer": [
                {"question":"What is the purpose of unit testing?", "options":["Test a small component in isolation", "Test only office hardware", "Replace requirements", "Guarantee no future bugs"], "answer":0, "skill":"Problem Solving"},
                {"question":"Why is version control useful?", "options":["It records and manages code changes", "It makes all code bug-free", "It replaces documentation", "It prevents all outages"], "answer":0, "skill":"Git"},
                {"question":"What should readable, maintainable code prioritize?", "options":["Clear structure and meaningful names", "One huge function for everything", "No tests", "Hidden variable meanings"], "answer":0, "skill":"Software Design"}
            ],
            "Software Tester / QA Engineer": [
                {"question":"What should a good bug report include?", "options":["Steps to reproduce and expected/actual results", "Only the tester's name", "A random screenshot without context", "No environment details ever"], "answer":0, "skill":"Bug Reporting"},
                {"question":"What is regression testing?", "options":["Checking existing functionality after changes", "Testing only new colours", "Deleting test cases", "Skipping a retest"], "answer":0, "skill":"Software Testing"},
                {"question":"What is the purpose of a test case?", "options":["Define conditions, steps, and expected result", "Store passwords", "Design a logo", "Replace all requirements"], "answer":0, "skill":"Test Cases"}
            ],
            "Cloud Engineer": [
                {"question":"What does least privilege mean in cloud security?", "options":["Grant only the access required", "Give everyone administrator access", "Disable all logs", "Share credentials"], "answer":0, "skill":"Security"},
                {"question":"Which is a common use of Linux in cloud environments?", "options":["Managing servers and services", "Editing only presentation slides", "Replacing networking", "Creating financial statements"], "answer":0, "skill":"Linux"},
                {"question":"Why automate infrastructure tasks?", "options":["Improve consistency and repeatability", "Guarantee zero risk", "Remove monitoring", "Avoid all documentation"], "answer":0, "skill":"Automation"}
            ],
            "Cybersecurity Analyst": [
                {"question":"What should an analyst do after detecting a suspicious security alert?", "options":["Investigate evidence and follow incident procedures", "Ignore it automatically", "Delete all logs", "Share credentials publicly"], "answer":0, "skill":"Cybersecurity"},
                {"question":"What does a firewall primarily help control?", "options":["Network traffic", "Spreadsheet formulas", "Document spelling", "CPU brand"], "answer":0, "skill":"Networking"},
                {"question":"Why are security logs useful?", "options":["They support investigation and auditing", "They prevent every attack by themselves", "They replace access control", "They remove the need for updates"], "answer":0, "skill":"Security"}
            ],
            "Financial Analyst": [
                {"question":"Which statement reports revenue and expenses over a period?", "options":["Income statement", "Network diagram", "Test plan", "Recruitment tracker"], "answer":0, "skill":"Financial Analysis"},
                {"question":"What is a budget variance?", "options":["Difference between actual and budgeted values", "A list of employee names", "A type of chart title", "A database key"], "answer":0, "skill":"Accounting"},
                {"question":"Which tool is commonly used for financial modelling and calculations?", "options":["Excel", "CSS", "Git only", "HTML"], "answer":0, "skill":"Excel"}
            ],
            "Banking Operations Executive": [
                {"question":"Why must banking customer details be verified before processing a transaction?", "options":["To support accuracy, identity checks and compliance", "To make processing slower without reason", "To publish customer details", "To skip records"], "answer":0, "skill":"Banking Operations"},
                {"question":"How should sensitive customer information be handled?", "options":["Keep it confidential and follow policy", "Share it in public groups", "Store it on personal public pages", "Ignore access rules"], "answer":0, "skill":"Documentation"},
                {"question":"What is good customer service?", "options":["Listen, verify the issue and respond clearly", "Promise anything without checking", "Ignore complaints", "Disclose other customers' data"], "answer":0, "skill":"Customer Service"}
            ],
            "Credit Analyst": [
                {"question":"What is the purpose of credit analysis?", "options":["Assess ability and likelihood to repay debt", "Measure website colours", "Recruit employees", "Design a marketing logo"], "answer":0, "skill":"Credit Analysis"},
                {"question":"Why examine a borrower's income and existing obligations?", "options":["To assess repayment capacity", "To choose office furniture", "To determine social media reach", "To skip risk assessment"], "answer":0, "skill":"Financial Analysis"},
                {"question":"What does risk analysis help an organization do?", "options":["Identify and evaluate possible losses", "Guarantee no losses", "Replace all records", "Ignore uncertainty"], "answer":0, "skill":"Risk Analysis"}
            ],
            "HR Executive": [
                {"question":"What is onboarding?", "options":["Helping a new employee join and understand the organization", "Closing a financial year", "Testing a software release", "Forecasting inventory"], "answer":0, "skill":"Human Resources"},
                {"question":"How should employee records be handled?", "options":["Accurately and confidentially", "Shared with everyone", "Kept incomplete", "Published publicly"], "answer":0, "skill":"Documentation"},
                {"question":"What is a key recruitment activity?", "options":["Screening applications against role requirements", "Forecasting cloud usage", "Writing SQL joins only", "Testing network firewalls"], "answer":0, "skill":"Recruitment"}
            ],
            "HR Analyst": [
                {"question":"Which metric can help measure employee turnover?", "options":["Separations relative to average headcount", "Website page colour", "Number of SQL joins", "Cloud CPU temperature"], "answer":0, "skill":"Human Resources"},
                {"question":"Why validate HR data before reporting?", "options":["To reduce errors and improve reliability", "To hide missing values", "To inflate results", "To avoid definitions"], "answer":0, "skill":"Data Analysis"},
                {"question":"Which tool is often used to summarize HR data?", "options":["Excel", "CSS", "HTML only", "Firewall"], "answer":0, "skill":"Excel"}
            ],
            "Talent Acquisition Specialist": [
                {"question":"What is candidate sourcing?", "options":["Finding potential candidates for a vacancy", "Approving a loan", "Testing software", "Forecasting stock"], "answer":0, "skill":"Candidate Sourcing"},
                {"question":"Why use consistent job-related screening criteria?", "options":["To support fair and relevant candidate comparisons", "To guarantee every candidate is hired", "To avoid reading applications", "To share personal data"], "answer":0, "skill":"Recruitment"},
                {"question":"What should interview coordination include?", "options":["Scheduling, clear communication and records", "Publishing private feedback", "Ignoring availability", "Deleting candidate details without policy"], "answer":0, "skill":"Interviewing"}
            ],
            "Marketing Analyst": [
                {"question":"What does conversion rate measure?", "options":["The share of users who complete a desired action", "The number of office chairs", "Only the campaign budget", "The number of file folders"], "answer":0, "skill":"Marketing"},
                {"question":"Why segment customers in campaign analysis?", "options":["To compare behaviour across meaningful groups", "To hide all results", "To remove every metric", "To guarantee sales"], "answer":0, "skill":"Data Analysis"},
                {"question":"Which tool can summarize campaign performance data?", "options":["Excel or an analytics dashboard", "CSS only", "A firewall only", "A compiler only"], "answer":0, "skill":"Reporting"}
            ],
            "Market Research Analyst": [
                {"question":"Why use a representative sample in a survey?", "options":["To improve how well findings reflect the target population", "To guarantee every answer is true", "To avoid defining a population", "To remove all uncertainty"], "answer":0, "skill":"Research Methods"},
                {"question":"What is a leading risk in a poorly worded survey question?", "options":["Biased responses", "Faster internet", "More secure passwords", "Lower file size"], "answer":0, "skill":"Survey Analysis"},
                {"question":"Why report research limitations?", "options":["To explain uncertainty and context", "To hide methods", "To guarantee conclusions", "To avoid evidence"], "answer":0, "skill":"Reporting"}
            ],
            "Digital Marketing Executive": [
                {"question":"What does SEO aim to improve?", "options":["Visibility in relevant organic search results", "Bank loan approval", "Database normalization", "Employee onboarding"], "answer":0, "skill":"SEO"},
                {"question":"Which metric directly counts completed desired actions in a campaign?", "options":["Conversions", "Font size", "File extension", "CPU cores"], "answer":0, "skill":"Analytics"},
                {"question":"Why tailor social media content to an audience?", "options":["To improve relevance and engagement", "To eliminate analytics", "To guarantee virality", "To avoid brand consistency"], "answer":0, "skill":"Social Media"}
            ],
            "Operations Analyst": [
                {"question":"What is a process bottleneck?", "options":["A step that limits overall process throughput", "A marketing slogan", "A database password", "An employee benefit"], "answer":0, "skill":"Process Analysis"},
                {"question":"Why track operational KPIs over time?", "options":["To monitor performance and identify changes", "To avoid measuring outcomes", "To guarantee no delays", "To replace process maps"], "answer":0, "skill":"Reporting"},
                {"question":"Which tool is commonly used for tabular operational analysis?", "options":["Excel", "CSS", "HTML", "Image editor only"], "answer":0, "skill":"Excel"}
            ],
            "Supply Chain Analyst": [
                {"question":"What is demand forecasting used for?", "options":["Estimating future demand to support planning", "Recording employee birthdays only", "Styling web pages", "Testing software bugs"], "answer":0, "skill":"Forecasting"},
                {"question":"Why monitor inventory levels?", "options":["To balance availability and holding costs", "To guarantee demand never changes", "To remove supplier records", "To skip replenishment planning"], "answer":0, "skill":"Inventory Management"},
                {"question":"What does supply-chain analysis commonly examine?", "options":["Supplier, inventory, transport and delivery performance", "Only website colours", "Only recruitment interviews", "Only source-code style"], "answer":0, "skill":"Supply Chain"}
            ],
            "Sales Executive": [
                {"question":"What is a useful first step when handling a customer's need?", "options":["Ask questions and understand the requirement", "Make unrelated promises", "Ignore objections", "Share another customer's private details"], "answer":0, "skill":"Communication"},
                {"question":"Why maintain a sales pipeline?", "options":["Track leads and the stages of potential deals", "Replace customer consent", "Guarantee every lead converts", "Remove sales records"], "answer":0, "skill":"Sales"},
                {"question":"What is a win-win negotiation?", "options":["Seeking an acceptable outcome for both parties", "Hiding all terms", "Ignoring customer needs", "Changing agreed facts"], "answer":0, "skill":"Negotiation"}
            ],
            "Research Assistant": [
                {"question":"Why document research methods?", "options":["To make the work transparent and reproducible", "To hide how results were obtained", "To guarantee the hypothesis", "To avoid citing sources"], "answer":0, "skill":"Research Methods"},
                {"question":"What should be done before analyzing collected data?", "options":["Check quality, completeness and consistency", "Change results to fit expectations", "Delete all metadata", "Skip consent requirements"], "answer":0, "skill":"Data Analysis"},
                {"question":"Why distinguish evidence from interpretation in a report?", "options":["To make conclusions clear and defensible", "To make every result certain", "To avoid limitations", "To remove methodology"], "answer":0, "skill":"Documentation"}
            ],
            "Administrative Executive": [
                {"question":"What is an effective way to manage multiple office deadlines?", "options":["Use a prioritized task list and calendar", "Rely only on memory", "Delete reminders", "Avoid communicating delays"], "answer":0, "skill":"Administration"},
                {"question":"Why keep office documents organized?", "options":["To retrieve accurate information efficiently", "To make records inaccessible", "To avoid version control", "To publish confidential files"], "answer":0, "skill":"Documentation"},
                {"question":"What is professional workplace communication?", "options":["Clear, respectful and audience-appropriate information", "Unverified rumours", "Unclear instructions", "Sharing private data freely"], "answer":0, "skill":"Communication"}
            ],
            "Quality Analyst": [
                {"question":"What is a corrective action intended to do?", "options":["Address the cause of a detected nonconformity", "Hide a defect", "Delete audit evidence", "Guarantee no future issue without verification"], "answer":0, "skill":"Quality Control"},
                {"question":"Why record quality defects consistently?", "options":["To identify patterns and track improvements", "To avoid corrective action", "To change standards secretly", "To prevent all measurement"], "answer":0, "skill":"Documentation"},
                {"question":"What can trend analysis of defect data reveal?", "options":["Recurring problems or process changes", "The exact future with certainty", "Employee passwords", "Web page colours"], "answer":0, "skill":"Statistics"}
            ]
        }
        # A safe fallback ensures every catalogued role has its own role-labelled quiz.
        questions = role_quiz_bank.get(job_role, [
            {"question":f"For a {job_role}, why is {job_requirements[job_role][0]} important?", "options":[f"It supports a core responsibility of the {job_role} role", "It replaces all role duties", "It guarantees a job offer", "It removes the need for training"], "answer":0, "skill":job_requirements[job_role][0]},
            {"question":f"What is a good practice for a {job_role}?", "options":["Follow documented procedures and verify work", "Ignore errors", "Share confidential information", "Skip communication"], "answer":0, "skill":job_requirements[job_role][1]},
            {"question":f"When learning a skill required for {job_role}, what is most useful?", "options":["Practice with realistic tasks and review results", "Memorize the title only", "Avoid feedback", "Assume competence without practice"], "answer":0, "skill":job_requirements[job_role][2]}
        ])
        with st.form(f"role_quiz_{sector}_{job_role}"):
            answers = []
            for i, question in enumerate(questions):
                st.markdown(f"**Question {i + 1}.** {question['question']}")
                selected = st.radio(
                    "Select your answer:", question["options"], index=None,
                    key=f"quiz_answer_{sector}_{job_role}_{i}"
                )
                answers.append(selected)
            submit_quiz = st.form_submit_button("📊 Submit Role Quiz")

        if submit_quiz:
            unanswered = [i + 1 for i, answer in enumerate(answers) if answer is None]
            if unanswered:
                st.warning("Please answer all questions before submitting the quiz.")
            else:
                quiz_score = 0
                skill_correct = {}
                skill_total = {}
                for i, question in enumerate(questions):
                    skill = question["skill"]
                    skill_total[skill] = skill_total.get(skill, 0) + 1
                    correct_answer = question["options"][question["answer"]]
                    if answers[i] == correct_answer:
                        quiz_score += 1
                        skill_correct[skill] = skill_correct.get(skill, 0) + 1
                overall_quiz_score = quiz_score / len(questions) * 100
                st.success(f"{job_role} Quiz Score: {overall_quiz_score:.2f}%")
                st.subheader("📊 Skill-wise Verification")
                skill_scores = {}
                for skill, total in skill_total.items():
                    score = skill_correct.get(skill, 0) / total * 100
                    skill_scores[skill] = score
                    st.write(f"**{skill}:** {score:.2f}%")

                # ------------------------------------------------
                # Career readiness
                # ------------------------------------------------
                st.divider()
                st.header("🎯 Career Readiness Score")
                career_readiness = (
                    similarity_percentage + resume_match_score + overall_quiz_score
                ) / 3
                st.metric("Career Readiness Score", f"{career_readiness:.2f}%")
                st.caption(
                    "This prototype score equally averages text similarity, preliminary resume/skill match, "
                    "and the selected role's short quiz. It is a provisional indicator, not a validated hiring score."
                )

                if career_readiness >= 80:
                    st.success("🟢 Strong preliminary readiness — continue practical validation.")
                elif career_readiness >= 60:
                    st.info("🟡 Developing readiness — address the identified gaps.")
                else:
                    st.warning("🔴 Further skill development is recommended.")

                # ------------------------------------------------
                # Existing course/project recommendations
                # ------------------------------------------------
                st.divider()
                st.header("📚 Recommended Courses & Practical Projects")
                weak_skills_found = False
                for skill, score in skill_scores.items():
                    if score < 60:
                        weak_skills_found = True
                        st.markdown(f"**⚠️ {skill} — {score:.2f}%**")
                        if skill in recommendations:
                            course_name = recommendations[skill]["course"]
                            project_name = recommendations[skill]["project"]
                        else:
                            course_name = f"{skill} fundamentals and practical training for {job_role}"
                            project_name = f"A portfolio mini-project demonstrating {skill} in a {job_role} scenario"
                        st.write("📘 **Recommended Learning:**", course_name)
                        st.write("💻 **Recommended Practical Project:**", project_name)
                # Also recommend development actions for role skills not tested by the short quiz.
                for skill in missing_job_skills:
                    if skill not in skill_scores:
                        weak_skills_found = True
                        st.markdown(f"**📌 Skill gap: {skill}**")
                        st.write("📘 **Recommended Learning:**", recommendations.get(skill, {}).get("course", f"{skill} fundamentals for {job_role}"))
                        st.write("💻 **Recommended Practical Project:**", recommendations.get(skill, {}).get("project", f"A practical {job_role} portfolio task using {skill}"))
                if not weak_skills_found:
                    st.success("No weak areas were identified in this short quiz. Continue practicing with realistic role-based tasks.")
