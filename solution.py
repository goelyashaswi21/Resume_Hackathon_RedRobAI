import math

candidates = [
    ("Arjun Sharma",    "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning"),
    ("Priya Nair",      "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS"),
    ("Rahul Gupta",     "Java, Spring Boot, MySql, Microservices, Docker, kubernates"),
    ("Sneha Patel",     "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib"),
    ("Vikram Singh",    "C++, Algoritms, Data Structure, competitive programming, python"),
    ("Ananya Krishnan", "javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD"),
    ("Karan Mehta",     "Python, Sklearn, XGboost, feature engineering, SQL, tableau"),
    ("Deepika Rao",     "Java, Android, Kotlin, Firebase, REST, UI/UX, figma"),
    ("Aditya Kumar",    "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest"),
    ("Meera Iyer",      "python, R, statistics, ML, regression, clustering, Power-BI"),
]

jds = {
    "JD-1 — Kakao (ML Engineer)":
        "Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, SQL, Data Visualization, NLP, BERT, Feature Engineering, Statistics",
    "JD-2 — Naver (Backend Engineer)":
        "Java, Spring Boot, MySQL, PostgreSQL, Microservices, Docker, Kubernetes, REST API, CI/CD, Redis",
    "JD-3 — Line (Frontend Engineer)":
        "JavaScript, React, Vue, TypeScript, REST API, HTML/CSS, Node.js, GraphQL, Redux, Jest, AWS",
}

SKILL_ALIASES = {
    "python": "python", "pyhton": "python",
    "java": "java",
    "javascript": "javascript", "javascrpit": "javascript", "js": "javascript",
    "typescript": "typescript", "typescrpit": "typescript",
    "c++": "cpp", "cpp": "cpp",
    "r": "r",
    "kotlin": "kotlin",
    "machinelearning": "machine_learning", "machine learning": "machine_learning",
    "ml": "machine_learning", "sklearn": "machine_learning",
    "deeplearning": "deep_learning", "deep learning": "deep_learning",
    "deep-learning": "deep_learning",
    "tensorflow": "tensorflow", "pytorch": "pytorch", "keras": "keras",
    "nlp": "nlp", "bert": "bert", "xgboost": "xgboost",
    "feature engineering": "feature_engineering",
    "statistics": "statistics", "stats": "statistics",
    "regression": "regression", "clustering": "clustering",
    "data-viz": "data_visualization", "data visualization": "data_visualization",
    "data viz": "data_visualization", "matplotlib": "data_visualization",
    "tableau": "data_visualization", "power-bi": "data_visualization",
    "power bi": "data_visualization", "powerbi": "data_visualization",
    "pandas": "pandas", "numpy": "numpy",
    "react": "react", "reacts": "react", "reactjs": "react",
    "vue": "vue", "vue.js": "vue", "vuejs": "vue",
    "redux": "redux", "tailwind": "tailwind",
    "html/css": "html_css", "html css": "html_css",
    "html": "html_css", "css": "html_css",
    "jest": "jest", "graphql": "graphql",
    "node.js": "nodejs", "nodejs": "nodejs", "node js": "nodejs",
    "flask": "flask",
    "spring boot": "spring_boot", "springboot": "spring_boot",
    "rest api": "rest_api", "rest": "rest_api", "restapi": "rest_api",
    "microservices": "microservices",
    "sql": "sql", "mysql": "mysql", "mysq": "mysql",
    "postgresql": "postgresql", "postgres": "postgresql",
    "mongodb": "mongodb", "redis": "redis",
    "docker": "docker",
    "kubernetes": "kubernetes", "kubernates": "kubernetes", "k8s": "kubernetes",
    "ci/cd": "ci_cd", "cicd": "ci_cd", "ci cd": "ci_cd",
    "aws": "aws",
    "android": "android", "firebase": "firebase",
    "algorithms": "algorithms", "algoritms": "algorithms",
    "data structure": "data_structures", "data structures": "data_structures",
    "competitive programming": "competitive_programming",
    "ui/ux": "ui_ux", "ui ux": "ui_ux", "figma": "figma",
}

multi_word_keys = sorted([k for k in SKILL_ALIASES if " " in k], key=lambda x: -len(x))


def normalize(raw):
    tokens = [t.strip().lower() for t in raw.split(",")]
    result = []
    for token in tokens:
        found = False
        for phrase in multi_word_keys:
            if token == phrase:
                result.append(SKILL_ALIASES[phrase])
                found = True
                break
        if not found:
            if token in SKILL_ALIASES:
                result.append(SKILL_ALIASES[token])
    return result


def deduplicate(skills):
    seen = set()
    out = []
    for s in skills:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


resumes = []
for name, raw in candidates:
    skills = deduplicate(normalize(raw))
    resumes.append((name, skills))

all_skills = set()
for name, skills in resumes:
    all_skills.update(skills)

vocab = sorted(all_skills)
vocab_index = {s: i for i, s in enumerate(vocab)}
V = len(vocab)

df = {s: 0 for s in vocab}
for name, skills in resumes:
    for s in skills:
        df[s] += 1

idf = {s: math.log(10 / df[s]) for s in vocab}

tfidf = []
for name, skills in resumes:
    n = len(skills)
    vec = [0.0] * V
    for s in skills:
        vec[vocab_index[s]] = (1.0 / n) * idf[s]
    tfidf.append((name, vec))


def get_jd_vector(raw):
    tokens = [t.strip().lower() for t in raw.split(",")]
    matched = []
    for token in tokens:
        found = False
        for phrase in multi_word_keys:
            if token == phrase:
                matched.append(SKILL_ALIASES[phrase])
                found = True
                break
        if not found and token in SKILL_ALIASES:
            matched.append(SKILL_ALIASES[token])
    matched = set(matched)
    return [1 if vocab[i] in matched else 0 for i in range(V)]


def cosine(a, b):
    dot = sum(a[i] * b[i] for i in range(len(a)))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


for jd_name, raw_skills in jds.items():
    jd_vec = get_jd_vector(raw_skills)
    scores = [(name, cosine(vec, jd_vec)) for name, vec in tfidf]
    scores.sort(key=lambda x: (-x[1], x[0]))
    print(jd_name)
    print(", ".join(f"{name}({score:.2f})" for name, score in scores[:3]))
    print()
