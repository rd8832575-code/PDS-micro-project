import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz


# =========================
# LOAD CSV
# =========================

df = pd.read_csv("movie_dataset.csv")

print("\n===== DATA CLEANING =====")
print("Records before cleaning:", len(df))


# =========================
# DATA CLEANING
# =========================

df = df.fillna("")

df.drop_duplicates(inplace=True)
df.drop_duplicates("title", keep="first", inplace=True)

columns = [
    "title", "genre", "description",
    "director", "cast", "industry", "keywords"
]

for col in columns:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )

df.reset_index(drop=True, inplace=True)

print("Records after cleaning :", len(df))
print("Records removed        :", len(pd.read_csv(
    "movie_dataset.csv"
)) - len(df))


# =========================
# TF-IDF + COSINE SIMILARITY
# =========================

df["features"] = (
    df["genre"] + " " +
    df["keywords"] + " " +
    df["cast"] + " " +
    df["director"] + " " +
    df["description"]
)

tfidf = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

matrix = tfidf.fit_transform(df["features"])

similarity = cosine_similarity(matrix)


# =========================
# SEARCH DATA
# =========================

movies = df["title"].tolist()

actors = (
    df["cast"]
    .str.split(",")
    .explode()
    .str.strip()
    .unique()
    .tolist()
)

directors = (
    df["director"]
    .str.split(",")
    .explode()
    .str.strip()
    .unique()
    .tolist()
)

genres = (
    df["genre"]
    .str.split(",")
    .explode()
    .str.strip()
    .unique()
    .tolist()
)

industries = df["industry"].unique().tolist()


# =========================
# RAPIDFUZZ SEARCH
# =========================

def find_match(text, choices):

    result = process.extractOne(
        text.lower(),
        choices,
        scorer=fuzz.ratio
    )

    if result and result[1] >= 75:
        return result[0], result[1]

    return None, 0


# =========================
# MOVIE SEARCH
# =========================

def movie_search(name, count):

    match, score = find_match(name, movies)

    if not match:
        return False

    index = df.index[
        df["title"] == match
    ][0]

    scores = similarity[index].argsort()[::-1]

    print("\n===== MOVIE SEARCH =====")
    print("Matched Movie:", match.title())
    print("Match Score  :", round(score, 2), "%")

    print("\nRecommended Movies:")

    shown = 0

    for i in scores:

        if i == index:
            continue

        movie = df.iloc[i]

        print(
            f"{shown + 1}. {movie['title'].title()} "
            f"({movie['year']})"
        )

        print(
            "   Genre:",
            movie["genre"].title(),
            "| Similarity:",
            round(similarity[index][i] * 100, 2),
            "%"
        )

        shown += 1

        if shown == count:
            break

    return True


# =========================
# ACTOR SEARCH
# =========================

def actor_search(name, count):

    match, score = find_match(name, actors)

    if not match:
        return False

    result = df[
        df["cast"].apply(
            lambda x: match in [
                a.strip() for a in x.split(",")
            ]
        )
    ]

    if result.empty:
        return False

    print("\n===== ACTOR SEARCH =====")
    print("Matched Actor:", match.title())
    print("Match Score  :", round(score, 2), "%")

    for n, (_, movie) in enumerate(
        result.head(count).iterrows(), 1
    ):
        print(
            f"{n}. {movie['title'].title()} "
            f"({movie['year']})"
        )

    return True


# =========================
# DIRECTOR SEARCH
# =========================

def director_search(name, count):

    match, score = find_match(name, directors)

    if not match:
        return False

    result = df[df["director"] == match]

    if result.empty:
        return False

    print("\n===== DIRECTOR SEARCH =====")
    print("Matched Director:", match.title())
    print("Match Score     :", round(score, 2), "%")

    for n, (_, movie) in enumerate(
        result.head(count).iterrows(), 1
    ):
        print(
            f"{n}. {movie['title'].title()} "
            f"({movie['year']})"
        )

    return True


# =========================
# GENRE SEARCH
# =========================

def genre_search(name, count):

    match, score = find_match(name, genres)

    if not match:
        return False

    result = df[
        df["genre"].apply(
            lambda x: match in [
                g.strip() for g in x.split(",")
            ]
        )
    ]

    if result.empty:
        return False

    print("\n===== GENRE SEARCH =====")
    print("Matched Genre:", match.title())
    print("Match Score  :", round(score, 2), "%")

    for n, (_, movie) in enumerate(
        result.head(count).iterrows(), 1
    ):
        print(
            f"{n}. {movie['title'].title()} "
            f"({movie['year']})"
        )

    return True


# =========================
# INDUSTRY SEARCH
# =========================

def industry_search(name, count):

    match, score = find_match(name, industries)

    if not match:
        return False

    result = df[df["industry"] == match]

    if result.empty:
        return False

    print("\n===== INDUSTRY SEARCH =====")
    print("Matched Industry:", match.title())
    print("Match Score     :", round(score, 2), "%")

    for n, (_, movie) in enumerate(
        result.head(count).iterrows(), 1
    ):
        print(
            f"{n}. {movie['title'].title()} "
            f"({movie['year']})"
        )

    return True


# =========================
# MAIN SEARCH
# =========================

def recommend(text, count):

    text = text.strip().lower()

    if industry_search(text, count):
            return

    if genre_search(text, count):
            return

    if director_search(text, count):
            return

    if actor_search(text, count):
        return

    if movie_search(text, count):
            return

    print("\nNo match found.")
    print("Minimum match required: 75%")


# =========================
# PROGRAM
# =========================

print("\n======================================")
print("      MOVIE RECOMMENDATION SYSTEM")
print("======================================")

print("""
Search by:

Movie     : Jawan
Actor     : Shah Rukh Khan
Director  : Atlee
Genre     : Action
Industry  : Bollywood

Spelling mistakes are supported.
Minimum match: 75%

Type 'exit' to stop.
""")

while True:

    text = input(
        "\nEnter movie / actor / director / genre / industry: "
    )

    if text.lower() == "exit":
        print("Program terminated.")
        break

    number = input(
        "How many movies? (default 5): "
    )

    try:
        count = int(number) if number else 5
    except:
        count = 5

    recommend(text, count)