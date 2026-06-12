import re

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="E-Commerce Product Recommender",
    layout="wide",
)


st.markdown(
    """
    <style>
      :root {
        --primary: #0066cc;
        --ink: #1d1d1f;
        --muted: #7a7a7a;
        --parchment: #f5f5f7;
        --hairline: #e0e0e0;
      }

      .stApp {
        background: #ffffff;
        color: var(--ink);
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }

      .hero {
        padding: 56px 40px;
        border-radius: 0;
        background: var(--parchment);
        text-align: center;
      }

      .hero h1 {
        max-width: 900px;
        margin: 0 auto;
        color: var(--ink);
        font-size: 56px;
        font-weight: 600;
        line-height: 1.07;
        letter-spacing: 0;
      }

      .hero p {
        max-width: 760px;
        margin: 18px auto 0;
        color: var(--ink);
        font-size: 22px;
        line-height: 1.35;
      }

      .metric-strip {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 16px;
        margin: 24px 0;
      }

      .metric-card {
        padding: 18px 20px;
        border: 1px solid var(--hairline);
        border-radius: 18px;
        background: #ffffff;
      }

      .metric-card strong {
        display: block;
        color: var(--ink);
        font-size: 26px;
        font-weight: 600;
      }

      .metric-card span {
        color: var(--muted);
        font-size: 14px;
      }

      .section-label {
        margin: 8px 0;
        color: var(--muted);
        font-size: 14px;
        font-weight: 600;
      }

      .stButton > button {
        border: 1px solid var(--primary);
        border-radius: 9999px;
        background: var(--primary);
        color: #ffffff;
        font-size: 17px;
        padding: 0.65rem 1.4rem;
      }

      .stButton > button:hover {
        border-color: #0071e3;
        background: #0071e3;
        color: #ffffff;
      }

      div[data-testid="stTextInput"] input,
      div[data-testid="stNumberInput"] input {
        border-radius: 9999px;
      }

      div[data-testid="stImage"] img {
        object-fit: contain;
      }

      .score-row {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 8px;
        margin: 10px 0;
      }

      .score-pill {
        padding: 8px 10px;
        border: 1px solid var(--hairline);
        border-radius: 9999px;
        background: #fafafc;
        color: var(--ink);
        font-size: 12px;
        text-align: center;
      }

      .reason-box {
        min-height: 56px;
        margin-top: 10px;
        padding: 10px 12px;
        border-radius: 12px;
        background: #f5f5f7;
        color: var(--muted);
        font-size: 13px;
        line-height: 1.35;
      }

      @media (max-width: 760px) {
        .hero {
          padding: 40px 20px;
        }

        .hero h1 {
          font-size: 34px;
        }

        .metric-strip {
          grid-template-columns: 1fr;
        }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    products = pd.read_csv("models/clean_data.csv")
    trending = pd.read_csv("models/trending_products.csv")
    products["Name"] = products["Name"].fillna("")
    products["Tags"] = products["Tags"].fillna("")
    products["Brand"] = products["Brand"].fillna("Unknown")
    products["Category"] = products["Category"].fillna("")
    products["ImageURL"] = products["ImageURL"].fillna("")
    products["Rating"] = pd.to_numeric(products["Rating"], errors="coerce").fillna(0)
    products["ReviewCount"] = pd.to_numeric(products["ReviewCount"], errors="coerce").fillna(0)
    return products, trending


@st.cache_resource
def build_similarity_model(search_text):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(search_text)
    similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return similarity


def first_image_url(image_urls):
    if pd.isna(image_urls):
        return None

    image_url = str(image_urls).split("|")[0].strip()
    return image_url or None


def resolve_product_name(products, search_text):
    search_text = (search_text or "").strip()
    if not search_text:
        return None

    names = products["Name"].dropna()
    exact_matches = names[names.str.lower() == search_text.lower()]
    if not exact_matches.empty:
        return exact_matches.iloc[0]

    partial_matches = names[names.str.contains(search_text, case=False, regex=False)]
    if not partial_matches.empty:
        return partial_matches.iloc[0]

    return None


def normalize_series(values):
    values = pd.Series(values).fillna(0).astype(float)
    max_value = values.max()
    if max_value <= 0:
        return values * 0
    return values / max_value


def tokenize(text):
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(token) > 2
    }


def candidate_products(products, search_text, limit=25):
    search_text = (search_text or "").strip()
    base = products.copy()
    base["popularity_score"] = (
        0.55 * normalize_series(base["Rating"])
        + 0.45 * normalize_series(base["ReviewCount"])
    )

    if not search_text:
        ranked = base.sort_values(
            ["popularity_score", "Rating", "ReviewCount"],
            ascending=False,
        ).head(limit).copy()
        ranked["selector_score"] = ranked["popularity_score"]
        ranked["selector_reason"] = "popular catalog item"
    else:
        query = search_text.lower()
        query_tokens = tokenize(query)
        searchable = (
            base["Name"].str.lower()
            + " "
            + base["Brand"].str.lower()
            + " "
            + base["Category"].str.lower()
            + " "
            + base["Tags"].str.lower()
        )
        matches = base[searchable.str.contains(query, regex=False)].copy()

        if matches.empty:
            matches = base.copy()
            matches["selector_reason"] = "broad catalog fallback"
        else:
            matches["selector_reason"] = "text match"

        name_lower = matches["Name"].str.lower()
        brand_lower = matches["Brand"].str.lower()
        category_lower = matches["Category"].str.lower()
        tags_lower = matches["Tags"].str.lower()

        matches["selector_score"] = matches["popularity_score"].copy()
        matches.loc[name_lower == query, "selector_score"] += 1.0
        matches.loc[name_lower.str.startswith(query), "selector_score"] += 0.7
        matches.loc[
            name_lower.str.contains(query, regex=False)
            | brand_lower.str.contains(query, regex=False)
            | category_lower.str.contains(query, regex=False)
            | tags_lower.str.contains(query, regex=False),
            "selector_score",
        ] += 0.4

        if query_tokens:
            matches["selector_score"] += matches.apply(
                lambda row: (
                    len(tokenize(row.get("Name", "")) & query_tokens)
                    + len(tokenize(row.get("Brand", "")) & query_tokens)
                    + len(tokenize(row.get("Category", "")) & query_tokens)
                    + len(tokenize(row.get("Tags", "")) & query_tokens)
                ) * 0.04,
                axis=1,
            )

        ranked = matches.sort_values(
            ["selector_score", "Rating", "ReviewCount"],
            ascending=False,
        ).head(limit).copy()

        def selector_reason(row):
            reasons = []
            row_name = str(row.get("Name", "")).lower()
            row_brand = str(row.get("Brand", "")).lower()
            row_category = str(row.get("Category", "")).lower()
            row_tags = str(row.get("Tags", "")).lower()

            if row_name == query:
                reasons.append("exact product match")
            elif row_name.startswith(query):
                reasons.append("name starts with search term")
            elif query in row_name or query in row_brand or query in row_category or query in row_tags:
                reasons.append("contains the search term")

            if float(row.get("Rating", 0)) >= 4.3:
                reasons.append("strong customer rating")
            if float(row.get("ReviewCount", 0)) >= 1000:
                reasons.append(f"{int(round(float(row.get('ReviewCount', 0)))):,}+ reviews")

            return "; ".join(reasons[:3]) or "popular catalog item"

        ranked["selector_reason"] = ranked.apply(selector_reason, axis=1)

    ranked["selector_key"] = ranked.index.astype(str)
    ranked["selector_label"] = ranked.apply(
        lambda row: f"{row.get('Name', 'Untitled product')} · {row.get('Brand', 'Unknown')} · {row.get('Category', 'Uncategorized')}",
        axis=1,
    )
    return ranked


def recommendation_reason(source_product, recommended_product):
    reasons = []

    source_brand = str(source_product.get("Brand", "")).strip().lower()
    recommended_brand = str(recommended_product.get("Brand", "")).strip().lower()
    if source_brand and source_brand != "unknown" and source_brand == recommended_brand:
        reasons.append("same brand")

    source_name = str(source_product.get("Name", "")).strip().lower()
    recommended_name = str(recommended_product.get("Name", "")).strip().lower()
    if source_name and source_name == recommended_name:
        reasons.append("same product variant family")

    source_categories = tokenize(source_product.get("Category", ""))
    recommended_categories = tokenize(recommended_product.get("Category", ""))
    shared_categories = sorted(source_categories & recommended_categories)
    if shared_categories:
        reasons.append("shared category: " + ", ".join(shared_categories[:3]))

    source_tags = tokenize(source_product.get("Tags", ""))
    recommended_tags = tokenize(recommended_product.get("Tags", ""))
    shared_tags = [
        tag for tag in sorted(source_tags & recommended_tags)
        if tag not in shared_categories
    ]
    if shared_tags:
        reasons.append("shared tags: " + ", ".join(shared_tags[:4]))

    if float(recommended_product.get("Rating", 0)) >= 4.3:
        reasons.append("high customer rating")

    review_count = float(recommended_product.get("ReviewCount", 0))
    if review_count >= 1000:
        reasons.append(f"{int(round(review_count)):,}+ reviews")

    return "; ".join(reasons[:3]) or "similar product text profile"


def recommend_products(products, similarity, selected_product_index, count, weights):
    if selected_product_index not in products.index:
        return pd.DataFrame(), None

    product_index = selected_product_index
    source_product = products.loc[product_index]
    similar_items = list(enumerate(similarity[product_index]))
    candidates = pd.DataFrame(similar_items, columns=["product_index", "content_similarity"])
    candidates = candidates[candidates["product_index"] != product_index]

    recommendation_frame = products.loc[candidates["product_index"]].copy()
    recommendation_frame["content_similarity"] = candidates["content_similarity"].values
    recommendation_frame["rating_score"] = pd.to_numeric(recommendation_frame["Rating"], errors="coerce").fillna(0).clip(0, 5) / 5
    recommendation_frame["review_score"] = normalize_series(recommendation_frame["ReviewCount"])

    total_weight = sum(weights.values()) or 1.0
    content_weight = weights["content"] / total_weight
    rating_weight = weights["rating"] / total_weight
    review_weight = weights["review"] / total_weight

    recommendation_frame["hybrid_score"] = (
        content_weight * recommendation_frame["content_similarity"]
        + rating_weight * recommendation_frame["rating_score"]
        + review_weight * recommendation_frame["review_score"]
    )
    recommendation_frame["why_recommended"] = recommendation_frame.apply(
        lambda row: recommendation_reason(source_product, row),
        axis=1,
    )

    columns = [
        "Name",
        "ReviewCount",
        "Brand",
        "ImageURL",
        "Rating",
        "Category",
        "content_similarity",
        "hybrid_score",
        "why_recommended",
    ]
    return recommendation_frame.sort_values("hybrid_score", ascending=False).head(count)[columns], source_product


def render_product_card(product, show_scores=False):
    image_url = first_image_url(product.get("ImageURL"))
    if image_url:
        st.image(image_url, use_container_width=True)
    else:
        st.image("static/img/img_1.png", use_container_width=True)

    st.markdown(f"**{product.get('Name', 'Untitled product')}**")
    st.caption(f"Brand: {product.get('Brand', 'Unknown')}")
    st.caption(f"Rating: {product.get('Rating', 0)} - {product.get('ReviewCount', 0)} reviews")
    if show_scores:
        st.markdown(
            f"""
            <div class="score-row">
              <div class="score-pill">Similarity {product.get('content_similarity', 0):.2f}</div>
              <div class="score-pill">Hybrid {product.get('hybrid_score', 0):.2f}</div>
                            <div class="score-pill">Rating {float(product.get('Rating', 0)):.1f}</div>
            </div>
            <div class="reason-box">Why: {product.get('why_recommended', 'similar product text profile')}</div>
            """,
            unsafe_allow_html=True,
        )


products, trending = load_data()
model_text = (
    products["Name"].astype(str)
    + " "
    + products["Brand"].astype(str)
    + " "
    + products["Category"].astype(str)
    + " "
    + products["Tags"].astype(str)
).tolist()
similarity_model = build_similarity_model(tuple(model_text))

st.markdown(
    """
    <section class="hero">
      <h1>E-Commerce Product Recommendation System</h1>
      <p>Search a product, pick the best catalog match, and get hybrid recommendations powered by TF-IDF, cosine similarity, rating, and review signals.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="metric-strip">
      <div class="metric-card"><strong>{len(products):,}</strong><span>Products in catalog</span></div>
      <div class="metric-card"><strong>{products['Brand'].nunique():,}</strong><span>Brands represented</span></div>
      <div class="metric-card"><strong>Hybrid</strong><span>Similarity + rating + reviews</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Model details", expanded=False):
    st.write(
        "The app ranks a seed product from the query, builds TF-IDF features from product name, brand, category, and tags, then blends content similarity with rating and review signals."
    )
    st.code(
        "hybrid_score = w_content * content_similarity + w_rating * normalized_rating + w_review * normalized_review_count",
        language="text",
    )

with st.sidebar:
    st.header("Hybrid ranking")
    content_weight = st.slider("Content similarity", 0.40, 0.85, 0.68, 0.01)
    rating_weight = st.slider("Rating signal", 0.05, 0.35, 0.17, 0.01)
    review_weight = st.slider("Review signal", 0.05, 0.35, 0.15, 0.01)
    st.caption("Weights are normalized automatically before scoring.")

weights = {
    "content": content_weight,
    "rating": rating_weight,
    "review": review_weight,
}

left, right = st.columns([3, 1])
with left:
    query = st.text_input(
        "Product search",
        value="",
        placeholder="Try Hempz, Gillette, Crest, lipstick, toothpaste...",
    )
with right:
    result_count = st.number_input("Results", min_value=1, max_value=20, value=6, step=1)

candidate_frame = candidate_products(products, query)

if candidate_frame.empty:
    st.warning("No product candidates available. Try a brand or category keyword like Crest, Gillette, nail, lipstick, or shampoo.")
else:
    selector_lookup = candidate_frame.set_index("selector_key")
    selected_key = st.selectbox(
        "Select the catalog product to recommend from",
        selector_lookup.index.tolist(),
        format_func=lambda key: selector_lookup.loc[key, "selector_label"],
    )
    selected_product_row = selector_lookup.loc[selected_key]
    st.caption(
        f"Selector confidence: {selected_product_row['selector_reason']} • score {float(selected_product_row['selector_score']):.2f}"
    )
    recommendations, source_product = recommend_products(
        products,
        similarity_model,
        int(selected_key),
        int(result_count),
        weights,
    )

    if source_product is not None and not recommendations.empty:
        st.markdown('<p class="section-label">Selected product</p>', unsafe_allow_html=True)
        with st.container(border=True):
            source_left, source_right = st.columns([1, 2])
            with source_left:
                image_url = first_image_url(source_product.get("ImageURL"))
                st.image(image_url or "static/img/img_1.png", use_container_width=True)
            with source_right:
                st.subheader(source_product.get("Name", "Selected product"))
                st.write(f"Brand: {source_product.get('Brand', 'Unknown')}")
                st.write(f"Category: {source_product.get('Category', 'Unknown')}")
                st.write(f"Rating: {source_product.get('Rating', 0)} - {source_product.get('ReviewCount', 0)} reviews")
                st.caption(f"Selected because: {selected_product_row['selector_reason']}")

        st.markdown("## Recommended products")
        columns = st.columns(3)
        for index, (_, product) in enumerate(recommendations.iterrows()):
            with columns[index % 3]:
                with st.container(border=True):
                    render_product_card(product, show_scores=True)
    else:
        st.warning("No recommendations were generated for the selected product.")

st.markdown("## Trending products")
trend_columns = st.columns(4)
for index, (_, product) in enumerate(trending.head(8).iterrows()):
    with trend_columns[index % 4]:
        with st.container(border=True):
            render_product_card(product)

with st.expander("Good demo searches"):
    st.write(
        [
            "Hempz",
            "Gillette",
            "Crest",
            "Pantene",
            "lipstick",
            "toothpaste",
            "nail polish",
        ]
    )
