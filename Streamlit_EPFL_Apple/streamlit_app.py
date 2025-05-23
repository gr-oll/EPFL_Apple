# ───────────────────────────────────────────────────────────────
#  Streamlit “Book-Recommendations” app
#  • Grid of top-10 covers per user
#  • Click → modal pop-up with details
#  • Clean meta fields (no brackets / accents)
#  • Robust to missing / bad image URLs
# ───────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import re, ast, unicodedata, io
from PIL import Image, ImageDraw, ImageFont
from urllib.parse import urlparse

# ──────────────────
#  DATA
# ──────────────────
recommendations = pd.read_csv("recommendations_collab_weight_pct_upscale_1_nitems_2.csv")
items           = pd.read_csv("merged_with_ids.csv")
interactions     = pd.read_csv("kaggle_data/interactions_train.csv")

# ──────────────────
#  HELPERS
# ──────────────────
def create_placeholder_image(text, width=128, height=192):
    """Grey rectangle with the (possibly truncated) title text."""
    img  = Image.new("RGB", (width, height), (220, 220, 220))
    draw = ImageDraw.Draw(img)
    try:
        font_path = "DejaVuSans-Bold.ttf"
        base_size = 12
        font      = ImageFont.truetype(font_path, base_size)
    except Exception:
        font      = ImageFont.load_default()
        base_size = 10

    words = text.split()
    for size in range(base_size, 7, -1):
        try:
            f = ImageFont.truetype(font_path, size)
        except Exception:
            f = ImageFont.load_default()
        lines, line, maxw = [], "", width - 10
        for w in words:
            test = (line + " " + w).strip()
            if f.getlength(test) <= maxw:
                line = test
            else:
                lines.append(line)
                line = w
        lines.append(line)
        h_needed = len(lines) * (f.size + 2)
        if h_needed <= height - 10:
            font = f
            break

    y = (height - h_needed) // 2
    for l in lines:
        draw.text(((width - font.getlength(l)) / 2, y), l, fill=(60, 60, 60), font=font)
        y += font.size + 2

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def valid_http_url(u) -> bool:
    """Return True for strings that start with http:// or https://"""
    return isinstance(u, str) and u.lower().startswith(("http://", "https://"))


def pretty(val):
    """
    Human-readable version of DB fields:
    • list-strings → comma list
    • floats that are ints → int
    • remove quotes, brackets, accents
    """
    if pd.isna(val):
        return None

    if isinstance(val, float) and val.is_integer():
        return str(int(val))

    if isinstance(val, str):
        txt = val.strip()

        # remove stray trailing 's' like  "]s"
        if txt.endswith("]s"):
            txt = txt[:-1]

        # list-string to comma list
        if txt.startswith("[") and txt.endswith("]"):
            try:
                parsed = ast.literal_eval(txt)
                if isinstance(parsed, list):
                    txt = ", ".join(str(x) for x in parsed)
            except Exception:
                txt = txt.strip("[]")

        txt = txt.strip(" '\"")
        txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode()
        return txt.rstrip("/")  # Ensure trailing slashes are removed

    return str(val)


def strip_br_tags(text: str) -> str | None:
    """Remove every <br>, <br/>, <br />; return None for NaN/None."""
    if not isinstance(text, str) or text.lower() == "nan":
        return None
    return re.sub(r"<br\s*/?>", "", text, flags=re.I).strip()


# ──────────────────
#  UI
# ──────────────────

st.title("📚 Book Recommendations")
search_type = st.radio("Search by", ["User", "Title", "Keywords"], horizontal=True)

if search_type == "User":
    filtered_user_ids = recommendations["user_id"].loc[recommendations["user_id"] != 0]
    user_id = st.selectbox("Select a user", filtered_user_ids)

    if user_id:
        user_recs = recommendations.loc[recommendations.user_id == user_id, "recommendation"]

        if user_recs.empty or pd.isna(user_recs.iat[0]):
            st.warning("No recommendations available for the selected user.")
        else:
            recs     = user_recs.iat[0]
            book_ids = recs.split()[:10]
            max_cols = 5

            for i in range(0, len(book_ids), max_cols):
                cols = st.columns(max_cols)
                for j, bid in enumerate(book_ids[i : i + max_cols]):
                    info = items.loc[items.i == int(bid)].iloc[0]

                    title     = info["Title"].rstrip("/")
                    short_t   = " ".join(title.split()[:3]) + ("…" if len(title.split()) > 3 else "")
                    cover_src = info["image"] if valid_http_url(info["image"]) else create_placeholder_image(title)

                    with cols[j]:
                        st.image(cover_src, use_container_width=True)
                        if st.button(short_t, key=f"btn_{bid}", use_container_width=True, help=title):
                            st.session_state["dialog_book_id"] = bid
                st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

elif search_type == "Title":
    title_options = sorted([pretty(title).rstrip("/") for title in items["Title"].dropna().unique()])
    title_query = st.selectbox("Select a book title", title_options)

    if title_query:
        original_title = items.loc[items["Title"].apply(lambda x: pretty(x).rstrip("/")) == title_query, "Title"].iat[0]
        selected_book = items.loc[items["Title"] == original_title].iloc[0]

        # Display the selected book
        st.markdown("### Selected Book")
        col = st.columns([1, 2])
        with col[0]:
            cover_src = selected_book["image"] if valid_http_url(selected_book["image"]) else create_placeholder_image(pretty(selected_book["Title"]))
            st.image(cover_src, width=128)
        with col[1]:
            st.markdown(f"**Title:** {pretty(selected_book['Title'])}")
            st.markdown(f"**Authors:** {pretty(selected_book.get('authors'))}")
            if st.button("More Info", key=f"btn_info_{selected_book['i']}"):
                st.session_state["dialog_book_id"] = selected_book["i"]

        st.markdown("#### Similar Books")
        similar_books = recommendations.loc[recommendations["recommendation"].str.contains(str(selected_book["i"]), na=False)]

        if not similar_books.empty:
            book_ids = similar_books.iloc[0]["recommendation"].split()[:10]
            max_cols = 5

            for i in range(0, len(book_ids), max_cols):
                cols = st.columns(max_cols)
                for j, bid in enumerate(book_ids[i : i + max_cols]):
                    info = items.loc[items.i == int(bid)].iloc[0]

                    title     = pretty(info["Title"]).rstrip("/")
                    short_t   = " ".join(title.split()[:3]) + ("…" if len(title.split()) > 3 else "")
                    cover_src = info["image"] if valid_http_url(info["image"]) else create_placeholder_image(title)

                    with cols[j]:
                        st.image(cover_src, use_container_width=True)
                        if st.button(short_t, key=f"btn_{bid}", use_container_width=True, help=title):
                            st.session_state["dialog_book_id"] = bid
                st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        else:
            st.warning("No similar books found for the selected title.")

# Add a new search option: Search by Semantic
elif search_type == "Keywords":
    title_query = st.text_input("Enter keywords to search for a book title")

    if title_query:
        # Perform a case-insensitive search for titles containing the query words
        # Limit the keyword search to the 30 most relevant results
        matching_titles = items[items["Title"].str.contains(title_query, case=False, na=False)].head(30)

        if not matching_titles.empty:
            st.markdown("### Search Results")
            max_cols = 5
            for i in range(0, len(matching_titles), max_cols):
                cols = st.columns(max_cols)
                for j, (_, book) in enumerate(matching_titles.iloc[i : i + max_cols].iterrows()):
                    title     = pretty(book["Title"]).rstrip("/")
                    short_t   = " ".join(title.split()[:3]) + ("…" if len(title.split()) > 3 else "")
                    cover_src = book["image"] if valid_http_url(book["image"]) else create_placeholder_image(title)

                    with cols[j]:
                        st.image(cover_src, use_container_width=True)
                        if st.button(short_t, key=f"keywords_{book['i']}", use_container_width=True, help=title):
                            st.session_state["dialog_book_id"] = book["i"]
                st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        else:
            st.warning("No books found matching your query.")


def show_book_dialog(info: pd.Series):
    """Modal dialog with book details (Streamlit ≥ 1.32: st.dialog)."""
    @st.dialog(f"📖 {info['Title'].rstrip('/')}", width=800)
    def _modal():
        col1, col2 = st.columns([1, 2])

        # left column: cover + meta
        with col1:
            cover_src = info["image"] if valid_http_url(info["image"]) else create_placeholder_image(info["Title"])
            st.image(cover_src, width=192)
            meta = {
                "Authors":     info.get("authors"),
                "Pages":       info.get("pages"),
                "Published":   info.get("date_published"),
                "Language":    info.get("language"),
                "Publisher":   info.get("Publisher"),
                "Subjects":    info.get("subjects"),
            }
            syn = strip_br_tags(info.get("synopsis"))
            if syn:
                for k, v in meta.items():
                    nice = pretty(v)
                    if nice:
                        st.markdown(f"**{k}:** {nice}")

        # right column: synopsis or details
        with col2:
            if syn:
                st.markdown("#### Synopsis")
                st.write(syn)
            else:
                for k, v in meta.items():
                    nice = pretty(v)
                    if nice:
                        st.markdown(f"**{k}:** {nice}")

            # Update the Google Books link to use only the title since ISBN is not available
            google_books_url = f"https://books.google.com/books?q=title:{info['Title'].replace(' ', '+')}"

            st.markdown(f"[Search on Google Books]({google_books_url})", unsafe_allow_html=True)            
    _modal()

# Calculate the most-read books only on page refresh
if "most_read_books" not in st.session_state:
    st.session_state["most_read_books"] = interactions["i"].value_counts().head(50).sample(5).index.tolist()

most_read_books = st.session_state["most_read_books"]
most_read_info = items[items["i"].isin(most_read_books)]

# Display the most-read books
st.markdown("## 📖 Most Read Books")
max_cols = 5
for i in range(0, len(most_read_books), max_cols):
    cols = st.columns(max_cols)
    for j, book_id in enumerate(most_read_books[i : i + max_cols]):
        info = most_read_info[most_read_info["i"] == book_id].iloc[0]

        title     = info["Title"].rstrip("/")
        short_t   = " ".join(title.split()[:3]) + ("…" if len(title.split()) > 3 else "")
        cover_src = info["image"] if valid_http_url(info["image"]) else create_placeholder_image(title)

        with cols[j]:
            st.image(cover_src, use_container_width=True)
            if st.button(short_t, key=f"most_read_{book_id}", use_container_width=True, help=title):
                st.session_state["dialog_book_id"] = book_id
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)


# Handle modal dialog for book details
if "dialog_book_id" in st.session_state:
    bid  = st.session_state.pop("dialog_book_id")
    info = items.loc[items.i == int(bid)].iloc[0]
    show_book_dialog(info)