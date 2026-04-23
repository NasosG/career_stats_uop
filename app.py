import re
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


DATA_FILE = Path("monthly_metrics.csv")
WEBINARS_FILE = Path("webinars.csv")
PAGES_FILE = Path("page_performance.csv")


def normalize_month(s) -> str:
    """Ensure month is YYYY-MM. Handles 2025-10, 202601, 2025-10-01 (date). Returns '' if invalid."""
    if pd.isna(s) or s is None:
        return ""
    s = str(s).strip()
    if not s or s.lower() in ("nan", "nat"):
        return ""
    # Already YYYY-MM
    if "-" in s:
        parts = s.split("-")
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            return f"{parts[0]}-{parts[1].zfill(2)}"
    # 202601 -> 2026-01
    if len(s) == 6 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}"
    return ""


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip BOM and spaces from column names so 'month' is always found."""
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    # Force first column (month) as string so pandas doesn't parse 2025-10 as date/number
    df = pd.read_csv(path, dtype={0: str})
    df = _normalize_columns(df)
    if "month" not in df.columns:
        raise ValueError("Το αρχείο πρέπει να έχει στήλη 'month' σε μορφή YYYY-MM.")
    df = df.assign(month=df["month"].astype(str).str.strip())
    df = df[~df["month"].isin(["", "nan", "NaN"])]
    df = df.sort_values("month")
    return df


def parse_duration_to_seconds(x):
    """
    Accept values like:
    - '41s'
    - '2m 32s'
    - '1m'
    - numeric seconds
    """
    if pd.isna(x):
        return None
    if isinstance(x, (int, float)):
        return float(x)

    s = str(x).strip().lower()
    total = 0
    # minutes
    if "m" in s:
        parts = s.split("m")
        try:
            minutes = int(parts[0].strip())
        except ValueError:
            minutes = 0
        total += minutes * 60
        s = parts[1]
    # seconds
    s = s.replace("s", "").strip()
    if s:
        try:
            total += int(s)
        except ValueError:
            pass
    return total if total > 0 else None


def format_seconds_to_label(sec):
    if sec is None or pd.isna(sec):
        return "—"
    sec = int(round(sec))
    m, s = divmod(sec, 60)
    if m == 0:
        return f"{s}s"
    if s == 0:
        return f"{m}m"
    return f"{m}m {s}s"


def pct_change(new, old):
    if old is None or old == 0 or pd.isna(old):
        return None
    return (new - old) / old * 100.0


def format_change_arrow(pct):
    if pct is None or pd.isna(pct):
        return "—"
    pct_round = round(pct)
    if pct_round > 0:
        return f"⬆️ +{pct_round}%"
    if pct_round < 0:
        return f"⬇️ {pct_round}%"
    return "➖ Σταθερό"


def month_label_gr(month_str: str) -> str:
    months_gr = [
        "Ιανουαρίου",
        "Φεβρουαρίου",
        "Μαρτίου",
        "Απριλίου",
        "Μαΐου",
        "Ιουνίου",
        "Ιουλίου",
        "Αυγούστου",
        "Σεπτεμβρίου",
        "Οκτωβρίου",
        "Νοεμβρίου",
        "Δεκεμβρίου",
    ]
    s = str(month_str).strip()
    parts = s.split("-")
    if len(parts) != 2:
        # Fallback: e.g. "202601" -> 2026-01
        if len(s) == 6 and s.isdigit():
            return f"{months_gr[int(s[4:6]) - 1]} {s[:4]}"
        return s
    year, m = parts[0], parts[1]
    m_int = int(m)
    return f"{months_gr[m_int - 1]} {year}"


def build_text_summary(df: pd.DataFrame, month: str) -> str:
    row = df[df["month"] == month].iloc[0]
    prev_df = df[df["month"] < month]
    if prev_df.empty:
        prev = None
    else:
        prev = prev_df.iloc[-1]

    # Compute duration in seconds
    cur_dur_sec = parse_duration_to_seconds(row.get("avg_session_duration", None))
    prev_dur_sec = (
        parse_duration_to_seconds(prev.get("avg_session_duration", None)) if prev is not None else None
    )

    # Basic metrics (safe handling of NaN)
    def safe_int(val):
        if val is None or pd.isna(val):
            return 0
        try:
            return int(val)
        except (TypeError, ValueError):
            return 0

    new_users = safe_int(row.get("new_users", 0))
    returning_users = safe_int(row.get("returning_users", 0))
    total_users = new_users + returning_users
    share_new = (new_users / total_users * 100.0) if total_users > 0 else None
    share_ret = (returning_users / total_users * 100.0) if total_users > 0 else None

    # Use pd.to_numeric so we get numbers even if CSV has strings or empty cells
    impressions = pd.to_numeric(row.get("impressions"), errors="coerce")
    clicks = pd.to_numeric(row.get("clicks"), errors="coerce")
    ctr = pd.to_numeric(row.get("ctr"), errors="coerce")
    avg_position = pd.to_numeric(row.get("avg_position"), errors="coerce")
    if pd.isna(impressions):
        impressions = None
    else:
        impressions = float(impressions)
    if pd.isna(clicks):
        clicks = None
    else:
        clicks = float(clicks)
    if pd.isna(ctr):
        ctr = None
    else:
        ctr = float(ctr)
    if pd.isna(avg_position):
        avg_position = None

    # Previous month metrics (handle NaN safely)
    if prev is not None:
        imp_prev = pd.to_numeric(prev.get("impressions"), errors="coerce")
        clk_prev = pd.to_numeric(prev.get("clicks"), errors="coerce")
        ctr_prev = pd.to_numeric(prev.get("ctr"), errors="coerce")
        imp_prev = None if pd.isna(imp_prev) or imp_prev < 0 else float(imp_prev)
        clk_prev = None if pd.isna(clk_prev) or clk_prev < 0 else float(clk_prev)
        ctr_prev = None if pd.isna(ctr_prev) or ctr_prev < 0 else float(ctr_prev)
    else:
        imp_prev = clk_prev = ctr_prev = None

    imp_change = pct_change(impressions, imp_prev) if impressions is not None and imp_prev is not None else None
    clk_change = pct_change(clicks, clk_prev) if clicks is not None and clk_prev is not None else None
    ctr_change = (
        pct_change(ctr, ctr_prev) if ctr is not None and ctr_prev is not None else None
    )

    # Channel breakdown
    direct = safe_int(row.get("direct_sessions", 0))
    organic = safe_int(row.get("organic_search_sessions", 0))
    social = safe_int(row.get("organic_social_sessions", 0))
    referral = safe_int(row.get("referral_sessions", 0))
    unassigned = safe_int(row.get("unassigned_sessions", row.get("unassigned", 0)))

    total_sessions = direct + organic + social + referral + unassigned
    def share(x):
        return round(x / total_sessions * 100.0) if total_sessions > 0 else 0

    direct_p = share(direct)
    organic_p = share(organic)
    social_p = share(social)
    referral_p = share(referral)
    unassigned_p = share(unassigned)

    # Devices
    mobile_imp = safe_int(row.get("mobile_impressions", 0))
    desktop_imp = safe_int(row.get("desktop_impressions", 0))
    tablet_imp = safe_int(row.get("tablet_impressions", 0))

    # Text construction
    month_title = month_label_gr(month)
    lines = []

    lines.append(f"**{month_title} – Σύνοψη**")
    lines.append("")
    lines.append(f"Περίοδος: {month_title}")
    lines.append("")

    if total_users > 0 and share_new is not None:
        lines.append("Τα analytics δείχνουν ότι είχαμε:")
        lines.append("")
        lines.append(f"- Νέοι χρήστες: {new_users}")
        lines.append(f"- Returning users: {returning_users}")
        lines.append("")
        lines.append(
            f"Περίπου ~{round(share_new)}% νέοι / {round(share_ret)}% επιστρέφοντες χρήστες."
        )
        lines.append("")

    # Avg session duration
    lines.append("---")
    lines.append("")
    lines.append(f"**Avg session duration:** {row.get('avg_session_duration', '—')}")
    lines.append("")
    if cur_dur_sec is not None and prev_dur_sec is not None:
        dur_change = pct_change(cur_dur_sec, prev_dur_sec)
        arrow = format_change_arrow(dur_change)
        lines.append(
            f"Ο μέσος χρόνος παραμονής στη σελίδα είναι {format_seconds_to_label(cur_dur_sec)} ({arrow} σε σχέση με τον προηγούμενο μήνα)."
        )
    else:
        lines.append(
            "Ο μέσος χρόνος παραμονής στη σελίδα είναι ικανοποιητικός και δείχνει ότι οι χρήστες αφιερώνουν χρόνο για να δουν το περιεχόμενο."
        )
    lines.append("")

    # Search results
    lines.append("---")
    lines.append("")
    lines.append("🔎 **Συνολικά αποτελέσματα (Google Search)**")
    lines.append("")

    if impressions is not None:
        imp_str = f"{int(impressions):,}".replace(",", ".")
    else:
        imp_str = "—"
    if clicks is not None:
        clk_str = f"{int(clicks):,}".replace(",", ".")
    else:
        clk_str = "—"

    lines.append(f"- Impressions: {imp_str}")
    lines.append(f"- Clicks: {clk_str}")
    if ctr is not None and not pd.isna(ctr):
        lines.append(f"- CTR: ~{round(ctr, 2)}%")
    if avg_position is not None and not pd.isna(avg_position):
        lines.append(f"- Average Position: ~{round(avg_position, 2)}")
    lines.append("")

    # Comparison simple text
    if prev is not None:
        lines.append("📊 **Σύγκριση με προηγούμενο μήνα**")
        lines.append("")
        imp_arrow = format_change_arrow(imp_change)
        clk_arrow = format_change_arrow(clk_change)
        ctr_arrow = format_change_arrow(ctr_change)
        lines.append(f"- Impressions: {imp_arrow}")
        lines.append(f"- Clicks: {clk_arrow}")
        lines.append(f"- CTR: {ctr_arrow}")
        lines.append("")
        lines.append("Τι σημαίνει αυτό πρακτικά;")
        lines.append("")
        lines.append(
            "Το visibility στη Google συνεχίζει να εξελίσσεται. "
            "Τα impressions και τα clicks δείχνουν πώς κινείται η οργανική παρουσία του site στον χρόνο."
        )
        lines.append("")

    # Channels
    lines.append("**Κανάλια**")
    lines.append("")
    lines.append(f"- Direct: {direct} (~{direct_p}%)")
    lines.append(f"- Organic Search: {organic} (~{organic_p}%)")
    lines.append(f"- Organic Social: {social} (~{social_p}%)")
    lines.append(f"- Referral: {referral} (~{referral_p}%)")
    if unassigned > 0:
        lines.append(f"- Unassigned: {unassigned} (~{unassigned_p}%)")
    lines.append("")
    lines.append(
        "Η οργανική αναζήτηση (Google) αποτελεί βασικό κανάλι εισόδου στο site, κάτι ιδιαίτερα θετικό "
        "γιατί δείχνει ότι το site ενισχύει σταδιακά την παρουσία του στα αποτελέσματα αναζήτησης."
    )
    lines.append("")

    # Devices
    lines.append("📱 **Ανάλυση συσκευών (Impressions)**")
    lines.append("")
    lines.append(f"- Mobile: {mobile_imp:,}".replace(",", "."))
    lines.append(f"- Desktop: {desktop_imp:,}".replace(",", "."))
    lines.append(f"- Tablet: {tablet_imp:,}".replace(",", "."))
    lines.append("")
    lines.append(
        "Το mobile αποτελεί πολύ σημαντικό μέρος της επισκεψιμότητας, "
        "με τις εμφανίσεις από κινητές συσκευές να είναι σε παρόμοια ή υψηλότερα επίπεδα από το desktop."
    )
    lines.append("")

    # Conclusion
    lines.append("**Συμπερασματικά**")
    lines.append("")
    lines.append(
        "Η πορεία του site δείχνει σταθερή ανοδική τάση ως προς το visibility και την επισκεψιμότητα. "
        "Η οργανική αναζήτηση λειτουργεί ως βασικός πυλώνας προσέλκυσης χρηστών, ενώ τα social media και το direct traffic "
        "συμπληρώνουν το σύνολο της κίνησης."
    )

    return "\n".join(lines)


def main():
    st.set_page_config(page_title="Career.uop.gr – Monthly Stats", layout="wide")
    st.title("📈 Monthly Reports – career.uop.gr")

    st.write(
        "Φόρτωσε το αρχείο `monthly_metrics.csv` (ή άφησέ το στον ίδιο φάκελο με την εφαρμογή) "
        "και διάλεξε μήνα για αυτόματη δημιουργία report και γραφημάτων."
    )

    uploaded = st.file_uploader(
        "Άνοιγμα Excel/CSV με μηνιαία δεδομένα (π.χ. monthly_metrics.csv – όχι webinars.csv)",
        type=["csv", "xlsx"],
        key="metrics_uploader",
    )

    required_cols = [
        "month",
        "new_users",
        "returning_users",
        "impressions",
        "clicks",
        "ctr",
        "avg_position",
        "avg_session_duration",
        "direct_sessions",
        "organic_search_sessions",
        "organic_social_sessions",
        "referral_sessions",
        "mobile_impressions",
        "desktop_impressions",
        "tablet_impressions",
    ]

    df = None
    if uploaded is not None:
        if uploaded.name.lower().endswith(".xlsx"):
            df = pd.read_excel(uploaded)
        else:
            df = pd.read_csv(uploaded)
        df = _normalize_columns(df)
        if "month" in df.columns:
            df["month"] = df["month"].astype(str).str.strip()
            df = df.sort_values("month")
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            looks_like_webinars = "webinar_title" in df.columns or "registered" in df.columns
            if looks_like_webinars:
                st.info(
                    "Φαίνεται ότι ανεβάσατε το αρχείο **webinars**. Για το report χρειάζεται το **monthly_metrics.csv**. Χρησιμοποιώ το αρχείο από το project."
                )
            else:
                st.warning(f"Λείπουν στήλες: {', '.join(missing)}. Δοκίμασε το monthly_metrics.csv.")
            df = None

    if df is None or df.empty:
        df = load_data(DATA_FILE)

    if df.empty:
        st.warning(
            "Δεν βρέθηκαν δεδομένα. Βάλε το `monthly_metrics.csv` στον ίδιο φάκελο με την εφαρμογή ή ανέβασέ το παραπάνω."
        )
        st.stop()

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Λείπουν στήλες από τα δεδομένα: {', '.join(missing)}. Χρειάζεται αρχείο τύπου monthly_metrics.csv.")
        st.stop()

    # Normalize month to YYYY-MM when possible; otherwise keep original (so we never lose rows)
    def safe_month(x):
        if pd.isna(x):
            return ""
        s = str(x).strip()
        if not s or s.lower() in ("nan", "nat"):
            return ""
        n = normalize_month(x)
        return n if n else s

    month_series = df["month"].apply(safe_month)
    df = df.assign(month=month_series)
    df = df[~df["month"].isin(["", "nan", "NaN"])]
    df = df.sort_values("month").reset_index(drop=True)
    metrics_df = df.copy()

    # Sidebar controls: unique months for dropdown
    months = sorted(metrics_df["month"].unique().tolist())
    if not months:
        st.error("Δεν βρέθηκαν γραμμές με μήνα. Έλεγξε ότι η στήλη 'month' έχει τιμές (π.χ. 2025-10, 2026-01).")
        st.stop()
    default_month = months[-1]
    month = st.sidebar.selectbox("Επίλεξε μήνα για report", options=months, index=len(months) - 1)

    st.sidebar.write("Τρέχων μήνας report:", month_label_gr(month))

    tab_metrics, tab_pages, tab_webinars = st.tabs(["📊 Monthly report", "📄 Απόδοση Περιεχομένου", "🎓 Webinars"])

    with tab_metrics:
        # Main layout
        col_text, col_charts = st.columns([1.2, 1.8])

        with col_text:
            summary_md = build_text_summary(metrics_df, month)
            st.markdown(summary_md)

        with col_charts:
            st.subheader("Χρονοσειρές βασικών metrics")

            # Line charts for impressions, clicks, ctr
            fig_imp = px.line(
                metrics_df,
                x="month",
                y="impressions",
                markers=True,
                title="Impressions ανά μήνα",
            )
            fig_imp.update_traces(mode="lines+markers")
            st.plotly_chart(fig_imp, use_container_width=True)

            fig_clk = px.line(
                metrics_df,
                x="month",
                y="clicks",
                markers=True,
                title="Clicks ανά μήνα",
            )
            fig_clk.update_traces(mode="lines+markers")
            st.plotly_chart(fig_clk, use_container_width=True)

            fig_ctr = px.line(
                metrics_df,
                x="month",
                y="ctr",
                markers=True,
                title="CTR (%) ανά μήνα",
            )
            fig_ctr.update_traces(mode="lines+markers")
            st.plotly_chart(fig_ctr, use_container_width=True)

            st.subheader("Κανάλια ανά μήνα (sessions)")
            channels_cols = [
                "direct_sessions",
                "organic_search_sessions",
                "organic_social_sessions",
                "referral_sessions",
            ]
            df_channels = metrics_df[["month"] + channels_cols].melt(
                id_vars="month", var_name="channel", value_name="sessions"
            )
            ch_map = {
                "direct_sessions": "Direct",
                "organic_search_sessions": "Organic Search",
                "organic_social_sessions": "Organic Social",
                "referral_sessions": "Referral",
            }
            df_channels = df_channels.assign(channel=df_channels["channel"].map(ch_map))
            fig_ch = px.bar(
                df_channels,
                x="month",
                y="sessions",
                color="channel",
                barmode="stack",
                title="Sessions ανά κανάλι και μήνα",
            )
            st.plotly_chart(fig_ch, use_container_width=True)

            st.subheader("Συσκευές – Impressions ανά μήνα")
            dev_cols = ["mobile_impressions", "desktop_impressions", "tablet_impressions"]
            df_dev = metrics_df[["month"] + dev_cols].melt(
                id_vars="month", var_name="device", value_name="impressions"
            )
            dev_map = {
                "mobile_impressions": "Mobile",
                "desktop_impressions": "Desktop",
                "tablet_impressions": "Tablet",
            }
            df_dev = df_dev.assign(device=df_dev["device"].map(dev_map))
            fig_dev = px.bar(
                df_dev,
                x="month",
                y="impressions",
                color="device",
                barmode="group",
                title="Impressions ανά συσκευή και μήνα",
            )
            st.plotly_chart(fig_dev, use_container_width=True)

    with tab_pages:
        st.subheader("Απόδοση Περιεχομένου – Top 5 Σελίδες")
        st.write("Οι πιο δημοφιλείς σελίδες του website βάσει προβολών και συμμετοχής επισκεπτών.")

        uploaded_pages = st.file_uploader(
            "Άνοιγμα CSV με page performance (προαιρετικό)",
            type=["csv"],
            key="pages_uploader",
        )

        if uploaded_pages is not None:
            pages_df = pd.read_csv(uploaded_pages)
        elif PAGES_FILE.exists():
            pages_df = pd.read_csv(PAGES_FILE, dtype={"month": str})
        else:
            pages_df = pd.DataFrame()

        if not pages_df.empty:
            pages_df = _normalize_columns(pages_df)
            if "month" in pages_df.columns:
                pages_df = pages_df.assign(month=pages_df["month"].astype(str).str.strip())
                month_pages = pages_df[pages_df["month"] == month]
            else:
                month_pages = pd.DataFrame()

            if not month_pages.empty:
                st.markdown(f"### {month_label_gr(month)}")

                display_cols = [c for c in ["page_title", "views", "sessions", "engagement_rate", "avg_engagement_duration"] if c in month_pages.columns]
                display_df = month_pages[display_cols].reset_index(drop=True)
                display_df.index = display_df.index + 1
                col_rename = {
                    "page_title": "Σελίδα",
                    "views": "Views",
                    "sessions": "Sessions",
                    "engagement_rate": "Engagement %",
                    "avg_engagement_duration": "Avg Duration",
                }
                display_df = display_df.rename(columns=col_rename)
                st.dataframe(display_df, use_container_width=True)

                if "views" in month_pages.columns:
                    fig_pages = px.bar(
                        month_pages,
                        x="page_title",
                        y="views",
                        title=f"Views ανά σελίδα – {month_label_gr(month)}",
                        labels={"page_title": "Σελίδα", "views": "Views"},
                        color="page_title",
                    )
                    fig_pages.update_layout(showlegend=False, xaxis_tickangle=-30)
                    st.plotly_chart(fig_pages, use_container_width=True)

                if "engagement_rate" in month_pages.columns:
                    fig_engage = px.bar(
                        month_pages,
                        x="page_title",
                        y="engagement_rate",
                        title=f"Engagement Rate (%) ανά σελίδα – {month_label_gr(month)}",
                        labels={"page_title": "Σελίδα", "engagement_rate": "Engagement %"},
                        color="page_title",
                    )
                    fig_engage.update_layout(showlegend=False, xaxis_tickangle=-30)
                    st.plotly_chart(fig_engage, use_container_width=True)

            else:
                st.info(f"Δεν υπάρχουν δεδομένα σελίδων για **{month_label_gr(month)}**.")

            all_months_pages = sorted(pages_df["month"].unique().tolist()) if "month" in pages_df.columns else []
            if len(all_months_pages) > 1:
                st.markdown("---")
                st.markdown("### Σύγκριση Views ανά μήνα (Top σελίδες)")
                top_titles = pages_df.groupby("page_title")["views"].sum().nlargest(5).index.tolist()
                compare_df = pages_df[pages_df["page_title"].isin(top_titles)]
                fig_compare = px.bar(
                    compare_df,
                    x="month",
                    y="views",
                    color="page_title",
                    barmode="group",
                    title="Views ανά μήνα – Top σελίδες",
                    labels={"page_title": "Σελίδα", "views": "Views", "month": "Μήνας"},
                )
                st.plotly_chart(fig_compare, use_container_width=True)
        else:
            st.info("Δεν βρέθηκαν δεδομένα. Βάλε αρχείο `page_performance.csv` στο project ή κάνε upload.")

    with tab_webinars:
        st.subheader("Webinars")
        uploaded_webinars = st.file_uploader(
            "Άνοιγμα CSV με webinars (προαιρετικό – διαφορετικό από το monthly_metrics.csv)",
            type=["csv"],
            key="webinars_uploader",
        )

        if uploaded_webinars is not None:
            webinars_df = pd.read_csv(uploaded_webinars)
        else:
            if WEBINARS_FILE.exists():
                webinars_df = pd.read_csv(WEBINARS_FILE)
            else:
                webinars_df = pd.DataFrame()

        if webinars_df.empty:
            st.info("Δεν βρέθηκαν δεδομένα webinars. Βάλε αρχείο `webinars.csv` στο project ή κάνε upload εδώ.")
        else:
            if "month" in webinars_df.columns:
                webinars_df = webinars_df.assign(month=webinars_df["month"].astype(str).str.strip())
                month_webinars = webinars_df[webinars_df["month"] == month]
            else:
                month_webinars = pd.DataFrame()

            if not month_webinars.empty:
                st.markdown(f"Webinars για **{month_label_gr(month)}**")
                mw_display = month_webinars.reset_index(drop=True)
                mw_display.index = mw_display.index + 1
                st.dataframe(mw_display, use_container_width=True)
            else:
                st.markdown(f"Δεν υπάρχουν webinars καταχωρημένα για **{month_label_gr(month)}**.")

            st.markdown("---")
            st.markdown("**Όλα τα webinars:**")
            all_display = webinars_df.reset_index(drop=True)
            all_display.index = all_display.index + 1
            st.dataframe(all_display, use_container_width=True)


if __name__ == "__main__":
    main()

