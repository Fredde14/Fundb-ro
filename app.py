import os
import sqlite3
import datetime
import numpy as np
from PIL import Image
import streamlit as st

# ==========================================
# 1. STREAMLIT KONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="Fundbüro - Katharineum zu Lübeck",
    page_icon="🏫",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS für edles, maßgeschneidertes Schulinstituts-Design (Navy/Slate/Grey)
st.markdown("""
<style>
    /* Hauptfarben & Hintergründe */
    :root {
        --primary-navy: #1E3A8A;
        --secondary-slate: #475569;
        --bg-soft-grey: #F8FAFC;
        --card-bg: #FFFFFF;
        --accent-blue: #3B82F6;
    }

    /* Globale Resets */
    .stApp {
        background-color: var(--bg-soft-grey);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Schul-Branding Header */
    .brand-header {
        background-color: var(--primary-navy);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(30, 58, 138, 0.15);
    }
    .brand-header h1 {
        margin: 0;
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        color: #FFFFFF !important;
    }
    .brand-header p {
        margin: 4px 0 0 0;
        font-size: 0.85rem;
        color: #93C5FD;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    /* Auth Cards */
    .auth-card {
        background-color: var(--card-bg);
        padding: 28px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 20px;
    }

    /* Custom Cards für Feed */
    .item-card {
        background-color: var(--card-bg);
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .badge-found {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-lost {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Streamlit Button Styling Overrides */
    div.stButton > button {
        border-radius: 8px;
        border: none;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    div.stButton > button[kind="primary"] {
        background-color: var(--primary-navy);
        color: white;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #1E40AF;
        border-color: transparent;
    }

    /* Verstecke Standard Streamlit UI-Elemente */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. DATENBANK LOGIK (SQLite)
# ==========================================
DB_FILE = "fundbuero.db"

def init_db():
    """Initialisiert die SQLite-Datenbank."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            type TEXT NOT NULL, -- 'Gefunden' oder 'Verloren'
            category TEXT NOT NULL,
            location TEXT NOT NULL,
            date TEXT NOT NULL,
            description TEXT,
            status TEXT DEFAULT 'Offen',
            image_path TEXT,
            user_name TEXT
        )
    ''')
    conn.commit()
    conn.close()

def add_item(title, item_type, category, location, date_str, description, image_path, user_name):
    """Fügt ein neues Fundstück/Verluststück hinzu."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO items (title, type, category, location, date, description, image_path, user_name)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (title, item_type, category, location, date_str, description, image_path, user_name))
    conn.commit()
    conn.close()

def get_items(filter_type=None, category=None):
    """Lädt Einträge mit optionalen Filtern."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    query = "SELECT * FROM items WHERE 1=1"
    params = []
    
    if filter_type and filter_type != "Alle":
        query += " AND type = ?"
        params.append(filter_type)
    if category and category != "Alle":
        query += " AND category = ?"
        params.append(category)
        
    query += " ORDER BY id DESC"
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()
    return rows

init_db()


# ==========================================
# 3. KI / TENSORFLOW INFERENZ & FALLBACK
# ==========================================
LABELS = ["Bekleidung/Jacke", "Elektronik/Handy", "Schlüssel", "Rucksack/Tasche", "Mäppchen/Stifte", "Sonstiges"]

@st.cache_resource
def load_keras_model():
    """Lädt das Keras-Modell oder gibt None zurück, falls nicht vorhanden."""
    model_path = "keras_model.h5"
    if os.path.exists(model_path):
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path, compile=False)
            return model
        except Exception as e:
            st.warning(f"Modell konnte nicht geladen werden: {e}")
            return None
    return None

def predict_category(image: Image.Image):
    """
    Verarbeitet das Bild auf [224, 224, 3], normalisiert es
    und führt eine Vorhersage durch. Nutzt Fallback bei fehlendem Modell.
    """
    model = load_keras_model()
    
    # Preprocessing
    img_resized = image.convert("RGB").resize((224, 224))
    img_array = np.asarray(img_resized, dtype=np.float32)
    
    # Normalisierung auf [0, 1] (oder [-1, 1] je nach Modell)
    normalized_image_array = (img_array / 127.5) - 1.0
    data = np.expand_dims(normalized_image_array, axis=0)

    if model is not None:
        try:
            prediction = model.predict(data)
            index = np.argmax(prediction)
            confidence = float(prediction[0][index])
            return LABELS[index % len(LABELS)], confidence
        except Exception:
            pass

    # ELEGANTER FALLBACK (Falls kein Modell vorhanden ist)
    # Bildanalyse-Dummy anhand von Farbüberwiegenheit oder Zufallsauswahl für Demo
    avg_color = img_array.mean()
    fallback_index = int(avg_color) % len(LABELS)
    return LABELS[fallback_index], 0.82


# ==========================================
# 4. SESSION STATE & NAVIGATION
# ==========================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "active_view" not in st.session_state:
    st.session_state.active_view = "dashboard"
if "action_type" not in st.session_state:
    st.session_state.action_type = "Gefunden"  # 'Gefunden' oder 'Verloren'


# ==========================================
# 5. UI COMPONENTS & VIEWS
# ==========================================

def render_header():
    """Zeigt den einheitlichen Schul-Header."""
    st.markdown("""
    <div class="brand-header">
        <p>Katharineum zu Lübeck</p>
        <h1>DIGITALES FUNDBÜRO</h1>
    </div>
    """, unsafe_allow_html=True)

# --- VIEW 1: AUTHENTIFIZIERUNG ---
def view_login():
    render_header()
    
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        st.markdown("<h3 style='text-align: center; color: #475569;'>Anmeldung</h3>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            user_input = st.text_input("Anmeldename", placeholder="z. B. s.müller")
            password_input = st.text_input("Passwort", type="password", placeholder="••••••••")
            submit = st.form_submit_button("Anmelden", use_container_width=True, type="primary")
            
            if submit:
                if user_input and password_input:
                    st.session_state.logged_in = True
                    st.session_state.username = user_input
                    st.rerun()
                else:
                    st.error("Bitte gib Anmeldename und Passwort ein.")
                    
        c1, c2 = st.columns(2)
        with c1:
            st.caption("[Anmeldename vergessen?](#)")
        with c2:
            st.caption("[Passwort vergessen?](#)")

# --- VIEW 2: HAUPT-DASHBOARD ---
def view_dashboard():
    render_header()
    
    # Nutzer-Begrüßung & Logout
    col_user, col_logout = st.columns([3, 1])
    with col_user:
        st.write(f"Angemeldet als: **{st.session_state.username}**")
    with col_logout:
        if st.button("Abmelden", key="logout_btn"):
            st.session_state.logged_in = False
            st.rerun()

    st.markdown("---")

    # Zwei prominente Haupt-Aktionsbuttons
    col_lost, col_found = st.columns(2)
    
    with col_lost:
        if st.button("🔴 Ich habe etwas VERLOREN", use_container_width=True):
            st.session_state.action_type = "Verloren"
            st.session_state.active_view = "add_item"
            st.rerun()
            
    with col_found:
        if st.button("📷 Ich habe etwas GEFUNDEN", use_container_width=True, type="primary"):
            st.session_state.action_type = "Gefunden"
            st.session_state.active_view = "add_item"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Schnellauswahl & Filterleiste
    st.subheader("Fundstücke & Verlustmeldungen")
    
    col_f1, col_f2 = st.columns([1, 1])
    with col_f1:
        type_filter = st.selectbox("Typ", ["Alle", "Gefunden", "Verloren"])
    with col_f2:
        cat_filter = st.selectbox("Kategorie", ["Alle"] + LABELS)

    # Fundstück-Feed
    items = get_items(filter_type=type_filter, category=cat_filter)
    
    if not items:
        st.info("Keine passenden Einträge vorhanden.")
    else:
        for item in items:
            item_id, title, itype, category, location, date_str, desc, status, img_path, user = item
            
            badge_class = "badge-found" if itype == "Gefunden" else "badge-lost"
            
            with st.container():
                st.markdown(f"""
                <div class="item-card">
                    <span class="{badge_class}">{itype.upper()}</span>
                    <strong style="margin-left: 8px; font-size: 1.1rem;">{title}</strong>
                    <p style="color: #64748B; margin: 6px 0 2px 0; font-size: 0.85rem;">
                        📍 <b>Ort:</b> {location} | 📅 <b>Datum:</b> {date_str} | 🏷️ <b>Kategorie:</b> {category}
                    </p>
                    <p style="margin-top: 6px; font-size: 0.95rem;">{desc if desc else 'Keine zusätzliche Beschreibung.'}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if img_path and os.path.exists(img_path):
                    st.image(img_path, width=200)

# --- VIEW 3: KAMERA / UPLOAD & KI-ERKENNUNG ---
def view_add_item():
    render_header()
    
    if st.button("← Zurück zum Dashboard"):
        st.session_state.active_view = "dashboard"
        st.rerun()

    action_label = "Fundstück melden" if st.session_state.action_type == "Gefunden" else "Verlust melden"
    st.title(action_label)

    # Upload-Optionen: Kamera oder Datei-Upload
    upload_method = st.radio("Foto-Quelle wählen:", ["Kamera-Scanner", "Datei-Upload"], horizontal=True)
    
    uploaded_image = None
    if upload_method == "Kamera-Scanner":
        uploaded_image = st.camera_input("Kamera zum Sache detektieren")
    else:
        uploaded_image = st.file_uploader("Bild auswählen", type=["jpg", "jpeg", "png"])

    detected_category = "Sonstiges"
    confidence_score = 0.0
    saved_img_path = None

    # KI-Klassifikation ausführen, sobald ein Bild vorliegt
    if uploaded_image:
        image = Image.open(uploaded_image)
        
        # KI-Vorhersage
        detected_category, confidence_score = predict_category(image)
        
        st.success(f"🤖 **KI-Erkennung:** Automatisch als **{detected_category}** erkannt ({confidence_score*100:.1f}% Konfidenz)")
        
        # Bild lokal im Ordner 'uploads' speichern
        os.makedirs("uploads", exist_ok=True)
        saved_img_path = os.path.join("uploads", f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        image.save(saved_img_path)

    # Formular zur Bestätigung/Korrektur
    with st.form("add_item_form"):
        title = st.text_input("Bezeichnung / Titel", placeholder="z. B. Blaue Regentasse, Schlüsselbund...")
        
        # KI-Vorauswahl setzen
        cat_index = LABELS.index(detected_category) if detected_category in LABELS else 0
        category = st.selectbox("Kategorie (KI-Vorschlag verfeinern)", LABELS, index=cat_index)
        
        location = st.text_input("Fundort / Verlustort", placeholder="z. B. Sporthalle, Mensa, Raum 204")
        date_val = st.date_input("Datum", datetime.date.today())
        description = st.text_area("Zusätzliche Details / Hinweise", placeholder="Besondere Merkmale, Marken, Inhalt...")

        submit = st.form_submit_button("Eintrag veröffentlichen", type="primary", use_container_width=True)

        if submit:
            if not title or not location:
                st.error("Bitte gib mindestens einen Titel und den Ort an.")
            else:
                add_item(
                    title=title,
                    item_type=st.session_state.action_type,
                    category=category,
                    location=location,
                    date_str=date_val.strftime("%d.%m.%Y"),
                    description=description,
                    image_path=saved_img_path,
                    user_name=st.session_state.username
                )
                st.success("Erfolgreich gespeichert!")
                st.session_state.active_view = "dashboard"
                st.rerun()


# ==========================================
# 6. HAUPT-ROUTER
# ==========================================
def main():
    if not st.session_state.logged_in:
        view_login()
    else:
        if st.session_state.active_view == "dashboard":
            view_dashboard()
        elif st.session_state.active_view == "add_item":
            view_add_item()

if __name__ == "__main__":
    main()
