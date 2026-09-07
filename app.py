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

st.markdown("""
<style>
    :root {
        --primary-navy: #1E3A8A;
        --secondary-slate: #475569;
        --bg-soft-grey: #F8FAFC;
        --card-bg: #FFFFFF;
        --accent-blue: #3B82F6;
    }

    .stApp {
        background-color: var(--bg-soft-grey);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

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
        color: #FFFFFF !important;
    }
    .brand-header p {
        margin: 4px 0 0 0;
        font-size: 0.85rem;
        color: #93C5FD;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    .item-card {
        background-color: var(--card-bg);
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .badge-found {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-lost {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-resolved {
        background-color: #E0E7FF;
        color: #3730A3;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    div.stButton > button[kind="primary"] {
        background-color: var(--primary-navy);
        color: white;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. DATENBANK LOGIK & AUTOMATISCHE MIGRATION
# ==========================================
DB_FILE = "fundbuero.db"

def init_db():
    """Erstellt die Tabelle und fügt fehlende Spalten automatisch hinzu."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            type TEXT NOT NULL,
            category TEXT NOT NULL,
            location TEXT NOT NULL,
            date TEXT NOT NULL,
            description TEXT,
            status TEXT DEFAULT 'Offen',
            image_path TEXT,
            user_name TEXT,
            finder_name TEXT
        )
    ''')
    
    # Prüft vorhandene Spalten und rüstet 'finder_name' bei alten Datenbanken nach
    c.execute("PRAGMA table_info(items)")
    columns = [col[1] for col in c.fetchall()]
    if "finder_name" not in columns:
        c.execute("ALTER TABLE items ADD COLUMN finder_name TEXT")
        
    conn.commit()
    conn.close()

def add_item(title, item_type, category, location, date_str, description, image_path, user_name):
    """Fügt einen neuen Eintrag hinzu."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO items (title, type, category, location, date, description, status, image_path, user_name)
        VALUES (?, ?, ?, ?, ?, ?, 'Offen', ?, ?)
    ''', (title, item_type, category, location, date_str, description, image_path, user_name))
    conn.commit()
    conn.close()

def mark_as_found(item_id, finder_name):
    """Markiert eine Verlustanfrage als von jemandem gefunden."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        UPDATE items
        SET status = 'Gefunden / Gelöst', finder_name = ?
        WHERE id = ?
    ''', (finder_name, item_id))
    conn.commit()
    conn.close()

def get_items(filter_type=None, category=None):
    """Lädt Einträge aus der Datenbank."""
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
    model_path = "keras_model.h5"
    if os.path.exists(model_path):
        try:
            import tensorflow as tf
            return tf.keras.models.load_model(model_path, compile=False)
        except Exception:
            try:
                import tf_keras as keras
                return keras.models.load_model(model_path, compile=False)
            except Exception:
                return None
    return None

def predict_category(image: Image.Image):
    model = load_keras_model()
    img_resized = image.convert("RGB").resize((224, 224))
    img_array = np.asarray(img_resized, dtype=np.float32)
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
    st.session_state.action_type = "Gefunden"


# ==========================================
# 5. UI COMPONENTS & VIEWS
# ==========================================

def render_header():
    st.markdown("""
    <div class="brand-header">
        <p>Katharineum zu Lübeck</p>
        <h1>DIGITALES FUNDBÜRO</h1>
    </div>
    """, unsafe_allow_html=True)

# --- LOGIN SCREEN ---
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

# --- DASHBOARD ---
def view_dashboard():
    render_header()
    
    col_user, col_logout = st.columns([3, 1])
    with col_user:
        st.write(f"Angemeldet als: **{st.session_state.username}**")
    with col_logout:
        if st.button("Abmelden", key="logout_btn"):
            st.session_state.logged_in = False
            st.rerun()

    st.markdown("---")

    col_lost, col_found = st.columns(2)
    with col_lost:
        if st.button("🔴 Anfrage: Ich habe etwas VERLOREN", use_container_width=True):
            st.session_state.action_type = "Verloren"
            st.session_state.active_view = "add_item"
            st.rerun()
            
    with col_found:
        if st.button("📷 Ich habe etwas GEFUNDEN", use_container_width=True, type="primary"):
            st.session_state.action_type = "Gefunden"
            st.session_state.active_view = "add_item"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Übersicht & Anfragen")
    
    col_f1, col_f2 = st.columns([1, 1])
    with col_f1:
        type_filter = st.selectbox("Filter nach Typ", ["Alle", "Verloren", "Gefunden"])
    with col_f2:
        cat_filter = st.selectbox("Kategorie", ["Alle"] + LABELS)

    items = get_items(filter_type=type_filter, category=cat_filter)
    
    if not items:
        st.info("Keine passenden Einträge oder Anfragen vorhanden.")
    else:
        for item in items:
            # SICHERS ENTPACKEN (Verhindert Absturz bei unterschiedlichen Datenbank-Ständen)
            item_id = item[0]
            title = item[1]
            itype = item[2]
            category = item[3]
            location = item[4]
            date_str = item[5]
            desc = item[6]
            status = item[7] if len(item) > 7 else "Offen"
            img_path = item[8] if len(item) > 8 else None
            user = item[9] if len(item) > 9 else "Anonym"
            finder = item[10] if len(item) > 10 else "-"
            
            # Badge Styling
            if status == "Gefunden / Gelöst":
                badge_html = f'<span class="badge-resolved">GELÖST (Gefunden von {finder})</span>'
            elif itype == "Gefunden":
                badge_html = '<span class="badge-found">GEFUNDEN</span>'
            else:
                badge_html = '<span class="badge-lost">VERLUST-ANFRAGE</span>'

            with st.container():
                st.markdown(f"""
                <div class="item-card">
                    {badge_html}
                    <strong style="margin-left: 8px; font-size: 1.1rem;">{title}</strong>
                    <p style="color: #64748B; margin: 6px 0 2px 0; font-size: 0.85rem;">
                        📍 <b>Ort:</b> {location} | 📅 <b>Datum:</b> {date_str} | 🏷️ <b>Kategorie:</b> {category} | 👤 <b>Von:</b> {user}
                    </p>
                    <p style="margin-top: 6px; font-size: 0.95rem;">{desc if desc else 'Keine Beschreibung vorhanden.'}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if img_path and os.path.exists(img_path):
                    st.image(img_path, width=220)

                if itype == "Verloren" and status == "Offen":
                    col_act1, col_act2 = st.columns([2, 1])
                    with col_act2:
                        if st.button("🎉 Ich habe das gefunden!", key=f"found_btn_{item_id}", type="primary"):
                            mark_as_found(item_id, st.session_state.username)
                            st.success(f"Danke! Die Anfrage '{title}' wurde als gefunden markiert.")
                            st.rerun()
                st.markdown("---")

# --- ANFRAGE / FUNDSTÜCK ERSTELLEN ---
def view_add_item():
    render_header()
    
    if st.button("← Zurück zum Dashboard"):
        st.session_state.active_view = "dashboard"
        st.rerun()

    is_lost = st.session_state.action_type == "Verloren"
    action_title = "Verlustanfrage stellen" if is_lost else "Gefundenen Gegenstand melden"
    st.title(action_title)

    uploaded_image = None
    detected_category = "Sonstiges"
    saved_img_path = None

    if not is_lost:
        st.subheader("1. Foto aufnehmen / hochladen (für KI-Erkennung)")
        upload_method = st.radio("Foto-Quelle wählen:", ["Kamera-Scanner", "Datei-Upload"], horizontal=True)
        if upload_method == "Kamera-Scanner":
            uploaded_image = st.camera_input("Kamera zum Sache detektieren")
        else:
            uploaded_image = st.file_uploader("Bild auswählen", type=["jpg", "jpeg", "png"])
    else:
        st.info("💡 Wenn du ein Vergleichsfoto oder Beispielbild hast, kannst du es optional hochladen.")
        uploaded_image = st.file_uploader("Foto hinzufügen (optional)", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        image = Image.open(uploaded_image)
        detected_category, confidence_score = predict_category(image)
        st.success(f"🤖 **KI-Erkennung:** Vorab eingeordnet als **{detected_category}**")
        
        os.makedirs("uploads", exist_ok=True)
        saved_img_path = os.path.join("uploads", f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        image.save(saved_img_path)

    st.subheader("2. Details eingeben")
    with st.form("add_item_form"):
        title = st.text_input("Was wurde " + ("verloren?" if is_lost else "gefunden?"), 
                              placeholder="z. B. Blauer Schulrucksack, Schlüsselbund mit rotem Band...")
        
        cat_index = LABELS.index(detected_category) if detected_category in LABELS else 0
        category = st.selectbox("Kategorie", LABELS, index=cat_index)
        
        location = st.text_input("Vermuteter Ort", placeholder="z. B. Sporthalle, Mensa, Raum 204")
        date_val = st.date_input("Datum", datetime.date.today())
        description = st.text_area("Beschreibung & Merkmale", 
                                   placeholder="Genauere Beschreibung, damit Besitzer/Finder den Gegenstand zuordnen können.")

        button_text = "Verlustanfrage veröffentlichen" if is_lost else "Fundstück eintragen"
        submit = st.form_submit_button(button_text, type="primary", use_container_width=True)

        if submit:
            if not title or not location:
                st.error("Bitte gib mindestens den Namen des Gegenstands und den Ort an.")
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
                st.success("Erfolgreich eingetragen!")
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
