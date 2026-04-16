"""
BuildScan — AI-Powered Building Assessment
Streamlit app with project management, floor plans, condition ratings,
confidence indicators, priority tagging, before/after comparison,
photo thumbnails, manual fixture editing, and full export.
"""

import streamlit as st
import anthropic
import base64
import json
import pandas as pd
from datetime import datetime
import io
import os
from PIL import Image

# PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.utils import ImageReader


# =============================================================================
# CONFIG
# =============================================================================

FIXTURE_TYPES = [
    'light_fixture','outlet','switch','smoke_detector','sink','toilet','door','window',
    'vent','thermostat','fire_extinguisher','sprinkler_head','cabinet','countertop',
    'bathtub','staircase','garage_door','electrical_panel','water_heater','hvac_unit','sump_pump'
]
ROOM_TYPES = [
    'Kitchen','Living Room','Bedroom','Bathroom','Basement','Garage','Office',
    'Dining Room','Hallway','Stairwell','Mechanical Room','Utility Room','Attic',
    'Exterior','Roof','Crawl Space','Storage','Conference Room','Break Room','Other'
]
MAX_SNAPSHOTS = 5

def get_api_key():
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass
    env_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if env_key:
        return env_key
    return st.session_state.get("user_api_key", "")

def fmt_type(s):
    return (s or '').replace('_', ' ').title()

def uid():
    import random, time
    return hex(int(time.time()))[2:] + hex(random.randint(0, 0xFFFF))[2:]


# =============================================================================
# SESSION STATE INIT
# =============================================================================

def init_state():
    if 'projects' not in st.session_state:
        st.session_state.projects = {}
    if 'active_project' not in st.session_state:
        st.session_state.active_project = None
    if 'user_api_key' not in st.session_state:
        st.session_state.user_api_key = ""

init_state()


# =============================================================================
# DATA HELPERS
# =============================================================================

def get_project():
    pid = st.session_state.active_project
    return st.session_state.projects.get(pid) if pid else None

def get_building(proj=None):
    proj = proj or get_project()
    if not proj:
        return None
    bids = list(proj.get('buildings', {}).keys())
    return proj['buildings'][bids[0]] if bids else None

def get_rooms(building=None):
    building = building or get_building()
    if not building:
        return {}
    return building.get('rooms', {})

def get_room(room_id):
    rooms = get_rooms()
    return rooms.get(room_id)

def latest_snapshot(room):
    snaps = room.get('snapshots', [])
    return snaps[-1] if snaps else None

def is_assessed(room):
    snap = latest_snapshot(room)
    return bool(snap and snap.get('fixtures'))

def ensure_snapshot(room):
    if not room.get('snapshots'):
        room['snapshots'] = [{'takenAt': datetime.now().isoformat(), 'photos': [], 'fixtures': []}]
    return room['snapshots'][-1]

def auto_room_id(building, floor_id=None):
    floors = building.get('floors', [])
    floor = next((f for f in floors if f['id'] == floor_id), None) if floor_id else None
    floor_label = floor['label'] if floor else '0'
    rooms = building.get('rooms', {})
    existing = [r for r in rooms.values() if r.get('floorId') == floor_id]
    seq = len(existing) + 1
    return f"{floor_label}-{seq:02d}"


# =============================================================================
# IMAGE HELPERS
# =============================================================================

def encode_image(uploaded_file):
    return base64.standard_b64encode(uploaded_file.getvalue()).decode("utf-8")

def make_thumbnail(uploaded_file, max_width=200):
    uploaded_file.seek(0)
    img = Image.open(uploaded_file)
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    ratio = max_width / img.width
    new_size = (max_width, int(img.height * ratio))
    img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=50)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def compress_for_api(uploaded_file, max_width=1600, max_bytes=800*1024):
    uploaded_file.seek(0)
    img = Image.open(uploaded_file)
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    if img.width > max_width or img.height > max_width:
        ratio = min(max_width / img.width, max_width / img.height)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)
    quality = 70
    while quality > 10:
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=quality)
        if buf.tell() <= max_bytes:
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        quality -= 10
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=10)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# =============================================================================
# CLAUDE AI ANALYSIS
# =============================================================================

def analyze_room(client, images_base64, room_label):
    num = len(images_base64)
    ctx = f"This is a photo of the {room_label}." if num == 1 else f"These are {num} photos of the SAME room: the {room_label}, taken from different angles."
    dedup = "This is a single view — count everything visible." if num == 1 else "Cross-reference all photos to build ONE unified inventory. Do NOT double-count items visible in multiple photos."

    prompt = f"""{ctx}

You are a construction building assessment specialist. Analyze {"this photo" if num == 1 else "ALL photos together"} to create a comprehensive inventory.

IMPORTANT: {dedup}

SURFACES (rate condition good/fair/poor):
- ceiling_type, wall_type, flooring_type

FIXTURES (count each, rate condition good/fair/poor, rate confidence high/medium/low):
light_fixture, outlet, switch, smoke_detector, sink, toilet, door, window, vent, thermostat,
fire_extinguisher, sprinkler_head, cabinet, countertop, bathtub, staircase, electrical_panel,
water_heater, hvac_unit

CONDITION: good = no issues, fair = minor wear, poor = damaged/needs repair
CONFIDENCE: high = clearly visible, medium = partially visible/likely, low = uncertain

MATERIALS: List visible building materials (copper pipe, PVC, etc.)
EQUIPMENT: Note HVAC, water heater, electrical panel with manufacturer/model if readable

Respond ONLY with valid JSON:
{{
  "room_type": "string",
  "ceiling_type": "string", "ceiling_condition": "good|fair|poor",
  "wall_type": "string", "wall_condition": "good|fair|poor",
  "flooring_type": "string", "flooring_condition": "good|fair|poor",
  "fixtures": [{{"type":"string","count":1,"subtype":"string","description":"string","condition":"good|fair|poor","confidence":"high|medium|low","condition_notes":"string or null"}}],
  "materials_noted": ["string"],
  "equipment": [{{"type":"string","manufacturer":"string or null","model":"string or null","notes":"string"}}],
  "general_notes": "2-3 sentence summary",
  "total_fixture_count": 0
}}"""

    content = []
    for img in images_base64:
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img}})
    content.append({"type": "text", "text": prompt})

    response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": content}]
    )

    text = response.content[0].text
    try:
        start = text.index('{')
        end = text.rindex('}') + 1
        result = json.loads(text[start:end])
        for f in result.get('fixtures', []):
            f['id'] = uid()
            f['source'] = 'ai'
            f['override'] = False
            f['priority'] = None
            f.setdefault('quantity', f.get('count', 1))
        return result
    except:
        return {"fixtures": [], "error": "Failed to parse response"}


# =============================================================================
# DIFF ENGINE
# =============================================================================

def diff_snapshots(older, newer):
    key = lambda f: f"{f.get('type','')}|{f.get('subtype','')}"
    old_map = {key(f): f for f in (older.get('fixtures') or [])}
    new_map = {key(f): f for f in (newer.get('fixtures') or [])}
    added = [f for k, f in new_map.items() if k not in old_map]
    removed = [f for k, f in old_map.items() if k not in new_map]
    changed = []
    for k, f in new_map.items():
        if k in old_map:
            o = old_map[k]
            if o.get('quantity') != f.get('quantity') or o.get('condition') != f.get('condition'):
                changed.append({'old': o, 'new': f})
    return {'added': added, 'removed': removed, 'changed': changed}


# =============================================================================
# EXPORT HELPERS
# =============================================================================

def create_csv(rooms_dict):
    rows = []
    for r in rooms_dict.values():
        snap = latest_snapshot(r)
        for f in (snap.get('fixtures') or []):
            rows.append({
                'Room': r.get('label', r.get('id', '')),
                'Room ID': r.get('id', ''),
                'Fixture': fmt_type(f.get('type', '')),
                'Quantity': f.get('quantity', f.get('count', 1)),
                'Subtype': f.get('subtype', ''),
                'Condition': f.get('condition', ''),
                'Confidence': f.get('confidence', ''),
                'Priority': f.get('priority', ''),
                'Source': f.get('source', ''),
            })
    return pd.DataFrame(rows).to_csv(index=False) if rows else ""

def create_excel(rooms_dict):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        summary_rows = []
        fixture_rows = []
        equip_rows = []
        for r in rooms_dict.values():
            snap = latest_snapshot(r)
            if not snap:
                continue
            label = r.get('label', r.get('id', ''))
            total = sum(f.get('quantity', f.get('count', 1)) for f in snap.get('fixtures', []))
            summary_rows.append({
                'Room': label, 'Room ID': r.get('id', ''), 'Priority': r.get('priority', ''),
                'Ceiling': f"{snap.get('ceiling_type','N/A')} ({snap.get('ceiling_condition','N/A')})",
                'Walls': f"{snap.get('wall_type','N/A')} ({snap.get('wall_condition','N/A')})",
                'Flooring': f"{snap.get('flooring_type','N/A')} ({snap.get('flooring_condition','N/A')})",
                'Total Fixtures': total,
                'Materials': ', '.join(snap.get('materials_noted', [])),
                'Notes': snap.get('general_notes', ''),
            })
            for f in snap.get('fixtures', []):
                fixture_rows.append({
                    'Room': label, 'Fixture': fmt_type(f.get('type', '')),
                    'Qty': f.get('quantity', f.get('count', 1)), 'Subtype': f.get('subtype', ''),
                    'Condition': f.get('condition', ''), 'Confidence': f.get('confidence', ''),
                    'Priority': f.get('priority', ''), 'Source': f.get('source', ''),
                    'Description': f.get('description', ''),
                })
            for e in snap.get('equipment', []):
                equip_rows.append({
                    'Room': label, 'Type': e.get('type', ''),
                    'Manufacturer': e.get('manufacturer', ''), 'Model': e.get('model', ''),
                    'Notes': e.get('notes', ''),
                })
        if summary_rows:
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name='Room Summary', index=False)
        if fixture_rows:
            pd.DataFrame(fixture_rows).to_excel(writer, sheet_name='Fixtures', index=False)
        if equip_rows:
            pd.DataFrame(equip_rows).to_excel(writer, sheet_name='Equipment', index=False)
    return output.getvalue()

def create_pdf(rooms_dict, room_photos):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=60, leftMargin=60, topMargin=60, bottomMargin=60)
    elements = []
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('T', parent=styles['Heading1'], fontSize=22, textColor=colors.HexColor('#1E3A5F'), spaceAfter=20, alignment=TA_CENTER)
    h_style = ParagraphStyle('H', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor('#1E3A5F'), spaceAfter=10, spaceBefore=10)
    sub_style = ParagraphStyle('S', parent=styles['Heading3'], fontSize=11, textColor=colors.HexColor('#333'), spaceAfter=6)

    proj = get_project()
    elements.append(Spacer(1, 1 * inch))
    elements.append(Paragraph("BuildScan Assessment Report", title_style))
    if proj:
        elements.append(Paragraph(proj.get('name', ''), ParagraphStyle('Sub', parent=styles['Heading2'], alignment=TA_CENTER, textColor=colors.HexColor('#666'))))
    elements.append(Spacer(1, 0.3 * inch))
    total_rooms = len(rooms_dict)
    total_fix = sum(sum(f.get('quantity', f.get('count', 1)) for f in (latest_snapshot(r) or {}).get('fixtures', [])) for r in rooms_dict.values())
    summary_data = [['Date:', datetime.now().strftime('%B %d, %Y')], ['Rooms:', str(total_rooms)], ['Fixtures:', str(total_fix)]]
    t = Table(summary_data, colWidths=[1.5*inch, 3.5*inch])
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(0,-1),colors.HexColor('#f0f0f0')),('FONTNAME',(0,0),(0,-1),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),10),('GRID',(0,0),(-1,-1),0.5,colors.grey),('BOTTOMPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6)]))
    elements.append(t)
    elements.append(PageBreak())

    for room in rooms_dict.values():
        snap = latest_snapshot(room)
        if not snap:
            continue
        label = room.get('label', room.get('id', ''))
        elements.append(Paragraph(f"{label} ({room.get('id','')})", h_style))

        photos = room_photos.get(room.get('id', ''), [])
        for photo_b64 in photos[:3]:
            try:
                img_bytes = base64.b64decode(photo_b64)
                ir = ImageReader(io.BytesIO(img_bytes))
                iw, ih = ir.getSize()
                dw = 3 * inch
                dh = dw * (ih / iw)
                from reportlab.platypus import Image as RLImage
                elements.append(RLImage(io.BytesIO(img_bytes), width=dw, height=dh))
                elements.append(Spacer(1, 0.1 * inch))
            except:
                pass

        # Surfaces
        elements.append(Paragraph("Surfaces", sub_style))
        sd = [['Surface', 'Type', 'Condition']]
        for sl, sk, ck in [('Ceiling','ceiling_type','ceiling_condition'),('Walls','wall_type','wall_condition'),('Flooring','flooring_type','flooring_condition')]:
            sd.append([sl, snap.get(sk, 'N/A'), (snap.get(ck, '') or '').title()])
        st2 = Table(sd, colWidths=[1.2*inch, 2.5*inch, 1.5*inch])
        st2.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#4F46E5')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),9),('GRID',(0,0),(-1,-1),0.5,colors.grey),('BOTTOMPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5)]))
        elements.append(st2)
        elements.append(Spacer(1, 0.1 * inch))

        # Fixtures
        fixtures = snap.get('fixtures', [])
        if fixtures:
            elements.append(Paragraph("Fixtures", sub_style))
            fd = [['Type', 'Qty', 'Subtype', 'Condition', 'Confidence']]
            for f in sorted(fixtures, key=lambda x: -(x.get('quantity', x.get('count', 1)))):
                fd.append([fmt_type(f.get('type', '')), str(f.get('quantity', f.get('count', 1))), f.get('subtype', ''), (f.get('condition', '') or '').title(), (f.get('confidence', '') or '').title()])
            ft = Table(fd, colWidths=[1.8*inch, 0.5*inch, 1.3*inch, 0.8*inch, 0.8*inch])
            ft.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#4F46E5')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8),('GRID',(0,0),(-1,-1),0.5,colors.grey),('BOTTOMPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),4)]))
            elements.append(ft)

        notes = snap.get('general_notes', '')
        if notes:
            elements.append(Spacer(1, 0.1*inch))
            elements.append(Paragraph(f"<b>Notes:</b> {notes}", styles['Normal']))
        elements.append(PageBreak())

    elements.append(Paragraph("Generated by BuildScan — AI-assisted visual analysis.", ParagraphStyle('F', parent=styles['Normal'], fontSize=8, textColor=colors.grey, alignment=TA_CENTER)))
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(page_title="BuildScan", page_icon="🏗️", layout="wide")

st.markdown("""
<style>
    .cond-good { background: #dcfce7; color: #166534; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; }
    .cond-fair { background: #fef9c3; color: #854d0e; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; }
    .cond-poor { background: #fecaca; color: #991b1b; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; }
    .conf-high { background: #dcfce7; color: #166534; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; }
    .conf-medium { background: #f1f5f9; color: #475569; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; }
    .conf-low { background: #fef9c3; color: #854d0e; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; }
    .surface-tag { background: #e0e7ff; color: #3730a3; padding: 3px 10px; border-radius: 6px; font-size: 0.85rem; margin: 2px; display: inline-block; }
    .material-tag { background: #fef3c7; color: #92400e; padding: 3px 10px; border-radius: 6px; font-size: 0.85rem; margin: 2px; display: inline-block; }
    .priority-urgent { border-left: 4px solid #dc2626 !important; }
    .priority-watch { border-left: 4px solid #d97706 !important; }
</style>
""", unsafe_allow_html=True)

def cond_badge(c):
    c = (c or '').lower()
    if c == 'good': return '<span class="cond-good">Good</span>'
    if c == 'fair': return '<span class="cond-fair">Fair</span>'
    if c == 'poor': return '<span class="cond-poor">Poor</span>'
    return ''

def conf_badge(c):
    c = (c or '').lower()
    if c == 'high': return '<span class="conf-high">✓ High</span>'
    if c == 'medium': return '<span class="conf-medium">~ Medium</span>'
    return '<span class="conf-low">? Low</span>'

def priority_badge(p):
    if p == 'urgent': return '🔴 Urgent'
    if p == 'watch': return '🟡 Watch'
    if p == 'ok': return '🟢 OK'
    return ''


# =============================================================================
# HEADER
# =============================================================================

st.markdown('<h1 style="margin-bottom:0">🏗️ BuildScan</h1>', unsafe_allow_html=True)
st.caption("AI-Powered Building Assessment for Construction Companies")

api_key = get_api_key()


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("Projects")

    # Project selector
    projects = st.session_state.projects
    project_names = {pid: p['name'] for pid, p in projects.items()}

    if project_names:
        selected = st.selectbox(
            "Active Project",
            options=list(project_names.keys()),
            format_func=lambda x: project_names[x],
            index=list(project_names.keys()).index(st.session_state.active_project) if st.session_state.active_project in project_names else 0,
            key="project_selector"
        )
        st.session_state.active_project = selected

    # New project
    with st.expander("New Project", expanded=not bool(projects)):
        new_name = st.text_input("Project Name", placeholder="e.g., 123 Main St Renovation", key="new_proj_name")
        if st.button("Create Project", type="primary", disabled=not new_name):
            pid = uid()
            st.session_state.projects[pid] = {'id': pid, 'name': new_name, 'createdAt': datetime.now().isoformat(), 'buildings': {}}
            st.session_state.active_project = pid
            st.rerun()

    st.divider()

    # API key
    if api_key:
        st.success("API Connected")
    else:
        st.warning("No API Key")
        key_input = st.text_input("Anthropic API Key", type="password", key="api_key_input")
        if key_input:
            st.session_state.user_api_key = key_input
            st.rerun()

    # Stats
    rooms = get_rooms()
    if rooms:
        assessed = sum(1 for r in rooms.values() if is_assessed(r))
        total_fix = sum(
            sum(f.get('quantity', f.get('count', 1)) for f in (latest_snapshot(r) or {}).get('fixtures', []))
            for r in rooms.values()
        )
        st.divider()
        st.metric("Rooms", f"{assessed}/{len(rooms)} assessed")
        st.metric("Total Fixtures", total_fix)

    st.divider()

    # Backup
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export Backup", use_container_width=True):
            data = json.dumps({'projects': st.session_state.projects, 'active_project': st.session_state.active_project}, indent=2, default=str)
            st.download_button("Download", data, f"BuildScan_Backup_{datetime.now().strftime('%Y%m%d')}.json", "application/json", use_container_width=True)
    with col2:
        uploaded_backup = st.file_uploader("Import", type=['json'], key="backup_upload", label_visibility="collapsed")
        if uploaded_backup:
            try:
                data = json.loads(uploaded_backup.read())
                st.session_state.projects = data.get('projects', {})
                st.session_state.active_project = data.get('active_project')
                st.rerun()
            except:
                st.error("Invalid backup file")


# =============================================================================
# MAIN TABS
# =============================================================================

proj = get_project()
if not proj:
    st.info("Create a project in the sidebar to get started.")
    st.stop()

# Ensure building exists
building = get_building()
if not building:
    st.subheader(f"Project: {proj['name']}")
    bname = st.text_input("Building Name", value="Main Building", key="new_bldg_name")
    if st.button("Add Building", type="primary"):
        bid = uid()
        proj['buildings'][bid] = {'id': bid, 'name': bname, 'floors': [], 'rooms': {}}
        st.rerun()
    st.stop()

tab_rooms, tab_inventory, tab_report, tab_export = st.tabs(["🏢 Rooms", "📋 Inventory", "📊 Report", "📥 Export"])

rooms = get_rooms()
room_photos = {}  # Build photo map for export
for rid, r in rooms.items():
    snap = latest_snapshot(r)
    if snap and snap.get('photos'):
        room_photos[rid] = [p['thumbUri'] for p in snap['photos']]


# =============================================================================
# TAB: ROOMS
# =============================================================================

with tab_rooms:
    st.subheader(f"{building['name']} — Rooms")

    # Floor management
    floors = building.get('floors', [])
    assessed_count = sum(1 for r in rooms.values() if is_assessed(r))

    if rooms:
        st.progress(assessed_count / len(rooms) if rooms else 0, text=f"{assessed_count}/{len(rooms)} rooms assessed")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        with st.popover("+ Add Floor"):
            fl = st.text_input("Floor Label", placeholder="e.g., 1, 2, B1", key="add_floor_label")
            fsecs = st.text_input("Sections (optional, comma-separated)", placeholder="e.g., A, B, North", key="add_floor_secs")
            if st.button("Add Floor", key="add_floor_btn"):
                secs = [{'id': uid(), 'label': s.strip()} for s in fsecs.split(',') if s.strip()] if fsecs else []
                building.setdefault('floors', []).append({'id': uid(), 'label': fl, 'sections': secs})
                st.rerun()
    with col_b:
        with st.popover("+ Add Room"):
            room_type = st.selectbox("Type", ROOM_TYPES, key="add_room_type")
            room_label = st.text_input("Label (optional)", placeholder="e.g., Master Bath", key="add_room_label")
            floor_opts = {f['id']: f"Floor {f['label']}" for f in floors}
            floor_sel = st.selectbox("Floor", options=list(floor_opts.keys()), format_func=lambda x: floor_opts[x], key="add_room_floor") if floor_opts else None
            if st.button("Add Room", key="add_room_btn"):
                rid = auto_room_id(building, floor_sel)
                label = room_label.strip() if room_label.strip() else room_type
                building.setdefault('rooms', {})[rid] = {'id': rid, 'label': label, 'floorId': floor_sel, 'sectionId': None, 'priority': None, 'snapshots': []}
                st.rerun()
    with col_c:
        with st.popover("+ Bulk Add"):
            bulk_text = st.text_area("One room per line", placeholder="Room 101\nConference A\nBreak Room", key="bulk_rooms")
            bulk_floor = st.selectbox("Floor", options=[None] + [f['id'] for f in floors], format_func=lambda x: f"Floor {next((f['label'] for f in floors if f['id']==x), 'None')}" if x else "No floor", key="bulk_floor")
            if st.button("Add All", key="bulk_btn") and bulk_text:
                for line in bulk_text.strip().split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    rid = auto_room_id(building, bulk_floor)
                    building.setdefault('rooms', {})[rid] = {'id': rid, 'label': line, 'floorId': bulk_floor, 'sectionId': None, 'priority': None, 'snapshots': []}
                st.rerun()

    # Display rooms grouped by floor
    if floors:
        for floor in floors:
            floor_rooms = {rid: r for rid, r in rooms.items() if r.get('floorId') == floor['id']}
            floor_assessed = sum(1 for r in floor_rooms.values() if is_assessed(r))
            st.markdown(f"### Floor {floor['label']}  ({floor_assessed}/{len(floor_rooms)} assessed)")

            if floor_rooms:
                cols = st.columns(min(len(floor_rooms), 4))
                for i, (rid, room) in enumerate(floor_rooms.items()):
                    with cols[i % 4]:
                        assessed = is_assessed(room)
                        snap = latest_snapshot(room)
                        fix_count = sum(f.get('quantity', f.get('count', 1)) for f in (snap.get('fixtures', []) if snap else []))

                        with st.container(border=True):
                            st.markdown(f"**{room['label']}** `{room['id']}`")
                            if room.get('priority'):
                                st.markdown(priority_badge(room['priority']))
                            if assessed:
                                st.caption(f"{fix_count} fixtures")
                            else:
                                st.caption("_Not yet assessed_")
    else:
        # No floors, show flat list
        if rooms:
            cols = st.columns(min(len(rooms), 4))
            for i, (rid, room) in enumerate(rooms.items()):
                with cols[i % 4]:
                    with st.container(border=True):
                        st.markdown(f"**{room['label']}** `{room['id']}`")

    st.divider()

    # Room detail
    if rooms:
        selected_room_id = st.selectbox("Select room to view/assess", options=list(rooms.keys()), format_func=lambda x: f"{rooms[x]['label']} ({x})", key="room_detail_select")
        room = rooms[selected_room_id]
        snap = latest_snapshot(room) or {}
        fixtures = snap.get('fixtures', [])

        st.markdown(f"## {room['label']} — `{room['id']}`")

        # Priority
        new_pri = st.selectbox("Priority", ['', 'urgent', 'watch', 'ok'], index=['', 'urgent', 'watch', 'ok'].index(room.get('priority') or ''), format_func=lambda x: priority_badge(x) if x else 'None', key=f"pri_{selected_room_id}")
        room['priority'] = new_pri if new_pri else None

        # Photos
        st.markdown("### Photos")
        photos = snap.get('photos', [])
        if photos:
            photo_cols = st.columns(min(len(photos), 4))
            for i, p in enumerate(photos[:4]):
                with photo_cols[i % 4]:
                    st.image(base64.b64decode(p['thumbUri']), use_container_width=True)

        uploaded_files = st.file_uploader("Upload room photos", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True, key=f"upload_{selected_room_id}")
        if uploaded_files:
            preview_cols = st.columns(min(len(uploaded_files), 3))
            for i, f in enumerate(uploaded_files):
                with preview_cols[i % 3]:
                    st.image(f, use_container_width=True)

        # Analyze
        if st.button("🔍 Analyze Room with AI", type="primary", disabled=not uploaded_files or not api_key, key=f"analyze_{selected_room_id}"):
            with st.spinner("Analyzing... Claude is identifying fixtures, materials, and conditions..."):
                client = anthropic.Anthropic(api_key=api_key)
                images = [compress_for_api(f) for f in uploaded_files]
                result = analyze_room(client, images, room['label'])

            if result and 'error' not in result:
                # Save thumbnails
                thumbnails = []
                for f in uploaded_files:
                    try:
                        thumbnails.append({'thumbUri': make_thumbnail(f), 'takenAt': datetime.now().isoformat()})
                    except:
                        pass

                # Create new snapshot or update
                existing_snap = latest_snapshot(room)
                if existing_snap and existing_snap.get('fixtures') and len(room.get('snapshots', [])) < MAX_SNAPSHOTS:
                    # New snapshot for comparison
                    new_snap = {'takenAt': datetime.now().isoformat(), 'photos': thumbnails, 'fixtures': result.get('fixtures', []),
                                'ceiling_type': result.get('ceiling_type'), 'ceiling_condition': result.get('ceiling_condition'),
                                'wall_type': result.get('wall_type'), 'wall_condition': result.get('wall_condition'),
                                'flooring_type': result.get('flooring_type'), 'flooring_condition': result.get('flooring_condition'),
                                'materials_noted': result.get('materials_noted', []), 'equipment': result.get('equipment', []),
                                'general_notes': result.get('general_notes', '')}
                    room.setdefault('snapshots', []).append(new_snap)
                else:
                    s = ensure_snapshot(room)
                    s['photos'] = thumbnails
                    s['fixtures'] = result.get('fixtures', [])
                    s['ceiling_type'] = result.get('ceiling_type')
                    s['ceiling_condition'] = result.get('ceiling_condition')
                    s['wall_type'] = result.get('wall_type')
                    s['wall_condition'] = result.get('wall_condition')
                    s['flooring_type'] = result.get('flooring_type')
                    s['flooring_condition'] = result.get('flooring_condition')
                    s['materials_noted'] = result.get('materials_noted', [])
                    s['equipment'] = result.get('equipment', [])
                    s['general_notes'] = result.get('general_notes', '')
                    s['takenAt'] = datetime.now().isoformat()

                fix_count = sum(f.get('quantity', f.get('count', 1)) for f in result.get('fixtures', []))
                st.success(f"Found {fix_count} fixtures!")
                st.rerun()
            else:
                st.error("Analysis failed. Check your API key and try again.")

        # Surfaces
        if snap.get('ceiling_type') or snap.get('wall_type') or snap.get('flooring_type'):
            st.markdown("### Surfaces")
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.markdown(f"**Ceiling:** {snap.get('ceiling_type', 'N/A')} {cond_badge(snap.get('ceiling_condition'))}", unsafe_allow_html=True)
            with sc2:
                st.markdown(f"**Walls:** {snap.get('wall_type', 'N/A')} {cond_badge(snap.get('wall_condition'))}", unsafe_allow_html=True)
            with sc3:
                st.markdown(f"**Flooring:** {snap.get('flooring_type', 'N/A')} {cond_badge(snap.get('flooring_condition'))}", unsafe_allow_html=True)

        # Fixtures
        if fixtures:
            st.markdown("### Fixtures")
            for i, f in enumerate(sorted(fixtures, key=lambda x: -(x.get('quantity', x.get('count', 1))))):
                fc1, fc2, fc3, fc4, fc5 = st.columns([2, 0.5, 1.5, 1, 1])
                with fc1:
                    icon = "✏️ " if f.get('source') == 'manual' or f.get('override') else ""
                    st.markdown(f"{icon}**{fmt_type(f.get('type', ''))}** {priority_badge(f.get('priority', ''))}")
                with fc2:
                    st.write(f.get('quantity', f.get('count', 1)))
                with fc3:
                    st.caption(f"{f.get('subtype', '')} {f.get('description', '')}")
                with fc4:
                    st.markdown(cond_badge(f.get('condition')), unsafe_allow_html=True)
                with fc5:
                    st.markdown(conf_badge(f.get('confidence')), unsafe_allow_html=True)

        # Manual add/edit
        with st.expander("Edit Fixtures", expanded=False):
            # Delete
            if fixtures:
                for i, f in enumerate(fixtures):
                    dc1, dc2 = st.columns([3, 1])
                    with dc1:
                        st.write(f"{f.get('quantity', f.get('count', 1))}x {fmt_type(f.get('type', ''))}")
                    with dc2:
                        if st.button("Delete", key=f"del_{selected_room_id}_{i}"):
                            fixtures.pop(i)
                            st.rerun()

            # Add
            st.markdown("**Add Fixture**")
            with st.form(key=f"add_fix_{selected_room_id}"):
                af1, af2 = st.columns(2)
                with af1:
                    new_type = st.selectbox("Type", FIXTURE_TYPES, format_func=fmt_type, key=f"nft_{selected_room_id}")
                    new_qty = st.number_input("Quantity", min_value=1, value=1, key=f"nfq_{selected_room_id}")
                with af2:
                    new_sub = st.text_input("Subtype", key=f"nfs_{selected_room_id}")
                    new_cond = st.selectbox("Condition", ['good', 'fair', 'poor'], key=f"nfc_{selected_room_id}")
                new_desc = st.text_input("Description", key=f"nfd_{selected_room_id}")
                new_pri = st.selectbox("Priority", ['', 'urgent', 'watch', 'ok'], format_func=lambda x: priority_badge(x) if x else 'None', key=f"nfp_{selected_room_id}")
                if st.form_submit_button("Add Fixture"):
                    s = ensure_snapshot(room)
                    s['fixtures'].append({
                        'id': uid(), 'type': new_type, 'name': new_type, 'quantity': new_qty,
                        'subtype': new_sub, 'description': new_desc, 'condition': new_cond,
                        'confidence': 'high', 'source': 'manual', 'override': False,
                        'priority': new_pri if new_pri else None, 'condition_notes': ''
                    })
                    st.rerun()

        # Comparison
        if len(room.get('snapshots', [])) >= 2:
            with st.expander("Compare Assessments"):
                newer = room['snapshots'][-1]
                older = room['snapshots'][-2]
                st.markdown(f"**Previous:** {older.get('takenAt', 'N/A')[:16]}  →  **Current:** {newer.get('takenAt', 'N/A')[:16]}")
                diff = diff_snapshots(older, newer)
                if diff['added']:
                    st.markdown(f"**Added ({len(diff['added'])})**")
                    for f in diff['added']:
                        st.success(f"{fmt_type(f.get('type',''))} × {f.get('quantity', f.get('count', 1))}")
                if diff['removed']:
                    st.markdown(f"**Removed ({len(diff['removed'])})**")
                    for f in diff['removed']:
                        st.error(f"{fmt_type(f.get('type',''))} × {f.get('quantity', f.get('count', 1))}")
                if diff['changed']:
                    st.markdown(f"**Changed ({len(diff['changed'])})**")
                    for c in diff['changed']:
                        st.warning(f"{fmt_type(c['old'].get('type',''))}: {c['old'].get('quantity', c['old'].get('count',1))} → {c['new'].get('quantity', c['new'].get('count',1))}")
                if not diff['added'] and not diff['removed'] and not diff['changed']:
                    st.info("No changes between assessments.")

        # Materials & Equipment
        if snap.get('materials_noted'):
            st.markdown("### Materials")
            st.markdown(" ".join(f'<span class="material-tag">{m}</span>' for m in snap['materials_noted']), unsafe_allow_html=True)

        if snap.get('equipment'):
            st.markdown("### Equipment")
            for e in snap['equipment']:
                st.write(f"**{fmt_type(e.get('type', ''))}**: {e.get('manufacturer', '')} {e.get('model', '')}")
                if e.get('notes'):
                    st.caption(e['notes'])

        if snap.get('general_notes'):
            st.info(snap['general_notes'])


# =============================================================================
# TAB: INVENTORY
# =============================================================================

with tab_inventory:
    st.subheader("Fixture Inventory")

    all_fixtures = []
    for rid, r in rooms.items():
        s = latest_snapshot(r)
        for f in (s.get('fixtures', []) if s else []):
            all_fixtures.append({**f, '_room': r.get('label', rid), '_room_id': rid, '_room_pri': r.get('priority')})

    total = sum(f.get('quantity', f.get('count', 1)) for f in all_fixtures)
    types = set(f.get('type') for f in all_fixtures)

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.metric("Total Fixtures", total)
    with mc2:
        st.metric("Fixture Types", len(types))
    with mc3:
        st.metric("Rooms Assessed", sum(1 for r in rooms.values() if is_assessed(r)))

    if all_fixtures:
        # Priority view toggle
        priority_sort = st.toggle("Priority view (urgent first)", key="inv_priority_sort")

        if priority_sort:
            order = {'urgent': 0, 'watch': 1, 'ok': 2, None: 3, '': 3}
            all_fixtures.sort(key=lambda f: (order.get(f.get('priority') or f.get('_room_pri'), 3), -(f.get('quantity', f.get('count', 1)))))

        for f in all_fixtures:
            ic1, ic2, ic3, ic4, ic5 = st.columns([2, 0.5, 1.2, 1, 1])
            with ic1:
                icon = "✏️ " if f.get('source') == 'manual' or f.get('override') else ""
                st.markdown(f"{icon}**{fmt_type(f.get('type', ''))}** {priority_badge(f.get('priority', ''))}")
            with ic2:
                st.write(f.get('quantity', f.get('count', 1)))
            with ic3:
                st.caption(f['_room'])
            with ic4:
                st.markdown(cond_badge(f.get('condition')), unsafe_allow_html=True)
            with ic5:
                st.markdown(conf_badge(f.get('confidence')), unsafe_allow_html=True)
    else:
        st.info("No fixtures yet. Assess some rooms first.")


# =============================================================================
# TAB: REPORT
# =============================================================================

with tab_report:
    st.subheader("Building Report")

    if rooms:
        priority_view = st.toggle("Priority view", key="rpt_priority")
        room_list = list(rooms.values())
        if priority_view:
            order = {'urgent': 0, 'watch': 1, 'ok': 2, None: 3, '': 3}
            room_list.sort(key=lambda r: order.get(r.get('priority'), 3))

        floors = building.get('floors', [])
        if floors and not priority_view:
            for floor in floors:
                st.markdown(f"### Floor {floor['label']}")
                for room in room_list:
                    if room.get('floorId') != floor['id']:
                        continue
                    _render_report_room(room)
        else:
            for room in room_list:
                _render_report_room(room)
    else:
        st.info("No rooms to report on.")


# =============================================================================
# TAB: EXPORT
# =============================================================================

with tab_export:
    st.subheader("Export")

    if rooms:
        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            st.markdown("**PDF Report**")
            try:
                pdf_data = create_pdf(rooms, room_photos)
                st.download_button("Download PDF", pdf_data, f"BuildScan_{datetime.now().strftime('%Y%m%d')}.pdf", "application/pdf", use_container_width=True)
            except Exception as e:
                st.error(f"PDF failed: {e}")
        with ec2:
            st.markdown("**Excel**")
            excel_data = create_excel(rooms)
            st.download_button("Download Excel", excel_data, f"BuildScan_{datetime.now().strftime('%Y%m%d')}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        with ec3:
            st.markdown("**CSV**")
            csv_data = create_csv(rooms)
            if csv_data:
                st.download_button("Download CSV", csv_data, f"BuildScan_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv", use_container_width=True)

        # Delete project
        st.divider()
        st.markdown("### Delete Project")
        st.caption("Type the project name to confirm deletion.")
        del_confirm = st.text_input("Project name", key="del_confirm")
        if st.button("Delete Project", type="secondary"):
            if del_confirm == proj['name']:
                del st.session_state.projects[proj['id']]
                remaining = list(st.session_state.projects.keys())
                st.session_state.active_project = remaining[0] if remaining else None
                st.rerun()
            else:
                st.error("Name doesn't match.")
    else:
        st.info("No data to export.")


# =============================================================================
# REPORT ROOM RENDERER
# =============================================================================

def _render_report_room(room):
    snap = latest_snapshot(room)
    if not snap:
        return
    label = room.get('label', room.get('id', ''))
    fixtures = snap.get('fixtures', [])
    fix_count = sum(f.get('quantity', f.get('count', 1)) for f in fixtures)
    photos = snap.get('photos', [])

    with st.expander(f"{label} ({room['id']}) — {fix_count} fixtures {priority_badge(room.get('priority', ''))}", expanded=True):
        # Photos
        if photos:
            pcols = st.columns(min(len(photos), 4))
            for i, p in enumerate(photos[:4]):
                with pcols[i % 4]:
                    st.image(base64.b64decode(p['thumbUri']), use_container_width=True)

        # Surfaces
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown(f"**Ceiling:** {snap.get('ceiling_type', 'N/A')} {cond_badge(snap.get('ceiling_condition'))}", unsafe_allow_html=True)
        with sc2:
            st.markdown(f"**Walls:** {snap.get('wall_type', 'N/A')} {cond_badge(snap.get('wall_condition'))}", unsafe_allow_html=True)
        with sc3:
            st.markdown(f"**Flooring:** {snap.get('flooring_type', 'N/A')} {cond_badge(snap.get('flooring_condition'))}", unsafe_allow_html=True)

        # Fixtures
        for f in sorted(fixtures, key=lambda x: -(x.get('quantity', x.get('count', 1)))):
            st.markdown(f"{f.get('quantity', f.get('count',1))}× **{fmt_type(f.get('type',''))}** {f.get('subtype','')} {cond_badge(f.get('condition'))} {conf_badge(f.get('confidence'))} {priority_badge(f.get('priority',''))}", unsafe_allow_html=True)

        if snap.get('materials_noted'):
            st.markdown(" ".join(f'<span class="material-tag">{m}</span>' for m in snap['materials_noted']), unsafe_allow_html=True)

        if snap.get('general_notes'):
            st.caption(snap['general_notes'])


# Footer
st.divider()
st.caption(f"BuildScan — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
