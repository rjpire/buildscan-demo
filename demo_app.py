"""
BuildScan Demo - AI-Powered Building Assessment
A Streamlit app that uses Claude to analyze room photos for construction assessments.
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
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.utils import ImageReader


# =============================================================================
# API KEY SECURITY — uses Streamlit secrets, env var, or sidebar input
# =============================================================================

def get_api_key():
    """Get API key from Streamlit secrets, env var, or session state."""
    # 1. Streamlit Cloud secrets (highest priority)
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass
    # 2. Environment variable
    env_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if env_key:
        return env_key
    # 3. Session state (from sidebar input)
    return st.session_state.get("user_api_key", "")


# Page config
st.set_page_config(
    page_title="BuildScan - Building Assessment",
    page_icon="🏗️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-top: 0;
    }
    .fixture-badge {
        background: #4F46E5;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        margin: 2px;
        display: inline-block;
    }
    .surface-tag {
        background: #e0e7ff;
        color: #3730a3;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        margin: 2px;
        display: inline-block;
    }
    .material-tag {
        background: #fef3c7;
        color: #92400e;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        margin: 2px;
        display: inline-block;
    }
    .cond-good {
        background: #dcfce7;
        color: #166534;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        display: inline-block;
    }
    .cond-fair {
        background: #fef9c3;
        color: #854d0e;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        display: inline-block;
    }
    .cond-poor {
        background: #fecaca;
        color: #991b1b;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'room_analyses' not in st.session_state:
    st.session_state.room_analyses = []
if 'room_photos' not in st.session_state:
    st.session_state.room_photos = {}  # {room_label: [base64_thumbnail_strings]}
if 'user_api_key' not in st.session_state:
    st.session_state.user_api_key = ""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

FIXTURE_TYPES = [
    "light_fixture", "outlet", "switch", "smoke_detector", "sink", "toilet",
    "door", "window", "vent", "thermostat", "fire_extinguisher", "sprinkler_head",
    "cabinet", "countertop", "bathtub", "staircase", "garage_door", "electrical_panel",
    "water_heater", "hvac_unit", "sump_pump",
]


def condition_badge(condition):
    """Return HTML for a condition badge."""
    c = (condition or "").lower().strip()
    if c == "good":
        return '<span class="cond-good">Good</span>'
    elif c == "fair":
        return '<span class="cond-fair">Fair</span>'
    elif c == "poor":
        return '<span class="cond-poor">Poor</span>'
    return ""


def encode_image(uploaded_file):
    """Convert uploaded file to base64."""
    return base64.standard_b64encode(uploaded_file.getvalue()).decode("utf-8")


def make_thumbnail(uploaded_file, max_width=400):
    """Create a small JPEG thumbnail from an uploaded file."""
    uploaded_file.seek(0)
    img = Image.open(uploaded_file)
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    ratio = max_width / img.width
    new_size = (max_width, int(img.height * ratio))
    img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=70)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def auto_room_label(room_type):
    """Generate a unique room label like 'Kitchen #2' if duplicates exist."""
    existing = [a.get('room_name', '') for a in st.session_state.room_analyses]
    if room_type not in existing:
        return room_type
    count = sum(1 for n in existing if n.startswith(room_type)) + 1
    return f"{room_type} #{count}"


# =============================================================================
# CLAUDE AI ANALYSIS
# =============================================================================

def analyze_room_assessment(client, images_base64, room_name):
    """Send all room images to Claude in a single call for unified, deduplicated analysis."""

    num_photos = len(images_base64)
    photo_context = (
        f"This is a photo of the {room_name}."
        if num_photos == 1
        else f"These are {num_photos} photos of the SAME room: the {room_name}, taken from different angles."
    )

    prompt = f"""{photo_context}

You are a construction building assessment specialist. Analyze {"this photo" if num_photos == 1 else "ALL photos together"} to create a comprehensive inventory of building materials, fixtures, and structural elements visible in the room.

IMPORTANT: {"This is a single view — count everything visible." if num_photos == 1 else "These photos show the SAME room from different angles. Cross-reference the photos to build ONE unified inventory. Do NOT double-count items that appear in multiple photos — use spatial reasoning to determine if a fixture seen in one photo is the same one seen in another. However, DO count items that are only visible in one photo but not others."}

For this room, identify and count ALL of the following that are visible:

SURFACES (also rate condition as good/fair/poor):
- ceiling_type: What is the ceiling made of? (drywall, drop/suspended ceiling, plaster, exposed structure, tongue and groove wood, coffered, vaulted, concrete, metal/tin)
- wall_type: What are the walls made of? (drywall, plaster, brick, concrete block, wood paneling, tile, concrete, stone, stucco)
- flooring_type: What is the floor? (hardwood, laminate, tile, vinyl/LVP, carpet, concrete, linoleum, natural stone, epoxy)

FIXTURES (count each one you can see, and rate condition as good/fair/poor):
- light_fixture: Count and describe type (recessed/can, pendant, fluorescent, track, chandelier, sconce, flush mount, LED panel)
- outlet: Count visible electrical outlets, note if GFCI
- switch: Count light switches
- smoke_detector: Count smoke/CO detectors on ceiling
- sink: Count sinks, note type (undermount, drop-in, pedestal, etc.)
- toilet: Count if visible
- door: Count doors, note type (interior, exterior, sliding, pocket, French)
- window: Count windows, note type (single-hung, double-hung, casement, sliding)
- vent: Count HVAC vents/registers (supply, return, exhaust)
- thermostat: Note if visible
- fire_extinguisher: Note if visible
- sprinkler_head: Count if visible
- cabinet: Count/describe cabinets
- countertop: Describe material if visible
- bathtub: Note type (standard, walk-in shower, tub/shower combo)
- staircase: Note if visible with railing type

CONDITION GUIDE:
- "good" = No visible issues, well-maintained
- "fair" = Minor wear, cosmetic issues, functional but showing age
- "poor" = Damaged, needs repair or replacement, safety concern

MATERIALS:
- List any visible building materials (copper pipe, PVC, galvanized steel, etc.)

EQUIPMENT:
- Note any major equipment visible (HVAC unit, water heater, electrical panel) with manufacturer/model if readable

Respond ONLY with valid JSON:
{{
  "room_type": "string",
  "ceiling_type": "string",
  "ceiling_condition": "good | fair | poor",
  "wall_type": "string",
  "wall_condition": "good | fair | poor",
  "flooring_type": "string",
  "flooring_condition": "good | fair | poor",
  "fixtures": [
    {{
      "type": "string (e.g. light_fixture, outlet, switch, etc.)",
      "count": number,
      "subtype": "string (e.g. recessed, GFCI, double-hung)",
      "description": "string",
      "condition": "good | fair | poor",
      "condition_notes": "string or null"
    }}
  ],
  "materials_noted": ["string"],
  "equipment": [
    {{
      "type": "string",
      "manufacturer": "string or null",
      "model": "string or null",
      "notes": "string"
    }}
  ],
  "general_notes": "A 2-3 sentence summary of the room's overall condition and notable observations",
  "total_fixture_count": number
}}"""

    content = []
    for img_b64 in images_base64:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": img_b64
            }
        })
    content.append({"type": "text", "text": prompt})

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": content}]
    )

    response_text = response.content[0].text
    try:
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        return json.loads(response_text[start:end])
    except:
        return {"fixtures": [], "error": "Failed to parse response"}


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def create_assessment_excel(assessments):
    """Create Excel file with building assessment data."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        summary_rows = []
        for a in assessments:
            room = a.get('room_name', 'Unknown')
            total_fixtures = sum(f.get('count', 0) for f in a.get('fixtures', []))
            summary_rows.append({
                'Room': room,
                'Ceiling': f"{a.get('ceiling_type', 'N/A')} ({a.get('ceiling_condition', 'N/A')})",
                'Walls': f"{a.get('wall_type', 'N/A')} ({a.get('wall_condition', 'N/A')})",
                'Flooring': f"{a.get('flooring_type', 'N/A')} ({a.get('flooring_condition', 'N/A')})",
                'Total Fixtures': total_fixtures,
                'Materials Noted': ', '.join(a.get('materials_noted', [])),
                'Notes': a.get('general_notes', ''),
            })
        if summary_rows:
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name='Room Summary', index=False)

        fixture_rows = []
        for a in assessments:
            room = a.get('room_name', 'Unknown')
            for f in a.get('fixtures', []):
                fixture_rows.append({
                    'Room': room,
                    'Fixture Type': f.get('type', ''),
                    'Count': f.get('count', 0),
                    'Subtype': f.get('subtype', ''),
                    'Description': f.get('description', ''),
                    'Condition': f.get('condition', ''),
                    'Condition Notes': f.get('condition_notes', ''),
                })
        if fixture_rows:
            pd.DataFrame(fixture_rows).to_excel(writer, sheet_name='Fixture Inventory', index=False)

        equip_rows = []
        for a in assessments:
            room = a.get('room_name', 'Unknown')
            for e in a.get('equipment', []):
                equip_rows.append({
                    'Room': room,
                    'Equipment Type': e.get('type', ''),
                    'Manufacturer': e.get('manufacturer', ''),
                    'Model': e.get('model', ''),
                    'Notes': e.get('notes', ''),
                })
        if equip_rows:
            pd.DataFrame(equip_rows).to_excel(writer, sheet_name='Equipment', index=False)

    return output.getvalue()


def create_assessment_pdf(assessments, room_photos):
    """Create a PDF building assessment report with embedded photos."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=60, leftMargin=60, topMargin=60, bottomMargin=60)
    elements = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('Title2', parent=styles['Heading1'], fontSize=22, textColor=colors.HexColor('#1E3A5F'), spaceAfter=20, alignment=TA_CENTER)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor('#1E3A5F'), spaceAfter=10, spaceBefore=10)
    subheading_style = ParagraphStyle('SubHead', parent=styles['Heading3'], fontSize=11, textColor=colors.HexColor('#333'), spaceAfter=6)

    cond_colors = {
        'good': colors.HexColor('#166534'),
        'fair': colors.HexColor('#854d0e'),
        'poor': colors.HexColor('#991b1b'),
    }

    # Title page
    elements.append(Spacer(1, 1.5 * inch))
    elements.append(Paragraph("BuildScan", title_style))
    elements.append(Paragraph("Building Assessment Report", ParagraphStyle('Sub', parent=styles['Heading2'], alignment=TA_CENTER, textColor=colors.HexColor('#666'))))
    elements.append(Spacer(1, 0.5 * inch))

    total_rooms = len(assessments)
    total_fixtures = sum(sum(f.get('count', 0) for f in a.get('fixtures', [])) for a in assessments)

    summary_data = [
        ['Report Date:', datetime.now().strftime('%B %d, %Y')],
        ['Rooms Assessed:', str(total_rooms)],
        ['Total Fixtures:', str(total_fixtures)],
    ]
    summary_table = Table(summary_data, colWidths=[2 * inch, 3.5 * inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    elements.append(summary_table)
    elements.append(PageBreak())

    # Per-room pages
    for analysis in assessments:
        room_name = analysis.get('room_name', 'Unknown Room')
        elements.append(Paragraph(room_name, heading_style))

        # Photos
        photos = room_photos.get(room_name, [])
        if photos:
            for photo_b64 in photos[:4]:
                try:
                    img_bytes = base64.b64decode(photo_b64)
                    img_reader = ImageReader(io.BytesIO(img_bytes))
                    iw, ih = img_reader.getSize()
                    aspect = ih / iw
                    display_w = 3.5 * inch
                    display_h = display_w * aspect
                    from reportlab.platypus import RLImage
                    elements.append(RLImage(io.BytesIO(img_bytes), width=display_w, height=display_h))
                    elements.append(Spacer(1, 0.1 * inch))
                except:
                    pass

        # Surfaces
        elements.append(Paragraph("Surfaces", subheading_style))
        surface_data = [['Surface', 'Type', 'Condition']]
        for s_label, s_key, c_key in [('Ceiling', 'ceiling_type', 'ceiling_condition'),
                                       ('Walls', 'wall_type', 'wall_condition'),
                                       ('Flooring', 'flooring_type', 'flooring_condition')]:
            surface_data.append([s_label, analysis.get(s_key, 'N/A'), (analysis.get(c_key, '') or '').title()])

        s_table = Table(surface_data, colWidths=[1.2 * inch, 2.5 * inch, 1.5 * inch])
        s_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4F46E5')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(s_table)
        elements.append(Spacer(1, 0.15 * inch))

        # Fixtures
        fixtures = analysis.get('fixtures', [])
        if fixtures:
            elements.append(Paragraph("Fixtures", subheading_style))
            fix_data = [['Type', 'Count', 'Subtype', 'Condition']]
            for f in sorted(fixtures, key=lambda x: -x.get('count', 0)):
                fix_data.append([
                    f.get('type', '').replace('_', ' ').title(),
                    str(f.get('count', 0)),
                    f.get('subtype', '') or '',
                    (f.get('condition', '') or '').title(),
                ])
            f_table = Table(fix_data, colWidths=[2 * inch, 0.7 * inch, 1.8 * inch, 1 * inch])
            f_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4F46E5')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
            ]))
            # Color-code condition cells
            for i, f in enumerate(sorted(fixtures, key=lambda x: -x.get('count', 0)), start=1):
                cond = (f.get('condition', '') or '').lower()
                if cond in cond_colors:
                    f_table.setStyle(TableStyle([('TEXTCOLOR', (3, i), (3, i), cond_colors[cond])]))
            elements.append(f_table)
            elements.append(Spacer(1, 0.15 * inch))

        # Equipment
        equipment = analysis.get('equipment', [])
        if equipment:
            elements.append(Paragraph("Equipment", subheading_style))
            for e in equipment:
                mfg = e.get('manufacturer', '') or ''
                model = e.get('model', '') or ''
                elements.append(Paragraph(f"<b>{e.get('type', 'Unknown')}</b>: {mfg} {model}".strip(), styles['Normal']))
                if e.get('notes'):
                    elements.append(Paragraph(f"<i>{e['notes']}</i>", styles['Normal']))

        # Notes
        notes = analysis.get('general_notes', '')
        if notes:
            elements.append(Spacer(1, 0.1 * inch))
            elements.append(Paragraph(f"<b>Notes:</b> {notes}", styles['Normal']))

        materials = analysis.get('materials_noted', [])
        if materials:
            elements.append(Paragraph(f"<b>Materials:</b> {', '.join(materials)}", styles['Normal']))

        elements.append(PageBreak())

    # Footer
    footer_style = ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8, textColor=colors.grey, alignment=TA_CENTER)
    elements.append(Paragraph(
        "Report generated by BuildScan using AI-assisted visual analysis. "
        "This assessment is based on photographs and should be supplemented with on-site inspection.",
        footer_style
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


# =============================================================================
# MAIN APP
# =============================================================================

st.markdown('<p class="main-header">🏗️ BuildScan</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Building Assessment for Construction Companies</p>', unsafe_allow_html=True)

api_key = get_api_key()

# Sidebar
with st.sidebar:
    st.header("Assessment Summary")

    if api_key:
        st.success("API Connected")
    else:
        st.warning("No API key found")
        key_input = st.text_input("Enter Anthropic API Key", type="password", key="sidebar_api_key")
        if key_input:
            st.session_state.user_api_key = key_input
            st.rerun()

    st.divider()

    total_fixtures = sum(
        sum(f.get('count', 0) for f in a.get('fixtures', []))
        for a in st.session_state.room_analyses
    )

    st.metric("Rooms Assessed", len(st.session_state.room_analyses))
    st.metric("Total Fixtures", total_fixtures)

    if st.session_state.room_analyses:
        fixture_totals = {}
        for a in st.session_state.room_analyses:
            for f in a.get('fixtures', []):
                ftype = f.get('type', 'unknown')
                fixture_totals[ftype] = fixture_totals.get(ftype, 0) + f.get('count', 0)

        if fixture_totals:
            st.divider()
            st.subheader("Fixture Totals")
            for ftype, count in sorted(fixture_totals.items(), key=lambda x: -x[1]):
                label = ftype.replace('_', ' ').title()
                st.write(f"**{label}:** {count}")

    st.divider()

    if st.button("Clear All Data"):
        st.session_state.room_analyses = []
        st.session_state.room_photos = {}
        st.rerun()


# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["Add Room", "Fixture Inventory", "Building Report", "Export"])

# ==================== TAB 1: ADD ROOM ====================
with tab1:
    st.header("Assess a Room")

    col1, col2 = st.columns([1, 1])

    with col1:
        room_type = st.selectbox(
            "Room Type",
            ["Kitchen", "Living Room", "Bedroom", "Bathroom", "Basement",
             "Garage", "Office", "Dining Room", "Hallway", "Stairwell",
             "Mechanical Room", "Utility Room", "Attic", "Exterior",
             "Roof", "Crawl Space", "Storage", "Other"]
        )

    with col2:
        room_label = st.text_input(
            "Room Label (optional)",
            placeholder=f"e.g., Master Bath, Unit 2A {room_type}",
            help="Give this room a custom name to distinguish it from other rooms of the same type. Leave blank for auto-naming."
        )

    # Determine final room name
    if room_type == "Other" and not room_label:
        room_label = st.text_input("Enter custom room/area name")
    room_name = room_label.strip() if room_label.strip() else auto_room_label(room_type)

    # Check for duplicate
    existing_names = [a.get('room_name', '') for a in st.session_state.room_analyses]
    is_update = room_name in existing_names

    if is_update:
        st.info(f"A room named '{room_name}' already exists. Analyzing will update the existing assessment.")

    st.subheader("Upload Room Photos")
    st.caption("Upload multiple photos from different angles for the most complete assessment. All photos are analyzed together — duplicates are automatically handled.")

    uploaded_files = st.file_uploader(
        "Choose photos",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        key="room_photos_uploader"
    )

    if uploaded_files:
        cols = st.columns(min(len(uploaded_files), 3))
        for i, file in enumerate(uploaded_files):
            with cols[i % 3]:
                st.image(file, caption=f"Photo {i+1}", use_container_width=True)

    if st.button("Analyze Room", type="primary", disabled=not uploaded_files or not api_key):
        if not api_key:
            st.error("API key not configured.")
        else:
            try:
                client = anthropic.Anthropic(api_key=api_key)

                existing_idx = None
                for idx, a in enumerate(st.session_state.room_analyses):
                    if a.get('room_name') == room_name:
                        existing_idx = idx
                        break

                photo_count = len(uploaded_files)
                with st.spinner(f"Analyzing {photo_count} photo{'s' if photo_count > 1 else ''}... Claude is identifying fixtures, materials, and conditions..."):
                    images_base64 = [encode_image(f) for f in uploaded_files]
                    result = analyze_room_assessment(client, images_base64, room_name)

                if 'error' not in result:
                    result['room_name'] = room_name
                    result['room_type'] = room_type
                    result['photo_count'] = photo_count

                    # Store thumbnails
                    thumbnails = []
                    for f in uploaded_files:
                        try:
                            thumbnails.append(make_thumbnail(f))
                        except:
                            pass
                    st.session_state.room_photos[room_name] = thumbnails

                    if existing_idx is not None:
                        st.session_state.room_analyses[existing_idx] = result
                    else:
                        st.session_state.room_analyses.append(result)

                    fixture_count = sum(f.get('count', 0) for f in result.get('fixtures', []))
                    st.success(f"Assessment complete! Found {fixture_count} fixtures in {room_name} (from {photo_count} photo{'s' if photo_count > 1 else ''}).")
                    st.balloons()
                else:
                    st.error("Failed to analyze photos. Please try again.")

            except Exception as e:
                st.error(f"Error: {str(e)}")


# ==================== TAB 2: FIXTURE INVENTORY ====================
with tab2:
    st.header("Fixture Inventory")

    if st.session_state.room_analyses:
        all_fixtures = []
        for a in st.session_state.room_analyses:
            for f in a.get('fixtures', []):
                all_fixtures.append({'room': a.get('room_name', 'Unknown'), **f})

        total = sum(f.get('count', 0) for f in all_fixtures)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Fixtures", total)
        with col2:
            st.metric("Rooms Assessed", len(st.session_state.room_analyses))
        with col3:
            types = set(f.get('type') for f in all_fixtures)
            st.metric("Fixture Types", len(types))

        st.divider()

        rooms = list(set(a.get('room_name', 'Unknown') for a in st.session_state.room_analyses))
        selected_room = st.selectbox("Filter by Room", ["All Rooms"] + sorted(rooms), key="inv_room_filter")

        analyses_to_show = st.session_state.room_analyses
        if selected_room != "All Rooms":
            analyses_to_show = [a for a in analyses_to_show if a.get('room_name') == selected_room]

        for a_idx, analysis in enumerate(st.session_state.room_analyses):
            if selected_room != "All Rooms" and analysis.get('room_name') != selected_room:
                continue

            room = analysis.get('room_name', 'Unknown')
            fixtures = analysis.get('fixtures', [])
            fixture_count = sum(f.get('count', 0) for f in fixtures)

            with st.expander(f"{room} — {fixture_count} fixtures", expanded=True):
                # Surfaces with condition
                surfaces = []
                for s_label, s_key, c_key in [('Ceiling', 'ceiling_type', 'ceiling_condition'),
                                               ('Walls', 'wall_type', 'wall_condition'),
                                               ('Floor', 'flooring_type', 'flooring_condition')]:
                    s_val = analysis.get(s_key)
                    s_cond = analysis.get(c_key, '')
                    if s_val:
                        cond_html = f" {condition_badge(s_cond)}" if s_cond else ""
                        surfaces.append(f'<span class="surface-tag">{s_label}: {s_val}</span>{cond_html}')

                if surfaces:
                    st.markdown(" ".join(surfaces), unsafe_allow_html=True)
                    st.write("")

                # Fixtures with condition
                if fixtures:
                    for f_idx, f in enumerate(sorted(fixtures, key=lambda x: -x.get('count', 0))):
                        label = f.get('type', '').replace('_', ' ').title()
                        count = f.get('count', 0)
                        subtype = f.get('subtype', '')
                        desc = f.get('description', '')
                        cond = f.get('condition', '')
                        cond_notes = f.get('condition_notes', '')

                        col1, col2, col3, col4 = st.columns([1.2, 2, 1, 1.5])
                        with col1:
                            st.markdown(f"**{count}x** {label}")
                        with col2:
                            detail = subtype
                            if desc:
                                detail = f"{subtype} — {desc}" if subtype else desc
                            st.write(detail or "—")
                        with col3:
                            st.markdown(condition_badge(cond), unsafe_allow_html=True)
                        with col4:
                            st.write(cond_notes or "")

                # Materials
                materials = analysis.get('materials_noted', [])
                if materials:
                    st.write("")
                    st.markdown("**Materials:** " + " ".join(f'<span class="material-tag">{m}</span>' for m in materials), unsafe_allow_html=True)

                # Manual edit section
                st.write("")
                with st.expander("Edit Fixtures", expanded=False):
                    # Delete fixtures
                    if fixtures:
                        st.caption("Remove a fixture:")
                        for f_idx, f in enumerate(fixtures):
                            f_label = f.get('type', '').replace('_', ' ').title()
                            col_a, col_b = st.columns([3, 1])
                            with col_a:
                                st.write(f"{f.get('count', 0)}x {f_label}")
                            with col_b:
                                if st.button("Delete", key=f"del_{room}_{f_idx}"):
                                    st.session_state.room_analyses[a_idx]['fixtures'].pop(f_idx)
                                    st.rerun()

                    # Add fixture form
                    st.caption("Add a fixture:")
                    with st.form(key=f"add_fixture_{room}_{a_idx}"):
                        add_c1, add_c2 = st.columns(2)
                        with add_c1:
                            new_type = st.selectbox("Type", FIXTURE_TYPES, format_func=lambda x: x.replace('_', ' ').title(), key=f"new_type_{room}")
                            new_count = st.number_input("Count", min_value=1, value=1, key=f"new_count_{room}")
                        with add_c2:
                            new_subtype = st.text_input("Subtype", key=f"new_sub_{room}")
                            new_condition = st.selectbox("Condition", ["good", "fair", "poor"], key=f"new_cond_{room}")
                        new_desc = st.text_input("Description", key=f"new_desc_{room}")

                        if st.form_submit_button("Add Fixture"):
                            st.session_state.room_analyses[a_idx]['fixtures'].append({
                                'type': new_type,
                                'count': new_count,
                                'subtype': new_subtype,
                                'condition': new_condition,
                                'description': new_desc,
                                'condition_notes': '',
                            })
                            st.rerun()
    else:
        st.info("No assessments yet. Go to 'Add Room' to analyze a room photo.")


# ==================== TAB 3: BUILDING REPORT ====================
with tab3:
    st.header("Building Report")

    if st.session_state.room_analyses:
        for analysis in st.session_state.room_analyses:
            room_name = analysis.get('room_name', 'Unknown Room')
            with st.expander(f"{room_name}", expanded=True):

                # Photos
                photos = st.session_state.room_photos.get(room_name, [])
                if photos:
                    photo_cols = st.columns(min(len(photos), 4))
                    for i, photo_b64 in enumerate(photos[:4]):
                        with photo_cols[i % 4]:
                            st.image(base64.b64decode(photo_b64), use_container_width=True)
                    st.write("")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Surfaces")
                    for s_label, s_key, c_key in [('Ceiling', 'ceiling_type', 'ceiling_condition'),
                                                   ('Walls', 'wall_type', 'wall_condition'),
                                                   ('Flooring', 'flooring_type', 'flooring_condition')]:
                        s_val = analysis.get(s_key, 'N/A')
                        s_cond = analysis.get(c_key, '')
                        cond_html = f" {condition_badge(s_cond)}" if s_cond else ""
                        st.markdown(f"**{s_label}:** {s_val}{cond_html}", unsafe_allow_html=True)

                with col2:
                    st.subheader("Fixture Summary")
                    fixtures = analysis.get('fixtures', [])
                    for f in sorted(fixtures, key=lambda x: -x.get('count', 0))[:8]:
                        label = f.get('type', '').replace('_', ' ').title()
                        cond = f.get('condition', '')
                        cond_html = f" {condition_badge(cond)}" if cond else ""
                        st.markdown(f"**{f.get('count', 0)}** {label}{cond_html}", unsafe_allow_html=True)

                equipment = analysis.get('equipment', [])
                if equipment:
                    st.divider()
                    st.subheader("Equipment")
                    for e in equipment:
                        mfg = e.get('manufacturer', '')
                        model = e.get('model', '')
                        st.write(f"**{e.get('type', 'Unknown')}**: {mfg} {model}".strip())
                        if e.get('notes'):
                            st.caption(e['notes'])

                st.divider()
                st.subheader("Assessment Notes")
                st.info(analysis.get('general_notes', 'No notes available'))

                materials = analysis.get('materials_noted', [])
                if materials:
                    st.write(f"**Materials observed:** {', '.join(materials)}")
    else:
        st.info("No building analyses yet. Go to 'Add Room' to analyze a room photo.")


# ==================== TAB 4: EXPORT ====================
with tab4:
    st.header("Export")

    if st.session_state.room_analyses:
        # Preview table
        st.subheader("Fixture Inventory Preview")
        preview_rows = []
        for a in st.session_state.room_analyses:
            room = a.get('room_name', 'Unknown')
            for f in a.get('fixtures', []):
                preview_rows.append({
                    'Room': room,
                    'Fixture': f.get('type', '').replace('_', ' ').title(),
                    'Count': f.get('count', 0),
                    'Subtype': f.get('subtype', ''),
                    'Condition': f.get('condition', ''),
                    'Description': f.get('description', ''),
                })

        if preview_rows:
            df_preview = pd.DataFrame(preview_rows)
            st.dataframe(df_preview, use_container_width=True)

        st.divider()

        # Download buttons
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.subheader("PDF Report")
            try:
                pdf_data = create_assessment_pdf(st.session_state.room_analyses, st.session_state.room_photos)
                st.download_button(
                    label="Download PDF",
                    data=pdf_data,
                    file_name=f"BuildScan_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
            except Exception as e:
                st.error(f"PDF generation failed: {e}")

        with col2:
            st.subheader("Excel")
            excel_data = create_assessment_excel(st.session_state.room_analyses)
            st.download_button(
                label="Download Excel",
                data=excel_data,
                file_name=f"BuildScan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        with col3:
            st.subheader("CSV")
            if preview_rows:
                csv_data = df_preview.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"BuildScan_Fixtures_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

        with col4:
            st.subheader("JSON")
            json_data = json.dumps(st.session_state.room_analyses, indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"BuildScan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    else:
        st.info("No data to export yet. Analyze some rooms first!")

# Footer
st.divider()
st.caption(f"Assessment generated {datetime.now().strftime('%Y-%m-%d %H:%M')} | BuildScan - AI Building Assessment")
