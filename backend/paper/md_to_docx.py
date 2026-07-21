#!/usr/bin/env python3
"""
Convert methodology.md to a formatted Word document using pandoc.
- Times New Roman 12pt throughout
- Mathematical equations rendered as native Word OMML equations (via pandoc)
- Headings, tables, code blocks properly formatted
Usage: python md_to_docx.py
"""
import os
import glob
import subprocess
import sys
from docx import Document
from docx.shared import Pt, Inches
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
import lxml.etree as etree

def convert_with_pandoc(md_path, docx_path):
    """Use pandoc to convert markdown to docx with proper math equations."""
    cmd = [
        'pandoc',
        md_path,
        '-o', docx_path,
        '--from', 'markdown',
        '--to', 'docx',
        '--mathml',  # Convert LaTeX math to MathML then to OMML
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Pandoc stderr: {result.stderr}")
        # Try without --mathml flag
        cmd2 = [
            'pandoc',
            md_path,
            '-o', docx_path,
            '--from', 'markdown',
            '--to', 'docx',
        ]
        print(f"Retrying: {' '.join(cmd2)}")
        result = subprocess.run(cmd2, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Pandoc failed: {result.stderr}")
            return False
    
    return True

def set_times_new_roman(docx_path):
    """Post-process the docx to set all fonts to Times New Roman 12pt."""
    doc = Document(docx_path)
    
    # Set Normal style
    style = doc.styles['Normal']
    style.font.name = 'Times New Roman'
    style.font.size = Pt(12)
    
    rPr = style.element.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = etree.SubElement(rPr, qn('w:rFonts'))
    rFonts.set(qn('w:ascii'), 'Times New Roman')
    rFonts.set(qn('w:hAnsi'), 'Times New Roman')
    rFonts.set(qn('w:eastAsia'), 'Times New Roman')
    rFonts.set(qn('w:cs'), 'Times New Roman')
    
    # Set heading styles
    heading_sizes = {
        'Title': 16,
        'Heading 1': 14,
        'Heading 2': 13,
        'Heading 3': 12,
        'Heading 4': 12,
    }
    
    for heading_name, size in heading_sizes.items():
        try:
            h_style = doc.styles[heading_name]
            h_style.font.name = 'Times New Roman'
            h_style.font.size = Pt(size)
            h_style.font.bold = True
            rPr = h_style.element.get_or_add_rPr()
            rFonts = rPr.find(qn('w:rFonts'))
            if rFonts is None:
                rFonts = etree.SubElement(rPr, qn('w:rFonts'))
            rFonts.set(qn('w:ascii'), 'Times New Roman')
            rFonts.set(qn('w:hAnsi'), 'Times New Roman')
            rFonts.set(qn('w:eastAsia'), 'Times New Roman')
        except KeyError:
            pass
    
    # Set margins
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)
    
    # Walk through all paragraphs and set font on each run
    for paragraph in doc.paragraphs:
        for run in paragraph.runs:
            run.font.name = 'Times New Roman'
            run.font.size = Pt(12)
            rPr = run._element.get_or_add_rPr()
            rFonts = rPr.find(qn('w:rFonts'))
            if rFonts is None:
                rFonts = etree.SubElement(rPr, qn('w:rFonts'))
            rFonts.set(qn('w:ascii'), 'Times New Roman')
            rFonts.set(qn('w:hAnsi'), 'Times New Roman')
            rFonts.set(qn('w:eastAsia'), 'Times New Roman')
            rFonts.set(qn('w:cs'), 'Times New Roman')
    
    # Walk through all tables and set font
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.name = 'Times New Roman'
                        run.font.size = Pt(10)
                        rPr = run._element.get_or_add_rPr()
                        rFonts = rPr.find(qn('w:rFonts'))
                        if rFonts is None:
                            rFonts = etree.SubElement(rPr, qn('w:rFonts'))
                        rFonts.set(qn('w:ascii'), 'Times New Roman')
                        rFonts.set(qn('w:hAnsi'), 'Times New Roman')
                        rFonts.set(qn('w:eastAsia'), 'Times New Roman')
                        rFonts.set(qn('w:cs'), 'Times New Roman')
    
    doc.save(docx_path)

def add_page_numbers(doc):
    """Add page numbers to the footer of the document."""
    for section in doc.sections:
        footer = section.footer
        footer.is_linked_to_previous = False
        
        # Clear existing footer content
        for p in footer.paragraphs:
            p.clear()
        
        # Create a centered paragraph with page number field
        footer_para = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
        footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Add "Page " text
        run = footer_para.add_run("Page ")
        run.font.name = 'Times New Roman'
        run.font.size = Pt(10)
        
        # Add PAGE field code
        fldChar1 = etree.SubElement(run._element, qn('w:fldChar'))
        fldChar1.set(qn('w:fldCharType'), 'begin')
        
        instrText = etree.SubElement(run._element, qn('w:instrText'))
        instrText.set(qn('xml:space'), 'preserve')
        instrText.text = ' PAGE '
        
        fldChar2 = etree.SubElement(run._element, qn('w:fldChar'))
        fldChar2.set(qn('w:fldCharType'), 'end')
        
        # Add " of " text
        run2 = footer_para.add_run(" of ")
        run2.font.name = 'Times New Roman'
        run2.font.size = Pt(10)
        
        # Add NUMPAGES field code
        fldChar3 = etree.SubElement(run2._element, qn('w:fldChar'))
        fldChar3.set(qn('w:fldCharType'), 'begin')
        
        instrText2 = etree.SubElement(run2._element, qn('w:instrText'))
        instrText2.set(qn('xml:space'), 'preserve')
        instrText2.text = ' NUMPAGES '
        
        fldChar4 = etree.SubElement(run2._element, qn('w:fldChar'))
        fldChar4.set(qn('w:fldCharType'), 'end')

def _make_title_paragraph(body, title_text, font_size_half_pts=28):
    """Create a centered, bold title paragraph in Times New Roman."""
    title = etree.SubElement(body, qn('w:p'))
    title_pPr = etree.SubElement(title, qn('w:pPr'))
    title_jc = etree.SubElement(title_pPr, qn('w:jc'))
    title_jc.set(qn('w:val'), 'center')
    
    title_rPr = etree.SubElement(title_pPr, qn('w:rPr'))
    title_rFonts = etree.SubElement(title_rPr, qn('w:rFonts'))
    title_rFonts.set(qn('w:ascii'), 'Times New Roman')
    title_rFonts.set(qn('w:hAnsi'), 'Times New Roman')
    title_rFonts.set(qn('w:eastAsia'), 'Times New Roman')
    title_b = etree.SubElement(title_rPr, qn('w:b'))
    title_sz = etree.SubElement(title_rPr, qn('w:sz'))
    title_sz.set(qn('w:val'), str(font_size_half_pts))
    
    title_run = etree.SubElement(title, qn('w:r'))
    title_run_rPr = etree.SubElement(title_run, qn('w:rPr'))
    title_run_rFonts = etree.SubElement(title_run_rPr, qn('w:rFonts'))
    title_run_rFonts.set(qn('w:ascii'), 'Times New Roman')
    title_run_rFonts.set(qn('w:hAnsi'), 'Times New Roman')
    title_run_rFonts.set(qn('w:eastAsia'), 'Times New Roman')
    title_run_b = etree.SubElement(title_run_rPr, qn('w:b'))
    title_run_sz = etree.SubElement(title_run_rPr, qn('w:sz'))
    title_run_sz.set(qn('w:val'), str(font_size_half_pts))
    
    title_text_elem = etree.SubElement(title_run, qn('w:t'))
    title_text_elem.text = title_text
    return title


def _make_toc_field_paragraph(body, toc_switch):
    """
    Create a TOC field paragraph with the given switch (e.g. \\o "1-3" or \\c "Figure").
    The field will auto-populate when the user updates fields in Word.
    """
    toc_para = etree.SubElement(body, qn('w:p'))
    
    # Begin field
    toc_run = etree.SubElement(toc_para, qn('w:r'))
    fldChar_begin = etree.SubElement(toc_run, qn('w:fldChar'))
    fldChar_begin.set(qn('w:fldCharType'), 'begin')
    
    # Field instruction
    instr_run = etree.SubElement(toc_para, qn('w:r'))
    instrText = etree.SubElement(instr_run, qn('w:instrText'))
    instrText.set(qn('xml:space'), 'preserve')
    instrText.text = f' TOC {toc_switch} \\h \\z \\u '
    
    # Separator
    sep_run = etree.SubElement(toc_para, qn('w:r'))
    fldChar_sep = etree.SubElement(sep_run, qn('w:fldChar'))
    fldChar_sep.set(qn('w:fldCharType'), 'separate')
    
    # Placeholder text
    placeholder_run = etree.SubElement(toc_para, qn('w:r'))
    placeholder_t = etree.SubElement(placeholder_run, qn('w:t'))
    placeholder_t.text = 'Right-click and select "Update Field" to generate this list.'
    
    # End field
    end_run = etree.SubElement(toc_para, qn('w:r'))
    fldChar_end = etree.SubElement(end_run, qn('w:fldChar'))
    fldChar_end.set(qn('w:fldCharType'), 'end')
    
    return toc_para


def _make_page_break(body):
    """Create a page break paragraph."""
    pb_para = etree.SubElement(body, qn('w:p'))
    pb_run = etree.SubElement(pb_para, qn('w:r'))
    pb_br = etree.SubElement(pb_run, qn('w:br'))
    pb_br.set(qn('w:type'), 'page')
    return pb_para


def _add_bookmarks_to_captions(doc):
    """
    Scan all paragraphs for figure and table caption text (e.g., 'Figure 1:' or 'Table 1:')
    and apply Word's Caption style + add SEQ fields so that TOC \\c fields can reference them.
    Also adds bookmarks for PAGEREF cross-references.
    
    Returns lists of (caption_text, bookmark_name) for figures and tables.
    """
    import re
    # Only match actual caption paragraphs: "**Figure N:**" or "**Table N:**" at the start
    figure_pattern = re.compile(r'^\*{0,2}Figure\s+(\d+)\s*[:.]', re.IGNORECASE)
    table_pattern = re.compile(r'^\*{0,2}Table\s+(\d+)\s*[:.]', re.IGNORECASE)
    
    figures = []
    tables = []
    seen_fig_bookmarks = set()
    seen_tbl_bookmarks = set()
    
    # Ensure Caption style exists
    try:
        caption_style = doc.styles['Caption']
    except KeyError:
        caption_style = doc.styles.add_style('Caption', 1)  # 1 = paragraph style
    caption_style.font.name = 'Times New Roman'
    caption_style.font.size = Pt(11)
    caption_style.font.bold = True
    
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        
        fig_match = figure_pattern.match(text)
        tbl_match = table_pattern.match(text)
        
        if fig_match:
            fig_num = fig_match.group(1)
            bookmark_name = f'_Fig{fig_num}'
            if bookmark_name in seen_fig_bookmarks:
                continue
            seen_fig_bookmarks.add(bookmark_name)
            figures.append((text, bookmark_name))
            _apply_caption_style_with_seq(para, 'Figure', bookmark_name)
        elif tbl_match:
            tbl_num = tbl_match.group(1)
            bookmark_name = f'_Tbl{tbl_num}'
            if bookmark_name in seen_tbl_bookmarks:
                continue
            seen_tbl_bookmarks.add(bookmark_name)
            tables.append((text, bookmark_name))
            _apply_caption_style_with_seq(para, 'Table', bookmark_name)
    
    # Sort by figure/table number
    figures.sort(key=lambda x: int(re.search(r'\d+', x[1]).group()))
    tables.sort(key=lambda x: int(re.search(r'\d+', x[1]).group()))
    
    return figures, tables


def _apply_caption_style_with_seq(paragraph, label, bookmark_name):
    """
    Apply the Caption style to a paragraph and add a SEQ field + bookmark.
    The SEQ field enables TOC \\c to list these paragraphs.
    """
    # Apply Caption style
    paragraph.style = 'Caption'
    
    p_elem = paragraph._element
    
    # Add bookmarkStart right after pPr
    pPr = p_elem.find(qn('w:pPr'))
    if pPr is None:
        pPr = etree.SubElement(p_elem, qn('w:pPr'))
        p_elem.insert(0, pPr)
    
    bm_id = str(abs(hash(bookmark_name)) % 100000)
    bm_start = etree.Element(qn('w:bookmarkStart'))
    bm_start.set(qn('w:id'), bm_id)
    bm_start.set(qn('w:name'), bookmark_name)
    pPr.addnext(bm_start)
    
    # Add bookmarkEnd at the end of the paragraph
    bm_end = etree.Element(qn('w:bookmarkEnd'))
    bm_end.set(qn('w:id'), bm_id)
    p_elem.append(bm_end)


def _build_manual_list(body, items, list_title):
    """
    Build a manual list of figures or tables with PAGEREF fields for page numbers.
    Each item is (caption_text, bookmark_name).
    """
    # Title
    _make_title_paragraph(body, list_title)
    
    # Empty paragraph for spacing
    etree.SubElement(body, qn('w:p'))
    
    for caption_text, bookmark_name in items:
        # Create a paragraph for each item
        entry_para = etree.SubElement(body, qn('w:p'))
        
        # Add tab stops: one right-aligned with dot leader for page numbers
        pPr = etree.SubElement(entry_para, qn('w:pPr'))
        tabs = etree.SubElement(pPr, qn('w:tabs'))
        tab = etree.SubElement(tabs, qn('w:tab'))
        tab.set(qn('w:val'), 'right')
        tab.set(qn('w:leader'), 'dot')
        tab.set(qn('w:pos'), '9360')  # ~6.5 inches
        
        # Truncate caption text to a reasonable length for the list
        display_text = caption_text
        if len(display_text) > 120:
            display_text = display_text[:117] + '...'
        
        # Add the caption text
        text_run = etree.SubElement(entry_para, qn('w:r'))
        text_rPr = etree.SubElement(text_run, qn('w:rPr'))
        text_rFonts = etree.SubElement(text_rPr, qn('w:rFonts'))
        text_rFonts.set(qn('w:ascii'), 'Times New Roman')
        text_rFonts.set(qn('w:hAnsi'), 'Times New Roman')
        text_rFonts.set(qn('w:eastAsia'), 'Times New Roman')
        text_sz = etree.SubElement(text_rPr, qn('w:sz'))
        text_sz.set(qn('w:val'), '24')  # 12pt
        
        text_t = etree.SubElement(text_run, qn('w:t'))
        text_t.text = display_text
        text_t.set(qn('xml:space'), 'preserve')
        
        # Tab to the right
        tab_run = etree.SubElement(entry_para, qn('w:r'))
        tab_char = etree.SubElement(tab_run, qn('w:tab'))
        
        # PAGEREF field for page number
        pageref_run = etree.SubElement(entry_para, qn('w:r'))
        pageref_rPr = etree.SubElement(pageref_run, qn('w:rPr'))
        pageref_rFonts = etree.SubElement(pageref_rPr, qn('w:rFonts'))
        pageref_rFonts.set(qn('w:ascii'), 'Times New Roman')
        pageref_rFonts.set(qn('w:hAnsi'), 'Times New Roman')
        pageref_rFonts.set(qn('w:eastAsia'), 'Times New Roman')
        pageref_sz = etree.SubElement(pageref_rPr, qn('w:sz'))
        pageref_sz.set(qn('w:val'), '24')
        
        fld_begin = etree.SubElement(pageref_run, qn('w:fldChar'))
        fld_begin.set(qn('w:fldCharType'), 'begin')
        
        instr_run = etree.SubElement(entry_para, qn('w:r'))
        instr_rPr = etree.SubElement(instr_run, qn('w:rPr'))
        instr_rFonts = etree.SubElement(instr_rPr, qn('w:rFonts'))
        instr_rFonts.set(qn('w:ascii'), 'Times New Roman')
        instr_rFonts.set(qn('w:hAnsi'), 'Times New Roman')
        instr_rFonts.set(qn('w:eastAsia'), 'Times New Roman')
        instr_sz = etree.SubElement(instr_rPr, qn('w:sz'))
        instr_sz.set(qn('w:val'), '24')
        
        instr_text = etree.SubElement(instr_run, qn('w:instrText'))
        instr_text.set(qn('xml:space'), 'preserve')
        instr_text.text = f' PAGEREF {bookmark_name} \\h '
        
        fld_sep = etree.SubElement(instr_run, qn('w:fldChar'))
        fld_sep.set(qn('w:fldCharType'), 'separate')
        
        # Placeholder page number
        num_run = etree.SubElement(entry_para, qn('w:r'))
        num_rPr = etree.SubElement(num_run, qn('w:rPr'))
        num_rFonts = etree.SubElement(num_rPr, qn('w:rFonts'))
        num_rFonts.set(qn('w:ascii'), 'Times New Roman')
        num_rFonts.set(qn('w:hAnsi'), 'Times New Roman')
        num_rFonts.set(qn('w:eastAsia'), 'Times New Roman')
        num_sz = etree.SubElement(num_rPr, qn('w:sz'))
        num_sz.set(qn('w:val'), '24')
        
        num_t = etree.SubElement(num_run, qn('w:t'))
        num_t.text = '1'
        
        fld_end_run = etree.SubElement(entry_para, qn('w:r'))
        fld_end = etree.SubElement(fld_end_run, qn('w:fldChar'))
        fld_end.set(qn('w:fldCharType'), 'end')


def _extract_label(caption_text, label_type):
    """Extract just 'Figure N' or 'Table N' from a full caption string."""
    import re
    if label_type == 'Figure':
        m = re.search(r'(Figure\s+\d+)', caption_text, re.IGNORECASE)
    else:
        m = re.search(r'(Table\s+\d+)', caption_text, re.IGNORECASE)
    return m.group(1) if m else caption_text


def _make_pageref_run(parent_elem, bookmark_name):
    """Create a PAGEREF field run inside a parent XML element."""
    fld_begin_run = etree.SubElement(parent_elem, qn('w:r'))
    fld_begin = etree.SubElement(fld_begin_run, qn('w:fldChar'))
    fld_begin.set(qn('w:fldCharType'), 'begin')
    
    instr_run = etree.SubElement(parent_elem, qn('w:r'))
    instr_text = etree.SubElement(instr_run, qn('w:instrText'))
    instr_text.set(qn('xml:space'), 'preserve')
    instr_text.text = f' PAGEREF {bookmark_name} \\h '
    
    fld_sep_run = etree.SubElement(parent_elem, qn('w:r'))
    fld_sep = etree.SubElement(fld_sep_run, qn('w:fldChar'))
    fld_sep.set(qn('w:fldCharType'), 'separate')
    
    num_run = etree.SubElement(parent_elem, qn('w:r'))
    num_t = etree.SubElement(num_run, qn('w:t'))
    num_t.text = '1'
    
    fld_end_run = etree.SubElement(parent_elem, qn('w:r'))
    fld_end = etree.SubElement(fld_end_run, qn('w:fldChar'))
    fld_end.set(qn('w:fldCharType'), 'end')


def _build_list_table(doc, items, label_type):
    """
    Build a 2-column table: left column = 'Figure N' / 'Table N',
    right column = page number (PAGEREF field linked to bookmark).
    """
    from docx.shared import Inches as DocxInches
    
    n_rows = len(items) + 1  # +1 for header
    table = doc.add_table(rows=n_rows, cols=2)
    table.alignment = 1  # center
    
    # Add borders manually (pandoc-generated docx may not have 'Table Grid' style)
    tbl = table._element
    tblPr = tbl.find(qn('w:tblPr'))
    if tblPr is None:
        tblPr = etree.SubElement(tbl, qn('w:tblPr'))
        tbl.insert(0, tblPr)
    tblBorders = etree.SubElement(tblPr, qn('w:tblBorders'))
    for border_name in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
        border = etree.SubElement(tblBorders, qn(f'w:{border_name}'))
        border.set(qn('w:val'), 'single')
        border.set(qn('w:sz'), '4')
        border.set(qn('w:space'), '0')
        border.set(qn('w:color'), '000000')
    
    # Set column widths
    for row in table.rows:
        row.cells[0].width = DocxInches(4.0)
        row.cells[1].width = DocxInches(2.0)
    
    # Header row
    hdr_cell0 = table.rows[0].cells[0]
    hdr_cell1 = table.rows[0].cells[1]
    hdr_cell0.paragraphs[0].text = label_type
    hdr_cell1.paragraphs[0].text = 'Page No.'
    for cell in [hdr_cell0, hdr_cell1]:
        for para in cell.paragraphs:
            for run in para.runs:
                run.font.name = 'Times New Roman'
                run.font.size = Pt(12)
                run.font.bold = True
    
    # Data rows
    for i, (caption_text, bookmark_name) in enumerate(items):
        row = table.rows[i + 1]
        label = _extract_label(caption_text, label_type)
        
        # Left cell: label text
        left_cell = row.cells[0]
        left_para = left_cell.paragraphs[0]
        left_run = left_para.add_run(label)
        left_run.font.name = 'Times New Roman'
        left_run.font.size = Pt(12)
        
        # Right cell: PAGEREF field
        right_cell = row.cells[1]
        right_para = right_cell.paragraphs[0]
        right_para_elem = right_para._element
        
        _make_pageref_run(right_para_elem, bookmark_name)
        
        # Set font on the PAGEREF runs
        for run in right_para.runs:
            run.font.name = 'Times New Roman'
            run.font.size = Pt(12)
    
    return table._element


def add_table_of_contents(doc):
    """
    Insert an auto-updating Table of Contents at the beginning of the document,
    followed by a List of Figures page and a List of Tables page.
    When the user opens the file in Word and presses Ctrl+A then F9 (or right-click
    -> Update Field), all fields will populate with headings, figures, tables and
    their page numbers.
    """
    body = doc.element.body
    
    # Find the first paragraph element in the body
    first_para = None
    for child in body:
        if child.tag == qn('w:p'):
            first_para = child
            break
    
    if first_para is None:
        return
    
    # --- Step 1: Add bookmarks to figure and table captions ---
    figures, tables = _add_bookmarks_to_captions(doc)
    
    # --- Step 2: Build the front matter (TOC, LOF, LOT) ---
    # We'll create all elements first (appended at end of body), then move them
    # to before the first content paragraph.
    
    elements_to_move = []
    
    # === Table of Contents page ===
    toc_title = _make_title_paragraph(body, 'Table of Contents')
    elements_to_move.append(toc_title)
    
    # Empty paragraph for spacing
    toc_spacer = etree.SubElement(body, qn('w:p'))
    elements_to_move.append(toc_spacer)
    
    toc_field = _make_toc_field_paragraph(body, '\\o "1-3"')
    elements_to_move.append(toc_field)
    
    # Page break after TOC
    toc_pb = _make_page_break(body)
    elements_to_move.append(toc_pb)
    
    # === List of Figures page ===
    lof_title = _make_title_paragraph(body, 'List of Figures')
    elements_to_move.append(lof_title)
    
    lof_spacer = etree.SubElement(body, qn('w:p'))
    elements_to_move.append(lof_spacer)
    
    lof_table = _build_list_table(doc, figures, 'Figure')
    elements_to_move.append(lof_table)
    
    # Page break after LOF
    lof_pb = _make_page_break(body)
    elements_to_move.append(lof_pb)
    
    # === List of Tables page ===
    lot_title = _make_title_paragraph(body, 'List of Tables')
    elements_to_move.append(lot_title)
    
    lot_spacer = etree.SubElement(body, qn('w:p'))
    elements_to_move.append(lot_spacer)
    
    lot_table = _build_list_table(doc, tables, 'Table')
    elements_to_move.append(lot_table)
    
    # Page break after LOT
    lot_pb = _make_page_break(body)
    elements_to_move.append(lot_pb)
    
    # Move all front matter elements to before the first content paragraph
    for elem in elements_to_move:
        body.remove(elem)
        first_para.addprevious(elem)

def add_toc_and_page_numbers(docx_path):
    """Add Table of Contents and page numbers to the document."""
    doc = Document(docx_path)
    
    # Add page numbers to footer
    print("Adding page numbers to footer...")
    add_page_numbers(doc)
    
    # Add Table of Contents
    print("Adding Table of Contents...")
    add_table_of_contents(doc)
    
    doc.save(docx_path)

def md_to_docx(md_path, docx_path):
    """Full conversion pipeline: pandoc for content + post-process for fonts."""
    
    # Step 1: Convert with pandoc (handles math equations natively)
    print("Step 1: Converting with pandoc (LaTeX math -> Word OMML equations)...")
    if not convert_with_pandoc(md_path, docx_path):
        print("ERROR: Pandoc conversion failed!")
        return False
    
    # Step 2: Post-process to set Times New Roman 12pt
    print("Step 2: Setting Times New Roman 12pt throughout...")
    set_times_new_roman(docx_path)
    
    # Step 3: Add Table of Contents and page numbers
    print("Step 3: Adding Table of Contents and page numbers...")
    add_toc_and_page_numbers(docx_path)
    
    # Step 4: Verify
    doc = Document(docx_path)
    print(f"Done!")
    print(f"  Paragraphs: {len(doc.paragraphs)}")
    print(f"  Tables: {len(doc.tables)}")
    print(f"  Output: {docx_path}")
    print()
    print("NOTE: Open the file in Word and press Ctrl+A then F9 to update the")
    print("      Table of Contents with actual page numbers.")
    
    return True

if __name__ == '__main__':
    md_path = r'D:\project\biotech-evo2\backend\paper\methodology.md'
    paper_dir = r'D:\project\biotech-evo2\backend\paper'
    
    # Find the latest version number
    existing = glob.glob(os.path.join(paper_dir, 'methodology_v*.docx'))
    max_version = 22
    for f in existing:
        basename = os.path.basename(f)
        try:
            vnum = int(basename.replace('methodology_v', '').replace('.docx', ''))
            if vnum > max_version:
                max_version = vnum
        except ValueError:
            continue
    
    next_version = max_version + 1
    docx_path = os.path.join(paper_dir, f'methodology_v{next_version}.docx')
    
    print(f"Latest version: v{max_version}")
    print(f"Generating: v{next_version}")
    print()
    md_to_docx(md_path, docx_path)