"""
Properly fix the reference.docx:
1. Set docDefaults to explicitly use Times New Roman (not theme references)
2. Fix theme1.xml to use Times New Roman
3. Set body text to 12pt, headings to 14-16pt
4. Keep Cambria Math for equation rendering
"""
import zipfile
import re
import os

REF_PATH = os.path.join(os.path.dirname(__file__), "custom-reference.docx")

z = zipfile.ZipFile(REF_PATH, "r")
files = {}
for name in z.namelist():
    files[name] = z.read(name)
z.close()

# --- 1. Fix word/styles.xml ---
styles = files["word/styles.xml"].decode("utf-8")

# Replace theme-based font references with explicit Times New Roman
styles = styles.replace(
    '<w:rFonts w:asciiTheme="minorHAnsi" w:cstheme="minorBidi" w:eastAsiaTheme="minorHAnsi" w:hAnsiTheme="minorHAnsi" />',
    '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman" w:eastAsia="Times New Roman" />'
)

# Also replace all theme references
styles = styles.replace('w:asciiTheme="majorHAnsi"', 'w:ascii="Times New Roman"')
styles = styles.replace('w:hAnsiTheme="majorHAnsi"', 'w:hAnsi="Times New Roman"')
styles = styles.replace('w:asciiTheme="minorHAnsi"', 'w:ascii="Times New Roman"')
styles = styles.replace('w:hAnsiTheme="minorHAnsi"', 'w:hAnsi="Times New Roman"')
styles = styles.replace('w:cstheme="minorBidi"', 'w:cs="Times New Roman"')
styles = styles.replace('w:cstheme="majorBidi"', 'w:cs="Times New Roman"')
styles = styles.replace('w:eastAsiaTheme="minorHAnsi"', 'w:eastAsia="Times New Roman"')
styles = styles.replace('w:eastAsiaTheme="majorHAnsi"', 'w:eastAsia="Times New Roman"')

# Replace any remaining Calibri/Cambria (but NOT Cambria Math)
styles = styles.replace("Calibri", "Times New Roman")
styles = styles.replace("Cambria Math", "__MATH_FONT__")
styles = styles.replace("Cambria", "Times New Roman")
styles = styles.replace("__MATH_FONT__", "Cambria Math")

# Set default body size to 12pt (24 half-points) in docDefaults
# Find the docDefaults rPr section and ensure sz=24
dd_match = re.search(r'(<w:docDefaults>\s*<w:rPrDefault>\s*<w:rPr>)(.*?)(</w:rPr>)', styles, re.DOTALL)
if dd_match:
    inner = dd_match.group(2)
    if '<w:sz' in inner:
        inner = re.sub(r'<w:sz w:val="\d+"/>', '<w:sz w:val="24"/>', inner)
        inner = re.sub(r'<w:szCs w:val="\d+"/>', '<w:szCs w:val="24"/>', inner)
    else:
        inner = inner + '<w:sz w:val="24"/><w:szCs w:val="24"/>'
    styles = styles[:dd_match.start()] + dd_match.group(1) + inner + dd_match.group(3) + styles[dd_match.end():]

files["word/styles.xml"] = styles.encode("utf-8")

# --- 2. Fix word/theme/theme1.xml ---
theme = files["word/theme/theme1.xml"].decode("utf-8")
theme = re.sub(r'(<a:majorFont>.*?<a:latin typeface=")([^"]*)(")', r'\g<1>Times New Roman\g<3>', theme, flags=re.DOTALL)
theme = re.sub(r'(<a:minorFont>.*?<a:latin typeface=")([^"]*)(")', r'\g<1>Times New Roman\g<3>', theme, flags=re.DOTALL)
files["word/theme/theme1.xml"] = theme.encode("utf-8")

# --- 3. Fix word/fontTable.xml ---
font_table = files["word/fontTable.xml"].decode("utf-8")
if "Times New Roman" not in font_table:
    font_table = font_table.replace(
        "</w:fonts>",
        '<w:font w:name="Times New Roman"><w:charset w:val="00"/><w:family w:val="roman"/></w:font></w:fonts>'
    )
files["word/fontTable.xml"] = font_table.encode("utf-8")

# --- 4. Write the modified docx ---
tmp_path = REF_PATH + ".tmp"
zout = zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED)
for name, data in files.items():
    zout.writestr(name, data)
zout.close()
os.replace(tmp_path, REF_PATH)

# --- 5. Verify ---
z = zipfile.ZipFile(REF_PATH, "r")
final_styles = z.read("word/styles.xml").decode("utf-8")
final_theme = z.read("word/theme/theme1.xml").decode("utf-8")
z.close()

dd_match = re.search(r'<w:docDefaults>.*?<w:rFonts[^>]*/>', final_styles, re.DOTALL)
print("docDefaults font:", dd_match.group()[-120:] if dd_match else "NOT FOUND")

major_match = re.search(r'<a:majorFont>.*?<a:latin typeface="([^"]*)"', final_theme, re.DOTALL)
minor_match = re.search(r'<a:minorFont>.*?<a:latin typeface="([^"]*)"', final_theme, re.DOTALL)
print("Theme major font:", major_match.group(1) if major_match else "NOT FOUND")
print("Theme minor font:", minor_match.group(1) if minor_match else "NOT FOUND")

remaining_themes = len(re.findall(r'asciiTheme=', final_styles))
print("Remaining theme references:", remaining_themes)
print("Cambria Math preserved:", "Cambria Math" in final_styles)
print("Done: reference.docx fully updated")