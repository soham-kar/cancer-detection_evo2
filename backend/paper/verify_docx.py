"""Verify the generated methodology.docx has proper fonts and math."""
import zipfile
import re

z = zipfile.ZipFile("methodology.docx", "r")
names = z.namelist()

# Check for math-related files
math_files = [n for n in names if "math" in n.lower() or "equation" in n.lower() or "omml" in n.lower()]
print("Math-related files:", math_files)

# Check document content
doc = z.read("word/document.xml").decode("utf-8")
fonts = set(re.findall(r'w:ascii="([^"]+)"', doc))
print("Fonts in document:", fonts)

# Count OMML math elements (native Word equations)
math_count = doc.count("<m:oMath")
print("OMML math elements:", math_count)

# Check styles
styles = z.read("word/styles.xml").decode("utf-8")
style_fonts = set(re.findall(r'w:ascii="([^"]+)"', styles))
print("Fonts in styles:", style_fonts)

# Check default font in docDefaults
doc_defaults_match = re.search(r'<w:docDefaults>.*?<w:rFonts[^>]*>', styles, re.DOTALL)
if doc_defaults_match:
    print("Default font setting:", doc_defaults_match.group()[:200])

print("Total files in docx:", len(names))
z.close()