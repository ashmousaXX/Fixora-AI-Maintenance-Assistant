import json
import re
from pathlib import Path
import pymupdf
import pdfplumber
import pytesseract
from PIL import Image
from config import (MANUALS,MANUALS_DIR,PROCESSED_DIR,)

import shutil
tesseract_path = shutil.which("tesseract")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

def clean_text(text):
    """
    Basic cleanup for extracted PDF text.
    """
    if not text:
        return ""
    text = text.replace("\u00ad", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def clean_cell(value):
    """
    Clean a PDF table cell.
    """
    if value is None:
        return ""

    value = str(value).replace("\n", " ")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\bpage(\d)",r"page \1", value,flags=re.IGNORECASE,)
    return value.strip()

def fix_medical_terms(text):
    """
    Normalize common medical terms.
    """
    if not text:
        return ""
    text = re.sub( r"\bSpO\s+2\b","SpO2",text,flags=re.IGNORECASE,)
    text = re.sub(r"\betCO\s+2\b","etCO2",text,flags=re.IGNORECASE,)
    text = re.sub(r"\bCO\s+2\b","CO2",text,flags=re.IGNORECASE,)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def detect_error_code(text):
    """
    Detect common medical-device error formats.
    Examples:
        E37
        E-37
        ERR37
        Error 37
        Error code 37
        Code 37
        Fault 37

    The current retrieval.py expects the numeric part.
    """
    patterns = [
        r"\bE[-\s]?(\d{1,5})\b",
        r"\bERR[-\s]?(\d{1,5})\b",
        r"\berror\s+code\s+(\d{1,5})\b",
        r"\berror\s+(\d{1,5})\b",
        r"\bcode\s+(\d{1,5})\b",
        r"\bfault\s+(\d{1,5})\b",
    ]

    for pattern in patterns:
        match = re.search(pattern,text,flags=re.IGNORECASE,)
        if match:
            return match.group(1)
    return None

def detect_chunk_type(text):
    """
    Assign a simple chunk type.
    """
    upper_text = text.upper()
    if any(
        term in upper_text
        for term in [ "DANGER","WARNING","CAUTION", ]
    ):
        return "safety"
    if detect_error_code(text) is not None:
        return "error_code"
    return "text"

def extract_with_pymupdf(pdf_path):
    """
    Try to extract text from all pages using PyMuPDF.

    Returns:
        list[dict]
    """

    pages = []
    try:
        document = pymupdf.open(
            pdf_path
        )
        for page_number, page in enumerate(
            document,
            start=1,
        ):
            try:
                text = page.get_text(
                    "text"
                ).strip()
            except Exception as error:
                print(
                    f"    PyMuPDF page "
                    f"{page_number} error: {error}"
                )
                text = ""
            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )
        document.close()

    except Exception as error:
        print(
            f"    PyMuPDF failed: {error}"
        )

        return []
    return pages

def extract_with_pdfplumber(pdf_path):
    """
    Try to extract text from all pages using pdfplumber.

    Returns:
        list[dict]
    """
    pages = []
    try:
        with pdfplumber.open(
            pdf_path
        ) as pdf:
            for page_number, page in enumerate(pdf.pages,start=1,):
                try:
                    text = (page.extract_text( x_tolerance=2,y_tolerance=2,)or "")
                except Exception as error:
                    print(
                        f"    pdfplumber page "
                        f"{page_number} error: {error}"
                    )
                    text = ""
                pages.append(
                    {
                        "page": page_number,
                        "text": text.strip(),
                    }
                )

    except Exception as error:
        print(f"  pdfplumber failed: {error}")
        return []
    return pages

# OCR extraction
def extract_with_ocr(pdf_path):
    """
    OCR fallback.

    Used only when normal text extraction fails.
    """
    pages = []
    print("  Starting OCR fallback...")

    try:
        document = pymupdf.open(
            pdf_path
        )
        for page_number, page in enumerate(document,start=1,):
            try:
                pix = page.get_pixmap(
                    matrix=pymupdf.Matrix(
                        2.0,
                        2.0,
                    ),
                    alpha=False,
                )
                image = Image.frombytes(
                    "RGB",
                    (
                        pix.width,
                        pix.height,
                    ),
                    pix.samples,
                )
                text = pytesseract.image_to_string(
                    image,
                    lang="eng",
                ).strip()
            except Exception as error:
                print(
                    f"  OCR page "
                    f"{page_number} error: {error}"
                )
                text = ""
            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )
            if page_number % 10 == 0:
                print(
                    f"    OCR processed "
                    f"{page_number} pages..."
                )
        document.close()

    except Exception as error:
        print(f" OCR failed: {error}")
        return []
    return pages

def extract_pdf_text(pdf_path):
    """
    Unified PDF text extraction pipeline.

    Order:
        1. PyMuPDF
        2. pdfplumber
        3. OCR

    The file is only passed to OCR when both normal
    extraction methods return no usable text.
    """
    print(f" Extracting: {pdf_path.name}")
    pages = extract_with_pymupdf(
        pdf_path
    )
    pymupdf_non_empty = sum(
        bool(page["text"].strip())
        for page in pages
    )
    if pymupdf_non_empty > 0:
        return pages
    
    print("  No usable PyMuPDF text.")
    print("  Trying pdfplumber...")
    
    pages = extract_with_pdfplumber(
        pdf_path
    )
    pdfplumber_non_empty = sum(
        bool(page["text"].strip())
        for page in pages
    )

    if pdfplumber_non_empty > 0:
        return pages
    print(" No usable pdfplumber text.")
    pages = extract_with_ocr(
        pdf_path
    )
    return pages

def load_device_inventory():
    """
    Load metadata generated by build_inventory.py.
    """
    inventory_path = (
        MANUALS_DIR.parent
        / "device_inventory.json"
    )
    if not inventory_path.exists():
        raise FileNotFoundError(
            f"Device inventory not found: "
            f"{inventory_path}"
        )
    with open(inventory_path,"r",encoding="utf-8",) as file:
        return json.load(file)

def get_stable_device_info(pdf_path, device_info):
    """
    Keep stable device IDs for manuals that are referenced
    directly by config.py.

    The inventory uses automatically generated IDs, while
    retrieval.py/rag.py use stable IDs such as:
        servo_ventilator
        sc6002xl
        philips_g40

    If a specialized manual exists in MANUALS, its stable
    ID is used. Otherwise the inventory entry is unchanged.
    """

    updated_info = dict(device_info)
    for stable_id in ("servo_ventilator","sc6002xl","philips_g40",):
        manual = MANUALS.get(stable_id)
        if not manual:
            continue

        configured_file = Path(
            manual["file"]
        ).name
        if pdf_path.name == configured_file:
            updated_info["device_id"] = stable_id
            updated_info["device"] = manual.get(
                "device",
                updated_info.get("device", pdf_path.stem),
            )
            updated_info["manufacturer"] = manual.get(
                "manufacturer",
                updated_info.get("manufacturer", "Unknown"),
            )
            break
    return updated_info

def make_generic_chunk(
    pdf_path,
    device_info,
    page_number,
    text,
    chunk_number,
):
    """
    Create a chunk compatible with retrieval.py.
    """
    text = fix_medical_terms( text)
    return {
        "chunk_id":
            f"{device_info['device_id']}"
            f"_p{page_number}"
            f"_c{chunk_number}",
        "device_id":
            device_info["device_id"],
        "device":
            device_info["device"],
        "manufacturer":
            device_info["manufacturer"],
        "page":
            page_number,
        "section":
            "General",
        "chunk_type":
            detect_chunk_type(
                text
            ),
        "error_code":
            detect_error_code(
                text
            ),
        "manual":
            pdf_path.name,
        "text":
            text,
    }

def extract_generic_chunks(
    pdf_path,
    device_info,
    pages=None,
):
    """
    Generic fallback parser.

    If pages are provided, reuse them.
    This avoids re-reading OCR PDFs.
    """
    if pages is None:
        pages = extract_pdf_text(
            pdf_path
        )
    chunks = []
    max_chars = 1200

    for page_data in pages:
        page_number = page_data["page"]
        raw_text = page_data["text"]
        if not raw_text.strip():
            continue

        text = clean_text(
            raw_text
        )
        if not text:
            continue
        
        parts = re.split(r"(?<=[.!?])\s+",text,)
        current_chunk = ""
        chunk_number = 0
        for part in parts:
            part = part.strip()
            if not part:
                continue
            candidate_length = (
                len(current_chunk)
                + len(part)
                + 1
            )

            if candidate_length <= max_chars:
                if current_chunk:
                    current_chunk += " "
                current_chunk += part

            else:
                if current_chunk:
                    chunk_number += 1
                    chunks.append(
                        make_generic_chunk(
                            pdf_path=pdf_path,
                            device_info=device_info,
                            page_number=page_number,
                            text=current_chunk,
                            chunk_number=chunk_number,
                        )
                    )
                current_chunk = part

        if current_chunk:
            chunk_number += 1
            chunks.append(
                make_generic_chunk(
                    pdf_path=pdf_path,
                    device_info=device_info,
                    page_number=page_number,
                    text=current_chunk,
                    chunk_number=chunk_number,
                )
            )
    return chunks

def extract_servo_error_chunks( pages,manual,):
    """
    Specialized parser for Servo technical error codes.
    """
    chunks = []
    inside_error_table = False

    for page in pages:
        page_number = page["page"]
        raw_text = page["text"]
        if not raw_text:
            continue

        text = clean_text(raw_text)
        if (
            "Technical error codes" in text
            and "Error code" in text
            and "Error message / Possible cause" in text
            and "Recommended action" in text
        ):
            inside_error_table = True
        if not inside_error_table:
            continue

        if (
            "Preventive maintenance" in text
            and "Technical error codes" not in text
        ):
            break
        pattern = r"""
        (?<!\d)
        (\d{1,5})
        \s+
        ([A-Z][A-Z0-9_ ]+?)
        (?=
            \s+\d+\.
            |
            \s+N/A
            |
            \s+\d{1,5}\s+[A-Z]
            |
            $
        )
        """
        matches = list(re.finditer(pattern,text,re.VERBOSE,))
        for index, match in enumerate(
            matches
        ):
            error_code = (
                match.group(1)
                .strip()
            )
            if error_code == "382":
                continue

            error_message = (match.group(2).strip())
            error_message = re.sub(r"_\s+","_",error_message,)
            start = match.end()

            if index + 1 < len(matches):
                end = (matches[index + 1].start())
            else:
                end = len(text)
            action_text = (text[start:end].strip())
            action_text = re.sub(r"^Recommended action\s*", "" , action_text,flags=re.IGNORECASE,)
            chunk_text = (
                f"Error code: {error_code}. "
                f"Error message / possible cause: "
                f"{error_message}. "
                f"Recommended action: "
                f"{action_text if action_text else 'Not specified.'}"
            )
            chunks.append(
                {
                    "device_id":
                        "servo_ventilator",
                    "device":
                        manual["device"],
                    "manufacturer":
                        manual["manufacturer"],
                    "page":
                        page_number,
                    "section":
                        "Technical error codes",
                    "chunk_type":
                        "error_code",
                    "error_code":
                        error_code,
                    "error_message":
                        error_message,
                    "recommended_action":
                        action_text,
                    "manual":
                        Path(manual["file"]).name,
                    "text":
                        chunk_text,
                }
            )
    return chunks

def extract_philips_troubleshooting_chunks(
    manual,
):
    """
    Specialized Philips G30/G40 troubleshooting parser.
    This parser is retained for the known G40 document.
    If it returns zero chunks, build_all_chunks()
    will use generic fallback.
    """
    chunks = []
    current_symptom = None

    try:
        with pdfplumber.open(
            manual["file"]
        ) as pdf:
            for page_number in range(
                45,
                51,
            ):
                if page_number > len(pdf.pages):
                    continue
                page = pdf.pages[page_number - 1]
                tables = page.extract_tables()
                for table_number, table in enumerate(
                    tables,
                    start=1,
                ):
                    if not table:
                        continue

                    current_symptom = None
                    section = "Troubleshooting"
                    if page_number in [45, 46]:
                        section = "Power Problems"

                    elif page_number == 47:
                        section = "Display Problems"

                    elif page_number == 48:
                        if table_number == 1:
                            section = "Alarm Problems"
                        elif table_number == 2:
                            section = "NIBP Problems"

                    elif page_number == 49:
                        if table_number == 1:
                            section = "NIBP Problems"
                        elif table_number == 2:
                            section = "Temperature Problems"

                    elif page_number == 50:
                        if table_number == 1:
                            section = "SpO2 Problems"
                        elif table_number == 2:
                            section = "etCO2 Problems"
                        elif table_number == 3:
                            section = "C.O. Problems"

                    for row in table:
                        if not row:
                            continue
                        if len(row) < 3:
                            continue
                        symptom = clean_cell(row[0])
                        cause = clean_cell(row[1])
                        action = clean_cell(row[2])

                        if (
                            symptom.lower()
                            == "symptom"
                            and
                            "possible cause"
                            in cause.lower()
                        ):
                            continue
                        if symptom:
                            current_symptom = (symptom)
                        if not current_symptom:
                            continue
                        if not cause or not action:
                            continue

                        symptom = fix_medical_terms(current_symptom)
                        cause = fix_medical_terms(cause)
                        action = fix_medical_terms(action)
                        chunk_text = (
                            f"Symptom: {symptom}. "
                            f"Possible cause: {cause}. "
                            f"Action: {action}"
                        )
                        chunks.append(
                            {
                                "device_id":
                                    "philips_g40",
                                "device":
                                    manual["device"],
                                "manufacturer":
                                    manual["manufacturer"],
                                "page":
                                    page_number,
                                "section":
                                    section,
                                "chunk_type":
                                    "troubleshooting",
                                "error_code":
                                    None,
                                "symptom":
                                    symptom,
                                "possible_cause":
                                    cause,
                                "action":
                                    action,
                                "manual":
                                    Path(manual["file"]).name,
                                "text":
                                    chunk_text,
                            }
                        )

    except Exception as error:
        print(
            f"    Philips specialized parser "
            f"failed: {error}"
        )
    return chunks

def extract_sc6002xl_troubleshooting_chunks(
    manual,
):
    """
    Specialized SC6002XL troubleshooting parser.
    """
    chunks = []
    try:
        with pdfplumber.open(
            manual["file"]
        ) as pdf:
            for page_number in range(
                73,
                79,
            ):
                if page_number > len(pdf.pages):
                    continue
                page = pdf.pages[page_number - 1]
                tables = page.extract_tables()
                
                for table_number, table in enumerate(
                    tables,
                    start=1,
                ):
                    if not table:
                        continue
                    section = "Troubleshooting"
                    if page_number in [73, 74]:
                        section = "Power Problems"
                    elif page_number == 75:
                        section_map = {
                            1: "Power-off Alarm Malfunction",
                            2: "Power-up Process Malfunction",
                            3: "Rotary Knob Malfunction",
                            4: "LCD Display Malfunction",
                        }
                        section = section_map.get(
                            table_number,
                            "Troubleshooting",
                        )
                    elif page_number == 76:
                        section_map = {
                            1: "LCD Display Malfunction",
                            2: "Fixed Key Malfunction",
                            3: "Alarm Malfunctions",
                        }
                        section = section_map.get(
                            table_number,
                            "Troubleshooting",
                        )
                    elif page_number == 77:
                        section_map = {
                            1: "NBP Malfunctions",
                            2: "etCO2 Malfunctions",
                        }
                        section = section_map.get(
                            table_number,
                            "Troubleshooting",
                        )
                    elif page_number == 78:
                        section = "Recorder Malfunctions"
                    for row in table:
                        if not row:
                            continue
                        if len(row) < 3:
                            continue
                        symptom = clean_cell(row[0])
                        cause = clean_cell(row[1])
                        action = clean_cell(row[2])
                        if not symptom:
                            continue
                        if not cause or not action:
                            continue
                        symptom_lower = (symptom.lower())
                        if (
                            symptom_lower
                            in ["conditions","symptom(s)", "symptoms",] and
                            "possible cause"
                            in cause.lower()
                        ):
                            continue
                        symptom = fix_medical_terms(symptom)
                        cause = fix_medical_terms(cause)
                        action = fix_medical_terms(action)
                        chunk_text = (
                            f"Symptom or condition: "
                            f"{symptom}. "
                            f"Possible cause: "
                            f"{cause}. "
                            f"Troubleshooting and remedial action: "
                            f"{action}"
                        )

                        chunks.append(
                            {
                                "device_id":
                                    "sc6002xl",
                                "device":
                                    manual["device"],
                                "manufacturer":
                                    manual["manufacturer"],
                                "page":
                                    page_number,
                                "section":
                                    section,
                                "chunk_type":
                                    "troubleshooting",
                                "error_code":
                                    None,
                                "symptom":
                                    symptom,
                                "possible_cause":
                                    cause,
                                "action":
                                    action,
                                "manual":
                                    Path(manual["file"]).name,
                                "text":
                                    chunk_text,
                            }
                        )

    except Exception as error:
        print(
            f"    SC6002XL specialized parser "
            f"failed: {error}"
        )
    return chunks

def build_all_chunks():

    all_chunks = []
    inventory = load_device_inventory()
    pdf_files = sorted(MANUALS_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files.")

    for index, pdf_path in enumerate(
        pdf_files,
        start=1,
    ):
        print()
        print(
            f"[{index}/{len(pdf_files)}] "
            f"{pdf_path.name}"
        )

        device_info = inventory.get(pdf_path.name)
        if device_info is None:
            print(
                "  WARNING: No inventory entry. "
                "Skipping."
            )
            continue

        device_info = get_stable_device_info(pdf_path,device_info,)
        try:
            chunks = []
            if ("servo_ventilator" in MANUALS and pdf_path.name == Path(
                    MANUALS[
                        "servo_ventilator"
                    ]["file"]
                ).name
            ):
                print( "  Using Servo specialized parser...")
                pages = extract_pdf_text(
                    pdf_path
                )
                chunks = (
                    extract_servo_error_chunks(
                        pages,
                        MANUALS["servo_ventilator"],
                    )
                )
                if not chunks:
                    print(
                        "  Specialized Servo parser "
                        "returned 0 chunks."
                    )
                    print(
                        "  Using generic fallback "
                        "with already extracted pages..."
                    )
                    chunks = extract_generic_chunks(
                        pdf_path,
                        device_info,
                        pages=pages,
                    )
            elif ( "philips_g40" in MANUALS and pdf_path.name ==
                Path(
                    MANUALS[
                        "philips_g40"
                    ]["file"]
                ).name
            ):
                print("Using Philips G40 specialized parser...")
                chunks = (
                    extract_philips_troubleshooting_chunks(
                        MANUALS[
                            "philips_g40"
                        ]
                    )
                )
                if not chunks:
                    print(
                        "  Specialized Philips parser "
                        "returned 0 chunks."
                    )
                    print(" Using generic fallback...")
                    chunks = extract_generic_chunks(pdf_path,device_info,)

            elif ( "sc6002xl" in MANUALS and pdf_path.name ==
                Path(
                    MANUALS[
                        "sc6002xl"
                    ]["file"]
                ).name
            ):
                print("Using SC6002XL specialized parser...")
                chunks = (
                    extract_sc6002xl_troubleshooting_chunks(
                        MANUALS[
                            "sc6002xl"
                        ]
                    )
                )
                if not chunks:
                    print(
                        "  Specialized SC6002XL parser "
                        "returned 0 chunks."
                    )
                    print(" Using generic fallback...")
                    chunks = extract_generic_chunks(pdf_path,device_info,)
            else:
                print("  Using generic parser...")
                chunks = extract_generic_chunks(pdf_path,device_info,)
            all_chunks.extend(chunks)
            print(
                f"  Chunks created: "
                f"{len(chunks)}"
            )

        except Exception as error:
            print(
                f"  ERROR while processing "
                f"{pdf_path.name}: {error}"
            )
    for index, chunk in enumerate(all_chunks,start=1,):
        chunk["chunk_id"] = (
            f"chunk_{index:05d}"
        )
    return all_chunks

def validate_chunks(chunks):

    print()
    print("=" * 70)
    print("CHUNK VALIDATION")
    print("=" * 70)

    required_fields = [
        "chunk_id",
        "device_id",
        "device",
        "manufacturer",
        "page",
        "section",
        "chunk_type",
        "text",
    ]
    missing_field_count = 0
    empty_text_count = 0
    duplicate_ids = 0

    seen_ids = set()
    for chunk in chunks:
        for field in required_fields:
            if field not in chunk:
                print(
                    f"Missing field '{field}' "
                    f"in chunk "
                    f"{chunk.get('chunk_id')}"
                )
                missing_field_count += 1

        if not chunk.get(
            "text",
            "",
        ).strip():
            empty_text_count += 1
        chunk_id = chunk.get(
            "chunk_id"
        )

        if chunk_id in seen_ids:
            duplicate_ids += 1
        seen_ids.add( chunk_id)
    print()
    print(
        f"Missing fields: "
        f"{missing_field_count}"
    )
    print(
        f"Empty texts: "
        f"{empty_text_count}"
    )
    print(
        f"Duplicate IDs: "
        f"{duplicate_ids}"
    )
    print()

    if (
        missing_field_count == 0
        and empty_text_count == 0
        and duplicate_ids == 0
    ):
        print("Validation passed.")
    else:
        print( "Validation found problems.")
    print("=" * 70)

def save_chunks_to_json(chunks):
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_file = (
        PROCESSED_DIR
        / "maintai_chunks.json"
    )

    with open( output_file,"w",encoding="utf-8",) as file:
        json.dump(
            chunks,
            file,
            indent=4,
            ensure_ascii=False,
        )
    print()
    print( f"Saved {len(chunks)} chunks")
    print( f"Output file: {output_file}")

def run_preprocessing():

    print("=" * 70)
    print("MAINTAI PREPROCESSING")
    print("=" * 70)
    chunks = build_all_chunks()
    print()
    print(
        f"Total chunks created: "
        f"{len(chunks)}"
    )
    validate_chunks(chunks)
    save_chunks_to_json(chunks)

if __name__ == "__main__":
    run_preprocessing()
