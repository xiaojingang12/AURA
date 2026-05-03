import argparse
import json
import os
from contextlib import suppress
from pathlib import Path

import torch
from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.data.data_reader_writer import FileBasedDataReader, FileBasedDataWriter
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze


def extract_title_from_content(content):
    """Extract a title from the first markdown heading in PDF-derived text."""
    for line in content.split("\n")[:20]:
        line = line.strip()
        if line.startswith("#"):
            return line[1:].strip()
    return "Unknown Title"


def process_pdf_with_magic_pdf(pdf_path, output_dir):
    """Convert a single PDF to markdown text with magic-pdf."""
    output_dir = Path(output_dir)
    local_image_dir = output_dir / "images"
    local_md_dir = output_dir

    local_image_dir.mkdir(parents=True, exist_ok=True)
    image_writer = FileBasedDataWriter(str(local_image_dir))
    md_writer = FileBasedDataWriter(str(local_md_dir))

    reader = FileBasedDataReader("")
    pdf_bytes = reader.read(str(pdf_path))

    dataset = PymuDocDataset(pdf_bytes)
    if dataset.classify() == SupportedPdfParseMethod.OCR:
        infer_result = dataset.apply(doc_analyze, ocr=True)
        pipe_result = infer_result.pipe_ocr_mode(image_writer)
    else:
        infer_result = dataset.apply(doc_analyze, ocr=False)
        pipe_result = infer_result.pipe_txt_mode(image_writer)

    md_filename = f"{Path(pdf_path).stem}.md"
    pipe_result.dump_md(md_writer, md_filename, str(local_image_dir))
    md_file_path = local_md_dir / md_filename
    content = md_file_path.read_text(encoding="utf-8")

    with suppress(Exception):
        md_file_path.unlink()

    return content


def check_gpu_availability():
    """Report CUDA availability."""
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"GPU count: {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
    else:
        print("Warning: no GPU detected; running on CPU")
    return cuda_available


def process_pdfs_to_json_with_magic_pdf(folder_path, output_json_path, output_dir):
    """Process all PDFs in a folder and save extracted content as JSON."""
    gpu_available = check_gpu_availability()
    device = torch.device("cuda" if gpu_available else "cpu")
    print(f"Current device: {device}")

    results = []
    folder_path = Path(folder_path)
    for pdf_path in sorted(folder_path.glob("*.pdf")):
        try:
            print(f"Processing: {pdf_path.name}")
            content = process_pdf_with_magic_pdf(pdf_path, output_dir)
            title = extract_title_from_content(content)
            results.append({"title": title, "context": content})
            print(f"Processed: {pdf_path.name}, title: {title[:50]}...")
        except Exception as exc:
            print(f"Error processing {pdf_path.name}: {exc}")

    output_json_path = Path(output_json_path)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Processing complete. Total PDFs processed: {len(results)}")
    print(f"Results saved to: {output_json_path}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Extract text from a folder of PDFs with magic-pdf.")
    parser.add_argument("--input-folder", required=True, help="Folder containing PDF files.")
    parser.add_argument("--output-json", required=True, help="Output JSON file.")
    parser.add_argument("--work-dir", default="output/pdf_extract", help="Temporary markdown/image output directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.input_folder):
        raise SystemExit(f"Input folder does not exist: {args.input_folder}")

    processed_data = process_pdfs_to_json_with_magic_pdf(args.input_folder, args.output_json, args.work_dir)
    print("\nSample processed PDFs:")
    for index, item in enumerate(processed_data[:3]):
        print(f"\nPDF {index + 1}:")
        print(f"  Title: {item['title'][:100]}...")
        print(f"  Content length: {len(item['context'])} characters")


if __name__ == "__main__":
    main()
