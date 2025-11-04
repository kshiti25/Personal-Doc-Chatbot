# docling_loader.py
import os
from typing import List
from langchain_core.documents import Document
from docling.document_converter import DocumentConverter

class DoclingLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.converter = DocumentConverter()

    def load(self) -> List[Document]:
        result = self.converter.convert(self.file_path)
        text = result.document.export_to_text()
        return [
            Document(
                page_content=text,
                metadata={"source": os.path.basename(self.file_path)}
            )
        ]
