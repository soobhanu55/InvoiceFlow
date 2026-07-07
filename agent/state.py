"""Pydantic schemas + the shared LangGraph state for the invoice pipeline."""
from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict

from pydantic import BaseModel, Field

DocType = Literal["invoice", "receipt", "credit_note", "unknown"]
Status = Literal["pending", "auto_approved", "approved", "needs_review", "rejected"]
HumanDecision = Literal["approve", "edit", "reject"]


class DocumentClassification(BaseModel):
    """Output of the classification node's LLM call."""

    doc_type: DocType = Field(description="Type of document")
    vendor_name: Optional[str] = Field(default=None, description="Vendor/supplier name as printed on the document")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence in this classification")


class LineItem(BaseModel):
    sku: Optional[str] = Field(default=None, description="Vendor SKU / item code if present")
    description: str
    quantity: float
    unit_price: float
    line_total: float


class ExtractedInvoice(BaseModel):
    """Output of the extraction node's LLM call."""

    invoice_number: Optional[str] = None
    po_number: Optional[str] = None
    vendor_name: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    currency: str = "USD"
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: Optional[float] = None
    tax_rate: Optional[float] = Field(default=None, description="e.g. 0.08 for 8%")
    tax_amount: Optional[float] = None
    total: Optional[float] = None
    extraction_confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ValidationIssue(BaseModel):
    reason_code: str
    message: str
    severity: Literal["error", "warning"] = "error"


class MatchLineItemResult(BaseModel):
    sku: Optional[str]
    description: str
    invoice_qty: float
    invoice_unit_price: float
    po_qty: Optional[float] = None
    po_unit_price: Optional[float] = None
    status: Literal["match", "qty_mismatch", "price_mismatch", "not_in_po", "not_in_catalog"]
    detail: Optional[str] = None


class MatchResult(BaseModel):
    po_found: bool
    po_status: Optional[str] = None
    vendor_verified: bool = False
    line_item_results: list[MatchLineItemResult] = Field(default_factory=list)
    po_total: Optional[float] = None
    invoice_total_vs_po_delta: Optional[float] = None
    all_matched: bool = False


class InvoiceState(TypedDict, total=False):
    invoice_id: str
    file_path: str

    raw_text: str
    ocr_lines: list[str]
    ocr_confidence: float

    doc_type: DocType
    vendor_name: Optional[str]
    vendor_id: Optional[str]
    classification_confidence: float

    # NOTE: stored as plain dicts (via .model_dump()), not live pydantic
    # instances -- LangGraph's sqlite checkpointer serializes state with
    # msgpack and warns/blocks on unregistered custom types. Nodes that
    # need the pydantic validation/behavior reconstruct it on the way in
    # (e.g. ExtractedInvoice(**state["extracted"])) and dump it back to a
    # dict on the way out.
    extracted: dict[str, Any]
    extraction_confidence: float

    validation_issues: list[dict[str, Any]]
    validation_passed: bool
    po_lookup: Optional[dict[str, Any]]

    match_result: dict[str, Any]

    overall_confidence: float
    status: Status

    human_decision: Optional[HumanDecision]
    human_corrections: Optional[dict[str, Any]]

    audit_log: list[str]
