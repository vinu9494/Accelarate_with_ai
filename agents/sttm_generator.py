"""STTM generation agent — unified autonomous ReAct version.

A single autonomous STTM agent holds four tools:
  - inspect_context_tool       : previews source data context for any layer
  - generate_bronze_sttm_tool  : generates Bronze ingestion rules
  - generate_silver_sttm_tool  : generates Silver cleansing rules
  - generate_gold_sttm_tool    : generates Gold materialisation rules

The orchestrator sends a goal stating which STTM to generate. The agent
inspects the relevant context, decides which generation tool matches the
request, executes it, and returns the saved STTM CSV path.

Business intent is consumed ONLY by Gold STTM generation. Bronze and Silver
are intent-agnostic — Bronze maps every source column as-is, Silver applies
standard cleansing rules to every Bronze column.

I/O contract:
    generate_bronze_sttm(profile_path, run_id, task_description) -> str
    generate_silver_sttm(bronze_output_paths, bronze_sttm_path, run_id, task_description) -> str
    generate_gold_sttm(silver_output_paths, silver_sttm_path, business_intent, run_id, task_description) -> str
"""

import json
import os
import pandas as pd
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from core.config import STTM_DIR, ANTHROPIC_API_KEY, ANTHROPIC_MODEL, LLM_MAX_TOKENS
from core.llm_retry import invoke_with_retry
from core.audit import AuditLogger
from core.observability import AgentTrace


# ---------------------------------------------------------------------------
# Unified autonomous STTM agent prompt
# ---------------------------------------------------------------------------

STTM_AGENT_PROMPT = """You are a Data Engineering Architect generating Source-to-Target Mappings (STTM) for Medallion pipelines.

Call inspect_context_tool first, then call the ONE generation tool that matches the requested layer. Tools take no arguments.

Row fields: source_schema, source_table, source_column, target_schema, target_table, target_column, transformation_type, transformation_logic.

Bronze: "Direct" pass-through, "Indirect" for renamed/cast; add _load_timestamp and _source_file rows; no surrogate key.
Silver: first row = surrogate key (pk_<stem>_silver_id); apply null handling, type casting, date standardisation per column; id columns get type cast only.
Gold: first row = surrogate key (pk_gold_id); join Silver tables on key columns; shape tables for the business intent.

Bronze/Silver are intent-agnostic. Gold is intent-driven."""


# ---------------------------------------------------------------------------
# Pure Python context helpers — no LLM
# ---------------------------------------------------------------------------

def _prepare_bronze_context(profile_path: str) -> dict:
    """Read the dataset profile JSON produced by the profiler."""
    with open(profile_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _prepare_silver_context(bronze_output_paths: list[str], bronze_sttm_path: str) -> list[dict]:
    """Load Bronze Parquet metadata filtered to STTM-approved columns."""
    try:
        sttm_df = pd.read_csv(bronze_sttm_path)
    except Exception as e:
        raise ValueError(f"Failed to read Bronze STTM file '{bronze_sttm_path}': {e}")
    if "target_column" not in sttm_df.columns:
        raise ValueError(
            f"Bronze STTM file '{bronze_sttm_path}' missing required column 'target_column'. "
            f"Found columns: {list(sttm_df.columns)}. Check that the STTM generator produced a valid CSV with 'target_column'."
        )
    approved_cols = set(sttm_df.fillna("")["target_column"].unique())
    result = []
    for bp in bronze_output_paths:
        df = pd.read_parquet(bp)
        kept = [c for c in df.columns if c in approved_cols or c.startswith("_")]
        df = df[kept] if kept else df.iloc[:, :0]
        result.append({
            "filename": os.path.basename(bp),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "sample": df.head(5).to_dict(orient="records"),
        })
    return result


def _prepare_gold_context(silver_output_paths: list[str], silver_sttm_path: str) -> list[dict]:
    """Load Silver Parquet metadata filtered to STTM-approved columns."""
    try:
        sttm_df = pd.read_csv(silver_sttm_path)
    except Exception as e:
        raise ValueError(f"Failed to read Silver STTM file '{silver_sttm_path}': {e}")
    if "target_column" not in sttm_df.columns:
        raise ValueError(
            f"Silver STTM file '{silver_sttm_path}' missing required column 'target_column'. "
            f"Found columns: {list(sttm_df.columns)}. Check that the STTM generator produced a valid CSV with 'target_column'."
        )
    approved_cols = set(sttm_df.fillna("")["target_column"].unique())
    result = []
    for sp in silver_output_paths:
        df = pd.read_parquet(sp)
        kept = [c for c in df.columns if c in approved_cols or c.startswith("_")]
        df = df[kept] if kept else df.iloc[:, :0]
        result.append({
            "filename": os.path.basename(sp),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "sample": df.head(5).to_dict(orient="records"),
        })
    return result


def _generate_bronze_sttm_deterministic(profile_path: str, run_id: str) -> str:
    """Generate Bronze STTM from profile JSON without LLM — pure pass-through, always correct.

    Bronze is mechanical: every source column maps 1:1 to same-named target column.
    No LLM judgment is needed or wanted here.
    """
    context = _prepare_bronze_context(profile_path)
    rows = []
    for tbl_name, tbl_data in context.get("datasets", context.get("tables", {})).items():
        file_path = tbl_data.get("file", "")
        for col_name in tbl_data.get("columns", {}).keys():
            rows.append({
                "source_schema": "landing",
                "source_table": tbl_name,
                "source_column": col_name,
                "target_schema": "bronze",
                "target_table": tbl_name,
                "target_column": col_name,
                "transformation_type": "Direct",
                "transformation_logic": "Pass-through",
            })
        rows.append({
            "source_schema": "landing",
            "source_table": tbl_name,
            "source_column": "",
            "target_schema": "bronze",
            "target_table": tbl_name,
            "target_column": "_load_timestamp",
            "transformation_type": "Indirect",
            "transformation_logic": "Current timestamp",
        })
        rows.append({
            "source_schema": "landing",
            "source_table": tbl_name,
            "source_column": "",
            "target_schema": "bronze",
            "target_table": tbl_name,
            "target_column": "_source_file",
            "transformation_type": "Indirect",
            "transformation_logic": file_path,
        })
    sttm_path = str(STTM_DIR / f"sttm_bronze_{run_id[:8]}.csv")
    pd.DataFrame(rows).to_csv(sttm_path, index=False)
    print(f"[STTM] Bronze STTM saved (deterministic): {sttm_path} ({len(rows)} rows)")
    return sttm_path


def _generate_silver_sttm_deterministic(bronze_output_paths: list[str], run_id: str) -> str:
    """Generate Silver STTM from Bronze parquets without LLM — rule-based cleansing.

    Rules applied per dtype:
    - _meta columns (_load_timestamp, _source_file): Direct pass-through
    - *_date* / *_time* columns: standardize to ISO 8601
    - *_id columns: pass-through cast to VARCHAR
    - string/object dtype: COALESCE null to empty string
    - numeric dtype: COALESCE null to 0
    """
    rows = []
    for bronze_path in bronze_output_paths:
        df = pd.read_parquet(bronze_path)
        filename = os.path.basename(bronze_path)
        stem = filename.replace("_bronze.parquet", "").replace(".parquet", "")
        silver_table = f"{stem}_silver"

        rows.append({
            "source_schema": "bronze",
            "source_table": filename,
            "source_column": "",
            "target_schema": "silver",
            "target_table": silver_table,
            "target_column": f"pk_{stem}_silver_id",
            "transformation_type": "Indirect",
            "transformation_logic": "Auto-generated sequential surrogate primary key starting from 1",
        })

        for col in df.columns:
            col_lower = col.lower()
            dtype = str(df[col].dtype)

            if col.startswith("_"):
                t_type = "Direct"
                logic = "Pass-through"
            elif "date" in col_lower or "time" in col_lower:
                t_type = "Indirect"
                logic = "Standardize date format to ISO 8601 (YYYY-MM-DD); retain NULL as NULL"
            elif col_lower.endswith("_id") or col_lower == "id":
                t_type = "Direct"
                logic = "Cast to VARCHAR; pass through as-is; retain NULL"
            elif dtype in ("object", "string"):
                t_type = "Indirect"
                logic = "COALESCE(column, '') — replace NULL with empty string"
            elif "int" in dtype or "float" in dtype:
                t_type = "Indirect"
                logic = "COALESCE(column, 0) — replace NULL with 0; retain numeric precision"
            else:
                t_type = "Direct"
                logic = "Pass-through"

            rows.append({
                "source_schema": "bronze",
                "source_table": filename,
                "source_column": col,
                "target_schema": "silver",
                "target_table": silver_table,
                "target_column": col,
                "transformation_type": t_type,
                "transformation_logic": logic,
            })

    sttm_path = str(STTM_DIR / f"sttm_silver_{run_id[:8]}.csv")
    pd.DataFrame(rows).to_csv(sttm_path, index=False)
    print(f"[STTM] Silver STTM saved (deterministic): {sttm_path} ({len(rows)} rows)")
    return sttm_path


def _extract_sttm_rows(result: dict) -> list[dict]:
    """Scan agent message history (reverse order) for a JSON array of STTM rows."""
    for msg in reversed(result.get("messages", [])):
        content = getattr(msg, "content", "")
        if not isinstance(content, str):
            continue
        text = content
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        text = text.strip()
        start, end = text.find("["), text.rfind("]")
        if start == -1 or end == -1:
            continue
        try:
            rows = json.loads(text[start: end + 1])
            if isinstance(rows, list) and rows:
                return rows
        except (json.JSONDecodeError, ValueError):
            continue
    return []


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _make_llm():
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(api_key=ANTHROPIC_API_KEY, model=ANTHROPIC_MODEL, temperature=0, max_tokens=LLM_MAX_TOKENS)


# ---------------------------------------------------------------------------
# Unified tool factory — all 4 tools built from the caller's context
# ---------------------------------------------------------------------------

def _make_sttm_tools(
    profile_path: str | None,
    bronze_output_paths: list[str] | None,
    bronze_sttm_path: str | None,
    silver_output_paths: list[str] | None,
    silver_sttm_path: str | None,
    business_intent: str | None,
    run_id: str,
    scratchpad: dict,
):
    """Build all four STTM tools bound to the caller's context via closure.

    Only the context relevant to the requested layer will be populated; the
    others will be None and the agent should not call those generation tools.
    """

    @tool
    def inspect_context_tool(confirmation: str = "execute") -> str:
        """Preview the source data context — returns compact table/column listing only.
        Call this FIRST. The generation tools have the full context internally.
        """
        if profile_path:
            try:
                context = _prepare_bronze_context(profile_path)
                lines = []
                for tbl, tbl_data in context.get("datasets", context.get("tables", {})).items():
                    cols = list(tbl_data.get("columns", {}).keys())
                    lines.append(f"Table: {tbl} | columns: {', '.join(cols)}")
                return "layer: bronze\n" + "\n".join(lines)
            except Exception as e:
                return f"layer: bronze | error reading profile: {e}"
        if bronze_output_paths and bronze_sttm_path:
            try:
                context = _prepare_silver_context(bronze_output_paths, bronze_sttm_path)
                lines = [f"Table: {t['filename']} | columns: {', '.join(t.get('columns', []))}" for t in context]
                return "layer: silver\n" + "\n".join(lines)
            except Exception as e:
                return f"layer: silver | error: {e}"
        if silver_output_paths and silver_sttm_path:
            try:
                context = _prepare_gold_context(silver_output_paths, silver_sttm_path)
                lines = [f"Table: {t['filename']} | columns: {', '.join(t.get('columns', []))}" for t in context]
                return "layer: gold\n" + "\n".join(lines)
            except Exception as e:
                return f"layer: gold | error: {e}"
        return "error: no source context available"

    @tool
    def generate_bronze_sttm_tool(confirmation: str = "execute") -> str:
        """Generate Bronze STTM. Returns JSON with sttm_path and row_count."""
        if not profile_path:
            return json.dumps({"error": "No profile_path available for Bronze STTM"})

        context = _prepare_bronze_context(profile_path)

        # Build compact column listing to avoid dumping full profile JSON
        try:
            tables_summary = []
            for tbl_name, tbl_data in context.get("datasets", context.get("tables", {})).items():
                cols = list(tbl_data.get("columns", {}).keys())
                tables_summary.append(f"Table: {tbl_name} | columns: {', '.join(cols)}")
            context_summary = "\n".join(tables_summary) or json.dumps(context, default=str)[:2000]
        except Exception:
            context_summary = json.dumps(context, default=str)[:2000]

        inner_prompt = (
            "Generate a complete Bronze STTM JSON array. Map EVERY column listed below.\n"
            f"Source tables:\n{context_summary}\n\n"
            "CRITICAL RULES — follow exactly:\n"
            "1. source_column AND target_column MUST be the EXACT column name shown in the listing above. "
            "Do NOT invent, rename, abbreviate, or substitute any column name. "
            "Examples of what NOT to do: if the source has 'total_amount' do NOT write 'sale_amount', 'amount', or 'revenue'; "
            "if the source has 'transaction_id' do NOT write 'sale_id' or 'id'; "
            "if the source has 'region' do NOT write 'location' or 'area'.\n"
            "2. Map EVERY column in the listing — do not skip any column.\n"
            "3. All Bronze mappings are Direct pass-through: transformation_type='Direct', transformation_logic='Pass-through'.\n"
            "4. Add one _load_timestamp row and one _source_file row per table "
            "(source_column='', transformation_type='Indirect').\n"
            "Return ONLY a JSON array. Each row: source_schema, source_table, source_column, "
            "target_schema, target_table, target_column, transformation_type, transformation_logic. "
            "No markdown, no prose."
        )
        llm = _make_llm()
        response = llm.invoke(inner_prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        # Strip fences if present
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        raw = raw.strip()
        start, end = raw.find("["), raw.rfind("]")
        rows = []
        if start != -1 and end != -1:
            try:
                rows = json.loads(raw[start: end + 1])
            except (json.JSONDecodeError, ValueError):
                rows = []

        sttm_path = str(STTM_DIR / f"sttm_bronze_{run_id[:8]}.csv")
        pd.DataFrame(rows).to_csv(sttm_path, index=False)
        scratchpad["sttm_path"] = sttm_path
        print(f"[STTM] Bronze STTM saved: {sttm_path} ({len(rows)} rows)")
        return json.dumps({"sttm_path": sttm_path, "row_count": len(rows)})

    @tool
    def generate_silver_sttm_tool(confirmation: str = "execute") -> str:
        """Generate Silver STTM. Returns JSON with sttm_path and row_count."""
        if not (bronze_output_paths and bronze_sttm_path):
            return json.dumps({"error": "No bronze_output_paths/bronze_sttm_path available for Silver STTM"})

        context = _prepare_silver_context(bronze_output_paths, bronze_sttm_path)

        # Build a concise summary of the Bronze metadata (filenames + columns)
        try:
            context_summary = "\n".join(
                f"Table: {t['filename']} | columns: {', '.join(t.get('columns', []))}"
                for t in context
            )
        except Exception:
            context_summary = "(unable to summarise bronze metadata)"

        # Silver is intent-agnostic: apply standard cleansing to every Bronze column.
        inner_prompt = (
            "Generate a complete Silver STTM as a JSON array of rows. Map EVERY column below.\n"
            f"Source tables:\n{context_summary[:4000]}\n\n"
            "CRITICAL RULES — follow exactly:\n"
            "1. source_column MUST be the EXACT column name from the listing above. "
            "Do NOT rename, abbreviate, or substitute any column name.\n"
            "2. Map EVERY column in the listing — do not skip any column.\n"
            "3. First row per table: surrogate key (source_column='', target_column='pk_<stem>_silver_id', "
            "transformation_type='Indirect', transformation_logic='Auto-generated sequential surrogate primary key starting from 1').\n"
            "4. Apply null handling, type casting, date standardisation per column. Id columns: type cast only.\n"
            "5. source_table must be the exact Bronze filename as shown in the listing above.\n"
            "6. target_table must be '<stem>_silver' where <stem> is the Bronze filename without extension and without '_bronze'.\n"
            "Return ONLY a JSON array. Each row: source_schema, source_table, source_column, "
            "target_schema, target_table, target_column, transformation_type, transformation_logic. No markdown."
        )
        llm = _make_llm()
        response = llm.invoke(inner_prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        raw = raw.strip()
        start, end = raw.find("["), raw.rfind("]")
        rows = []
        if start != -1 and end != -1:
            try:
                rows = json.loads(raw[start: end + 1])
            except (json.JSONDecodeError, ValueError):
                rows = []

        sttm_path = str(STTM_DIR / f"sttm_silver_{run_id[:8]}.csv")
        pd.DataFrame(rows).to_csv(sttm_path, index=False)
        scratchpad["sttm_path"] = sttm_path
        print(f"[STTM] Silver STTM saved: {sttm_path} ({len(rows)} rows)")
        return json.dumps({"sttm_path": sttm_path, "row_count": len(rows)})

    @tool
    def generate_gold_sttm_tool(confirmation: str = "execute") -> str:
        """Generate Gold STTM. Returns JSON with sttm_path and row_count."""
        if not (silver_output_paths and silver_sttm_path):
            return json.dumps({"error": "No silver_output_paths/silver_sttm_path available for Gold STTM"})
        if not business_intent:
            return json.dumps({"error": "business_intent is required for Gold STTM generation"})

        context = _prepare_gold_context(silver_output_paths, silver_sttm_path)

        # Rich context: table name + every column with dtype + 3 sample values
        # so the LLM can identify geographic, numeric, and key columns from the
        # actual data regardless of what the files are named.
        try:
            context_lines = []
            for t in context:
                col_details = []
                for col in t.get("columns", []):
                    dtype = t.get("dtypes", {}).get(col, "?")
                    samples = [str(row.get(col, "")) for row in t.get("sample", [])[:3] if row.get(col) not in (None, "")]
                    sample_str = f"  samples: {', '.join(samples)}" if samples else ""
                    col_details.append(f"    {col} ({dtype}){sample_str}")
                context_lines.append(f"Table: {t['filename']}\n" + "\n".join(col_details))
            context_summary = "\n\n".join(context_lines)
        except Exception:
            context_summary = json.dumps(context, default=str)[:2000]

        inner_prompt = (
            "Generate a complete Gold STTM JSON array shaped for the business intent.\n"
            f"Business intent: {business_intent}\n"
            f"Silver tables (table name, column name, dtype, sample values):\n{context_summary[:5000]}\n\n"
            "CRITICAL RULES — follow exactly:\n"
            "1. source_column MUST be the EXACT column name from the listing above. "
            "Do NOT invent, rename, or substitute any column name.\n"
            "2. First row: surrogate key (source_column='', target_column='pk_gold_id', "
            "transformation_type='Indirect', transformation_logic='Auto-generated sequential surrogate primary key starting from 1').\n"
            "3. source_table must be the exact Silver table name as shown in the listing above.\n"
            "4. JOINS: look for columns with matching names across tables (typically columns ending in _id). "
            "Join on those shared key columns — do not assume any specific table or column name.\n"
            "5. GEOGRAPHIC GROUPING: if the business intent involves regions, locations, or territories, "
            "inspect the sample values above to identify which column contains geographic grouping data "
            "(e.g. a column whose samples are region/territory/zone names). Use that column directly. "
            "Do NOT substitute a place-name column (like a city or store name) for a region/territory column.\n"
            "6. REVENUE AGGREGATIONS: inspect dtypes and sample values to identify the pre-computed numeric "
            "column that represents sales/revenue/amount. Use SUM aggregation on that column. "
            "Do NOT write derived expressions like 'SUM(a * b)' — "
            "the ingestion engine sums only the source column as-is and cannot evaluate formulas.\n"
            "Return ONLY a JSON array. Each row: source_schema, source_table, source_column, "
            "target_schema, target_table, target_column, transformation_type, transformation_logic. No markdown."
        )
        llm = _make_llm()
        response = llm.invoke(inner_prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        raw = raw.strip()
        start, end = raw.find("["), raw.rfind("]")
        rows = []
        if start != -1 and end != -1:
            try:
                rows = json.loads(raw[start: end + 1])
            except (json.JSONDecodeError, ValueError):
                rows = []

        sttm_path = str(STTM_DIR / f"sttm_gold_{run_id[:8]}.csv")
        pd.DataFrame(rows).to_csv(sttm_path, index=False)
        scratchpad["sttm_path"] = sttm_path
        print(f"[STTM] Gold STTM saved: {sttm_path} ({len(rows)} rows)")
        return json.dumps({"sttm_path": sttm_path, "row_count": len(rows)})

    return inspect_context_tool, generate_bronze_sttm_tool, generate_silver_sttm_tool, generate_gold_sttm_tool


# ---------------------------------------------------------------------------
# Shared agent runner
# ---------------------------------------------------------------------------

def _run_sttm_agent(
    trace_name: str,
    run_id: str,
    task_description: str,
    tools: list,
    scratchpad: dict,
    audit_action: str,
    audit_kwargs: dict,
    expected_filename_fragment: str,
) -> str:
    """Instantiate the unified STTM agent, invoke it, extract and return STTM path."""
    trace = AgentTrace(trace_name, run_id)
    trace.set_input(**audit_kwargs)

    audit = AuditLogger(run_id)
    audit.log("sttm_generator", audit_action, **audit_kwargs)

    llm = _make_llm()
    agent = create_agent(llm, tools, system_prompt=STTM_AGENT_PROMPT)

    try:
        result = invoke_with_retry(agent, {"messages": [HumanMessage(content=task_description)]})
    except Exception as e:
        trace.fail(str(e))
        raise

    messages = result.get("messages", [])
    trace.extract_from_messages(messages)

    # Primary: path captured by the generation tool via scratchpad
    sttm_path = scratchpad.get("sttm_path", "")

    # Fallback: scan messages for the path string if scratchpad was not populated
    if not sttm_path:
        for msg in reversed(messages):
            content = getattr(msg, "content", "")
            if isinstance(content, str) and expected_filename_fragment in content:
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and "sttm_path" in parsed:
                        sttm_path = parsed["sttm_path"]
                        break
                except (json.JSONDecodeError, ValueError):
                    pass

    audit.log("sttm_generator", audit_action.replace("started", "completed"), output_file=sttm_path)
    trace.set_output(sttm_path=sttm_path).complete()
    return sttm_path


# ---------------------------------------------------------------------------
# Public entry points — I/O contract UNCHANGED
# ---------------------------------------------------------------------------

def generate_bronze_sttm(
    profile_path: str,
    run_id: str,
    task_description: str,
) -> str:
    """Bronze STTM entry point — deterministic, no LLM.

    Bronze is pure pass-through: every source column maps 1:1 without any
    transformation. An LLM is not needed and introduces hallucination risk.
    """
    print(f"[STTM] Generating Bronze STTM (deterministic) for run_id: {run_id}")

    audit = AuditLogger(run_id)
    audit.log("sttm_generator", "started_bronze", profile_path=profile_path)
    trace = AgentTrace("sttm_bronze", run_id)
    trace.set_input(profile_path=profile_path)

    sttm_path = _generate_bronze_sttm_deterministic(profile_path, run_id)

    audit.log("sttm_generator", "completed_bronze", output_file=sttm_path)
    trace.set_output(sttm_path=sttm_path).complete()
    return sttm_path


def generate_silver_sttm(
    bronze_output_paths: list[str],
    bronze_sttm_path: str,
    run_id: str,
    task_description: str,
) -> str:
    """Silver STTM entry point — deterministic, no LLM.

    Silver applies standard rule-based cleansing (null handling, date standardisation,
    type casting) derived from Bronze parquet dtypes. An LLM is not needed here.
    """
    print(f"[STTM] Generating Silver STTM (deterministic) for run_id: {run_id}")

    audit = AuditLogger(run_id)
    audit.log("sttm_generator", "started_silver", bronze_paths=bronze_output_paths)
    trace = AgentTrace("sttm_silver", run_id)
    trace.set_input(bronze_paths=bronze_output_paths)

    sttm_path = _generate_silver_sttm_deterministic(bronze_output_paths, run_id)

    audit.log("sttm_generator", "completed_silver", output_file=sttm_path)
    trace.set_output(sttm_path=sttm_path).complete()
    return sttm_path


def generate_gold_sttm(
    silver_output_paths: list[str],
    silver_sttm_path: str,
    business_intent: str,
    run_id: str,
    task_description: str,
) -> str:
    """Gold STTM agent entry point — autonomous ReAct version.

    Args:
        silver_output_paths: Silver Parquet file paths to use as source schema context.
        silver_sttm_path: Approved Silver STTM CSV (used to filter to approved columns).
        business_intent: Analytical goal guiding Gold table structure.
        run_id: Unique identifier for this pipeline run.
        task_description: High-level goal message from the orchestrator.

    Returns:
        str: Path to the saved Gold STTM CSV.
    """
    print(f"[STTM] Generating Gold STTM for run_id: {run_id}")
    scratchpad: dict = {}
    tools = list(_make_sttm_tools(
        profile_path=None,
        bronze_output_paths=None,
        bronze_sttm_path=None,
        silver_output_paths=silver_output_paths,
        silver_sttm_path=silver_sttm_path,
        business_intent=business_intent,
        run_id=run_id,
        scratchpad=scratchpad,
    ))
    return _run_sttm_agent(
        trace_name="sttm_gold",
        run_id=run_id,
        task_description=task_description,
        tools=tools,
        scratchpad=scratchpad,
        audit_action="started_gold",
        audit_kwargs={"silver_paths": silver_output_paths, "business_intent": business_intent},
        expected_filename_fragment=f"sttm_gold_{run_id[:8]}",
    )
