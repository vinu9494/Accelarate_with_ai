"""Silver layer AI agent — fully autonomous ReAct version.

The agent receives a goal from the orchestrator, uses inspect_task_tool to
preview Bronze Parquet schemas and STTM rules, forms a cleansing plan, then
executes via silver_ingestion_tool.

I/O contract (UNCHANGED — UI and orchestrator safe):
    execute_silver(input_files, sttm_path, run_id, task_description) -> list[str]
"""

import os
import json
import pandas as pd
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from core.config import SILVER_DIR, ANTHROPIC_API_KEY, ANTHROPIC_MODEL, LLM_MAX_TOKENS
from core.llm_retry import invoke_with_retry
from core.audit import AuditLogger
from core.observability import AgentTrace


SILVER_AGENT_PROMPT = """You are an autonomous Data Quality Engineer specialising in the Silver layer of a
Medallion data pipeline. You operate independently: you receive a goal from the
orchestrator, inspect the Bronze Parquet inputs and STTM cleansing rules, form a
concrete cleansing plan, execute it, and verify your output.

## Your operating mode — follow this EXACT sequence every time

1. THINK: Read the task. Identify the Bronze Parquet input files, the STTM path,
   and what the Silver layer is expected to produce (cleansed Parquet files with
   surrogate keys and standardised columns).

2. INSPECT: Call inspect_task_tool FIRST. This previews each Bronze Parquet file's
   schema and lists all STTM cleansing rules.
   State your observations: which columns need null handling, deduplication, type
   casting, or date standardisation.

3. PLAN: Based on the inspection output, write your explicit cleansing plan:
   - For each file, which rules apply?
   - How will nulls be handled per column (drop, fill mean/median/mode/constant)?
   - Which columns will be type-cast or date-standardised?
   - Which surrogate key will be injected as the first column?

4. ACT: Call silver_ingestion_tool to execute the full cleansing workflow across
   all Bronze input files using the approved STTM rules.

5. VERIFY: Confirm the list of Silver Parquet output paths returned and report
   what was cleansed (file count, rows per file, rules applied).

## Tool-calling rule — read this before calling anything

Every tool below takes NO meaningful arguments. Call each one with either no
arguments at all, or `confirmation="execute"` if your client requires an
argument. NEVER pass file paths, run IDs, or any other value you see in this
prompt as a tool argument — the tool already has that data bound internally.

## Available tools

- **inspect_task_tool**: Previews each Bronze Parquet input file (schema, column
  names, dtypes, row count, sample values) and lists all STTM cleansing rules.
  Call this FIRST to form your cleansing plan. Returns a JSON summary.

- **silver_ingestion_tool**: Executes the full Silver layer cleansing workflow.
  Reads each Bronze Parquet, applies approved STTM rules (null handling,
  deduplication, type casting, text normalisation), injects surrogate keys,
  filters to approved columns, and writes Silver Parquet artifacts.
  Returns a JSON list of output file paths.

## Output
After execution, report: (1) your cleansing plan, (2) which rules were applied to
each file, (3) the list of Silver Parquet output paths."""


# ---------------------------------------------------------------------------
# Pure Python helpers — no LLM
# ---------------------------------------------------------------------------

def _inspect_task(input_files: list[str], sttm_path: str) -> dict:
    """Preview Bronze Parquet schemas and STTM cleansing rules. No LLM."""
    sttm_df = pd.read_csv(sttm_path)
    # Normalize STTM column names to expected canonical names so downstream
    # code doesn't KeyError when different CSV variants are used.
    sttm_df.columns = [c.strip() for c in sttm_df.columns]
    col_map = {}
    lower_cols = {c.lower(): c for c in sttm_df.columns}
    def find(col_options):
        for opt in col_options:
            if opt in lower_cols:
                return lower_cols[opt]
        return None

    mapping_candidates = {
        "source_table": ["source_table", "source table", "table", "sourcetable"],
        "source_column": ["source_column", "source column", "source", "sourcecolumn"],
        "target_column": ["target_column", "target column", "target", "targetcolumn", "target_col"],
        "transformation_type": ["transformation_type", "transformation type", "type", "transformationtype"],
        "transformation_logic": ["transformation_logic", "transformation logic", "logic", "transformation"],
    }

    for canonical, opts in mapping_candidates.items():
        found = find([o.lower() for o in opts])
        if found and found != canonical:
            col_map[found] = canonical

    if col_map:
        sttm_df = sttm_df.rename(columns=col_map)

    sttm_df = sttm_df.fillna("")
    file_summaries = []
    for fp in input_files:
        try:
            df = pd.read_parquet(fp)
            col_info = {}
            for col in df.columns:
                col_info[col] = {
                    "dtype": str(df[col].dtype),
                    "null_count": int(df[col].isnull().sum()),
                    "sample_values": df[col].dropna().head(3).tolist(),
                }
            file_summaries.append({
                "file": os.path.basename(fp),
                "rows": df.shape[0],
                "columns": list(df.columns),
                "column_info": col_info,
            })
        except Exception as e:
            file_summaries.append({"file": os.path.basename(fp), "error": str(e)})

    rules_summary = []
    for _, row in sttm_df.iterrows():
        rules_summary.append({
            "source_table": str(row.get("source_table", "")),
            "source_column": str(row.get("source_column", "")),
            "target_column": str(row.get("target_column", "")),
            "transformation_type": str(row.get("transformation_type", "")),
            "transformation_logic": str(row.get("transformation_logic", "")),
        })

    return {
        "files": file_summaries,
        "sttm_rules": rules_summary,
        "total_files": len(input_files),
        "total_rules": len(rules_summary),
    }


def _safe_numeric_cast(original: pd.Series, casted: pd.Series) -> bool:
    """Guard against STTM rules that mis-classify alphanumeric business keys
    (e.g. store_id="S01", product_id="P001") as numeric. Coercing those with
    pd.to_numeric(errors="coerce") doesn't raise — it silently turns every
    value into null, which then breaks every downstream join on that column.
    Only allow the cast if it doesn't newly null out a large share of values
    that were previously populated.
    """
    was_present = original.notna()
    now_missing = was_present & casted.isna()
    if was_present.sum() == 0:
        return True
    return (now_missing.sum() / was_present.sum()) <= 0.1


def _apply_silver_rules(input_files: list[str], sttm_path: str, run_id: str) -> list[str]:
    """Read each Bronze Parquet, apply STTM cleansing rules, inject surrogate key, write Silver Parquet."""
    audit = AuditLogger(run_id)
    audit.log("silver_agent", "started", input_files=input_files, sttm_path=sttm_path)

    sttm_df = pd.read_csv(sttm_path).fillna("")
    output_paths = []

    for file_path in input_files:
        df = pd.read_parquet(file_path)
        original_shape = df.shape
        file_name = os.path.basename(file_path)
        file_stem = os.path.splitext(file_name)[0]
        file_stem_base = file_stem.replace("_bronze", "")

        # Match STTM rows to this file leniently: the LLM is inconsistent about
        # what it puts in "source_table" (sometimes the real filename, sometimes
        # a generic placeholder like "parquet"). "source_schema" and
        # "target_table" have been reliable identifiers even when source_table
        # is wrong, so accept a match on any of the three rather than requiring
        # source_table alone - an exact-match-only STTM row set silently
        # producing zero rules collapses the whole table to just its
        # surrogate key and metadata columns.
        def _matches_file(row) -> bool:
            source_table = str(row.get("source_table", "")).strip()
            if source_table in ("", file_name, file_stem):
                return True
            source_schema = str(row.get("source_schema", "")).strip()
            if source_schema in (file_name, file_stem, file_stem_base):
                return True
            target_table = str(row.get("target_table", "")).strip()
            if target_table and target_table.replace("_silver", "") == file_stem_base:
                return True
            return False

        file_rules = (
            sttm_df[sttm_df.apply(_matches_file, axis=1)]
            if len(sttm_df) and "source_table" in sttm_df.columns
            else sttm_df
        )
        if len(sttm_df) and file_rules.empty:
            print(
                f"[SILVER] WARNING: no STTM rules matched '{file_name}' "
                f"(source_table/source_schema/target_table didn't identify this file) - "
                "output will only have the surrogate key and metadata columns"
            )

        for _, rule in file_rules.iterrows():
            source_col = str(rule.get("source_column", "")).strip()
            target_col = str(rule.get("target_column", "")).strip()
            logic = str(rule.get("transformation_logic", "")).lower()

            if source_col and target_col and source_col in df.columns and source_col != target_col:
                df = df.rename(columns={source_col: target_col})

            working_col = target_col if target_col in df.columns else source_col

            if "deduplic" in logic:
                # STTM rules mention "deduplicate" on plenty of columns that are
                # supposed to repeat across rows (region, category, payment_method,
                # etc.) - the LLM says it per-column almost reflexively. Treating
                # that literally as drop_duplicates(subset=[that_one_column]) keeps
                # only the FIRST row per unique value, destroying the whole table.
                # Only honor a single-column dedup for genuine identifier columns
                # (or when the logic explicitly says so); otherwise fall back to
                # whole-row dedup, which is what "remove duplicates" safely means.
                is_identifier_col = working_col.endswith("_id") or working_col == "id"
                explicit_all_cols = "all column" in logic or "entire row" in logic or "exact duplicate" in logic
                if is_identifier_col and not explicit_all_cols:
                    before = len(df)
                    deduped = df.drop_duplicates(subset=[working_col])
                    if before == 0 or (before - len(deduped)) / before <= 0.5:
                        df = deduped
                    else:
                        print(
                            f"[SILVER] Skipping single-column dedup on '{working_col}' - "
                            f"would drop {before - len(deduped)}/{before} rows; "
                            "falling back to whole-row dedup instead"
                        )
                        df = df.drop_duplicates()
                else:
                    df = df.drop_duplicates()
                continue

            if not working_col or working_col not in df.columns:
                continue

            try:
                if "drop null" in logic or "remove null" in logic:
                    df = df.dropna(subset=[working_col])
                elif "fill null" in logic and "mean" in logic:
                    df[working_col] = df[working_col].fillna(
                        pd.to_numeric(df[working_col], errors="coerce").mean()
                    )
                elif "fill null" in logic and "median" in logic:
                    df[working_col] = df[working_col].fillna(
                        pd.to_numeric(df[working_col], errors="coerce").median()
                    )
                elif "fill null" in logic and "mode" in logic:
                    mode_val = df[working_col].mode()
                    if not mode_val.empty:
                        df[working_col] = df[working_col].fillna(mode_val.iloc[0])
                elif "fill null" in logic or "default" in logic:
                    df[working_col] = df[working_col].fillna("")

                if "date" in logic or "datetime" in logic:
                    _date_fmts = [
                        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
                        "%Y%m%d", "%d-%b-%Y", "%d-%B-%Y",
                        "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                        "%m-%d-%Y", "%d.%m.%Y",
                    ]
                    _parsed = None
                    for _fmt in _date_fmts:
                        try:
                            _try = pd.to_datetime(df[working_col], format=_fmt, errors="coerce")
                            if _try.notna().sum() > 0:
                                _parsed = _try
                                break
                        except Exception:
                            continue
                    df[working_col] = _parsed if _parsed is not None else pd.to_datetime(
                        df[working_col], errors="coerce"
                    )
                elif "integer" in logic:
                    _casted = pd.to_numeric(df[working_col], errors="coerce")
                    if _safe_numeric_cast(df[working_col], _casted):
                        df[working_col] = _casted.astype("Int64")
                    else:
                        print(
                            f"[SILVER] Skipping integer cast on '{working_col}' - "
                            "values are not numeric (e.g. alphanumeric IDs like 'S01'); "
                            "keeping original values so joins on this column still work"
                        )
                elif "float" in logic or "decimal" in logic or "numeric" in logic:
                    _casted = pd.to_numeric(df[working_col], errors="coerce")
                    if _safe_numeric_cast(df[working_col], _casted):
                        df[working_col] = _casted
                    else:
                        print(
                            f"[SILVER] Skipping numeric cast on '{working_col}' - "
                            "values are not numeric; keeping original values"
                        )
                elif "text" in logic:
                    df[working_col] = df[working_col].astype(str)

                if "lowercase" in logic:
                    df[working_col] = df[working_col].astype(str).str.lower()
                elif "uppercase" in logic:
                    df[working_col] = df[working_col].astype(str).str.upper()
                elif "title case" in logic:
                    df[working_col] = df[working_col].astype(str).str.title()

                if "strip" in logic or "trim" in logic:
                    df[working_col] = df[working_col].astype(str).str.strip()
            except (ValueError, TypeError):
                pass

        # Inject surrogate primary key as first column
        pk_col = f"pk_{file_stem}_silver_id"
        if pk_col not in df.columns:
            df.insert(0, pk_col, range(1, len(df) + 1))

        # Filter columns to approved Silver targets + system metadata columns
        # Be defensive: allow missing 'target_column' by falling back to an empty set.
        if "target_column" in file_rules.columns:
            approved_target_cols = set(file_rules["target_column"].unique())
        else:
            approved_target_cols = set()
        columns_to_keep = [
            c for c in df.columns
            if c in approved_target_cols or c.startswith("_") or c.startswith("pk_")
        ]
        if pk_col in columns_to_keep:
            columns_to_keep.remove(pk_col)
            columns_to_keep.insert(0, pk_col)
        df = df[columns_to_keep]

        filename = os.path.basename(file_path).replace("_bronze.parquet", "_silver.parquet")
        output_path = str(SILVER_DIR / filename)
        df.to_parquet(output_path, index=False)
        output_paths.append(output_path)

        audit.log(
            "silver_agent", "file_processed",
            input_file=file_path,
            output_file=output_path,
            input_shape=list(original_shape),
            output_shape=list(df.shape),
        )

    audit.log("silver_agent", "completed", output_files=output_paths)
    return output_paths


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------

def _make_silver_tools(input_files: list[str], sttm_path: str, run_id: str):
    """Returns inspect + ingestion tools bound to this run's parameters via closure."""

    @tool
    def inspect_task_tool(confirmation: str = "execute") -> str:
        """Preview Bronze Parquet input schemas and STTM cleansing rules.

        Returns a JSON summary of each Bronze file's column names, dtypes, null counts,
        sample values, and all STTM cleansing rules (null handling, type casting, etc.).
        Call this FIRST to understand what needs to be cleansed and form your plan.
        """
        return json.dumps(_inspect_task(input_files, sttm_path), default=str)

    @tool
    def silver_ingestion_tool(confirmation: str = "execute") -> str:
        """Execute the full Silver layer cleansing workflow using approved STTM rules.

        Reads each Bronze Parquet file, applies null handling, deduplication, type
        casting and text normalisation rules from the STTM, injects surrogate keys,
        filters to approved columns, and writes Silver Parquet artifacts.
        Returns a JSON list of output file paths.
        """
        output_paths = _apply_silver_rules(input_files, sttm_path, run_id)
        return json.dumps(output_paths)

    return inspect_task_tool, silver_ingestion_tool


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _make_llm():
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(api_key=ANTHROPIC_API_KEY, model=ANTHROPIC_MODEL, temperature=0, max_tokens=LLM_MAX_TOKENS)


# ---------------------------------------------------------------------------
# Public entry point — I/O contract UNCHANGED
# ---------------------------------------------------------------------------

def execute_silver(
    input_files: list[str],
    sttm_path: str,
    run_id: str,
    task_description: str,
) -> list[str]:
    """Silver AI agent entry point — autonomous ReAct version.

    The agent inspects Bronze Parquet schemas and STTM rules, forms an explicit
    cleansing plan, then executes across all input files.

    Args:
        input_files: Bronze Parquet file paths to cleanse.
        sttm_path: Path to the approved Silver STTM CSV.
        run_id: Unique identifier for this pipeline run.
        task_description: High-level goal message from the orchestrator.

    Returns:
        list[str]: Silver Parquet output file paths.
    """
    trace = AgentTrace("silver_agent", run_id)
    trace.set_input(input_files=input_files, sttm_path=sttm_path)

    inspect_tool, ingestion_tool = _make_silver_tools(input_files, sttm_path, run_id)
    llm = _make_llm()

    print("[SILVER] Running autonomous ReAct agent (anthropic)")
    agent = create_agent(llm, [inspect_tool, ingestion_tool], system_prompt=SILVER_AGENT_PROMPT)

    try:
        result = invoke_with_retry(agent, {"messages": [HumanMessage(content=task_description)]})
    except Exception as e:
        trace.fail(str(e))
        raise

    messages = result.get("messages", [])
    trace.extract_from_messages(messages)

    output_paths: list[str] = []
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, list) and all(isinstance(p, str) for p in parsed):
                    output_paths = parsed
                    break
            except (json.JSONDecodeError, ValueError):
                continue

    trace.set_output(output_paths=output_paths).complete()
    return output_paths
