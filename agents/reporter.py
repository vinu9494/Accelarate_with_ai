"""Reporting AI agent — fully autonomous ReAct version.

The agent receives a goal from the orchestrator, inspects available Gold tables
first to understand their schemas, forms an analytical plan, writes and executes
SQL to answer the business question, and renders an HTML report.

I/O contract (UNCHANGED — UI and orchestrator safe):
    generate_report(gold_files, business_intent, run_id, task_description) -> str
"""

import json
import pandas as pd
import duckdb
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from core.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, REPORTS_DIR, LLM_MAX_TOKENS
from core.llm_retry import invoke_with_retry
from core.audit import AuditLogger
from core.observability import AgentTrace
from core.memory import store_document


REPORTER_AGENT_PROMPT = """You are an autonomous Senior Data Analyst and Business Intelligence Engineer
specialising in business-intent-driven reporting from Medallion Gold layer data.
You operate independently: you receive a goal from the orchestrator, inspect the Gold
tables, form an analytical plan, write and execute SQL, and return structured analysis.

## Your operating mode — follow this EXACT sequence every time

1. THINK: Read the task. Identify the business question, the Gold files available,
   and what kind of analysis (aggregation, trend, comparison, ranking) will answer it.

2. INSPECT: Call inspect_gold_tables_tool FIRST. This gives you a lightweight preview
   of each Gold table — column names, dtypes, row count, and 3 sample rows — without
   loading full data into DuckDB. State your observations: which tables are relevant,
   which columns can answer the business question, what joins may be needed.

3. PLAN: Based on the inspection, write your analytical plan:
   - Which Gold tables will you query?
   - What SQL approach will directly answer the business question?
   - What chart type(s) will best visualise the answer?
   - CRITICAL: "highest sales", "best-selling", "top product", or "most revenue" for a
     product/category/region means the SUM of the relevant amount column (e.g.
     total_amount) GROUPed BY that product/category/region across ALL matching
     transactions — never the single largest individual transaction row. A Gold
     table is transaction-grained (one row per sale), so ranking by a raw row-level
     column directly (e.g. ORDER BY total_amount DESC without SUM/GROUP BY) answers
     "which single transaction was biggest", not "which product sold the most" —
     these give different, both-plausible-looking numbers, so get this right the
     first time. Always aggregate with SUM(...) GROUP BY ... before ranking with
     ROW_NUMBER/RANK/ORDER BY, unless the business question explicitly asks about
     a single transaction/order.

4. ACT — two sub-steps in order:
   a. Call load_gold_data_tool to register Gold tables in DuckDB and get the full schema catalog.
   b. Call execute_query_tool(sql_query=<your_sql>) to execute your SQL and get results.

5. VERIFY & RESPOND: Analyse the query results and return ONLY a valid JSON object
   as your final answer (no markdown fences, no prose before or after).

## Tool-calling rule — read this before calling anything

inspect_gold_tables_tool and load_gold_data_tool take NO meaningful arguments —
call each with no arguments at all, or `confirmation="execute"` if your client
requires one. NEVER pass file paths, run IDs, or any other value you see in
this prompt as their argument — that data is already bound internally.
execute_query_tool is the ONLY tool that takes a real argument: exactly one
parameter, `sql_query`, containing your SQL string.

## Available tools

- **inspect_gold_tables_tool**: Quickly previews Gold Parquet files — table names,
  column names, dtypes, row count, and 3 sample rows per table. Call this FIRST
  to understand what is available before loading into DuckDB. Returns a JSON summary.

- **load_gold_data_tool**: Loads Gold Parquet files into an in-memory DuckDB database
  and returns a full catalog of table names, column names, types, row counts, and
  sample data. Call this before execute_query_tool.

- **execute_query_tool**: Executes a SQL SELECT query against the loaded Gold tables
  in DuckDB. Pass your SQL as the sql_query parameter. Returns query results as a
  JSON array. On error returns {"error": "..."}.

## Output format
Return ONLY a valid JSON object — no markdown fences, no prose:
{
  "direct_answer": {
    "question": "Restate the business question clearly",
    "answer": "Direct answer with specific numbers from the query results",
    "why": "Evidence and reasoning from the data",
    "approach": "Describe the SQL query and analytical method used"
  },
  "charts": [
    {
      "type": "bar|line|pie|scatter",
      "title": "Chart title",
      "x_column": "column from query result",
      "y_column": "column from query result (bar/line/scatter)",
      "labels_column": "column from query result (pie only)",
      "values_column": "column from query result (pie only)",
      "reason": "Why this chart directly answers the question"
    }
  ],
  "detailed_analysis": "2-3 paragraphs of additional insights and patterns"
}

## Rules
- Include only 1-2 charts that directly answer the business question.
- Use ACTUAL column names from the query result — not from the original Gold tables.
- Be specific with numbers in the direct_answer.
- Write standard ANSI SQL compatible with DuckDB.
- If execute_query_tool returns an error, fix the SQL and retry once."""


# ---------------------------------------------------------------------------
# Pure Python helpers — no LLM
# ---------------------------------------------------------------------------

def _inspect_gold_tables(gold_files: list[str]) -> dict:
    """Quick preview of Gold Parquet tables: schema + 3 sample rows. No LLM, no DuckDB."""
    summary = {}
    for fp in gold_files:
        try:
            df = pd.read_parquet(fp)
            stem = Path(fp).stem.replace("-", "_").replace(" ", "_")
            summary[stem] = {
                "file": fp,
                "table_name": stem,
                "row_count": len(df),
                "columns": list(df.columns),
                "dtypes": {c: str(t) for c, t in df.dtypes.items()},
                "sample_rows": df.head(3).to_dict(orient="records"),
            }
        except Exception as e:
            summary[Path(fp).stem] = {"file": fp, "error": str(e)}
    return summary


def _resolve_column(name: str | None, df: pd.DataFrame) -> str | None:
    """Match an LLM-specified column name to the real one, tolerating case/whitespace
    drift (e.g. chart_spec says 'Total_Sales' but the SQL result column is 'total_sales').
    Returns None if nothing matches."""
    if not name:
        return None
    if name in df.columns:
        return name
    lookup = {c.strip().lower(): c for c in df.columns}
    return lookup.get(name.strip().lower())


def generate_chart_from_spec(df: pd.DataFrame, chart_spec: dict, chart_id: int) -> str:
    """Render a single Plotly chart from an LLM-specified chart spec dict. Returns embedded HTML."""
    try:
        chart_type = chart_spec.get("type", "bar").lower()
        title = chart_spec.get("title", f"Chart {chart_id}")

        if chart_type == "bar":
            x_col = _resolve_column(chart_spec.get("x_column"), df)
            y_col = _resolve_column(chart_spec.get("y_column"), df)

            if x_col is None:
                print(
                    f"[REPORTER] Chart {chart_id}: x_column '{chart_spec.get('x_column')}' not found "
                    f"in query result (columns: {list(df.columns)}) - skipping chart"
                )
                return ""

            if not y_col:
                # y_column wasn't found under its exact/case-insensitive name. Before
                # giving up, check whether there's exactly one other numeric column
                # in the result - that's almost always the actual metric the query
                # was built to answer (e.g. spec says 'total_sales' but the SQL
                # aliased it 'total_sales_amount'). Only fall back to a meaningless
                # row-count chart if that recovery isn't possible, and never
                # mislabel a row-count chart with the intended metric's name.
                numeric_cols = [
                    c for c in df.columns
                    if c != x_col and pd.api.types.is_numeric_dtype(df[c])
                ]
                if len(numeric_cols) == 1:
                    print(
                        f"[REPORTER] Chart {chart_id}: y_column '{chart_spec.get('y_column')}' not found "
                        f"in query result (columns: {list(df.columns)}) - using the only numeric column "
                        f"'{numeric_cols[0]}' instead"
                    )
                    y_col = numeric_cols[0]

            if y_col:
                # Auto-detect a string column to annotate bars (e.g. product_name in a
                # top-per-region result). Spec can explicitly supply text_column; otherwise
                # use the first non-numeric, non-x column if one exists.
                text_col = _resolve_column(chart_spec.get("text_column"), df)
                if text_col is None:
                    candidates = [
                        c for c in df.columns
                        if c != x_col and not pd.api.types.is_numeric_dtype(df[c])
                    ]
                    text_col = candidates[0] if candidates else None

                if text_col:
                    # Pick the highest-y row per x group so the annotation matches the
                    # bar height without losing the associated label value.
                    top_rows = (
                        df.loc[df.groupby(x_col)[y_col].idxmax()]
                          .sort_values(y_col, ascending=False)
                          .head(10)
                    )
                    fig = go.Figure(data=[go.Bar(
                        x=top_rows[x_col],
                        y=top_rows[y_col],
                        text=top_rows[text_col],
                        textposition="outside",
                        marker_color="#667eea",
                        hovertemplate=(
                            f"<b>%{{x}}</b><br>"
                            f"{y_col}: %{{y:,.0f}}<br>"
                            f"{text_col}: %{{text}}<extra></extra>"
                        ),
                    )])
                else:
                    agg_data = df.groupby(x_col)[y_col].sum().sort_values(ascending=False).head(10)
                    fig = go.Figure(data=[go.Bar(x=agg_data.index, y=agg_data.values, marker_color="#667eea")])
                y_axis_title = y_col
            else:
                print(
                    f"[REPORTER] Chart {chart_id}: y_column '{chart_spec.get('y_column')}' not found "
                    f"in query result (columns: {list(df.columns)}) - falling back to row counts per "
                    f"'{x_col}', which is a different metric than what was intended"
                )
                value_counts = df[x_col].value_counts().head(10)
                fig = go.Figure(data=[go.Bar(x=value_counts.index, y=value_counts.values, marker_color="#667eea")])
                y_axis_title = "Count"

            fig.update_layout(title=title, xaxis_title=x_col, yaxis_title=y_axis_title,
                              height=450, template="plotly_white")
            return fig.to_html(full_html=False, include_plotlyjs=False, div_id=f"chart_{chart_id}")

        elif chart_type == "line":
            x_col = _resolve_column(chart_spec.get("x_column"), df)
            y_col = _resolve_column(chart_spec.get("y_column"), df)
            if x_col is None or y_col is None:
                print(f"[REPORTER] Chart {chart_id}: x/y column not found in query result "
                      f"(columns: {list(df.columns)}) - skipping chart")
                return ""
            fig = px.line(df, x=x_col, y=y_col, title=title)
            fig.update_layout(height=450, template="plotly_white")
            return fig.to_html(full_html=False, include_plotlyjs=False, div_id=f"chart_{chart_id}")

        elif chart_type == "pie":
            labels_col = _resolve_column(chart_spec.get("labels_column"), df)
            values_col = _resolve_column(chart_spec.get("values_column"), df)
            if labels_col is None or values_col is None:
                print(f"[REPORTER] Chart {chart_id}: labels/values column not found in query result "
                      f"(columns: {list(df.columns)}) - skipping chart")
                return ""
            agg_data = df.groupby(labels_col)[values_col].sum()
            fig = go.Figure(data=[go.Pie(labels=agg_data.index, values=agg_data.values)])
            fig.update_layout(title=title, height=450)
            return fig.to_html(full_html=False, include_plotlyjs=False, div_id=f"chart_{chart_id}")

        elif chart_type == "scatter":
            x_col = _resolve_column(chart_spec.get("x_column"), df)
            y_col = _resolve_column(chart_spec.get("y_column"), df)
            if x_col is None or y_col is None:
                print(f"[REPORTER] Chart {chart_id}: x/y column not found in query result "
                      f"(columns: {list(df.columns)}) - skipping chart")
                return ""
            fig = px.scatter(df, x=x_col, y=y_col, title=title)
            fig.update_layout(height=450, template="plotly_white")
            return fig.to_html(full_html=False, include_plotlyjs=False, div_id=f"chart_{chart_id}")

        return ""
    except Exception as e:
        print(f"[REPORTER] Error generating chart {chart_id}: {e}")
        return ""


def generate_table_html(df: pd.DataFrame, max_rows: int = 50) -> str:
    """Render a query result DataFrame as a styled HTML table."""
    if df is None or df.empty:
        return "<p>No data available.</p>"
    display = df.head(max_rows)
    header_cells = "".join(f"<th>{col}</th>" for col in display.columns)
    rows_html = ""
    for _, row in display.iterrows():
        cells = "".join(
            f"<td>{v:,.2f}</td>" if isinstance(v, float) else f"<td>{v}</td>"
            for v in row
        )
        rows_html += f"<tr>{cells}</tr>"
    return f"""
    <div style="overflow-x:auto">
      <table class="data-table">
        <thead><tr>{header_cells}</tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    """


def _extract_analysis(result: dict) -> dict:
    """Scan agent message history (reverse order) for a JSON object with 'direct_answer' key."""
    for msg in reversed(result.get("messages", [])):
        content = getattr(msg, "content", "")
        # LangChain Anthropic returns content as a list of typed blocks
        # (e.g. [{"type": "text", "text": "..."}]) — flatten to a plain string.
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
                if not isinstance(block, dict) or block.get("type") == "text"
            )
        if not isinstance(content, str) or not content.strip():
            continue
        text = content
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            continue
        try:
            parsed = json.loads(text[start: end + 1])
            if isinstance(parsed, dict) and "direct_answer" in parsed:
                return parsed
        except (json.JSONDecodeError, ValueError):
            continue
    return {}


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------

def _make_reporter_tools(gold_files: list[str], run_id: str):
    """Returns inspect + load + query tools sharing a DuckDB connection via closure."""
    conn = duckdb.connect(":memory:")
    scratchpad: dict = {}

    @tool
    def inspect_gold_tables_tool(confirmation: str = "execute") -> str:
        """Preview Gold Parquet tables before loading into DuckDB.

        Returns a JSON summary of each Gold table: table name, file path, row count,
        column names, dtypes, and 3 sample rows. Call this FIRST to understand what
        data is available and form your analytical plan.
        """
        return json.dumps(_inspect_gold_tables(gold_files), default=str)

    @tool
    def load_gold_data_tool(confirmation: str = "execute") -> str:
        """Load Gold Parquet files into DuckDB and return the full table catalog.

        Registers each Gold file as a DuckDB table and returns a catalog mapping table
        names to column names, types, row counts, and sample data — everything needed
        to write a precise SQL query. Call this before execute_query_tool.
        """
        catalog: dict = {}
        for fp in gold_files:
            df = pd.read_parquet(fp)
            stem = Path(fp).stem.replace("-", "_").replace(" ", "_")
            conn.register(stem, df)
            catalog[stem] = {
                "table_name": stem,
                "columns": list(df.columns),
                "dtypes": {c: str(t) for c, t in df.dtypes.items()},
                "sample": df.head(5).to_dict(orient="records"),
                "row_count": len(df),
            }
        scratchpad["catalog"] = catalog
        return json.dumps(catalog, default=str)

    @tool
    def execute_query_tool(sql_query: str) -> str:
        """Execute a SQL SELECT query against the loaded Gold tables in DuckDB.

        Call this after load_gold_data_tool. Pass your SQL as sql_query.
        Returns the query result as a JSON array of records (up to 100 rows).
        On SQL error returns {"error": "..."} — fix the SQL and retry once.
        """
        try:
            result_df = conn.execute(sql_query).fetchdf()
            scratchpad["result_df"] = result_df
            scratchpad["sql_query"] = sql_query
            return json.dumps(result_df.head(100).to_dict(orient="records"), default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    return inspect_gold_tables_tool, load_gold_data_tool, execute_query_tool, scratchpad, conn


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _make_llm():
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(api_key=ANTHROPIC_API_KEY, model=ANTHROPIC_MODEL, temperature=0, max_tokens=LLM_MAX_TOKENS)


# ---------------------------------------------------------------------------
# Public entry point — I/O contract UNCHANGED
# ---------------------------------------------------------------------------

def generate_report(
    gold_files: list[str],
    business_intent: str,
    run_id: str,
    task_description: str,
) -> str:
    """Reporter AI agent entry point — autonomous ReAct version.

    The agent inspects Gold tables, plans its SQL analysis, loads tables into
    DuckDB, executes the query, and renders a self-contained HTML report.

    Args:
        gold_files: Gold Parquet file paths to analyse.
        business_intent: The business question driving the analysis.
        run_id: Unique identifier for this pipeline run.
        task_description: High-level goal message from the orchestrator.

    Returns:
        str: Path to the saved HTML report.
    """
    trace = AgentTrace("reporter", run_id)
    trace.set_input(gold_files=gold_files, business_intent=business_intent)

    print(f"[REPORTER] Starting report generation for run_id: {run_id}")
    audit = AuditLogger(run_id)
    audit.log("reporter", "started", gold_files=gold_files, intent=business_intent)

    if not gold_files:
        audit.log("reporter", "error", detail="No gold files to report on")
        trace.fail("No gold files provided")
        return ""

    inspect_tool, load_tool, query_tool, scratchpad, conn = _make_reporter_tools(gold_files, run_id)
    llm = _make_llm()

    print("[REPORTER] Running autonomous ReAct agent (anthropic)")
    agent = create_agent(
        llm,
        [inspect_tool, load_tool, query_tool],
        system_prompt=REPORTER_AGENT_PROMPT,
    )

    try:
        result = invoke_with_retry(agent, {"messages": [HumanMessage(content=task_description)]})
    except Exception as e:
        trace.fail(str(e))
        conn.close()
        raise
    finally:
        conn.close()

    messages = result.get("messages", [])
    trace.extract_from_messages(messages)

    # Extract structured analysis from agent message history
    analysis_result = _extract_analysis(result)
    result_df: pd.DataFrame = scratchpad.get("result_df")  # type: ignore[assignment]
    query_code: str = scratchpad.get("sql_query", "-- No query executed")

    # Fallback: agent did not call execute_query_tool or query returned nothing
    if result_df is None or result_df.empty:
        print("[REPORTER] No query result in scratchpad - falling back to combined gold data")
        fallback_dfs = [pd.read_parquet(fp) for fp in gold_files]
        result_df = pd.concat(fallback_dfs, ignore_index=True) if fallback_dfs else pd.DataFrame()
        query_code = "-- Fallback: combined all Gold tables"

    # Fallback: agent produced HTML/prose instead of JSON — synthesise with a direct LLM call
    if not analysis_result and result_df is not None and not result_df.empty:
        print("[REPORTER] Agent did not return JSON — running synthesis fallback")
        from langchain_core.messages import SystemMessage
        synthesis_llm = _make_llm()
        data_preview = json.dumps(result_df.head(30).to_dict(orient="records"), default=str)
        synthesis_response = synthesis_llm.invoke([
            SystemMessage(content=(
                "You are a data analyst. Return ONLY a valid JSON object — "
                "no markdown fences, no HTML, no prose outside the JSON.\n"
                "Schema:\n"
                '{"direct_answer": {"question": "...", "answer": "...", "why": "...", "approach": "..."}, '
                '"charts": [{"type": "bar", "title": "...", "x_column": "...", "y_column": "..."}], '
                '"detailed_analysis": "..."}'
            )),
            HumanMessage(content=(
                f"Business question: {business_intent}\n\n"
                f"SQL used:\n{query_code}\n\n"
                f"Query results (top 30 rows):\n{data_preview}\n\n"
                "Using the ACTUAL column names from the query results above, "
                "return the JSON analysis object."
            )),
        ])
        syn_content = synthesis_response.content
        if isinstance(syn_content, list):
            syn_content = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in syn_content if not isinstance(b, dict) or b.get("type") == "text"
            )
        s, e = syn_content.find("{"), syn_content.rfind("}")
        if s != -1 and e != -1:
            try:
                analysis_result = json.loads(syn_content[s:e + 1])
            except (json.JSONDecodeError, ValueError):
                pass

    if not analysis_result:
        analysis_result = {
            "direct_answer": {
                "question": business_intent,
                "answer": "Analysis could not be structured.",
                "why": "Agent response did not contain a parseable JSON object.",
                "approach": "N/A",
            },
            "charts": [],
            "detailed_analysis": "No structured analysis available.",
        }

    print(f"[REPORTER] Query result: {result_df.shape[0]} rows x {result_df.shape[1]} columns")

    # Deduplicate chart specs that share the same x/y columns — the LLM occasionally
    # emits two near-identical specs with different titles but same data.
    seen_xy: set = set()
    unique_chart_specs = []
    for spec in analysis_result.get("charts", []):
        key = (spec.get("x_column", ""), spec.get("y_column", ""))
        if key not in seen_xy:
            seen_xy.add(key)
            unique_chart_specs.append(spec)

    # Generate charts from deduplicated specs
    charts_html = []
    for idx, chart_spec in enumerate(unique_chart_specs, 1):
        chart_html = generate_chart_from_spec(result_df, chart_spec, idx)
        if chart_html:
            charts_html.append(chart_html)
    print(f"[REPORTER] Generated {len(charts_html)} charts (deduplicated from {len(analysis_result.get('charts', []))} specs)")

    table_html = generate_table_html(result_df)
    direct_answer = analysis_result.get("direct_answer", {})
    detailed_analysis = analysis_result.get("detailed_analysis", "No additional analysis provided.")

    answer_html = f"""
    <div class="answer-section">
        <p>{direct_answer.get('answer', 'No answer provided')}</p>
    </div>
    """

    query_code_escaped = query_code.replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    approach_html = f"""
    <div class="approach-section">
        <h3>Query Code</h3>
        <pre class="code-block"><code>{query_code_escaped}</code></pre>
        <h3>Query Description</h3>
        <p>{direct_answer.get('approach', 'No methodology provided')}</p>
    </div>
    """

    charts_section = "\n".join(charts_html) if charts_html else "<p>No charts generated.</p>"

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Executive Report - {run_id[:8]}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
                color: #333;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
            }}
            .header h1 {{ margin: 0; font-size: 32px; }}
            .header p {{ margin: 10px 0 0 0; opacity: 0.9; }}
            .section {{
                background: white;
                padding: 25px;
                border-radius: 8px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .section h2 {{
                color: #667eea;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
                margin-top: 0;
            }}
            .answer-section {{
                background: #e8f4f8;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #28a745;
            }}
            .answer-section p {{ margin: 0; line-height: 1.6; font-size: 16px; color: #333; }}
            .approach-section {{ margin: 20px 0; }}
            .approach-section h3 {{ color: #667eea; font-size: 16px; margin: 20px 0 10px 0; }}
            .code-block {{
                background: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 6px;
                padding: 15px;
                overflow-x: auto;
                font-family: 'Courier New', monospace;
                font-size: 13px;
                line-height: 1.4;
                color: #333;
                margin: 0 0 15px 0;
            }}
            .code-block code {{ color: #667eea; }}
            .approach-section p {{ line-height: 1.6; color: #555; margin: 0 0 15px 0; }}
            .chart-container {{ margin: 20px 0; }}
            .data-table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 14px;
            }}
            .data-table th {{
                background-color: #667eea;
                color: white;
                padding: 10px 12px;
                text-align: left;
                font-weight: 600;
            }}
            .data-table td {{
                padding: 8px 12px;
                border-bottom: 1px solid #eee;
                color: #333;
            }}
            .data-table tr:nth-child(even) td {{ background-color: #f9f9fb; }}
            .data-table tr:hover td {{ background-color: #eef0fc; }}
            .insights-section {{
                background: #f9f9fb;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #764ba2;
                line-height: 1.7;
                color: #444;
            }}
            .footer {{
                text-align: center;
                color: #999;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #eee;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>&#128202; Executive Report</h1>
            <p><strong>Business Question:</strong> {business_intent}</p>
        </div>
        <div class="section">
            <h2>&#9989; Answer</h2>
            {answer_html}
        </div>
        <div class="section">
            <h2>&#128202; Approach &amp; Query</h2>
            {approach_html}
        </div>
        <div class="section">
            <h2>&#128201; Visual Evidence</h2>
            <div class="chart-container">
                {charts_section}
            </div>
        </div>
        <div class="section">
            <h2>&#128196; Query Results</h2>
            {table_html}
        </div>
        <div class="section">
            <h2>&#128270; Key Insights</h2>
            <div class="insights-section">
                {detailed_analysis}
            </div>
        </div>
        <div class="footer">
            <p>Generated by IDAMP (Intent-Driven Agentic Medallion Pipeline)</p>
            <p>Report Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </body>
    </html>
    """

    report_path = str(REPORTS_DIR / f"report_{run_id[:8]}.html")
    print(f"[REPORTER] Saving HTML report -> {report_path}")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    json_path = str(REPORTS_DIR / f"report_{run_id[:8]}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis_result, f, indent=2)

    store_document(
        doc_id=f"report_{run_id}",
        text=json.dumps(analysis_result),
        metadata={"type": "report", "run_id": run_id, "intent": business_intent},
    )

    audit.log("reporter", "completed", report_path=report_path)
    trace.set_output(report_path=report_path).complete()
    print(f"[REPORTER] Done - {report_path}")
    return report_path
