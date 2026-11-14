from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.token import Token

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

PAGE_W = 8.27
PAGE_H = 11.69
MARGIN_X = 0.32
MARGIN_TOP = 0.34
MARGIN_BOTTOM = 0.34
COL_GAP = 0.18
N_PAGES = 3
N_COLS = 2
COL_W = (PAGE_W - 2 * MARGIN_X - (N_COLS - 1) * COL_GAP) / N_COLS

TITLE_SIZE = 17.5
SUBTITLE_SIZE = 9.6
SECTION_SIZE = 10.5
SUBHEAD_SIZE = 9.2
CODE_SIZE = 9.0
TEXT_SIZE = 9.1
FOOTER_SIZE = 7.4

CODE_LH = CODE_SIZE * 1.18 / 72
TEXT_LH = TEXT_SIZE * 1.18 / 72
SECTION_H = 0.20
SECTION_GAP = 0.17
BLOCK_GAP = 0.045

NAVY = "#183451"
BLUE = "#2f5f8a"
CODE = "#102a43"
MUTED = "#536878"
RULE = "#c9d4df"
LIGHT = "#eef3f8"

KEYWORD = "#7b2d8e"
STRING = "#1b6b3a"
COMMENT = "#6b7280"
NUMBER = "#b45309"
FUNCTION = "#1f4e79"
OPERATOR = "#334155"
PUNCT = "#475569"

SECTION_COLORS = {
    "Setup and IO": "#1f3a5f",
    "Selection and indexing": "#245c73",
    "Types and missing values": "#5b3a72",
    "Strings": "#2f6b4f",
    "Regex (re module + Series.str)": "#8a4b2a",
    "Dates and time series": "#2e5c8a",
    "GroupBy": "#5a4b8a",
    "Joins and merging": "#8a3b3b",
    "Purity, performance, validation": "#2f6b5f",
    "Reshaping and MultiIndex": "#6b5b2e",
    "Top gotchas": "#3b3b3b",
}


def token_style(ttype):
    if ttype in Token.Comment:
        return COMMENT, "italic"
    if ttype in Token.String:
        return STRING, "normal"
    if ttype in Token.Number:
        return NUMBER, "normal"
    if ttype in Token.Keyword:
        return KEYWORD, "normal"
    if ttype in Token.Name.Builtin:
        return FUNCTION, "normal"
    if ttype in Token.Name.Function:
        return FUNCTION, "normal"
    if ttype in Token.Name.Attribute:
        return FUNCTION, "normal"
    if ttype in Token.Operator:
        return OPERATOR, "normal"
    if ttype in Token.Punctuation:
        return PUNCT, "normal"
    return CODE, "normal"


def token_spans(text):
    spans = []
    for ttype, value in lex(text, PythonLexer()):
        if not value or value == "\n":
            continue
        color, style = token_style(ttype)
        if spans and spans[-1][1:] == (color, style):
            spans[-1] = (spans[-1][0] + value, color, style)
        else:
            spans.append((value, color, style))
    return spans


def text_width_chars(width, size, factor=0.52):
    return max(18, int(width / (size * factor / 72)))


def code_width_chars(width):
    return max(18, int(width / (CODE_SIZE * 0.6018 / 72)))


def wrap_plain(text, width):
    return textwrap.wrap(
        text,
        width=width,
        break_long_words=False,
        break_on_hyphens=False,
        replace_whitespace=False,
        drop_whitespace=True,
    ) or [""]


class Block:
    def height(self, width):
        raise NotImplementedError

    def draw(self, ax, x, y, width):
        raise NotImplementedError


@dataclass
class CodeBlock(Block):
    lines: list[str]
    highlight: bool = True

    def wrapped(self, width):
        chars = code_width_chars(width)
        out = []
        for line in self.lines:
            if len(line) <= chars:
                out.append(line)
            else:
                out.extend(
                    textwrap.wrap(
                        line,
                        width=chars,
                        subsequent_indent="  ",
                        break_long_words=False,
                        break_on_hyphens=False,
                        replace_whitespace=False,
                        drop_whitespace=False,
                    )
                )
        return out

    def height(self, width):
        return len(self.wrapped(width)) * CODE_LH + BLOCK_GAP

    def draw(self, ax, x, y, width):
        char_w = CODE_SIZE * 0.6018 / 72
        for line in self.wrapped(width):
            if self.highlight:
                cursor = x
                for text, color, style in token_spans(line):
                    ax.text(
                        cursor,
                        y,
                        text,
                        family="DejaVu Sans Mono",
                        fontsize=CODE_SIZE,
                        color=color,
                        style=style,
                        va="top",
                        ha="left",
                    )
                    cursor += len(text) * char_w
            else:
                ax.text(
                    x,
                    y,
                    line,
                    family="DejaVu Sans Mono",
                    fontsize=CODE_SIZE,
                    color=CODE,
                    va="top",
                    ha="left",
                )
            y -= CODE_LH
        return y - BLOCK_GAP


@dataclass
class BulletBlock(Block):
    items: list[str]

    def height(self, width):
        chars = text_width_chars(width - 0.10, TEXT_SIZE)
        total = 0.0
        for item in self.items:
            total += len(wrap_plain(item, chars)) * TEXT_LH
            total += 0.018
        return total + BLOCK_GAP

    def draw(self, ax, x, y, width):
        chars = text_width_chars(width - 0.10, TEXT_SIZE)
        for item in self.items:
            lines = wrap_plain(item, chars)
            ax.text(
                x,
                y,
                "-",
                family="DejaVu Sans",
                fontsize=TEXT_SIZE,
                color=BLUE,
                va="top",
                ha="left",
            )
            for line in lines:
                ax.text(
                    x + 0.095,
                    y,
                    line,
                    family="DejaVu Sans",
                    fontsize=TEXT_SIZE,
                    color=CODE,
                    va="top",
                    ha="left",
                )
                y -= TEXT_LH
            y -= 0.018
        return y - BLOCK_GAP


@dataclass
class ParamBlock(Block):
    params: list[tuple[str, str]]

    def rows(self, width):
        chars = code_width_chars(width)
        rows = []
        for name, desc in self.params:
            name_w = len(name) * CODE_SIZE * 0.60 / 72 + 0.045
            desc_w = max(0.6, width - name_w)
            desc_chars = text_width_chars(desc_w, TEXT_SIZE)
            desc_lines = wrap_plain(desc, desc_chars)
            rows.append((name, name_w, desc_lines))
        return rows

    def height(self, width):
        total = 0.0
        for _, _, desc_lines in self.rows(width):
            total += len(desc_lines) * TEXT_LH
            total += 0.026
        return total + BLOCK_GAP

    def draw(self, ax, x, y, width):
        for name, name_w, desc_lines in self.rows(width):
            ax.text(
                x,
                y,
                name,
                family="DejaVu Sans Mono",
                fontsize=CODE_SIZE,
                color=BLUE,
                weight="bold",
                va="top",
                ha="left",
            )
            for idx, line in enumerate(desc_lines):
                ax.text(
                    x + name_w,
                    y,
                    line,
                    family="DejaVu Sans",
                    fontsize=TEXT_SIZE,
                    color=CODE,
                    va="top",
                    ha="left",
                )
                y -= TEXT_LH
            y -= 0.026
        return y - BLOCK_GAP


@dataclass
class SubheadBlock(Block):
    text: str

    def height(self, width):
        return SUBHEAD_SIZE * 1.20 / 72 + 0.035

    def draw(self, ax, x, y, width):
        ax.text(
            x,
            y,
            self.text,
            family="DejaVu Sans",
            fontsize=SUBHEAD_SIZE,
            color=BLUE,
            weight="bold",
            va="top",
            ha="left",
        )
        return y - self.height(width)


@dataclass
class NoteBlock(Block):
    text: str

    def height(self, width):
        chars = text_width_chars(width - 0.14, TEXT_SIZE)
        return len(wrap_plain(self.text, chars)) * TEXT_LH + 0.08

    def draw(self, ax, x, y, width):
        h = self.height(width)
        rect = Rectangle(
            (x, y - h),
            width,
            h,
            facecolor=LIGHT,
            edgecolor=RULE,
            linewidth=0.4,
        )
        ax.add_patch(rect)
        chars = text_width_chars(width - 0.14, TEXT_SIZE)
        yy = y - 0.035
        for line in wrap_plain(self.text, chars):
            ax.text(
                x + 0.06,
                yy,
                line,
                family="DejaVu Sans",
                fontsize=TEXT_SIZE,
                color=CODE,
                va="top",
                ha="left",
            )
            yy -= TEXT_LH
        return y - h - BLOCK_GAP


@dataclass
class Section:
    title: str
    blocks: list[Block]

    def height(self, width):
        return SECTION_H + 0.045 + sum(b.height(width) for b in self.blocks)

    def draw(self, ax, x, y, width):
        rect = Rectangle(
            (x, y - SECTION_H),
            width,
            SECTION_H,
            facecolor=SECTION_COLORS.get(self.title, NAVY),
            edgecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x + 0.055,
            y - SECTION_H / 2,
            self.title,
            family="DejaVu Sans",
            fontsize=SECTION_SIZE,
            color="white",
            weight="bold",
            va="center",
            ha="left",
        )
        y -= SECTION_H + 0.045
        for block in self.blocks:
            y = block.draw(ax, x, y, width)
            y -= 0.015
        return y


def code(*lines, highlight=True):
    return CodeBlock(list(lines), highlight=highlight)


def bullets(*items):
    return BulletBlock(list(items))


def params(*items):
    return ParamBlock(list(items))


def sub(text):
    return SubheadBlock(text)


def note(text):
    return NoteBlock(text)


SECTIONS = [
    Section(
        "Setup and IO",
        [
            code(
                "import pandas as pd",
                "import numpy as np",
                "pd.options.mode.copy_on_write = True",
            ),
            code(
                "pd.read_csv(filepath, sep=\",\", header=\"infer\",",
                "  names=None, usecols=None, dtype=None,",
                "  parse_dates=None, na_values=None,",
                "  nrows=None, chunksize=None, encoding=None)",
            ),
            params(
                ("dtype", "dict or type; use \"string\" for IDs."),
                ("parse_dates", "column(s) or list to parse."),
                ("na_values", "extra strings to treat as missing."),
                ("usecols/nrows", "reduce memory when exploring."),
                ("chunksize", "iterate large files in batches."),
            ),
            code(
                "df.to_csv(path, index=False)",
                "pd.read_parquet(path)",
                "df.to_parquet(path, index=False)",
            ),
            code(
                "df.info(); df.head(n); df.tail(n)",
                "df.shape; df.dtypes; df.columns; df.index",
                "df.describe(include=\"all\")",
                "df.isna().sum(); df.nunique()",
                "df.memory_usage(deep=True).sum()",
            ),
            code(
                "pd.set_option(\"display.max_columns\", 40)",
                "pd.set_option(\"display.width\", 140)",
                "df.info(memory_usage=\"deep\")",
                "df.dtypes.value_counts()",
                "df.select_dtypes(\"number\")",
            ),
        ],
    ),
    Section(
        "Selection and indexing",
        [
            code(
                "df[\"col\"]; df[[\"a\", \"b\"]]",
                "df.loc[row_labels, col_labels]   # inclusive",
                "df.iloc[row_pos, col_pos]        # exclusive",
                "df.at[label, \"col\"]; df.iat[0, 0]",
            ),
            code(
                "mask = (df[\"a\"] > 1) & (df[\"b\"].eq(\"x\"))",
                "df.loc[mask, [\"a\", \"b\"]]",
                "df.query(\"a > 1 and b == 'x'\")",
                "df.loc[df[\"a\"].between(1, 5)]",
                "df.loc[df[\"id\"].isin([\"A\", \"B\"])]",
            ),
            code(
                "df.sort_values(by, ascending=True,",
                "  na_position=\"last\", ignore_index=False,",
                "  key=None)",
                "df.nlargest(n, \"col\"); df.nsmallest(n, \"col\")",
                "df.sample(n, random_state=42)",
                "df.reset_index(drop=True)",
                "df.set_index(\"col\", drop=True,",
                "  verify_integrity=False)",
                "df.reindex(index=None, columns=None,",
                "  fill_value=np.nan, method=None, limit=None)",
                "df.rename(columns={...}, index={...})",
                "df.drop(columns=[...], index=[...],",
                "  errors=\"raise\")",
            ),
            code(
                "df.loc[:, \"a\":\"c\"]      # label slice",
                "df.iloc[:, 0:3]          # position slice",
                "df.filter(items=[\"a\", \"b\"])",
                "df.filter(like=\"price\")",
                "df.filter(regex=\"^order_\")",
            ),
            bullets(
                ".loc is label-inclusive; .iloc is position-exclusive.",
                "Use &, |, ~ and parenthesize every condition.",
                ".copy() for an independent slice; avoid chained assignment.",
            ),
        ],
    ),
    Section(
        "Types and missing values",
        [
            code(
                "df.astype({\"id\": \"string\", \"qty\": \"Int32\",",
                "  \"price\": \"Float64\", \"flag\": \"boolean\",",
                "  \"cat\": \"category\"}, errors=\"raise\")",
            ),
            note(
                "Nullable dtypes: string, Int8/16/32/64, Float32/64, "
                "boolean, category. object->string, int64->Int32, "
                "float64->Float32."
            ),
            code(
                "pd.to_numeric(arg, errors=\"coerce\", downcast=None)",
                "pd.to_datetime(arg, format=\"mixed\",",
                "  errors=\"coerce\", utc=False, dayfirst=False)",
                "df.isna(); df.notna(); df.isnull()",
                "df.fillna(value, method=None, axis=None,",
                "  limit=None)",
                "df.dropna(axis=0, how=\"any\", thresh=None,",
                "  subset=None)",
                "df.where(cond, other=np.nan)",
                "df.mask(cond, other=np.nan)",
                "df.clip(lower=None, upper=None)",
                "df.replace(to_replace, value=None, regex=False)",
                "s.map(dict_or_series, na_action=None)",
                "df.duplicated(subset=None, keep=\"first\")",
                "df.drop_duplicates(subset=None, keep=\"first\",",
                "  ignore_index=False)",
                "df.nunique(dropna=True)",
                "df.value_counts(normalize=False, dropna=True,",
                "  sort=True)",
            ),
            code(
                "pd.cut(x, bins, right=True, labels=None,",
                "  include_lowest=False, ordered=True)",
                "pd.qcut(x, q, labels=None, duplicates=\"raise\")",
            ),
            code(
                "df.select_dtypes(\"number\")",
                "df.select_dtypes(exclude=[\"object\", \"string\"])",
                "df.convert_dtypes()",
                "df.infer_objects()",
                "df.ffill(); df.bfill()",
                "df.interpolate(method=\"linear\",",
                "  limit_direction=\"both\")",
                "df.combine_first(other)",
                "df.any(axis=1); df.all(axis=1)",
            ),
            code(
                "df.isna().sum()",
                "df.isna().mean().mul(100).round(2)",
                "df.isna().any(axis=1)",
                "df.dropna(how=\"all\")",
                "df.dropna(thresh=len(df.columns) - 2)",
                "df.fillna({\"a\": 0, \"b\": \"unknown\"})",
            ),
            code(
                "df[\"id\"] = df[\"id\"].astype(\"string\")",
                "df[\"cat\"] = df[\"cat\"].astype(\"category\")",
                "pd.array(values, dtype=\"Int32\")",
                "df[\"flag\"].astype(\"boolean\")",
            ),
            bullets(
                "astype(bool) on NA gives True; use fillna(False).astype(bool).",
                "str.contains(\".\") is regex; use regex=False or r\"\\.\".",
                "pd.Series(values) resets the index; pass index=s.index.",
                "qcut on tied values can fail; rank(method=\"first\") first.",
                "NaN is float; pd.NA is nullable; None becomes NA in string dtypes.",
            ),
        ],
    ),
    Section(
        "Strings",
        [
            code(
                "s.str.strip(); s.str.lower(); s.str.upper()",
                "s.str.title(); s.str.len(); s.str[0]",
                "s.str.replace(pat, repl, n=-1, case=None,",
                "  flags=0, regex=False)",
                "s.str.contains(pat, case=True, flags=0,",
                "  na=None, regex=True)",
                "s.str.match(pat, case=True, flags=0, na=None)",
                "s.str.fullmatch(pat, case=True, flags=0,",
                "  na=None)",
                "s.str.extract(pat, flags=0, expand=True)",
                "s.str.split(pat=None, n=-1, expand=False,",
                "  regex=None)",
                "s.str.findall(pat, flags=0)",
                "s.str.count(pat, flags=0)",
                "s.str.startswith(pat, na=None)",
                "s.str.endswith(pat, na=None)",
                "s.str.pad(width, side=\"left\", fillchar=\" \")",
                "s.str.zfill(width)",
                "s.str.cat(sep=None, na_rep=None)",
                "s.str.get_dummies(sep=\"|\")",
            ),
            code(
                "s.str.extractall(pat)",
                "s.str.get(0); s.str.find(\"x\")",
                "s.str.rfind(\"x\"); s.str.index(\"x\")",
                "s.str.slice(start=0, stop=5)",
                "s.str.wrap(width)",
                "s.str.translate(table)",
                "s.str.normalize(\"NFKD\")",
                "s.str.partition(sep, expand=True)",
                "s.str.removeprefix(\"pre\")",
                "s.str.removesuffix(\".csv\")",
            ),
            bullets(
                "Normalize labels first: .str.strip().str.lower().",
                "Use regex=False for literal $, comma, and dot.",
                "Use na=False to avoid NA in boolean masks.",
            ),
        ],
    ),
    Section(
        "Regex (re module + Series.str)",
        [
            code(
                "re.search(pat, s)      # first anywhere",
                "re.match(pat, s)       # first at start",
                "re.fullmatch(pat, s)   # entire string",
                "re.findall(pat, s)     # list",
                "re.finditer(pat, s)    # iterator",
                "re.sub(pat, repl, s, count=0)",
                "re.split(pat, s, maxsplit=0)",
                "re.compile(pat, flags=0)",
            ),
            code(
                ".    any        ^  start     $  end",
                "\\d   digit      \\w word      \\s whitespace",
                "[abc] [^abc] [a-z]",
                "* 0+   + 1+   ? 0/1   {n} {n,m}",
                "|    or       () group    (?:) noncapture",
                "(?P<name>...) named       \\1 backref",
                "*? lazy",
                "flags: re.I  re.M  re.S",
                "raw strings: r\"\\d+\\.\\d+\"",
            ),
            code(
                "m = re.search(pat, s)",
                "m.group(0); m.group(1)",
                "m.groupdict(); m.groups()",
                "m.span(); m.start(); m.end()",
                "re.escape(literal)",
            ),
            code(
                "r\"^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$\"  # email",
                "r\"^\\d{4}-\\d{2}-\\d{2}$\"     # ISO date",
                "r\"^[A-Z]{2}\\d{4}$\"             # code",
            ),
            bullets(
                "Series.str uses regex by default.",
                "str.extract supports named groups.",
                "str.replace(..., regex=False) for literal text.",
            ),
        ],
    ),
    Section(
        "Dates and time series",
        [
            code(
                "d.dt.year; d.dt.month; d.dt.day",
                "d.dt.dayofweek      # Mon=0",
                "d.dt.weekday; d.dt.day_name()",
                "d.dt.quarter; d.dt.is_month_end",
                "d.dt.normalize(); d.dt.floor(\"D\")",
                "d.dt.ceil(\"D\")",
                "d.dt.to_period(\"M\"); d.dt.to_period(\"Q\")",
                "d.dt.strftime(\"%Y-%m\")",
                "d.dt.is_quarter_start; d.dt.days_in_month",
                "df.rolling(7, on=\"date\", min_periods=1).sum()",
                "df.groupby(pd.Grouper(key=\"date\",",
                "  freq=\"ME\")).sum()",
            ),
            code(
                "pd.date_range(start, end, periods=None,",
                "  freq=\"D\", tz=None, normalize=False,",
                "  inclusive=\"both\")",
                "pd.Timestamp(\"2024-01-31\")",
                "pd.Timedelta(days=7)",
                "df.set_index(\"date\").resample(rule,",
                "  closed=None, label=None,",
                "  origin=\"start_day\", offset=None).agg(...)",
                "df.rolling(window, min_periods=1, center=False,",
                "  closed=None).mean()",
                "df.expanding(min_periods=1).sum()",
                "df.shift(periods=1, freq=None, fill_value=None)",
                "df.diff(periods=1)",
                "df.pct_change(periods=1, fill_method=None)",
                "df.cumsum(); df.cummax()",
                "df.asfreq(\"D\", fill_value=0)",
                "df.reindex(pd.date_range(...), fill_value=0)",
            ),
            bullets(
                "Aliases: D, W, ME, QE, YE, B; older M/Q/Y are deprecated.",
                "Reindex a full calendar before resample; empty days vanish otherwise.",
                "resample creates empty bins; groupby only observed groups.",
            ),
        ],
    ),
    Section(
        "GroupBy",
        [
            code(
                "df.groupby(by, axis=0, level=None,",
                "  as_index=True, sort=True, dropna=True,",
                "  observed=False, group_keys=True)",
            ),
            code(
                "out = df.groupby(\"k\", observed=True).agg(",
                "  total=(\"x\", \"sum\"),",
                "  n=(\"id\", \"nunique\"),",
                "  avg=(\"x\", \"mean\"),",
                "  first=(\"date\", \"min\"))",
            ),
            code(
                "g = df.groupby(\"k\")[\"x\"]",
                "g.transform(\"sum\")      # aligned broadcast",
                "g.transform(lambda s: (s - s.mean()) / s.std())",
                "g.agg([\"sum\", \"mean\"])  # multi-index columns",
                "g.size(); g.count(); g.nunique()",
                "g.first(); g.last(); g.quantile(0.5)",
                "df.groupby(\"k\").filter(lambda x: len(x) > 2)",
                "df.groupby(\"k\", as_index=False).agg(...)",
                "df.groupby(\"k\", dropna=False).size()",
                "pd.Grouper(key=\"date\", freq=\"ME\")",
                "df.groupby(\"k\").cumsum()",
                "df.groupby(\"k\").diff()",
                "df.groupby(\"k\").cumcount()",
                "df.groupby(level=0).sum()",
            ),
            bullets(
                "agg reduces; transform keeps row count; filter keeps groups.",
                "size counts rows; count non-null; nunique distinct.",
                "observed=True for categoricals; dropna=False keeps NaN keys.",
                "Prefer transform over apply; avoid row-wise Python.",
            ),
        ],
    ),
    Section(
        "Joins and merging",
        [
            code(
                "pd.merge(left, right, how=\"inner\", on=None,",
                "  left_on=None, right_on=None,",
                "  left_index=False, right_index=False,",
                "  suffixes=(\"_x\", \"_y\"), validate=None,",
                "  indicator=False, sort=False)",
            ),
            params(
                ("how", "inner | left | right | outer | cross."),
                ("on", "shared key name(s)."),
                ("left_on/right_on", "different key names."),
                ("left_index/right_index", "join on index."),
                ("suffixes", "overlapping non-key columns."),
                ("validate", "one_to_one | one_to_many | many_to_one | many_to_many."),
                ("indicator", "adds _merge: both / left_only / right_only."),
                ("sort", "sort by join keys."),
            ),
            code(
                "df.join(other, on=None, how=\"left\",",
                "  lsuffix=\"\", rsuffix=\"\", validate=None)",
                "pd.concat(objs, axis=0, join=\"outer\",",
                "  ignore_index=False, keys=None,",
                "  verify_integrity=False, sort=False)",
            ),
            code(
                "# anti-join",
                "m = left.merge(right, on=\"id\", how=\"left\",",
                "  validate=\"many_to_one\", indicator=True)",
                "m.loc[m[\"_merge\"].eq(\"left_only\")]",
            ),
            code(
                "pd.merge_asof(left, right, on=None,",
                "  left_on=None, right_on=None, by=None,",
                "  suffixes=(\"_x\", \"_y\"), tolerance=None,",
                "  direction=\"backward\",",
                "  allow_exact_matches=True)",
            ),
            bullets(
                "merge_asof: sort both frames by the as-of key; by partitions per group.",
                "direction: backward | forward | nearest.",
                "Missing keys do not match; indicator surfaces left_only.",
                "Always state cardinality with validate= when known.",
            ),
        ],
    ),
    Section(
        "Reshaping and MultiIndex",
        [
            code(
                "df.pivot(index, columns, values)  # no aggregation",
                "df.pivot_table(values=None, index=None,",
                "  columns=None, aggfunc=\"mean\",",
                "  fill_value=None, margins=False,",
                "  margins_name=\"All\", dropna=True,",
                "  sort=True, observed=False)",
                "df.melt(id_vars=None, value_vars=None,",
                "  var_name=None, value_name=\"value\",",
                "  col_level=None, ignore_index=True)",
                "pd.crosstab(index, columns, values=None,",
                "  aggfunc=None, rownames=None,",
                "  colnames=None, normalize=False,",
                "  margins=False, margins_name=\"All\",",
                "  dropna=True)",
            ),
            code(
                "df.stack(level=-1, dropna=True)",
                "df.unstack(level=-1, fill_value=None)",
                "df.explode(\"col\", ignore_index=False)",
                "pd.get_dummies(data, prefix=None,",
                "  prefix_sep=\"_\", columns=None,",
                "  dummy_na=False, drop_first=False,",
                "  dtype=bool)",
                "df.T",
            ),
            code(
                "df.xs(key, level=None, axis=0,",
                "  drop_level=True)",
                "df.loc[(\"US\", \"Electronics\")]",
                "df.set_index([\"a\", \"b\"]); df.reset_index()",
                "df.swaplevel(); df.droplevel(level)",
                "df.sort_index(level=[...], ascending=[...])",
                "df.rename_axis(index=\"i\", columns=\"c\")",
            ),
            bullets(
                "pivot fails on duplicate index/column pairs; pivot_table aggregates.",
                "melt is inverse of pivot; stack/unstack move index levels.",
                "normalize=\"index\" for row shares; also \"columns\", \"all\", True.",
            ),
        ],
    ),
    Section(
        "Purity, performance, validation",
        [
            code(
                "df.assign(a=..., b=...)",
                "df.pipe(func, *args, **kwargs)",
                "df.loc[mask, \"col\"] = value   # avoid chained",
            ),
            code(
                "np.where(cond, x, y)",
                "np.select([c1, c2], [v1, v2], default=v3)",
                "df[\"new\"] = df[\"a\"] * df[\"b\"]   # vectorized",
            ),
            code(
                "pd.testing.assert_frame_equal(left, right,",
                "  check_dtype=True, check_like=False,",
                "  check_exact=False, rtol=1e-5,",
                "  atol=1e-8)",
                "pd.testing.assert_series_equal(left, right)",
                "df.equals(other)",
            ),
            code(
                "assert list(out.columns) == expected",
                "assert out.index.is_unique",
                "assert out[\"id\"].is_unique",
                "assert np.isclose(out[\"revenue\"].sum(), total)",
                "assert out[\"date\"].notna().all()",
                "assert out[\"qty\"].gt(0).all()",
            ),
            bullets(
                "Avoid iterrows, row-wise apply, inplace=True, chained assignment.",
                "CoW on: pd.options.mode.copy_on_write = True.",
                "Downcast with astype or to_numeric(downcast=...); check memory_usage.",
                "Assert columns, index, dtypes, uniqueness, totals.",
                "pipe for named steps; assign for multiple columns.",
            ),
        ],
    ),
    Section(
        "Top gotchas",
        [
            bullets(
                ".loc inclusive; .iloc exclusive.",
                "Parenthesize masks: (a) & (b), not a & b.",
                "Chained indexing may not assign; use .loc.",
                "astype(bool) on NA -> True.",
                "str.contains(\".\") is regex.",
                "merge can multiply rows; use validate=.",
                "Missing keys do not match; use indicator=True.",
                "groupby drops NaN keys unless dropna=False.",
                "Use observed=True for categoricals.",
                "resample creates empty bins; reindex full calendar.",
                "qcut duplicate edges: rank(method=\"first\").",
                "Series assignment aligns on index.",
                "inplace=True returns None; avoid it.",
                "merge_asof needs sorted keys.",
                "pct_change fill_method is deprecated.",
                "Use regex=False for literal string replace.",
                "Set dtype and na_values in read_csv upfront.",
                "to_datetime(format=\"mixed\") is pandas 2.0+.",
                "Use .loc[mask] for assignment, not df[mask].",
                "concat(axis=1) binds columns; check duplicate names.",
                "Sort before drop_duplicates(keep=\"last\").",
                "After merge, check row count and columns.",
                "Use pd.NA for nullable booleans.",
            ),
        ],
    ),
]


SECTION_ORDER = [
    "Setup and IO",
    "Selection and indexing",
    "Types and missing values",
    "Strings",
    "Regex (re module + Series.str)",
    "Dates and time series",
    "GroupBy",
    "Joins and merging",
    "Purity, performance, validation",
    "Reshaping and MultiIndex",
    "Top gotchas",
]
SECTIONS = sorted(SECTIONS, key=lambda section: SECTION_ORDER.index(section.title))


def make_columns():
    columns = []
    for page in range(N_PAGES):
        top = PAGE_H - (0.90 if page == 0 else 0.47)
        bottom = MARGIN_BOTTOM + 0.18
        for col in range(N_COLS):
            x = MARGIN_X + col * (COL_W + COL_GAP)
            columns.append(
                {
                    "page": page,
                    "col": col,
                    "x": x,
                    "top": top,
                    "bottom": bottom,
                    "width": COL_W,
                }
            )
    return columns


def flow(sections, columns):
    placements = []
    col_idx = 0
    y = columns[0]["top"]
    for section in sections:
        h = section.height(COL_W)
        full_h = columns[col_idx]["top"] - columns[col_idx]["bottom"]
        if h > full_h:
            raise ValueError(f"Section too tall for column: {section.title}")
        while y - h < columns[col_idx]["bottom"]:
            col_idx += 1
            if col_idx >= len(columns):
                raise ValueError(f"Cheatsheet overflows {N_PAGES} pages")
            y = columns[col_idx]["top"]
        placements.append((col_idx, section, y))
        y -= h + SECTION_GAP
    return placements


def draw_header(ax, page):
    if page == 0:
        ax.text(
            MARGIN_X,
            PAGE_H - 0.35,
            "Pandas Data-Wrangling Cheatsheet",
            family="DejaVu Sans",
            fontsize=TITLE_SIZE,
            color=NAVY,
            weight="bold",
            va="top",
            ha="left",
        )
        ax.text(
            MARGIN_X,
            PAGE_H - 0.58,
            "Jane Street Data Engineer interview prep  |  A4  |  three pages",
            family="DejaVu Sans",
            fontsize=SUBTITLE_SIZE,
            color=MUTED,
            va="top",
            ha="left",
        )
        legend = [
            ("keyword", KEYWORD),
            ("string", STRING),
            ("comment", COMMENT),
            ("number", NUMBER),
            ("function", FUNCTION),
        ]
        lx = PAGE_W - MARGIN_X
        for label, color in reversed(legend):
            ax.text(
                lx,
                PAGE_H - 0.58,
                label,
                family="DejaVu Sans Mono",
                fontsize=7.4,
                color=color,
                va="top",
                ha="right",
            )
            lx -= len(label) * 7.4 * 0.6018 / 72 + 0.10
        ax.plot(
            [MARGIN_X, PAGE_W - MARGIN_X],
            [PAGE_H - 0.73, PAGE_H - 0.73],
            color=NAVY,
            linewidth=1.0,
        )
    else:
        ax.text(
            MARGIN_X,
            PAGE_H - 0.27,
            "Pandas Data-Wrangling Cheatsheet  (continued)",
            family="DejaVu Sans",
            fontsize=10.5,
            color=NAVY,
            weight="bold",
            va="top",
            ha="left",
        )
        ax.plot(
            [MARGIN_X, PAGE_W - MARGIN_X],
            [PAGE_H - 0.38, PAGE_H - 0.38],
            color=NAVY,
            linewidth=0.8,
        )


def draw_footer(ax, page):
    y = 0.19
    ax.plot(
        [MARGIN_X, PAGE_W - MARGIN_X],
        [y + 0.07, y + 0.07],
        color=RULE,
        linewidth=0.4,
    )
    ax.text(
        MARGIN_X,
        y,
        "Pandas cheatsheet  |  Jane Street DE prep",
        family="DejaVu Sans",
        fontsize=FOOTER_SIZE,
        color=MUTED,
        va="top",
        ha="left",
    )
    ax.text(
        PAGE_W - MARGIN_X,
        y,
        f"page {page + 1} / {N_PAGES}",
        family="DejaVu Sans",
        fontsize=FOOTER_SIZE,
        color=MUTED,
        va="top",
        ha="right",
    )


def draw_column_rules(ax):
    for col in range(1, N_COLS):
        x = MARGIN_X + col * (COL_W + COL_GAP) - COL_GAP / 2
        ax.plot(
            [x, x],
            [MARGIN_BOTTOM + 0.25, PAGE_H - 0.82],
            color=RULE,
            linewidth=0.35,
        )


def main():
    out_dir = Path(__file__).resolve().parent
    pdf_path = out_dir / "PANDAS_CHEATSHEET.pdf"
    preview_dir = Path("/tmp/opencode/pandas_cheatsheet_preview")
    preview_dir.mkdir(parents=True, exist_ok=True)

    columns = make_columns()
    placements = flow(SECTIONS, columns)

    per_page = {page: [] for page in range(N_PAGES)}
    for col_idx, section, y in placements:
        per_page[columns[col_idx]["page"]].append((col_idx, section, y))

    with PdfPages(pdf_path) as pdf:
        pdf.infodict().update(
            {
                "Title": "Pandas Data-Wrangling Cheatsheet",
                "Author": "OpenCode",
                "Subject": "Pandas reference for Jane Street Data Engineer interview prep",
                "Keywords": "pandas, python, data engineering, cheatsheet",
            }
        )
        for page in range(N_PAGES):
            fig = plt.figure(figsize=(PAGE_W, PAGE_H))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, PAGE_W)
            ax.set_ylim(0, PAGE_H)
            ax.axis("off")
            draw_header(ax, page)
            draw_footer(ax, page)
            draw_column_rules(ax)

            for col_idx, section, y in sorted(
                per_page[page], key=lambda item: (item[0], -item[2])
            ):
                col = columns[col_idx]
                section.draw(ax, col["x"], y, col["width"])

            pdf.savefig(fig)
            fig.savefig(preview_dir / f"page_{page + 1}.png", dpi=150)
            plt.close(fig)

    print(f"wrote {pdf_path}")
    print(f"pages: {N_PAGES}")
    print(f"previews: {preview_dir}")


if __name__ == "__main__":
    main()
